# modified from https://github.com/ejmichaud/neural-verification

from typing import List, Dict, Union
import math
import random
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

@dataclass
class MLPConfig:
    in_dim: int = 2
    out_dim: int = 1
    width: int = 40
    depth: int = 2 # note: depth is the #(linear layers), and #(hidden layers) = #(linear layers) - 1.
    activation: type = nn.SiLU
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        shp = [config.in_dim] + [config.width]*(config.depth-1) + [config.out_dim]
        layers = []
        for i in range(config.depth):
            layers.append(nn.Linear(shp[i], shp[i+1]))
            if i < config.depth - 1:
                layers.append(config.activation())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
        # input shape = (batch_size, input_dim)
        # define activation here
        #f = lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        # f = torch.nn.SiLU()
        # for i in range(self.depth-1):
        #     x = f(self.linears[i](x))
        # x = self.linears[-1](x)
        # # output shape = (batch_size, output_dim)
        # return x
    
@dataclass
class RNNConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_dim: int = 40
    
    
class RNN(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.Wh = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.Wx = nn.Linear(config.input_dim, config.hidden_dim)
        self.Wy = nn.Linear(config.hidden_dim, config.output_dim)
        self.act = nn.Sigmoid()
        #self.act = lambda x: x
        self.device = device
    
    def forward(self, x):
        
        # x shape: (batch size, sequence length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        outs = []

        for i in range(seq_length):
            hidden = self.act(self.Wh(hidden) + self.Wx(x[:,i,:]))
            out = self.Wy(hidden)
            outs.append(out)
            
        # out shape: (batch size, sequence length, output_dim)
        return torch.stack(outs).permute(1,0,2)


# let's make a more general RNN where we can have an arbitrarily-complex MLP
# as the hidden state transition function and the output function
@dataclass
class GeneralRNNConfig:
    input_dim: int = 2
    output_dim: int = 1
    hidden_dim: int = 40
    hidden_mlp_depth: int = 2 # this would be 1 hidden layer
    hidden_mlp_width: int = 100
    output_mlp_depth: int = 2 # this would be 1 hidden layer
    output_mlp_width: int = 100
    activation: type = nn.SiLU

class GeneralRNN(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        hmlp_config = MLPConfig(
            config.hidden_dim + config.input_dim, 
            config.hidden_dim, 
            config.hidden_mlp_width, 
            config.hidden_mlp_depth, 
            config.activation
        )
        self.hmlp = MLP(hmlp_config).to(device)
        ymlp_config = MLPConfig(
            config.hidden_dim, 
            config.output_dim, 
            config.output_mlp_width, 
            config.output_mlp_depth, 
            config.activation
        )
        self.ymlp = MLP(ymlp_config).to(device)
        self.device = device

    def forward(self, x, h=None):
        """The transition is given by:
            h_t = f([h_{t-1}, x_t])
            y_t = g(h_t)
        where f and g are MLPs.

        This function takes in the input and the hidden state and 
        returns an output and a hidden state.
        """
        # x shape: (batch_size, input_dim)
        # h shape: (batch_size, hidden_dim)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        else:
            assert h.size(0) == x.size(0)
            assert h.size(1) == self.hidden_dim
            assert h.device == self.device
        assert x.size(1) == self.config.input_dim
        assert x.device == self.device
        hx = torch.cat((h, x), dim=1)
        h = self.hmlp(hx)
        y = self.ymlp(h)
        return y, h
    
    def forward_sequence(self, x):
        """This function takes in a sequence of inputs and returns a sequence of outputs
        as well as the final hidden state."""
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)
        hiddens = []
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        hiddens.append(hidden.data)
        assert x.size(2) == self.config.input_dim
        assert x.device == self.device
        outs = []
        for i in range(seq_length):
            out, hidden = self.forward(x[:,i,:], hidden)
            hiddens.append(hidden.data)
            outs.append(out)
        # out shape: (batch_size, sequence_length, output_dim)
        return torch.stack(outs).permute(1,0,2), hiddens