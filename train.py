import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm
import random
from torch.utils.data import DataLoader
from functools import *
from models import get_model_dataset
import os

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def run_experiment(model, full_dataset, config):

    train_size = int(config['frac'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    batch_size = config.get('batch_size', len(full_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    steps = 10001
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4,foreach=False)
    init_embed = model.embedding.cpu().detach().numpy()
    first_embeds = [init_embed]
    embeddings = [init_embed]
    final_embed = None

    for step in tqdm(range(steps)):
        
        CEL = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            test_loss = CEL(pred, labels)
            test_acc = torch.mean((torch.argmax(pred, dim=1) == labels).float())
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            loss = CEL(pred, labels)
            acc = torch.mean((torch.argmax(pred, dim=1) == labels).float())
            train_losses.append(loss.item())
            train_accs.append(acc.item())

            loss.backward()
            optimizer.step()

        if (step % 500 == 0):
            print("step = %d | train loss: %.2e | test loss %.2e | train acc: %.2e | test acc: %.2e "%(step, loss.cpu().detach().numpy(), test_loss.cpu().detach().numpy(), acc.cpu().detach().numpy(), test_acc.cpu().detach().numpy()))
        
        if step < 1000: 
            first_embeds.append(model.embed.W_E.cpu().detach().numpy())
        if step == steps - 1:
            final_embed = model.embed.W_E.cpu().detach().numpy()
            
        if step % 100 == 0:
            embeddings.append(model.embed.W_E.cpu().detach().numpy())
            
    returns = {
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'init_embed': init_embed,
        'final_embed': final_embed, 
        'first_embeds': first_embeds, 
        'embeddings': embeddings,
    }
    return returns

import string

while True:
    letters_and_numbers = string.ascii_lowercase + string.digits.replace('0', '')
    run_name = ''.join(random.choices(letters_and_numbers, k=10))
    print(run_name)
    C = 143
    n_layers = 1
    if random.randint(0,3):
        n_layers = random.randint(1,4)
    frac_coeff = random.uniform(0,1)
    diff_vocab = 0
    eqn_sign = 0
    if random.randint(0, 4)==0:
        diff_vocab = random.randint(0,1)
        eqn_sign = random.randint(0,1)
    d_model = 128
    if random.randint(0,2) == 0:
        d_model = int(2 ** random.uniform(5,9))
    print(f'd={d_model}')
    config = dict(
        name='modadd_'+str(C),
        funcs=f'lambda x: int(x[0]) * int(x[1]) % {C})',
        C=C,
        n_heads=4,
        d_model=d_model,
        n_layers=n_layers,
        attention_dir='casual',
        act_fn='GeLU' if random.randint(0,3)==0 else 'ReLU',
        epoch=20000,
        batch_size=C*C,
        lr=1e-3,
        weight_decay=2.,
        frac=0.8,
        attn_coeff=frac_coeff,
        runid=run_name,
        diff_vocab=diff_vocab,
        eqn_sign=eqn_sign,
    )

    model, full_dataset = get_model_dataset(config)

    returns = run_experiment(model, full_dataset, config)
    data_path = f"multiplication/data/modmul143/{run_name}.pt"

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.save(os.path.join(data_path, 'embeddings.npy'), returns['embeddings'])
    np.save(os.path.join(data_path, 'first_embeds.npy'), returns['first_embeds'])
    np.save(os.path.join(data_path, 'final_embed.npy'), returns['final_embed'])
    np.save(os.path.join(data_path, 'train_acc.npy'), returns['train_accs'])
    np.save(os.path.join(data_path, 'test_acc.npy'), returns['test_accs'])
    np.save(os.path.join(data_path, 'train_loss.npy'), returns['train_losses'])
    np.save(os.path.join(data_path, 'test_loss.npy'), returns['test_losses'])
    torch.save(model.state_dict(), os.path.join(data_path, 'model.pt'))

    
