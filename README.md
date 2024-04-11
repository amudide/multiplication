# How does a transformer compute $a \cdot b  (\text{mod } c)$?

When $c$ is prime, the algorithm is analogous to the modular addition algorithm described by [Nanda et al.](https://arxiv.org/abs/2301.05217) due to the existence of a primitive root $g$.

When $c$ is the product of primes, the algorithm computes $a \cdot b (\text{mod } p_i)$ using a primitive root $g_i$ for all $i$ and combines the solutions using the Chinese Remainder Theorem (CRT).

In the general case, the algorithm exploits primitive roots, multiplicative order and CRT simultaneously.

<p align="center">
  <img src="https://github.com/amudide/multiplication/blob/main/figure.png" alt="Figure"/>
</p>

