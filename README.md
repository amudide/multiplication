# How does a transformer compute a*b (mod c)?

When c is prime, the algorithm is analogous to the modular addition algorithm described by https://arxiv.org/abs/2301.05217 due to the existence of primitive roots.

When c is the product of primes, the algorithm computes a*b (mod p_i) for all i and combines the solutions using the Chinese Remainder Theorem (CRT).

In the general case, the algorithm exploits primitive roots, multiplicative order and the Chinese Remainder Theorem (CRT) simultaneously.

<p align="center">
  <img src="https://github.com/amudide/multiplication/figure.png?raw=true" alt="Figure"/>
</p>

