#!/usr/bin/python
import random

dim = 3
n = 1000
k = 2

print(dim, n, k)

for i in range(n):
    for j in range(dim):
        print( random.uniform(0.1,1), end='' )
    print()