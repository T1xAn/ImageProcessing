# -*- coding: utf-8 -*-
import numpy as np

rnd = np.random.default_rng()

a = rnd.integers(0, 10, (10,10))

b = rnd.integers(0, 10, (10,10))
print("\n-------------Матрица a--------------\n")
print(a)
print("\n-------------Матрица b--------------\n")
print(b)

c = np.where(a == b, a, 100)

print("\n-------------Результат--------------------\n")
print(c)
