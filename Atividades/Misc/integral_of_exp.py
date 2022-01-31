#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:27:30 2022

@author: valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

s = 0
n = 10000
a = 2
b = [10**i for i in range(1, 7)]
b.sort()

repeat = 1000

plt.figure(figsize=(7,5))

print(np.exp(-2))

for i in range(len(b)):
    s = []
    
    for j in range(repeat):
        
        samples = np.random.uniform(a, b[i], n)
    
        s.append(np.mean(np.exp(-samples))*(b[i]-a))
    
    print(np.mean(s))
    
    plt.errorbar(b[i], np.mean(s), np.vstack((np.mean(s)-np.min(s),np.max(s)-np.mean(s))), c="k", fmt="o", lw=1, capsize=2)

plt.xlabel("b")
plt.ylabel("Integral")
plt.xscale("log")
plt.yscale("log")
plt.hlines(np.exp(-2), b[0], b[-1], ls="--", color="grey")
plt.show()