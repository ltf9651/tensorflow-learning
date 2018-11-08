# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100) #正态分布

X = x.reshape(-1, 1) # 100行 1列

plt.scatter(x, y)
plt.show()

def J(theta,X_b,y):
    try:
        # MSE
        return ((y - X_b.dot(theta)) ** 2) / len(X_b)    
    except:
        return float('inf')
    
def dJ(theta, X_b,y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1, len(theta)):
        res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:,i]))
    return res * 2 /len(X_b)