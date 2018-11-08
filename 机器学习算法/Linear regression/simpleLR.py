# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:19:13 2018
线性回归
@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,3.,2.,3.,5.])

plt.scatter(x,y)

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0
d = 0.0

for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2
    
a = num / d
b = y_mean - a * x_mean

y_hat = a * x + b
plt.plot(x, y_hat, color="r")
plt.show()

x_predict = 6
y_predict = a * x_predict + b
print(y_predict)

from LRpackage.SimpleLinearRegression1 import SimpleLinearRegression1
reg1 = SimpleLinearRegression1()
reg1.fit(x, y)
print(reg1.predict(np.array([x_predict])))

"""向量化计算速度更快"""
from LRpackage.SimpleLinearRegression2 import SimpleLinearRegression2
reg1 = SimpleLinearRegression2()
reg1.fit(x, y)
print(reg1.predict(np.array([x_predict])))