# -*- coding: utf-8 -*-
#多元线性回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target
X = X[y < 50]
y = y[y < 50]

from LRpackage.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,seed=666)

from LRpackage.LinearRegression import LinearRegression
reg = LinearRegression()
reg.fit_normal(X_train, y_train)
reg.coef_
reg.interception_
score = reg.score(X_test, y_test)
print(score)