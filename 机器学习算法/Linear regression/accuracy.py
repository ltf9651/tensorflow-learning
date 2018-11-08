# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:17:26 2018
衡量标准
@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
# 第五列作为特征参数
x = boston.data[:, 5]
y = boston.target

x = x[y < 50]
y = y[y < 50]

plt.scatter(x, y)
plt.show()

from LRpackage.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,seed=666)

from LRpackage.SimpleLinearRegression2 import SimpleLinearRegression2
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)
reg.score(x_test, y_test)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color="r")
plt.show()

y_predict = reg.predict(x_test)
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)

from math import sqrt
rmse_test = sqrt(mse_test)
mae_test = np.sum(np.absolute(y_predict - y_test))/len(y_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

R = 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)
from sklearn.metrics import r2_score
R = r2_score(y_test, y_predict)
print(R)



"""多元"""
x = boston.data
y = boston.target
x = x[y < 50]
y = y[y < 50]
from LRpackage.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,seed=666)

""" LR in scikit"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
score = lin_reg.score(x_test, y_test)
print(score)

""" LR in KNN """
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(x_train, y_train)
score = knn_reg.score(x_test, y_test)
print(score)
"""
from sklearn.model_selection import GridSearchCV
param_grid  = [
    {
        'weights': ['uniform'], 
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)], 
        'p': [i for i in range(1, 6)]
    }
]
grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)
score = grid_search.best_estimator_.score(x_test, y_test)
print(score)"""

"""可解释性"""
lin_reg.fit(x ,y)
# 对特征所占权重由低到高排序(正相关)   强解释性
print(boston.feature_names[np.argsort(lin_reg.coef_)])