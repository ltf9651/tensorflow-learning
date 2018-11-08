# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:55:05 2018
数据归一化
@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt

"""最值归一化"""
x = np.random.randint(0, 100, size=100)
(x - np.min(x))/(np.max(x) - np.min(x))

X = np.random.randint(0, 100, (50,2))
X = np.array(X, dtype=float)

X[:, 0] = (X[:, 0] - np.min(X[:, 0])) / (np.max(X[:,0]) - np.min(X[:,0]))
X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:,1]) - np.min(X[:,1]))
plt.scatter(X[:,0], X[:,1])
plt.show()

"""均值归一化 均值为0，方差为1"""
X[:,0] = (X[:,0] - np.mean(X[:,0])) / np.std(X[:,0])
X[:,1] = (X[:,1] - np.mean(X[:,1])) / np.std(X[:,1])
plt.scatter(X[:,0], X[:,1])
plt.show()

"""Scikitlearn"""
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
standScaler.fit(X_train)
standScaler.scale_
standScaler.mean_
X_train_standard = standScaler.transform(X_train)
X_test_standard = standScaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_standard, y_train)
score = knn_clf.score(X_test_standard, y_test)
print(score)