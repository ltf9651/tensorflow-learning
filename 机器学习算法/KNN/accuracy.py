# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 08:53:52 2018
分类准确度
@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from KNN_package.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from KNN_package.KNN import KNNClassifier
knn_clf = KNNClassifier(k=3)
knn_clf.fit(X_train, y_train)
predict = knn_clf.predict(X_test)
chance = knn_clf.score(X_test, y_test)

print(chance)

"""scikit-learn"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
chance = accuracy_score(predict, y_test)

chance2 = clf.score(X_test, y_test)
print(chance)
print(chance2)

"""寻找最好的k"""
best_score = 0.0
best_k = -1
for k in range(1, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k
        
print("best_k = ", best_k)
print("best_score = ", best_score)

"""考虑距离?不考虑距离"""
best_method = ""
best_score = 0.0
best_k = -1
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_method = method

print("best_method = ", best_method)
print("best_k = ", best_k)
print("best_score = ", best_score)

"""闵科夫斯基距离  超参数p"""
best_p = -1
best_score = 0.0
best_k = -1
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = p

print("best_p = ", best_p)
print("best_k = ", best_k)
print("best_score = ", best_score)

"""Grid Search 网格搜索超参数"""
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
knn_clf = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=4)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
grid_search.best_score_
grid_search.best_params_
knn_clf = grid_search.best_estimator_
chance = knn_clf.score(X_test, y_test)
print(chance)