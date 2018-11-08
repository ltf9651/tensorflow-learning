# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 23:53:58 2018

@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
iris.keys()

X = iris.data
y = iris.target

from KNN_package.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from KNN_package.KNN import KNNClassifier

my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
predict = my_knn_clf.predict(X_test)

"""预测准确率"""
chance = sum(predict == y_test)/len(y_test)
print(chance)

"""scikit learn 的split"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)