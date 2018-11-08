# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:59:00 2018
KNN算法封装
@author: LTF
"""

import numpy as np
from math import sqrt
from collections import Counter
from KNN_package.accuracy_cal import accuracy_score

class KNNClassifier:
    
    def __init__(self, k):
        """初始化KNN分类器"""
        self.k = k
        self._X_train = None
        self._y_train = None
        
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        
        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]
    
    def score(self, X_test, y_test):
        predict = self.predict(X_test)
        return accuracy_score(y_test, predict)
    
    def __repr__(self):
        return "KNN(k=%d)" % self.k