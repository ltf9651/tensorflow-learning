# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:56:44 2018
均值归一化
@author: LTF
"""

import numpy as np

class StandardScale:
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """获取均值和方差"""
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])    
        return self
    
    def transform(self, X):
        """将X进行均值方差归一化处理"""
        resX = np.empty(shape = X.shape, dtype = float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX