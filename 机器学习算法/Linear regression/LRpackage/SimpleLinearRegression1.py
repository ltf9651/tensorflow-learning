# -*- coding: utf-8 -*-

import numpy as np
class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        num = 0.0
        d = 0.0
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
            
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        return self
    
    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def _repr_(self):
        return "SimpleLinearRegression1()"