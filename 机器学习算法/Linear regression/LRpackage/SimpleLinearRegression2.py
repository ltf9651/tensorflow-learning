# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import r2_score
class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
            
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        
        return self
    
    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    
    def _repr_(self):
        return "SimpleLinearRegression1()"# -*- coding: utf-8 -*-

