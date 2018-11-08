# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 09:05:56 2018

@author: LTF
"""

import numpy as np

def accuracy_score(data, predict):
    return sum(predict == data)/len(data)