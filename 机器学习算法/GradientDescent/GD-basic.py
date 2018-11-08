# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:52:19 2018

@author: LTF
"""

import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 140)
plot_y = (plot_x - 2.5)**2 - 1
plt.plot(plot_x, plot_y)

"""求导函数"""
def dJ(theta):
    return 2*(theta - 2.5)

def J(theta):
    try:
        return (theta - 2.5)**2 -1
    except:
        return float('inf')

#学习率
eta = 0.1
#精度
"""
epsilon = 1e-8
theta = 0.0
theta_history = [theta]
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    
    if(abs(J(theta) - J(last_theta)) < epsilon):
        break
    
print(theta)
print(J(theta))
print(theta_history)

plt.plot(np.array(theta_history), J(np.array(theta_history)),color="r", marker="+")
plt.show()
"""
def gardient_descent(initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0
    
    while i_iter < n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
        
        i_iter += 1
        
    return theta
    
def plot_theta_history():
    plt.plot(plot_x, J(plot_x), color="b")
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
    plt.show()
    
eta = 0.09
theta_history=[]
gardient_descent(0., eta)
plot_theta_history()
print(len(theta_history))


# 随机题第下降法
np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100) #正态分布

X = x.reshape(-1, 1) # 100行 1列

def dj_sgd(theta,X_b_i,y_i):
    return X_b_i.dot(X_b_i.dot(theta) - y_i) * 2

def sgd(X_b, y, initial_theta, n_iters):
    t0 = 5
    t1 = 50
    def learning_rate(t):
        return t0/(t+t1)
    
    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b))
        gradient = dj_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learning_rate(cur_iter) * gradient
        
    return theta
