# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

points_num = 100;
vector = [];

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vector.append([x1, y1])

x_data = [v[0] for v in vector]
y_data = [v[1] for v in vector]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20):
    sess.run(train)

print(sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'r*', label="points")
plt.legend()
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.show()

sess.close()