import numpy as np
import matplotlib.pyplot as plt

raw_data_X  = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color="r")
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color="g")

x = np.array([8.093607318, 3.365731514])
plt.scatter(x[0], x[1], color="b")
plt.show()

from math import sqrt
distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]

nearest = np.argsort(distances)
k = 6
topk_y = [y_train[n] for n in nearest[:k]]

from collections import Counter
counter = Counter(topk_y)

predict = counter.most_common(1)[0][0]

""" KNN function """
def KNN_function(k, X_train, y_train, x):
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)
    topk_y = [y_train[n] for n in nearest[:k]]
    return Counter(topk_y).most_common(1)[0][0]

predict = KNN_function(6, X_train, y_train, x)
print(predict)

"""
KNN是一个不需要训练过程的算法，没有模型，可认为训练数据集就是模型本身

scikit-learn
"""

from sklearn.neighbors import KNeighborsClassifier

KNN_classifier = KNeighborsClassifier(n_neighbors=6)
KNN_classifier.fit(X_train, y_train)
x_reshape = x.reshape(1, -1)
predict = KNN_classifier.predict(x_reshape)
print(predict[0])

from KNN_package.KNN import KNNClassifier
knn_clf = KNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
predict = knn_clf._predict(x_reshape)
print(predict)