import tensorflow as tf
import os
import pickle
import numpy as np

CIFAR_DIR = "CNN_RNN_GAN_pratice/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    """读取文件数据"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1  # 归一化 更快更好地找到误差最小值
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle  # 过完一遍数据之后进行shuffle使数据散乱
        self._indicator = 0

        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)  # 混排
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        # 每次返回 batch_size个样本，遍历完之后进行shuffle(if need shuffle)
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("所有数据已遍历完成，没有新的数据 ")

        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")

        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [
    os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)
]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, need_shuffle=True)
test_data = CifarData(test_filenames, False)

x = tf.placeholder(tf.float32, [None, 3072])  # 32 * 32 * 3 = 3072
y = tf.placeholder(tf.int64, [None])
# w = tf.get_variable(
#     'w', [x.get_shape()[-1], 10],
#     initializer=tf.random_normal_initializer(0, 1))  # shape = 3072 * 10
# b = tf.get_variable('b', [10], initializer=tf.constant_initializer(0.0))
# y_ = tf.matmul(x, w) + b  # shape: (None, 3072) * (3072, 10) = (None, 10)
hidden1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, 50, activation=tf.nn.relu)
y_ = tf.layers.dense(hidden3, 10)

p_y_1 = tf.nn.softmax(y_)  # 多维选用softmax函数  [[0.01,0.4,....,0.04],[..]]
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.losses.sparse_softmax_cross_entropy(
    labels=y, logits=y_)  # y 图片真实类别值， y_图片经过计算得到的内积值

predict = tf.argmax(y_, 1)  # y_ 最大的值作为predict  y_ 二维矩阵 第一位度：样本 第二维度：概率分布
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op],
                                             feed_dict={
                                                 x: batch_data,
                                                 y: batch_labels
                                             })
        if i & 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' %
                  (i + 1, loss_val, accuracy_val))

        # 使用测试数据进行检验
        if i % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(
                    batch_size)
                test_accuracy = sess.run([accuracy],
                                         feed_dict={
                                             x: test_batch_data,
                                             y: test_batch_labels
                                         })
                all_test_acc = []
                all_test_acc.append(test_accuracy)
            test_acc = np.mean(all_test_acc)
            print('[Test ] Step: %d, acc: %4.5f' % (i, test_acc))

sess.close()
