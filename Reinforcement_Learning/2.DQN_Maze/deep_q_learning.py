# -*- coding: UTF-8 -*-
"""
Deep Q Learning 算法  决策部分
"""

import numpy as np
import tensorflow as tf

# 伪随机数。为了复现结果
np.random.seed(1)
tf.set_random_seed(1)


class DeepQLearning:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 discount_factor=0.9,
                 e_greedy=0.1,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 output_graph=False):
        self.n_actions = n_actions  # action 数目
        self.n_features = n_features  # state/observation 里的特征数目
        self.lr = learning_rate
        self.gamma = discount_factor  # 折扣因子
        self.epsilon = e_greedy  # 贪婪度 Epsilon Greddy
        self.replace_target_iter = replace_target_iter  # 每隔 %s 步更新神经网络参数，将 Q_eval的值赋予 Q_target
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 随机选取记忆片段的大小

        # 学习次数 (用于判断是否更换 Q_target_net 参数)
        self.learning_steps = 0

        # 初始化全 0 记忆[s, a, r, s_]
        self.memory = np.zeros([self.memory_size, n_features * 2 + 2])

        # 构建神经网络
        self.construct_network()

        # 提取 Q_target_net 和 Q_eval_net 的参数
        t_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_eval_net')

        # 用 Q_eval_net 参数来替换 Q_target_net 参数
        with tf.variable_scope('target_replacement'):
            self.target_replace_op = [
                tf.assign(t, e) for t, e in zip(t_params, e_params)
            ]

        self.sess = tf.Session()

        if output_graph:
            # 输出 TensorBoard 日志文件
            tf.summary.FileWriter("logs", self.sess.graph)

        # 初始化全局变量
        self.sess.run(tf.global_variables_initializer())

    '''
    构建两个神经网络（Q_eval_net 和 Q_target_net）。
    固定住一个神经网络 (Q_target_net) 的参数（所谓 Fixed Q target）。
    Q_target_net 相当于 Q_eval_net 的一个历史版本, 拥有 Q_eval_net 之前的一组参数。
    这组参数被固定一段时间, 然后再被 Q_eval_net 的新参数所替换。
    Q_eval_net 的参数是不断在被提升的
    '''

    def construct_network(self):
        # 输入数据[s, a, r, s_]
        with tf.variable_scope('input'):
            self.s = tf.placeholder(
                tf.float32, [None, self.n_features], name='s')  # State
            self.a = tf.placeholder(
                tf.int32, [
                    None,
                ], name='a')  # Action
            self.r = tf.placeholder(
                tf.float32, [
                    None,
                ], name='r')  # Reward
            self.s_ = tf.placeholder(
                tf.float32, [None, self.n_features], name='s_')  # 下一个 State

        # 权重和偏差
        w_initializer, b_initializer = tf.random_normal_initializer(
            0, 0.3), tf.constant_initializer(0.1)

        # 创建 Q_eval 神经网络, 适时更新参数
        with tf.variable_scope('Q_eval_net'):
            e1 = tf.layers.dense(
                self.s,
                20,
                tf.nn.relu,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name="e1")  # eval_net第一层(全连接层)
            self.q_eval = tf.layers.dense(
                e1,
                self.n_actions,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name="e2")

        # 创建 Q_target 神经网络, 提供 target Q
        with tf.variable_scope('Q_target_net'):
            t1 = tf.layers.dense(
                self.s_,
                20,
                tf.nn.relu,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='t1')
            self.q_next = tf.layers.dense(
                t1,
                self.n_actions,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='t2')

        # 在 Q_target_net 中，计算下一个状态 s_j_next 的真实 Q 值
        with tf.variable_scope('Q_target'):
            q_target = self.r + self.gamma * tf.reduce_mean(
                self.q_next, axis=1)
            # tf.stop_gradient 使 q_target 不参与梯度计算的操作（ q_target 在 q_target_net中.让Q_eval_net的参数加入计算）
            self.q_target = tf.stop_gradient(q_target)

        # 在 Q_eval_net 中，计算状态 s_j 的估计 Q 值
        with tf.variable_scope('Q_eval'):
            a_indices = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                axis=1)
            # tf.gather_nd 用 indices 定义的形状来对 params 进行切片
            self.q_eval_by_a = tf.gather_nd(
                params=self.q_eval, indices=a_indices)

        # 计算真实值和估计值的误差（loss）
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(
                    self.q_target, self.q_eval_by_a, name='error'))

        # 梯度下降法优化参数
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(
                self.loss)
