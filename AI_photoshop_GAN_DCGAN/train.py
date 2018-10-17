# -*- coding: UTF-8 -*-
"""
DCGAN 训练
"""

import glob
import numpy as np
from scipy import misc
import tensorflow as tf
from network import *


def train():
    # 获取训练数据
    data = []
    for image in glob.glob('AI_photoshop_GAN_DCGAN/images/*'):
        image_data = misc.imread(image)  # 利用 PIL库读取图片数据
        data.append(image_data)

    input_data = np.array(data)

    # 数据标准化 [-1, 1]范围的取值   tanh激活函数->(-1,1)
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    # 构造 生成器 和 判别器
    g = generator_model()
    d = discriminator_model()

    # 构建 生成器 和 判别器 组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # Adam Optimizer 优化器
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, bera_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, bera_1=BETA_1)