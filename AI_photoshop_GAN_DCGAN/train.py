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
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # Sequential.compile() 配置生成器和判别器
    g.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    # 开始训练
    for epoch in range(EPOCHS):
        # 每经过 BATCH_SIZE 整数倍训练一次
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

            # 连续型均匀分布的随机数据（噪声）, 生成网络可以用噪声生成图片
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))  #连续型均匀分布

            # 生成器生成的图片数据
            generated_images = g.predict(random_data, verbose=0)
            input_batch = np.concatenate((input_batch, generated_images)) # 首尾相连
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE  # 1 通过判别器检测   0 不通过

            # 训练判别器，让其具备不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch, output_batch)
            d.trainable = False # 当训练 生成器 时，让 判别器 不可被训练

            # 再次重新生成随机数据（噪声），如果两次 random_data 一样的话不够随机化
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # 训练 生成器，并通过不可被训练的 判别器 去判别
            g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)

            # 训练完生成器后回复判别器可训练
            d.trainable = True

            # 打印损失
            print("Epoch {}, 第 {} 步, 生成器的损失: {:.3f}, 判别器的损失: {:.3f}".format(epoch, index, g_loss, d_loss))

            # Epoch 每轮训练完成后保存 生成器 和 判别器 的参数
            if epoch % 10 == 9:
                g.save_weights("generator_weight", True)
                d.save_weights("discriminator_weight", True)


if __name__ == "__main__":
    train()