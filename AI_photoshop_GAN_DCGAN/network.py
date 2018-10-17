# -*- coding: UTF-8 -*-
"""
DCGAN 深层卷积生成对抗网络
"""

import tensorflow as tf

# Hyperparameters 超参数
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5


def discriminator_model():
    """判别器模型"""
    model = tf.keras.models.Sequential()

    # 添加第一个卷积层
    model.add(
        tf.keras.layers.Conv2D(
            64,  # 64个 filters（过滤器）, 输出深度为64
            (5, 5),  # 过滤器在二维的大小（5 * 5）
            padding='same',  # 输出的大小不变，需要在外围补零 2 圈
            input_shape=(64, 64, 3),  # 输入形状 [64, 64, 3]。3 表示 RGB 三原色
        ))
    # 第二层：激活函数层
    model.add(tf.keras.layers.Activation('tanh'))
    # 第三层：池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 扁平化
    model.add(tf.keras.layers.Flatten())
    # 全连接层
    model.add(tf.keras.layers.Dense(1024))  # 1024个神经元
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model


def generator_model():
    """生成器模型，从随机数来生成图片"""
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(
        input_dim=100, units=1024))  # 输入的维度是 100, 输出维度（神经元个数）是1024 的全连接层
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(128 * 8 * 8, ))
    model.add(tf.keras.layers.BatchNormalization())  # 批标准化
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(
        tf.keras.layers.Reshape((8, 8, 128),
                                input_shape=(128 * 8 * 8, )))  # 变成 8 * 8像素图片
    model.add(tf.keras.layers.UpSampling2D(size=2))  # 反亚采样， 像素变成 16 * 16
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=2))  # 像素变成 32 * 32
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=2))  # 像素变成 64 * 64
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding='same'))  # RGB 3 输出深度
    model.add(tf.keras.layers.Activation('tanh'))

    return model


# 构造一个 Sequential 对象，包含一个 生成器 和一个 判别器
# 输入 -> 生成器 -> 判别器 -> 输出
def generator_containing_discriminator(generator, discriminator):
    """
    整合 G 和 D, 用来训练生成器（数据通过判别器判别
    """
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时 判别器 不可被训练
    model.add(discriminator)
    return model