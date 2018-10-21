# -*- coding: UTF-8 -*-
"""
用 DCGAN 的生成器模型和训练得到的参数文件生成图片
"""
import tensorflow as tf
from PIL import Image
import numpy as np

from network import *


def generate():
    # 配置生成器并加载训练好的参数
    g = generator_model()
    g.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))
    g.load_weights("generator_weight")

    # 连续均匀分布随即数据（噪声）
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    # 用随机数据作为输入，生成图片数据
    images = g.predict(random_data, verbose=1)

    # 用生成的图片数据生成真正的图片(PNG)
    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save('image-%s.png' % i)


if __name__ == "__main__":
    generate()