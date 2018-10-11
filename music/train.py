"""
训练神经网络，将参数（Weight）存入 HDF5 文件
"""

import numpy as np
import tensorflow as tf

from utils import *
from network import *

"""
==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代

# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch
"""
