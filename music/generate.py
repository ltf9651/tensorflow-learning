"""
用训练好的神经网络模型参数来作曲
"""

import pickle
import tensorflow as tf
import numpy as np

from utils import *
from network import *


def prepare_sequences(notes, pitch_names, num_pitch):
    """为神经网络准备供训练的序列"""
    sequence_length = 100

    #得到所有不同的音调的名字
    pitch_names = sorted(set(item for item in notes))  # set集合元素不允许重复

    # 创建字典, 用于映射音调和整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    #创建神经网络的输入序列和输出序列
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]  # 多个
        sequence_out = notes[i + sequence_length]  # 单个
        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append([pitch_to_int[sequence_out]])

    n_patterns = len(network_input)

    # 将network_input输入序列的形状转成 LSTM 模型可以接受的形式
    normalized_input = np.reshape(network_input,
                                  (n_patterns, sequence_length, 1))

    # 将输入 标准化（归一化）处理
    normalized_input = normalized_input / float(num_pitch)

    return (network_input, normalized_input)


def generate_notes(model, network_input, pitch_names, num_pitch):
    """基于一序列音符用神经网络来生成新的音符"""

    # 从输入里随机选择一个序列作为'预测'生成的音乐的起始点
    start = np.random.randint(0, len(network_input) - 1)

    # 创建字典用于映射 整数 和 音调
    int_to_pitch = dict((num, pitch) for num, pitch in enumerate(pitch_names))

    pattern = network_input[start]

    # 神经网络实际生成的音调
    prediction_output = []

    # 生成 700 个音调/音符
    for not_index in range(700):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # 输入归一化
        prediction_input = prediction_input / float(num_pitch)

        # 用载入了训练所得最佳参数文件的神经网络来 预测/生成 新的音调
        predication = model.predict(prediction_input, verbose=0)

        # argmax 取出参数里最大维度的值
        index = np.argmax(predication)

        # 将 整数 转成音调
        result = int_to_pitch[index]

        prediction_output.append(result)

        # pattern往后移动一位
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# 以之前训练所得的最佳参数来生成音乐
def generate():
    # 加载用于训练神经网络的音乐数据
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # 得到所有音调的名字
    pitch_names = sorted(set(item for item in notes))

    # 得到所有不重复的音调数
    num_pitch = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitch_names, num_pitch)

    # 载入之前训练时得到的最好的参数(最小的 loss)，生成神经网络模型
    model = network_model(normalized_input, num_pitch, 'best-weights.hdf5')

    # 用神经网络来生成音乐数据
    predication = generate_notes(model, network_input, pitch_names, num_pitch)

    # 用预测的音乐数据生成 MIDI 文件，转成MP3
    create_music(predication)