import tensorflow as tf
import numpy as np
"""
RNN-LSTM 循环神经网络
"""


#num_pitch 音符数目
def network_model(inputs, num_pitch, weights_file=None):
    #添加汉堡层
    model = tf.keras.models.Sequential()

    #第一层
    model.add(
        tf.keras.layers.LSTM(
            512,  #神经元数目,输出维度
            input_shape=(inputs.shape[1],
                         inputs.shape[2]),  #输入的形状,对第一个LSTM层必须设置
            return_sequences=True,  #返回所有序列的输出 False->最后一个输出,堆叠LSTM必须设置，最后一层无需设置
        ))

    #第二层dropout，丢弃30%，防止过拟合
    model.add(tf.keras.layers.Dropout(0.3))

    #第三层
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))

    #第四层
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(512, return_sequences=False))

    #Dense层,全连接层,256个神经元
    model.add(tf.keras.layers.Dense(256))

    model.add(tf.keras.layers.Dropout(0.3))

    # 输出的数目等于所有不重复的音调的数目
    model.add(tf.keras.layers.Dense(num_pitch))

    # Softmax激活函数计算每个音调的比率，取最大值作为新音符
    model.add(tf.keras.layers.Activation('Softmax'))

    # Sequential model配置 ,交叉熵作为损失函数, rmsprop优化器,
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    #如果生成音乐时，从 HDF5 文件中加载所有神经网络层的参数
    if weights_file is not None:
        model.load_weights(weights_file)

    return model