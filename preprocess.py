# preprocess.py
import pandas as pd
import numpy as np
from keras.src.utils import to_categorical

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 图像和标签处理
    # np.fromstring() 将该字符串快速转为数值数组。.reshape(-1, 48, 48, 1) 将每张图像重塑为 48x48 的矩阵，并增加一个通道维度 1，以便于神经网络处理（即为灰度图）
    X = np.array([np.fromstring(image, sep=' ') for image in data['pixels']]).reshape(-1, 48, 48, 1) / 255.0  # 将像素值归一化
    # 使用 to_categorical 将情感标签转换为 one-hot 编码
    y = to_categorical(data['emotion'].values)

    # 根据 "Usage" 列划分训练集和测试集
    train_mask = data['Usage'] == 'Training'
    test_mask = data['Usage'] == 'PublicTest'

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, X_test, y_train, y_test
