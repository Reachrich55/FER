# encoding:utf-8
import pandas as pd
import numpy as np
from PIL import Image
import os

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'normal'  # 中性
}


def saveImageFromFer2013(file):
    # 读取CSV文件
    faces_data = pd.read_csv(file)
    image_count = 0

    # 遍历CSV文件内容，将图像数据按类别保存
    for index, row in faces_data.iterrows():
        try:
            emotion_data = row[0]
            image_data = row[1]
            usage_data = row[2]

            # 将图像数据转换成48*48矩阵
            data_array = np.fromstring(image_data, sep=' ', dtype=np.float32)
            image = data_array.reshape(48, 48)

            # 创建保存路径
            dir_name = usage_data
            emotion_name = emotions.get(str(emotion_data), "unknown")
            image_path = os.path.join(dir_name, emotion_name)
            os.makedirs(image_path, exist_ok=True)

            # 保存图像
            image_name = os.path.join(image_path, f"{index}.jpg")
            # 将图像数据（数组）保存为灰度图像
            Image.fromarray(image).convert('L').save(image_name)
            image_count += 1
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    print(f'总共有 {image_count} 张图片')

saveImageFromFer2013('fer2013.csv')