# FER
基于CNN的面部表情识别
## 1. 数据集选择  
FER2013包含七种情感，分别为愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性，共35887张图像。数据最初通过网络收集，包含不同年龄、种族和性别的面部图像，增加了数据的多样性，但同时也带来了噪声和标注错误。由于图像分辨率较低，表情的细微差异在视觉上不太明显。此外，不同个体的表情差异较大，加之数据集中各情感类别的分布不均匀，这为模型的训练带来了挑战。  
数据集的下载地址为：<https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data>  
## 2. 模型选择  
CNN具有局部特征不变性、权值共享等特点。在处理图像任务上表现优异，能够有效地提取出图像的多层次特征，具有高效的空间不变性和较强的泛化能力，是目前图像处理任务的主流模型。  

算法：用于面部表情识别的卷积神经网络（CNN）  
输入：大小为 48x48 的灰度图像  
输出：7 类情感类别中的一种  

步骤：  

Ⅰ初始化一个顺序模型。  

Ⅱ添加卷积层：  

&emsp;2x卷积层：64 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
&emsp;2x卷积层：128 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
&emsp;2x卷积层：256 个过滤器，3x3 卷积核，BatchNormalization，激活函数 ELU。  
&emsp;2x2 最大池化层。  
Ⅲ添加全连接层：  

&emsp;扁平化层。  
&emsp;全连接层：128 个单元，激活函数 ELU，BatchNormalization。  
&emsp;输出层：7 个单元，激活函数 softmax。  

Ⅳ编译模型：使用 Adam 优化器（学习率 $\alpha$=0.001），损失函数为 categorical_crossentropy，评估指标为准确率accuracy。  

Ⅴ训练模型：训练 100 个周期，在测试集上进行评估。  

输出：返回训练好的模型及其性能指标。  
```python
# model.py
from keras.src.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        # 第一组卷积层: 2个卷积层 (64 filters, 3x3), BatchNormalization, ELU
        Conv2D(64, (3, 3), padding='same', activation='elu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 第二组卷积层: 2个卷积层 (128 filters, 3x3), BatchNormalization, ELU
        Conv2D(128, (3, 3), padding='same', activation='elu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 第三组卷积层: 2个卷积层 (256 filters, 3x3), BatchNormalization, ELU
        Conv2D(256, (3, 3), padding='same', activation='elu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='elu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # 在模型的卷积层和全连接层之间添加扁平化层将多维特征转换为一维向量，以便全连接层能够接收这些特征进行进一步处理。
        Flatten(),

        # 全连接层
        Dense(128, activation='elu'),
        BatchNormalization(),

        # 输出层
        Dense(7, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

```
## 3. 训练和优化
### 3.1 数据集预处理
加载和预处理数据，进行图像归一化和标签编码，并划分训练集与测试集  
```python
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

```
### 3.2 损失函数选择
交叉熵损失函数categorical_crossentropy，计算的是预测的类别概率与真实类别之间的交叉熵，即模型输出概率分布与目标类别分布之间的距离。适用于多分类问题。公式为：  
  
$$Loss = -\sum_{i=1}^{N}y_{i}\log(p_{i})$$

其中：  
$y_{i}$是真实标签（one-hot编码）  
$p_{i}$是模型预测的该类的概率
### 3.3 早停设置
设置早停可以帮助防止模型在训练集上过拟合，从而提高泛化能力。特别是在深度学习模型的训练中，早停能有效节省时间。  
配置 EarlyStopping 回调函数，在验证损失 (val_loss) 停止降低后提前停止训练，并恢复最佳权重，以防止过拟合。  
```python
# train.py

# 在验证损失 (val_loss) 停止降低后提前停止训练，10个epoch内无提升后恢复最佳权重
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```
### 3.4 批量归一化
BatchNormalization批量归一化是一种深度学习中的正则化技术，主要用于加速神经网络的训练过程并提升模型的稳定性和准确性。  
在训练过程中，BatchNormalization 会对每个 batch 内的输入进行标准化处理，使其均值为 0，方差为 1  
Ⅰ对输入的 $x$ 求均值 $\mu$ 和方差 $\delta^{2}$  
Ⅱ标准化输入：  

$$\hat{x} = \frac{x-\mu}{\sqrt{\delta^{2}+\epsilon}}$$

其中， $\epsilon$ 是一个很小的常数，用于防止除零错误  

Ⅲ应用缩放和平移参数  

$$y=\gamma\hat{x}+\beta$$

其中， $\gamma$ 和 $\beta$ 是可学习的参数，用于恢复网络的表达能力  
## 4. 实时面部表情识别
### 4.1 人脸检测
检测视频帧中的人脸，可以使用OpenCV提供的Haar特征级联分类器，需要haarcascade_frontalface_default.xml文件。  
下载链接：<https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml>  
### 4.2 面部表情识别
使用已经训练好的模型进行面部表情识别，具体代码如下  
```python
# camera.py
import cv2
import numpy as np
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model('saved_model/emotion_model.keras')
print("模型加载成功")

# 定义情感类别
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("无法加载人脸检测器 XML 文件。请确保路径正确。")

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 是默认摄像头

if not cap.isOpened():
    raise IOError("无法打开摄像头。请检查摄像头是否连接正确。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧。")
        break

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 绘制人脸矩形框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 提取人脸区域
        face_roi = gray[y:y+h, x:x+w]
        try:
            # 调整人脸区域大小为 48x48
            face_roi = cv2.resize(face_roi, (48, 48))
        except:
            continue

        # 预处理图像
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)  # 添加通道维度

        # 进行预测
        predictions = model.predict(face_roi)
        max_index = int(np.argmax(predictions))
        predicted_emotion = emotion_labels[max_index]
        confidence = predictions[0][max_index]

        # 显示预测结果
        label = f"{predicted_emotion} ({confidence*100:.2f}%)"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # 显示结果帧
    cv2.imshow('Real-Time Facial Emotion Recognition', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

```
## 5. 演示视频
<https://www.bilibili.com/video/BV1jemYY1EYW/?share_source=copy_web&vd_source=b1907127417f1a85ec092926911acabc>
