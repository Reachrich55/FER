import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# 自定义模型，将第一层输入通道改为1
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入通道改为1
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 输出类别数调整为情感分类数

    def forward(self, x):
        return self.model(x)

# 加载自定义模型和权重
model = EmotionResNet18(num_classes=7)
model.load_state_dict(torch.load('saved_model/resnet18_model.pth'))
model.eval()
print("模型加载成功")

# 定义情感类别
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("无法加载人脸检测器 XML 文件。请确保路径正确。")

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("无法打开摄像头。请检查摄像头是否连接正确。")

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧。")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]

        try:
            face_tensor = preprocess(face_roi).unsqueeze(0)  # 扩展 batch 维度
        except:
            continue

        with torch.no_grad():
            predictions = model(face_tensor)
            probabilities = F.softmax(predictions, dim=1)
            max_index = int(torch.argmax(probabilities))
            predicted_emotion = emotion_labels[max_index]
            confidence = probabilities[0][max_index].item()

        label = f"{predicted_emotion} ({confidence*100:.2f}%)"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Real-Time Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
