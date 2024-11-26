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
