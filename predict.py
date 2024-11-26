# predict.py
import tensorflow as tf
from preprocess import load_and_preprocess_data

model = tf.keras.models.load_model('saved_model/emotion_model.keras')

print("模型加载成功")

# 编译指标信息
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 加载和预处理数据
X_train, X_test, y_train, y_test = load_and_preprocess_data('fer2013.csv')

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试集上的准确率: {accuracy:.2%}")
