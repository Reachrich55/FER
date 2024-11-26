# train.py
from keras.src.callbacks import EarlyStopping
from preprocess import load_and_preprocess_data
from model import create_model
import os

# 加载和预处理数据
X_train, X_test, y_train, y_test = load_and_preprocess_data('fer2013.csv')

# 创建模型
model = create_model()

# 设置早停
# 配置 EarlyStopping 回调函数，在验证损失 (val_loss) 停止降低后提前停止训练，允许10个epoch内无提升，并恢复最佳权重
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
# 使用 fit 方法进行模型训练，通过指定 epochs 和 batch_size，模型会在训练集上进行迭代学习
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=[early_stopping])

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试集上的准确率: {accuracy:.2%}")

# 保存模型
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/emotion_model.keras')
print("模型已保存至 'saved_model/emotion_model.keras'")
