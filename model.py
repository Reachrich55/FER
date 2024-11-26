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

        # 全连接层
        Flatten(),
        Dense(128, activation='elu'),
        BatchNormalization(),

        # 输出层
        Dense(7, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
