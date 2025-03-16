import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加載數據
def load_data(data_dir):
    images = []
    labels = []
    
    for identity_folder in Path(data_dir).iterdir():
        if not identity_folder.is_dir():
            continue
        
        identity_name = identity_folder.name  # 例如 'identity_50'
        img_count = 0
        for img_path in identity_folder.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"無法讀取圖片: {img_path}")
                continue
            img = cv2.resize(img, (224, 224))  # VGG16要求輸入圖片大小為224x224
            images.append(img)
            labels.append(identity_name)
            img_count += 1
        if img_count > 0:  # 只打印有圖片的身份
            print(f"從 {identity_folder} 加載了 {img_count} 張圖片")
    
    if not images:
        raise ValueError(f"在 {data_dir} 中未找到任何圖片，請檢查路徑或文件！")
    
    images = np.array(images, dtype=np.float32) / 255.0  # 規一化到 [0, 1]
    labels = np.array(labels)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    
    return images, labels_onehot, label_encoder

train_dir = r"C:\Biometrics\data\training\augmented"
test_dir = r"C:\Biometrics\data\testing\organized"

# 加載數據並檢查
print("加載訓練集...")
X_train, y_train, train_label_encoder = load_data(train_dir)
print("加載測試集...")
X_test, y_test_temp, test_label_encoder = load_data(test_dir)

print(f"訓練集樣本數: {X_train.shape[0]}, 測試集樣本數: {X_test.shape[0]}")
print(f"訓練集類別數: {y_train.shape[1]}")
print(f"測試集原始類別數: {y_test_temp.shape[1]}")

# 2. 調整測試集標籤以匹配訓練集的類別
train_labels = train_label_encoder.classes_  # 44個身份
test_labels = test_label_encoder.classes_    # 11個身份

# 確保測試集身份是訓練集身份的子集
test_labels_encoded = np.array([np.where(train_labels == label)[0][0] for label in test_label_encoder.inverse_transform(np.argmax(y_test_temp, axis=1))])
y_test = to_categorical(test_labels_encoded, num_classes=len(train_labels))

print(f"調整後測試集標籤形狀: {y_test.shape}")

# 3. 構建轉移學習模型 (VGG16)
def build_transfer_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # 凍結預訓練層

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = (224, 224, 3)  # VGG16要求輸入大小為224x224x3
num_classes = y_train.shape[1]  # 44

model = build_transfer_model(input_shape, num_classes)
model.summary()

# 4. 數據增強
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 5. 訓練模型
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

model.save("cnn_identity_model.h5")
print("模型已保存至 cnn_identity_model.h5")

# 6. 評估與可視化
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"測試集準確率: {test_accuracy:.4f}")

plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.title('訓練和驗證準確率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 7. 預測範例
sample_img = X_test[0:1]
prediction = model.predict(sample_img)
predicted_label = train_label_encoder.inverse_transform([np.argmax(prediction)])
true_label = train_label_encoder.inverse_transform([np.argmax(y_test[0])])
print(f"真實身份: {true_label[0]}")
print(f"預測身份: {predicted_label[0]}")

# 8. 打印測試集身份名稱
print("測試集身份名稱:", test_label_encoder.classes_)