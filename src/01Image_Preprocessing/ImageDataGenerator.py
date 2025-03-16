import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt

# 設置路徑
train_input_dir = r"C:\Biometrics\data\training\body"
train_output_dir = r"C:\Biometrics\data\training\augmented"
test_input_dir = r"C:\Biometrics\data\testing\organized"
test_output_dir = r"C:\Biometrics\data\testing\augmented"

# 創建輸出目錄
for directory in [train_output_dir, test_output_dir]:
    os.makedirs(directory, exist_ok=True)
    for identity in range(1, 83):  # 假設身份從 01 到 82
        identity_folder = os.path.join(directory, f"identity_{identity:02d}")
        os.makedirs(identity_folder, exist_ok=True)

# 數據增強參數
datagen = ImageDataGenerator(
    rotation_range=5,            # 0-5 度
    width_shift_range=0.05,      # 5% 圖片寬度
    height_shift_range=0.05,     # 5% 圖片高度
    horizontal_flip=True,        # 水平翻轉
    brightness_range=[0.95, 1.05], # 縮小亮度範圍
    zoom_range=0.05,            # 5% 縮放範圍
    fill_mode='constant',       # 常數填充
    cval=0                      # 填充值為黑色
)

# 函數：增強並保存圖片
def augment_images(input_dir, output_dir, num_augmentations=3):
    for identity_folder in Path(input_dir).iterdir():
        if not identity_folder.is_dir():
            continue
        
        identity_name = identity_folder.name
        output_identity_dir = os.path.join(output_dir, identity_name)
        
        # 讀取原始圖片
        images = []
        for img_path in identity_folder.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"無法讀取圖片: {img_path}")
                continue
            img = cv2.resize(img, (128, 128))  # 統一大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉為 RGB
            # 移除規一化，保持 [0, 255] 範圍
            images.append(img)
        
        if not images:
            print(f"未找到 {identity_folder} 中的圖片")
            continue
        
        # 對每張圖片進行增強
        for img in images:
            img = np.expand_dims(img, axis=0)  # 增加批量維度
            i = 0
            for batch in datagen.flow(img, batch_size=1, save_to_dir=output_identity_dir,
                                    save_prefix=f"{identity_name}_aug", save_format='jpg'):
                i += 1
                if i >= num_augmentations:  # 每個圖片生成指定數量的增強版本
                    break

# 執行增強
print("開始增強訓練集...")
augment_images(train_input_dir, train_output_dir, num_augmentations=4)
print("訓練集增強完成！")

print("開始增強測試集...")
augment_images(test_input_dir, test_output_dir, num_augmentations=4)
print("測試集增強完成！")

# 可視化增強效果（檢查用）
sample_img_path = r"C:\Biometrics\data\training\body\identity_01\image01.jpg"
sample_img = cv2.imread(sample_img_path)
if sample_img is not None:
    sample_img = cv2.resize(sample_img, (128, 128))
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img_expanded = np.expand_dims(sample_img, axis=0)
    for batch in datagen.flow(sample_img_expanded, batch_size=1, save_to_dir=r"C:\Biometrics\data\training\augmented\identity_01", save_prefix="test_aug", save_format='jpg'):
        break
    augmented_img = cv2.imread(r"C:\Biometrics\data\training\augmented\identity_01\test_aug_0.jpg")
    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img)
    plt.title("原始圖片")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_img)
    plt.title("增強後圖片")
    plt.axis('off')
    plt.show()

print("數據增強完成！請檢查增強後的數據夾：")
print(f"訓練集: {train_output_dir}")
print(f"測試集: {test_output_dir}")