import os
import shutil
from pathlib import Path

def parse_filename(filename):
    """解析檔案名稱，提取身份ID"""
    name = filename.split('.')[0]
    identity = name[5:7]  # 提取身份ID (如 '50')
    return identity

def organize_images_by_identity(input_dir, output_dir):
    """將圖片按身份整理到子文件夾"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍歷輸入目錄中的所有圖片
    for img_path in Path(input_dir).glob('*.jpg'):
        filename = img_path.name
        try:
            identity = parse_filename(filename)  # 提取身份ID
            identity_folder = f"identity_{identity}"  # 例如 'identity_50'
            
            # 創建身份子文件夾
            identity_dir = os.path.join(output_dir, identity_folder)
            os.makedirs(identity_dir, exist_ok=True)
            
            # 移動圖片到對應身份文件夾
            dest_path = os.path.join(identity_dir, filename)
            shutil.move(str(img_path), dest_path)
            print(f"移動 {filename} 到 {identity_folder}")
        
        except Exception as e:
            print(f"處理 {filename} 時出錯: {e}")

if __name__ == "__main__":
    # 訓練集和測試集的原始路徑
    train_input_dir = r"C:\Biometrics\data\training\mask"
    test_input_dir = r"C:\Biometrics\data\training\mask\test"
    
    # 整理後的輸出路徑（直接在原路徑下整理）
    train_output_dir = train_input_dir
    test_output_dir = test_input_dir
    
    # 整理訓練集圖片
    print("整理訓練集圖片...")
    organize_images_by_identity(train_input_dir, train_output_dir)
    
    # 整理測試集圖片
    print("整理測試集圖片...")
    organize_images_by_identity(test_input_dir, test_output_dir)
    
    print("圖片整理完成！")