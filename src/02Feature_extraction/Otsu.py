import cv2
import numpy as np
import os
from pathlib import Path

def apply_otsu_binarization(image):
    """應用Otsu二值化處理"""
    # 如果是彩色圖，轉為灰度圖
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 應用Otsu二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def process_images_with_otsu(input_dir, output_dir):
    """處理指定目錄中的圖片並應用Otsu二值化"""
    # 創建輸出目錄
    otsu_dir = os.path.join(output_dir, "otsu")
    os.makedirs(otsu_dir, exist_ok=True)
    
    # 獲取所有圖片
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            # 讀取圖片
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            # 應用Otsu二值化
            binary_image = apply_otsu_binarization(image)
            
            # 保存結果
            filename = file_path.stem
            output_path = os.path.join(otsu_dir, f"{filename}_otsu.jpg")
            cv2.imwrite(output_path, binary_image)
            
            print(f"已處理並保存圖片: {output_path}")
            
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Biometrics\data\results\body"
    output_directory = r"C:\Biometrics\data\results"
    
    process_images_with_otsu(input_directory, output_directory)
    print("Otsu二值化處理完成!")