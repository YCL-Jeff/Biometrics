import os
import cv2
import numpy as np
import pandas as pd
from openpifpaf import decoder, network, show
from openpifpaf.predictor import Predictor
from pathlib import Path
import openpifpaf
from multiprocessing import Pool

# 檢查 OpenPifPaf 版本
print(f"OpenPifPaf 版本: {openpifpaf.__version__}")

# 設定路徑
input_base_dir = r"C:\Biometrics\data\training\augmented"
output_base_dir = r"C:\Biometrics\data\results\landmarks_augmented"
os.makedirs(output_base_dir, exist_ok=True)

# 初始化 OpenPifPaf 預測器
try:
    predictor = Predictor(checkpoint="shufflenetv2k16")
except Exception as e:
    print(f"錯誤: 初始化 Predictor 失敗 - {e}")
    exit(1)

# 處理圖片並提取關節點
def process_image(args):
    image_path, output_csv = args
    print(f"處理圖片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖片 {image_path}")
        return False
    
    predictions, _, _ = predictor.numpy_image(image)
    if not predictions:
        print(f"警告: 未檢測到關節點 {image_path}")
        return False
    
    annotation = predictions[0]
    keypoints = annotation.data
    data = []
    for i, kp in enumerate(keypoints):
        visibility = kp[2]
        if visibility >= 0.2:
            data.append({
                'landmark_id': i,
                'x': kp[0],
                'y': kp[1],
                'visibility': visibility
            })
        else:
            data.append({
                'landmark_id': i,
                'x': 0.0,
                'y': 0.0,
                'visibility': 0.0
            })
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"已保存關節點到: {output_csv}")
    return True

# 準備任務
def prepare_tasks():
    tasks = []
    for root, dirs, files in os.walk(input_base_dir):
        for filename in files:
            if filename.endswith('.jpg'):
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_base_dir)
                output_sub_dir = os.path.join(output_base_dir, relative_path)
                output_csv = os.path.join(output_sub_dir, filename.replace('.jpg', '_landmarks.csv'))
                tasks.append((image_path, output_csv))
    return tasks

if __name__ == '__main__':
    # 準備任務
    tasks = prepare_tasks()
    
    # 使用多進程處理
    with Pool(4) as p:
        results = p.map(process_image, tasks)
    
    # 統計結果
    processed_count = sum(1 for r in results if r)
    failed_count = len(tasks) - processed_count
    
    print(f"關節點提取完成！總共處理 {processed_count} 張圖片，失敗 {failed_count} 張！")
    
    # 檢查輸出資料夾
    total_files = sum(len(files) for _, _, files in os.walk(output_base_dir))
    if total_files == 0:
        print("警告: 輸出資料夾中無檔案，請檢查圖片或參數設定！")
    else:
        print(f"輸出資料夾 {output_base_dir} 包含 {total_files} 個檔案。")