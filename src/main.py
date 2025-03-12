import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def compute_hu_moments(image):
    """計算圖片的Hu矩"""
    # 將圖像轉為灰度（如果不是灰度圖）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 找到輪廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 選擇最大輪廓
    cnt = max(contours, key=cv2.contourArea)
    
    # 計算矩和Hu矩
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # 對Hu矩取對數（因為原始值範圍很大）
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments_log

def preprocess_image(image):
    """預處理圖片：直方圖均衡化、高斯模糊並規一化"""
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l_eq = cv2.equalizeHist(l)  # 直方圖均衡化
    image_lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR)
    image_blurred = cv2.GaussianBlur(image_eq, (5, 5), 0)  # 高斯模糊
    image_normalized = image_blurred.astype(np.float32) / 255.0  # 規一化
    image_processed = (image_normalized * 255).astype(np.uint8)
    return image_processed

def parse_filename(filename):
    """解析檔案名稱，提取身份ID和視角"""
    name = filename.split('.')[0]  # 移除副檔名
    identity = name[5:7]  # 提取身份ID (如 '50')
    view = name[-1:]      # 提取視角 (如 'f', 's')
    return identity, view

def process_images(input_dir):
    """處理指定目錄中的圖片並提取Hu矩"""
    data = []
    image_names = []
    identities = []
    views = []
    
    # 遍歷指定目錄中的所有JPG圖片
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            # 讀取圖片
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            # 預處理圖片
            processed_image = preprocess_image(image)
            
            # 計算Hu矩
            hu_moments = compute_hu_moments(processed_image)
            if hu_moments is None:
                print(f"無法計算Hu矩: {file_path}")
                continue
            
            # 解析檔案資訊
            identity, view = parse_filename(file_path.name)
            data.append(hu_moments)
            image_names.append(file_path.name)
            identities.append(identity)
            views.append(view)
            print(f"已處理圖片: {file_path}")
            
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {e}")
    
    return np.array(data), image_names, identities, views

def visualize_comparison(hu_data, image_names, identities, views, output_dir):
    """可視化兩張圖片的Hu矩比較"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # 繪製每張圖片的Hu矩
    for i in range(len(hu_data)):
        plt.plot(range(1, 8), hu_data[i], marker='o', label=f"{image_names[i]} (ID: {identities[i]}, View: {views[i]})")
    
    # 添加標題和標籤
    plt.title('Comparison of Hu Moments for Different Views')
    plt.xlabel('Hu Moment Index')
    plt.ylabel('Log-transformed Value')
    plt.xticks(range(1, 8))
    plt.legend()
    plt.grid(True)
    
    # 保存圖表
    output_path = os.path.join(output_dir, 'hu_moments_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成比較圖表: {output_path}")

def save_to_csv(hu_data, image_names, identities, views, output_path):
    """將Hu矩數據保存為CSV"""
    df = pd.DataFrame(hu_data, columns=[f'h{i+1}' for i in range(7)], index=image_names)
    df['Identity'] = identities
    df['View'] = views
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"數據已保存到: {output_path}")
    return df

def main():
    # 定義輸入和輸出路徑
    input_directory = r"C:\Biometrics\data\results\mask"  # 假設訓練資料夾有兩張圖
    output_directory = r"C:\Biometrics\data\results\analysis"
    
    # 處理圖像並提取Hu矩
    hu_data, image_names, identities, views = process_images(input_directory)
    
    if len(hu_data) > 0:
        # 保存數據到CSV
        csv_path = os.path.join(output_directory, "hu_moments.csv")
        df = save_to_csv(hu_data, image_names, identities, views, csv_path)
        
        # 可視化比較
        visualize_comparison(hu_data, image_names, identities, views, output_directory)
        
        # 簡單的距離比較（正面與側面）
        if len(hu_data) == 2:
            from scipy.spatial.distance import euclidean
            distance = euclidean(hu_data[0], hu_data[1])
            print(f"正面與側面Hu矩的歐幾里得距離: {distance:.4f}")
    else:
        print("未找到可處理的圖像數據。")

if __name__ == "__main__":
    main()