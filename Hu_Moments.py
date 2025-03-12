import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def compute_hu_moments(image):
    """計算圖片的Hu矩"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments_log

def parse_filename(filename):
    """解析檔案名稱，提取身份ID和視角"""
    name = filename.split('.')[0]  # 移除副檔名
    identity = name[5:7]  # 提取身份ID (如 '50')
    view = name[-1:]      # 提取視角 (如 'f', 's')
    return identity, view

def process_images_for_hu_moments(input_dir):
    """處理指定資料夾中的所有圖片並計算Hu矩"""
    data = []
    image_names = []
    identities = []
    views = []
    
    # 自動遍歷資料夾中的所有JPG檔案
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            hu_moments = compute_hu_moments(image)
            if hu_moments is None:
                print(f"無法計算Hu矩: {file_path}")
                continue
            
            identity, view = parse_filename(file_path.name)
            data.append(hu_moments)
            image_names.append(file_path.name)
            identities.append(identity)
            views.append(view)
            print(f"已處理圖片: {file_path}")
            
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {e}")
    
    return np.array(data), image_names, identities, views

def save_to_csv(data, image_names, identities, views, output_path):
    """將Hu矩數據保存為CSV"""
    df = pd.DataFrame(data, columns=[f'h{i+1}' for i in range(7)], index=image_names)
    df['Identity'] = identities
    df['View'] = views
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"數據已保存到: {output_path}")
    return df

'''
    def plot_hu_moments_bar(df, output_dir):
    """生成每個圖片的Hu矩柱狀圖"""
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in df.iterrows():
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, 8), row[:7], color='skyblue')
        plt.title(f'Hu Moments for {idx} (ID: {row["Identity"]}, View: {row["View"]})')
        plt.xlabel('Hu Moment Index')
        plt.ylabel('Log-transformed Value')
        plt.xticks(range(1, 8))
        output_path = os.path.join(output_dir, f"{idx}_hu_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成柱狀圖: {output_path}")


def plot_hu_moments_heatmap(df, output_path):
    """生成Hu矩的熱圖"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.iloc[:, :7], annot=True, cmap='YlOrRd', fmt='.2f',
                xticklabels=[f'h{i+1}' for i in range(7)],
                yticklabels=[f'{idx} (ID:{row["Identity"]})' for idx, row in df.iterrows()])
    plt.title('Heatmap of Hu Moments by Image and Identity')
    plt.xlabel('Hu Moment Index')
    plt.ylabel('Image (Identity)')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成熱圖: {output_path}")

def plot_hu_moments_scatter(df, output_dir):
    """生成Hu矩的散點圖（h1 vs h2，按身份著色）"""
    plt.figure(figsize=(10, 8))
    colors = {'050': 'blue', '080': 'green', '081': 'red'}
    for idx, row in df.iterrows():
        plt.scatter(row['h1'], row['h2'], c=colors[row['Identity']], 
                   label=row['Identity'] if row['Identity'] not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.annotate(f"{idx[-2:]}", (row['h1'], row['h2']), fontsize=8)
    plt.title('Scatter Plot of h1 vs h2 by Identity')
    plt.xlabel('h1 (Log-transformed)')
    plt.ylabel('h2 (Log-transformed)')
    plt.legend(title='Identity')
    output_path = os.path.join(output_dir, 'hu_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成散點圖: {output_path}")
'''
if __name__ == "__main__":
    input_directory = r"C:\Biometrics\data\results\mask"  # 自動處理此資料夾中的所有JPG檔案
    output_directory = r"C:\Biometrics\data\results\analysis"
    
    # 移除file_list限制，自動遍歷資料夾
    hu_data, image_names, identities, views = process_images_for_hu_moments(input_directory)
    
    if len(hu_data) > 0:
        # 保存數據到CSV
        csv_path = os.path.join(output_directory, "hu_moments.csv")
        df = save_to_csv(hu_data, image_names, identities, views, csv_path)
        
        # 生成圖表
        plot_hu_moments_bar(df, os.path.join(output_directory, "bar_plots"))
        plot_hu_moments_heatmap(df, os.path.join(output_directory, "hu_heatmap.png"))
        plot_hu_moments_scatter(df, os.path.join(output_directory, "hu_scatter.png"))
        
        print("數據整理和圖表生成完成!")
    else:
        print("沒有可處理的數據")