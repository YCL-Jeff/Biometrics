import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

def find_contours_and_features(image):
    """找輪廓並計算幾何特徵"""
    # 轉為灰度圖
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 二值化處理
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    
    # 尋找輪廓
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # 選擇面積最大的輪廓
    main_contour = max(contours, key=cv.contourArea)
    
    # 計算幾何特徵
    area = cv.contourArea(main_contour)
    perimeter = cv.arcLength(main_contour, True)
    
    # 獲取最小外接矩形來計算長寬比
    x, y, w, h = cv.boundingRect(main_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    return main_contour, (area, perimeter, aspect_ratio), (x, y, w, h)

def visualize_features(image, contour, features, rect, output_path):
    """可視化輪廓和特徵並保存圖形"""
    if contour is None or features is None:
        return
    
    # 複製原始圖片以進行繪製
    vis_image = image.copy()
    
    # 繪製輪廓
    cv.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
    
    # 繪製外接矩形
    x, y, w, h = rect
    cv.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 創建新的圖形
    plt.figure(figsize=(10, 8))
    
    # 顯示處理後的圖片
    plt.imshow(cv.cvtColor(vis_image, cv.COLOR_BGR2RGB))
    
    # 添加特徵資訊文字
    area, perimeter, aspect_ratio = features
    text = f"Area: {area:.2f}\nPerimeter: {perimeter:.2f}\nAspect Ratio: {aspect_ratio:.2f}"
    plt.text(10, 30, text, color='white', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.5))
    
    plt.title('Contour Features')
    plt.axis('off')
    
    # 保存圖形
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_images(input_dir, output_dir):
    """處理指定目錄中的所有圖片"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    for image_file in image_files:
        # 讀取圖片
        img_path = os.path.join(input_dir, image_file)
        img = cv.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {image_file}")
            continue
            
        # 找輪廓並計算特徵
        contour, features, rect = find_contours_and_features(img)
        
        if features is None:
            print(f"No contours found in {image_file}")
            continue
            
        # 生成輸出路徑
        output_filename = f"features_{os.path.splitext(image_file)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 可視化並保存
        print(f"Processing {image_file}...")
        visualize_features(img, contour, features, rect, output_path)
        print(f"Saved visualization to {output_path}")

def main():
    input_dir = r"C:\Biometrics\data\results\mask"
    output_dir = r"C:\Biometrics\data\results\histograms"
    
    process_images(input_dir, output_dir)
    print("Processing completed!")

if __name__ == "__main__":
    main()