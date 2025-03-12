import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

def build_gabor_filters(k_size=31, sigma=4.0, theta_list=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       lamda=10.0, gamma=0.5, psi=0):
    """創建一組Gabor濾波器核"""
    filters = []
    for theta in theta_list:
        kern = cv.getGaborKernel((k_size, k_size), sigma, theta, lamda, gamma, psi, ktype=cv.CV_32F)
        filters.append(kern)
    return filters

def apply_gabor_filters(image, filters):
    """應用Gabor濾波器並返回結果"""
    # 轉為灰度圖
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 儲存每個濾波器的響應
    responses = []
    for kern in filters:
        filtered = cv.filter2D(gray, cv.CV_32F, kern)
        responses.append(filtered)
    return responses

def visualize_gabor_results(image, responses, filters, output_path):
    """可視化Gabor濾波器結果"""
    n_filters = len(filters)
    plt.figure(figsize=(15, 5))
    
    # 原始圖像
    plt.subplot(1, n_filters + 1, 1)
    plt.imshow(image if len(image.shape) == 3 else image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 每個濾波器的結果
    for i, (resp, kern) in enumerate(zip(responses, filters), 2):
        plt.subplot(1, n_filters + 1, i)
        plt.imshow(resp, cmap='gray')
        plt.title(f'Theta = {int(np.arctan2(kern[0,1], kern[0,0])*180/np.pi)}°')
        plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_images_with_gabor(input_dir, output_dir):
    """處理圖像並應用Gabor濾波器"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # 創建Gabor濾波器
    filters = build_gabor_filters()
    
    for image_file in image_files:
        img_path = os.path.join(input_dir, image_file)
        img = cv.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {image_file}")
            continue
        
        # 應用Gabor濾波器
        responses = apply_gabor_filters(img, filters)
        
        # 生成輸出路徑
        output_filename = f"gabor_{os.path.splitext(image_file)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 可視化並保存
        print(f"Processing {image_file} with Gabor filters...")
        visualize_gabor_results(img, responses, filters, output_path)
        print(f"Saved Gabor analysis to {output_path}")

def main():
    input_dir = r"C:\Biometrics\data\results\body"  # body和mask所在的目錄
    output_dir = r"C:\Biometrics\data\results\histograms"  # 輸出目錄
    
    process_images_with_gabor(input_dir, output_dir)
    print("Gabor analysis completed!")

if __name__ == "__main__":
    main()