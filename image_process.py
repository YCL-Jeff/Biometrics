import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path

def preprocess_image(image):
    """預處理圖片：直方圖均衡化、高斯模糊並規一化"""
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l_eq = cv2.equalizeHist(l)
    image_lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR)
    image_blurred = cv2.GaussianBlur(image_eq, (5, 5), 0)
    image_normalized = image_blurred.astype(np.float32) / 255.0
    image_processed = (image_normalized * 255).astype(np.uint8)
    return image_processed

def detect_pose_landmarks(image):
    """使用 MediaPipe 偵測人體關節點"""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=2, 
        min_detection_confidence=0.5,
        enable_segmentation=True
    ) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

def get_bounding_box_from_mask(mask):
    """從掩碼獲取外接矩形並轉換為正方形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 找到最大輪廓的外接矩形
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 將矩形轉換為正方形
    size = max(w, h)  # 使用較長的邊作為正方形邊長
    center_x, center_y = x + w//2, y + h//2
    
    # 計算正方形的新邊界
    new_x = max(0, center_x - size//2)
    new_y = max(0, center_y - size//2)
    
    # 確保不超出圖像邊界
    height, width = mask.shape
    if new_x + size > width:
        new_x = width - size
    if new_y + size > height:
        new_y = height - size
    
    return (new_x, new_y, size, size)

def crop_to_square(image, bbox):
    """根據給定的邊框裁切圖片為正方形"""
    if bbox is None:
        return image  # 如果沒有邊框，返回原圖
    
    x, y, size, _ = bbox
    return image[y:y+size, x:x+size]

def get_improved_body_mask(image, results):
    """使用 MediaPipe 的分割掩碼結合形態學運算獲得更精確的人體掩碼"""
    if results.segmentation_mask is None:
        return get_body_mask_with_grabcut(image, results.pose_landmarks)
    
    segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        final_mask = np.zeros_like(mask_cleaned)
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], 0, 1, -1)
        return final_mask
    return get_body_mask_with_grabcut(image, results.pose_landmarks)

def get_body_mask_with_grabcut(image, landmarks):
    """使用 GrabCut 基於 MediaPipe 關節點提取人體輪廓"""
    if landmarks is None:
        return None
    
    mask = np.zeros(image.shape[:2], np.uint8)
    height, width = image.shape[:2]
    landmarks_array = [[landmark.x * width, landmark.y * height] 
                      for landmark in landmarks.landmark 
                      if landmark.visibility > 0.5]
    
    if len(landmarks_array) < 5:
        return None
    
    landmarks_array = np.array(landmarks_array)
    min_x, min_y = np.min(landmarks_array, axis=0)
    max_x, max_y = np.max(landmarks_array, axis=0)
    
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    
    min_x = max(0, min_x - padding_x)
    min_y = max(0, min_y - padding_y)
    max_x = min(width, max_x + padding_x)
    max_y = min(height, max_y + padding_y)
    
    rect = (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    return mask_cleaned

def process_images(input_dir, output_dir):
    """處理指定目錄中的所有圖片"""
    body_dir = os.path.join(output_dir, "body")
    mask_dir = os.path.join(output_dir, "mask")
    
    os.makedirs(body_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            processed_image = preprocess_image(image)
            results = detect_pose_landmarks(processed_image)
            
            if results.pose_landmarks:
                body_mask = get_improved_body_mask(processed_image, results)
                
                if body_mask is not None:
                    # 獲取人體輪廓的外框並轉為正方形
                    bbox = get_bounding_box_from_mask(body_mask)
                    
                    # 裁切圖片和掩碼
                    cropped_body = crop_to_square(processed_image, bbox)
                    cropped_mask = crop_to_square(body_mask, bbox)
                    
                    # 應用掩碼到裁切後的圖片
                    body_only = cropped_body * cropped_mask[:, :, np.newaxis]
                    
                    # 保存結果
                    filename = file_path.stem
                    body_path = os.path.join(body_dir, f"{filename}.jpg")
                    mask_path = os.path.join(mask_dir, f"{filename}.jpg")
                    
                    cv2.imwrite(body_path, body_only)
                    cv2.imwrite(mask_path, cropped_mask * 255)
                    
                    print(f"已處理並保存圖片: {filename}")
                else:
                    print(f"無法創建掩碼: {file_path}")
            else:
                print(f"未檢測到關節點: {file_path}")
                
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {e}")

if __name__ == "__main__":
    input_directory = r"C:\Biometrics\data\training"
    output_directory = r"C:\Biometrics\data\results"
    
    process_images(input_directory, output_directory)
    print("處理完成!")