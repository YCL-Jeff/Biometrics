import cv2
import numpy as np
import os
import mediapipe as mp
from pathlib import Path
import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog
import joblib

# 設置日誌級別以隱藏 MediaPipe 的非關鍵警告
logging.getLogger('mediapipe').setLevel(logging.ERROR)

def parse_filename(filename):
    """解析檔案名稱，提取身份ID和視角"""
    name = filename.split('.')[0]  # 移除副檔名
    identity = name[5:7]  # 提取身份ID (如 '50')
    view = name[-1:]      # 提取視角 (如 'f', 's')
    return identity, view

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
        # 提供圖像尺寸資訊
        height, width = image.shape[:2]
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results.image_width = width
        results.image_height = height
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
    
    # 檢查正方形尺寸是否有效
    if size <= 0:
        return None
    
    return (new_x, new_y, size, size)

def crop_to_square(image, bbox):
    """根據給定的邊框裁切圖片為正方形"""
    if bbox is None:
        return None  # 返回 None 表示裁切失敗
    
    x, y, size, _ = bbox
    cropped = image[y:y+size, x:x+size]
    
    # 檢查裁切後的圖像是否為空
    if cropped.size == 0:
        return None
    
    return cropped

def resize_image(image, size=(256, 256)):
    """將圖像調整為指定大小"""
    if image is None:
        return None
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

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

def extract_landmarks(landmarks, image, filename, identity, view):
    """提取並格式化人體關節點數據"""
    if landmarks is None:
        return None
    
    height, width = image.shape[:2]
    landmarks_data = []
    for idx, landmark in enumerate(landmarks.landmark):
        landmarks_data.append({
            'filename': filename,
            'identity': identity,
            'view': view,
            'landmark_id': idx,
            'x': landmark.x * width,
            'y': landmark.y * height,
            'visibility': landmark.visibility
        })
    return landmarks_data

def extract_lbp_features(img):
    """提取 LBP 特徵"""
    if img is None:
        return None
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    return lbp.flatten()

def extract_hog_features(img):
    """提取 HOG 特徵"""
    if img is None:
        return None
    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

def train_and_save_svm(features, labels, output_dir, feature_type):
    """訓練 SVM 模型並保存到檔案"""
    if len(features) == 0 or len(labels) == 0:
        print(f"無法訓練 {feature_type} SVM：特徵或標籤為空")
        return None
    
    # 正規化特徵
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 訓練 SVM
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    clf.fit(features_scaled, labels)
    
    # 保存模型和正規化器
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{feature_type}_svm.joblib")
    scaler_path = os.path.join(models_dir, f"{feature_type}_scaler.joblib")
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"已保存 {feature_type} SVM 模型到: {model_path}")
    print(f"已保存 {feature_type} Scaler 到: {scaler_path}")
    return clf, scaler

def save_all_landmarks(landmarks_data, output_dir):
    """將所有關節點數據保存為單一 CSV 檔案"""
    if not landmarks_data:
        print("無關節點數據可保存")
        return
    
    landmarks_df = pd.DataFrame(landmarks_data)
    landmarks_path = os.path.join(output_dir, "all_landmarks.csv")
    landmarks_df.to_csv(landmarks_path, index=False)
    print(f"已保存所有關節點數據到: {landmarks_path}")

def process_images(input_dir, output_dir):
    """處理指定目錄中的所有圖片"""
    body_dir = os.path.join(output_dir, "body")
    mask_dir = os.path.join(output_dir, "mask")
    gray_dir = os.path.join(output_dir, "gray")
    
    os.makedirs(body_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)
    
    # 用於儲存特徵和標籤
    lbp_features = []
    hog_features = []
    labels = []
    all_landmarks_data = []  # 儲存所有人的關節點數據
    
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            # 預處理圖片
            processed_image = preprocess_image(image)
            
            # 轉換為灰度圖
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            
            # 檢測人體關節點
            results = detect_pose_landmarks(processed_image)
            
            if results.pose_landmarks:
                body_mask = get_improved_body_mask(processed_image, results)
                
                if body_mask is not None:
                    # 獲取人體輪廓的外框並轉為正方形
                    bbox = get_bounding_box_from_mask(body_mask)
                    if bbox is None:
                        print(f"無法生成有效的邊框: {file_path}")
                        continue
                    
                    # 裁切圖片、掩碼和灰度圖
                    cropped_body = crop_to_square(processed_image, bbox)
                    cropped_mask = crop_to_square(body_mask, bbox)
                    cropped_gray = crop_to_square(gray_image, bbox)
                    
                    # 檢查裁切結果
                    if cropped_body is None or cropped_mask is None or cropped_gray is None:
                        print(f"裁切失敗，圖像為空: {file_path}")
                        continue
                    
                    # 將裁切後的圖像調整為 256*256
                    cropped_body = resize_image(cropped_body, (256, 256))
                    cropped_mask = resize_image(cropped_mask, (256, 256))
                    cropped_gray = resize_image(cropped_gray, (256, 256))
                    
                    # 再次檢查調整大小後的結果
                    if cropped_body is None or cropped_mask is None or cropped_gray is None:
                        print(f"調整大小失敗，圖像為空: {file_path}")
                        continue
                    
                    # 應用掩碼到裁切後的圖片（body 和 gray）
                    body_only = cropped_body * cropped_mask[:, :, np.newaxis]
                    gray_only = cropped_gray * cropped_mask  # 灰度圖是單通道，直接相乘
                    
                    # 保存結果
                    filename = file_path.stem
                    identity, view = parse_filename(filename)
                    body_path = os.path.join(body_dir, f"{filename}.jpg")
                    mask_path = os.path.join(mask_dir, f"{filename}.jpg")
                    gray_path = os.path.join(gray_dir, f"{filename}.jpg")
                    
                    cv2.imwrite(body_path, body_only)
                    cv2.imwrite(mask_path, cropped_mask * 255)
                    cv2.imwrite(gray_path, gray_only)
                    
                    # 提取並儲存關節點數據
                    landmarks_data = extract_landmarks(results.pose_landmarks, processed_image, filename, identity, view)
                    if landmarks_data:
                        all_landmarks_data.extend(landmarks_data)
                    
                    # 提取特徵（使用灰度圖）
                    lbp_feature = extract_lbp_features(gray_only)
                    hog_feature = extract_hog_features(gray_only)
                    
                    if lbp_feature is not None and hog_feature is not None:
                        lbp_features.append(lbp_feature)
                        hog_features.append(hog_feature)
                        labels.append(identity)
                    
                    print(f"已處理並保存圖片: {filename}")
                else:
                    print(f"無法創建掩碼: {file_path}")
            else:
                print(f"未檢測到關節點: {file_path}")
                
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {str(e)}")
    
    # 訓練並保存 SVM 模型
    train_and_save_svm(lbp_features, labels, output_dir, "lbp")
    train_and_save_svm(hog_features, labels, output_dir, "hog")
    
    # 統一保存所有關節點數據
    save_all_landmarks(all_landmarks_data, output_dir)

if __name__ == "__main__":
    input_directory = r"C:\Biometrics\data\testing\organized"
    output_directory = r"C:\Biometrics\data\testing\processed"
    
    process_images(input_directory, output_directory)
    print("處理完成!")