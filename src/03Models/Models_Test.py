import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage.feature import local_binary_pattern, hog
import joblib
import logging

# 設置日誌級別以隱藏 MediaPipe 的非關鍵警告（如果需要）
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# 假設的預處理函數（與訓練時一致）
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
    """使用 MediaPipe 偵測人體關節點（僅用於獲取掩碼）"""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=2, 
        min_detection_confidence=0.5,
        enable_segmentation=True
    ) as pose:
        height, width = image.shape[:2]
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results.image_width = width
        results.image_height = height
        return results

def get_improved_body_mask(image, results):
    """使用 MediaPipe 的分割掩碼獲得人體掩碼"""
    if results.segmentation_mask is None:
        return None
    
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
    return None

def get_bounding_box_from_mask(mask):
    """從掩碼獲取外接矩形並轉換為正方形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    size = max(w, h)
    center_x, center_y = x + w//2, y + h//2
    new_x = max(0, center_x - size//2)
    new_y = max(0, center_y - size//2)
    height, width = mask.shape
    if new_x + size > width:
        new_x = width - size
    if new_y + size > height:
        new_y = height - size
    if size <= 0:
        return None
    return (new_x, new_y, size, size)

def crop_to_square(image, bbox):
    """根據給定的邊框裁切圖片為正方形"""
    if bbox is None:
        return None
    x, y, size, _ = bbox
    cropped = image[y:y+size, x:x+size]
    if cropped.size == 0:
        return None
    return cropped

def resize_image(image, size=(128, 128)):
    """將圖像調整為指定大小"""
    if image is None:
        return None
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def parse_filename(filename):
    """解析檔案名稱，提取身份ID"""
    name = filename.split('.')[0]
    identity = name[5:7]  # 提取身份ID (如 '50')
    return identity

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

def evaluate_model(test_dir, model_path, scaler_path, feature_extractor, model_name):
    """評估指定模型的性能"""
    # 載入模型和 Scaler
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 準備測試數據
    X_test = []
    y_test = []
    
    for file_path in Path(test_dir).glob('*.jpg'):
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            # 預處理圖像
            processed_image = preprocess_image(image)
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            
            # 檢測人體關節點並獲取掩碼
            results = detect_pose_landmarks(processed_image)
            body_mask = get_improved_body_mask(processed_image, results)
            
            if body_mask is not None:
                bbox = get_bounding_box_from_mask(body_mask)
                cropped_gray = crop_to_square(gray_image, bbox)
                if cropped_gray is not None:
                    cropped_gray = resize_image(cropped_gray, (128, 128))
                    if cropped_gray is not None:
                        # 應用掩碼（假設 mask 已調整到 128x128）
                        cropped_mask = resize_image(body_mask, (128, 128))
                        gray_only = cropped_gray * cropped_mask
                        
                        # 提取特徵
                        feature = feature_extractor(gray_only)
                        if feature is not None:
                            X_test.append(feature)
                            y_test.append(parse_filename(file_path.stem))
                    else:
                        print(f"調整大小失敗: {file_path}")
                else:
                    print(f"裁切失敗: {file_path}")
            else:
                print(f"無法創建掩碼: {file_path}")
                
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {str(e)}")
    
    if not X_test or not y_test:
        print(f"無有效測試數據，無法評估 {model_name} 模型")
        return None
    
    # 正規化特徵
    X_test_scaled = scaler.transform(X_test)
    
    # 預測
    y_pred = clf.predict(X_test_scaled)
    
    # 評估性能
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\n{model_name} 模型評估結果:")
    print(f"準確率: {accuracy * 100:.2f}%")
    print("混淆矩陣:\n", conf_matrix)
    print("分類報告:\n", class_report)
    
    return accuracy, conf_matrix, class_report

if __name__ == "__main__":
    # 定義路徑
    test_directory = r"C:\Biometrics\data\testing"  # 假設測試數據目錄
    model_paths = {
        "lbp": {
            "model": r"C:\Biometrics\data\results\models\lbp_svm.joblib",
            "scaler": r"C:\Biometrics\data\results\models\lbp_scaler.joblib",
            "extractor": extract_lbp_features
        },
        "hog": {
            "model": r"C:\Biometrics\data\results\models\hog_svm.joblib",
            "scaler": r"C:\Biometrics\data\results\models\hog_scaler.joblib",
            "extractor": extract_hog_features
        }
    }
    
    # 評估每個模型
    results = {}
    for model_name, paths in model_paths.items():
        accuracy, conf_matrix, class_report = evaluate_model(
            test_directory, paths["model"], paths["scaler"], paths["extractor"], model_name
        )
        if accuracy is not None:
            results[model_name] = {"accuracy": accuracy, "confusion_matrix": conf_matrix, "classification_report": class_report}
    
    # 總結結果
    if results:
        print("\n總結所有模型性能:")
        for model_name, result in results.items():
            print(f"{model_name} 模型準確率: {result['accuracy'] * 100:.2f}%")
    
    # 部署建議（根據準確率）
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0] if results else None
    if best_model:
        print(f"\n推薦部署模型: {best_model} (準確率: {results[best_model]['accuracy'] * 100:.2f}%)")
    else:
        print("\n無有效模型評估結果，建議檢查測試數據或模型訓練過程。")