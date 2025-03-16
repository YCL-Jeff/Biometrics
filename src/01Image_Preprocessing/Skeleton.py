import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import mediapipe as mp
import matplotlib.pyplot as plt

def parse_filename(filename):
    """解析檔案名稱，提取身份ID和視角"""
    name = filename.split('.')[0]  # 移除副檔名
    identity = name[5:7]  # 提取數字部分作為身份ID (改為兩個字符，如 '05')
    view = name[-2:]      # 提取視角 (如 'pf', 'ps')
    return identity, view

def detect_pose_landmarks(image):
    """使用 MediaPipe 偵測人體關節點"""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

def calculate_distance(point1, point2):
    """計算兩點間的歐幾里得距離"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_body_ratios(results, image_shape):
    """計算身體部位比例"""
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    height, width = image_shape[:2]
    
    points = {idx: (landmark.x * width, landmark.y * height) for idx, landmark in enumerate(landmarks)}
    
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    shoulder_width = calculate_distance(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    left_height = calculate_distance(points[LEFT_SHOULDER], points[LEFT_ANKLE])
    right_height = calculate_distance(points[RIGHT_SHOULDER], points[RIGHT_ANKLE])
    height = (left_height + right_height) / 2
    waist_width = calculate_distance(points[LEFT_HIP], points[RIGHT_HIP])
    left_ear_to_waist = calculate_distance(points[LEFT_EAR], points[LEFT_HIP])
    right_ear_to_waist = calculate_distance(points[RIGHT_EAR], points[RIGHT_HIP])
    ear_to_waist = (left_ear_to_waist + right_ear_to_waist) / 2
    left_waist_to_knee = calculate_distance(points[LEFT_HIP], points[LEFT_KNEE])
    right_waist_to_knee = calculate_distance(points[RIGHT_HIP], points[RIGHT_KNEE])
    waist_to_knee = (left_waist_to_knee + right_waist_to_knee) / 2
    
    shoulder_to_height_ratio = shoulder_width / height if height != 0 else 0
    shoulder_to_waist_ratio = shoulder_width / waist_width if waist_width != 0 else 0
    ear_to_waist_to_waist_to_knee_ratio = ear_to_waist / waist_to_knee if waist_to_knee != 0 else 0
    
    return {
        "shoulder_to_height_ratio": shoulder_to_height_ratio,
        "shoulder_to_waist_ratio": shoulder_to_waist_ratio,
        "ear_to_waist_to_waist_to_knee_ratio": ear_to_waist_to_waist_to_knee_ratio
    }

def draw_skeleton(image, results, output_path):
    """繪製骨架並標記關鍵部位"""
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width = image.shape[:2]
        key_points = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
        labels = ["Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", 
                  "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        
        for idx, point_idx in enumerate(key_points):
            x = int(landmarks[point_idx].x * width)
            y = int(landmarks[point_idx].y * height)
            cv2.putText(annotated_image, labels[idx], (x, y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated_image)
    print(f"已生成骨架圖: {output_path}")

def process_images(input_dir, output_dir):
    """處理資料夾中的所有圖片並計算比例"""
    ratios_data = []
    image_names = []
    identities = []
    skeleton_images = []
    all_views = []
    
    for file_path in Path(input_dir).glob('*.jpg'):
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue
            
            results = detect_pose_landmarks(image)
            if not results.pose_landmarks:
                print(f"未檢測到關節點: {file_path}")
                continue
            
            ratios = compute_body_ratios(results, image.shape)
            if ratios is None:
                print(f"無法計算比例: {file_path}")
                continue
            
            identity, view = parse_filename(file_path.name)
            ratios_data.append(ratios)
            image_names.append(file_path.name)
            identities.append(identity)
            all_views.append(view)
            # 儲存前三個身份的骨架圖
            if len(set(identities)) <= 3 and identity not in [i for i, _ in skeleton_images]:
                output_skeleton_path = os.path.join(output_dir, "skeleton_plots", f"skeleton_{identity}_{view}.jpg")
                draw_skeleton(image, results, output_skeleton_path)
                skeleton_images.append((identity, output_skeleton_path))
            
            print(f"已處理圖片: {file_path}")
            
        except Exception as e:
            print(f"處理圖片時出錯 {file_path}: {e}")
    
    return ratios_data, image_names, identities, all_views, skeleton_images

def save_ratios_to_csv(ratios_data, image_names, identities, all_views, output_path):
    """將比例數據保存為CSV，包括 View 列"""
    df = pd.DataFrame(ratios_data, index=image_names)
    df['Identity'] = identities
    df['View'] = all_views
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"比例數據已保存到: {output_path}")

def main():
    input_directory = r"C:\Biometrics\data\results\body"
    output_directory = r"C:\Biometrics\data\results\analysis"
    
    ratios_data, image_names, identities, all_views, skeleton_images = process_images(input_directory, output_directory)
    
    if ratios_data:
        csv_path = os.path.join(output_directory, "body_ratios.csv")
        save_ratios_to_csv(ratios_data, image_names, identities, all_views, csv_path)
        print("身體比例計算和骨架圖生成完成!")
    else:
        print("沒有可處理的數據")

if __name__ == "__main__":
    main()