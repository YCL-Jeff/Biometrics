import cv2
import numpy as np
import mediapipe as mp
import os

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def filter_green_background(image):
    """過濾綠幕背景，返回背景遮罩（綠幕區域為 0，其他為 255）"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 200, 200])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    background_mask = cv2.bitwise_not(green_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    return background_mask

def detect_pose_and_grabcut(image_path, output_path, scale_factor=0.5):
    """使用 MediaPipe 偵測關節 + GrabCut 提取人體輪廓，處理綠幕背景和地板問題"""
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法讀取圖片: {image_path}")
        return None

    h, w, _ = image.shape

    # 縮放圖片以加速檢測
    scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
    scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    scaled_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

    # MediaPipe 姿勢檢測（在縮放圖片上）
    results = pose.process(scaled_rgb)
    if not results.pose_landmarks:
        print(f"⚠️ MediaPipe 未偵測到人體姿勢: {image_path}")
        return None

    # 將關鍵點座標映射回原始尺寸
    keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]

    # 過濾綠幕背景
    green_mask = filter_green_background(image)

    # 改進初始矩形，減少地板包含
    head_y = min(keypoints[7][1], keypoints[8][1], keypoints[11][1], keypoints[12][1]) - 150  # 減少頭頂 padding
    ankle_y = max(keypoints[27][1], keypoints[28][1]) + 20  # 減少腳踝 padding，避免地板
    body_x_min = min(keypoints[11][0], keypoints[12][0], keypoints[15][0], keypoints[16][0], keypoints[23][0], keypoints[24][0]) - 120
    body_x_max = max(keypoints[11][0], keypoints[12][0], keypoints[15][0], keypoints[16][0], keypoints[23][0], keypoints[24][0]) + 120

    # 防止座標超出邊界
    rect = (
        max(0, body_x_min),
        max(0, head_y),
        min(w, body_x_max) - max(0, body_x_min),
        min(h, ankle_y) - max(0, head_y)
    )

    # 初始化 GrabCut 遮罩
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 將綠幕區域標記為確定背景
    mask[green_mask == 0] = cv2.GC_BGD

    # 執行 GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # 創建前景遮罩
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 找最大輪廓並過濾
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ 找不到輪廓: {image_path}")
        return None

    # 過濾輪廓（根據面積和高度範圍）
    min_area = h * w * 0.02  # 至少 2% 的圖像面積
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        # 檢查輪廓高度是否在頭頂到腳踝範圍內
        if y < head_y or (y + h_cnt) > ankle_y + 50:  # 允許小範圍超出
            continue
        # 檢查長寬比（假設人體長寬比範圍）
        aspect_ratio = w_cnt / h_cnt if h_cnt > 0 else 0
        if 0.02 < aspect_ratio < 4.0:  # 放寬範圍，適應側面姿勢
            filtered_contours.append(cnt)

    if not filtered_contours:
        print(f"⚠️ 找不到符合條件的輪廓: {image_path}")
        return None

    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # 創建乾淨的遮罩
    clean_mask = np.zeros_like(mask2)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 形態學操作（去除地板噪點）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    final_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

    # 應用遮罩
    result = image * (final_mask[:, :, np.newaxis] // 255)

    # 繪製輪廓和邊界框（可視化用）
    result_with_contour = image.copy()
    cv2.drawContours(result_with_contour, [largest_contour], -1, (0, 0, 255), 2)
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
    cv2.rectangle(result_with_contour, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

    # 保存結果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, result)
    if success:
        print(f"✅ GrabCut 處理後影像成功儲存至: {output_path}")
    else:
        print(f"❌ GrabCut 影像儲存失敗: {output_path}")

    # 保存帶輪廓的結果
    contour_output_path = output_path.replace(".jpg", "_contour.jpg")
    cv2.imwrite(contour_output_path, result_with_contour)
    print(f"✅ 輪廓結果儲存至: {contour_output_path}")

    # 保存最終遮罩
    mask_output_path = output_path.replace(".jpg", "_mask.jpg")
    cv2.imwrite(mask_output_path, final_mask)
    print(f"✅ 遮罩儲存至: {mask_output_path}")

    return result

def batch_process_images(input_dir, output_dir, scale_factor=0.5):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"grabcut_{filename}")
            print(f"🚀 處理圖片: {filename}")
            detect_pose_and_grabcut(input_path, output_path, scale_factor)

if __name__ == "__main__":
    input_directory = r"C:\Biometrics\data\training"
    output_directory = r"C:\Biometrics\data\results"
    batch_process_images(input_directory, output_directory, scale_factor=0.5)
    print("✅ 批量處理完成")