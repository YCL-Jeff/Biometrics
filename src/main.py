import cv2
import numpy as np
import mediapipe as mp
import os

def detect_pose_and_canny_edges(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法讀取圖片: {image_path}")
        return None

    h, w, _ = image.shape

    # 使用MediaPipe偵測關節
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print(f"⚠️ 無法偵測到人體姿勢: {image_path}")
        return None

    keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]

    # 建立精細的遮罩
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    hull = cv2.convexHull(np.array(keypoints))
    cv2.fillConvexPoly(mask, hull, 255)

    # 適度擴張遮罩範圍
    mask = cv2.dilate(mask, np.ones((50, 50), np.uint8), iterations=4)

    # 套用遮罩到原圖
    segmented = cv2.bitwise_and(image, image, mask=mask)

    # 灰度化處理並高斯平滑
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0.5)

    # 使用 Canny 生成邊緣影像
    edges = cv2.Canny(blurred, threshold1=70, threshold2=135)

    # 儲存結果
    edge_output_path = output_path.replace(".jpg", "_edges.jpg")
    cv2.imwrite(edge_output_path, edges)

    print(f"✅ Canny邊緣影像儲存至: {edge_output_path}")

    return edges

# 測試影像
if __name__ == "__main__":
    test_image_path = r"C:\Biometrics\data\training\016z050pf.jpg"
    canny_output_path = r"C:\Biometrics\data\results\016z050pf_canny_edges.jpg"

    edges = detect_pose_and_canny(test_image_path, canny_output_path)
    if edges is not None:
        cv2.imwrite(canny_output_path, edges)
        print(f"✅ Canny邊緣影像儲存至: {canny_output_path}")
    else:
        print("❌ 無法生成邊緣影像")
