import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from scipy import ndimage

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1)
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
mp_image = mp.Image  # 引入 mp.Image

def extract_features(image_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 無法讀取圖片: {image_path}")
        return None
    # 確保影像為正方形
    h, w = image.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image = cv2.resize(image, (256, 256))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 創建 mp.Image 並傳遞尺寸
    mp_img = mp_image(image_format=mp_image.FORMAT_RGB, data=image_rgb)
    image_size = (256, 256)  # 顯式指定影像尺寸

    # 面部模糊化
    face_results = face_detection.process(image_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = min(w - x, width + 2 * margin)
            height = min(h - y, height + 2 * margin)
            image[y:y+height, x:x+width] = cv2.GaussianBlur(image[y:y+height, x:x+width], (51, 51), 0)

    # 姿勢檢測
    try:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
            pose_results = pose.process(mp_img)  # 使用 mp.Image 處理
            if not pose_results.pose_landmarks:
                print(f"⚠️ 無法從 {image_path} 偵測人體姿勢")
                return None
    except Exception as e:
        print(f"⚠️ 姿勢檢測錯誤: {e}")
        return None

    landmarks = pose_results.pose_landmarks.landmark
    img_h, img_w, _ = image.shape
    keypoints = np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks])

    # 邊界框
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    valid_points = (x_coords > 0) & (y_coords > 0)
    if not np.any(valid_points):
        print(f"⚠️ 無法從 {image_path} 提取邊界框，可能未偵測到人體")
        return None
    x_min, x_max = np.min(x_coords[valid_points]), np.max(x_coords[valid_points])
    y_min, y_max = np.min(y_coords[valid_points]), np.max(y_coords[valid_points])
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = width / height if height > 0 else 0

    # 輪廓特徵 (Hu矩)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_moments = np.zeros(7)
    side_contour = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        side_contour = np.sum(mask > 0) / (img_h * img_w)
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)

    # Gabor濾波器
    def apply_gabor_filters(image):
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        frequencies = [0.1, 0.25]
        gabor_features = []
        for theta in orientations:
            for frequency in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 5.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                gabor_features.extend([np.mean(filtered), np.std(filtered)])
        return np.array(gabor_features)
    gabor_features = apply_gabor_filters(gray)

    # HOG特徵
    roi = gray[int(y_min):int(y_max), int(x_min):int(x_max)]
    if roi.size == 0:
        hog_features = np.zeros(36)
    else:
        try:
            roi_resized = cv2.resize(roi, (64, 128))
            hog_features = hog(roi_resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
            if len(hog_features) > 36:
                hog_features = hog_features[:36]
        except Exception as e:
            print(f"⚠️ HOG特徵提取錯誤: {e}")
            hog_features = np.zeros(36)

    # 姿勢特徵
    def extract_pose_features(keypoints):
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW = 0, 11, 12, 13, 14
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP = 15, 16, 23, 24
        LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 25, 26, 27, 28

        def calculate_angle(p1, p2, p3):
            if np.any(p1 <= 0) or np.any(p2 <= 0) or np.any(p3 <= 0):
                return 0
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            try:
                angle_rad = np.arccos((a**2 + c**2 - b**2) / (2 * a * c))
                return np.degrees(angle_rad)
            except:
                return 0

        def calculate_line_angle(p1, p2):
            if np.any(p1 <= 0) or np.any(p2 <= 0):
                return 0
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle_rad = np.arctan2(dy, dx)
            return np.degrees(angle_rad)

        height = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
        shoulder_width = np.linalg.norm(keypoints[LEFT_SHOULDER] - keypoints[RIGHT_SHOULDER])
        torso_length = np.linalg.norm((keypoints[LEFT_SHOULDER] + keypoints[RIGHT_SHOULDER])/2 - 
                                      (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP])/2)
        left_arm_length = (np.linalg.norm(keypoints[LEFT_SHOULDER] - keypoints[LEFT_ELBOW]) + 
                           np.linalg.norm(keypoints[LEFT_ELBOW] - keypoints[LEFT_WRIST]))
        right_arm_length = (np.linalg.norm(keypoints[RIGHT_SHOULDER] - keypoints[RIGHT_ELBOW]) + 
                            np.linalg.norm(keypoints[RIGHT_ELBOW] - keypoints[RIGHT_WRIST]))
        left_leg_length = (np.linalg.norm(keypoints[LEFT_HIP] - keypoints[LEFT_KNEE]) + 
                           np.linalg.norm(keypoints[LEFT_KNEE] - keypoints[LEFT_ANKLE]))
        right_leg_length = (np.linalg.norm(keypoints[RIGHT_HIP] - keypoints[RIGHT_KNEE]) + 
                            np.linalg.norm(keypoints[RIGHT_KNEE] - keypoints[RIGHT_ANKLE]))

        features = [
            shoulder_width / height if height > 0 else 0,
            torso_length / height if height > 0 else 0,
            ((left_arm_length + right_arm_length) / 2) / height if height > 0 else 0,
            ((left_leg_length + right_leg_length) / 2) / height if height > 0 else 0,
            calculate_angle(keypoints[LEFT_SHOULDER], keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST]),
            calculate_angle(keypoints[RIGHT_SHOULDER], keypoints[RIGHT_ELBOW], keypoints[RIGHT_WRIST]),
            calculate_angle(keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE]),
            calculate_angle(keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE]),
            calculate_line_angle(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]),
            calculate_line_angle(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])
        ]
        return np.array(features)

    pose_features = extract_pose_features(keypoints)

    # 最終特徵向量
    final_features = np.concatenate([
        np.array([aspect_ratio]),
        hu_moments,
        gabor_features,
        hog_features,
        pose_features
    ])
    scaler = MinMaxScaler()
    final_features = scaler.fit_transform([final_features])[0]

    print(f"Extracted features from {image_path}: aspect_ratio={aspect_ratio:.2f}, hu_moments={hu_moments[:2]}, "
          f"gabor_features={gabor_features[:2]:.2f}, hog_features={hog_features[:2]:.2f}, pose_features={pose_features[:2]:.2f}")
    return final_features

def process_folder(folder_path):
    features = {}
    failed_images = []
    print(f"Listing files in {folder_path}:")
    files = os.listdir(folder_path)
    print(files)
    for filename in files:
        if filename.endswith('.jpg'):
            parts = filename.split('z')
            if len(parts) != 2:
                print(f"Invalid filename format: {filename}")
                failed_images.append(filename)
                continue
            prefix, rest = parts
            if rest.endswith('pf.jpg'):
                subject_id = rest[:-6]
            elif rest.endswith('ps.jpg'):
                subject_id = rest[:-6]
            else:
                print(f"Invalid filename suffix: {filename}")
                failed_images.append(filename)
                continue
            image_path = os.path.join(folder_path, filename)
            try:
                feats = extract_features(image_path)
                if feats is not None:
                    if subject_id not in features:
                        features[subject_id] = []
                    features[subject_id].append(feats)
                else:
                    failed_images.append(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_images.append(image_path)
    if failed_images:
        print(f"Failed to process {len(failed_images)} images: {failed_images}")
    return features

def augment_image(image):
    augmented = []
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.flip(image, 1))
    return augmented

# 添加測試邏輯
if __name__ == "__main__":
    test_image_path = r"C:\Biometrics\data\training\016z050pf.jpg"
    if os.path.exists(test_image_path):
        print(f"Testing feature extraction on {test_image_path}")
        features = extract_features(test_image_path)
        if features is not None:
            print(f"Successfully extracted features: {features[:5]}...")
        else:
            print("Feature extraction failed.")
    else:
        print(f"Test image {test_image_path} does not exist.")