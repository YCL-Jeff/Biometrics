import cv2
import numpy as np


# 載入預訓練的臉部檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # 取第一個檢測到的臉部
        face_center = (x + w // 2, y + h // 2)  # 臉部中心
        return img, face_center
    return img, None

# 測試
img, face_center = detect_face('test/DSC00166.JPG')
print("臉部中心:", face_center)