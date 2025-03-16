import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from collections import Counter

# Load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return clf, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

# Extract Hu Moments
def extract_hu_moments(img):
    if img is None:
        return None
    _, img_binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(img_binary)
    hu_moments = cv2.HuMoments(moments)
    # Log transform to reduce dynamic range
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments.flatten()

# Extract LBP features
def extract_lbp_features(img):
    if img is None:
        return None
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    return lbp.flatten()

# Extract HOG features
def extract_hog_features(img):
    if img is None:
        return None
    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

# Prepare test data
def prepare_test_data(mask_base_dir, gray_base_dir):
    features_hu = []
    features_lbp = []
    features_hog = []
    true_labels = []
    available_identities = set()
    
    for identity_folder in Path(mask_base_dir).iterdir():
        if not identity_folder.is_dir():
            continue
        identity = identity_folder.name.split('_')[-1]
        gray_folder = os.path.join(gray_base_dir, f"identity_{identity}")
        if not os.path.exists(gray_folder):
            print(f"Gray folder not found for identity {identity}: {gray_folder}")
            continue
        
        for file_path in identity_folder.glob('*.jpg'):
            try:
                # Read mask image (grayscale)
                mask_img = cv2.imread(str(file_path), 0)
                if mask_img is None:
                    print(f"Unable to read mask: {file_path}")
                    continue
                
                # Read corresponding gray image
                gray_file_path = str(file_path).replace("mask", "gray")
                gray_img = cv2.imread(gray_file_path, 0)  # Read directly as grayscale
                if gray_img is None:
                    print(f"Unable to read gray image: {gray_file_path}")
                    continue
                
                # Extract features
                hu_feature = extract_hu_moments(mask_img)
                lbp_feature = extract_lbp_features(gray_img)
                hog_feature = extract_hog_features(gray_img)
                
                if hu_feature is not None and lbp_feature is not None and hog_feature is not None:
                    features_hu.append(hu_feature)
                    features_lbp.append(lbp_feature)
                    features_hog.append(hog_feature)
                    true_labels.append(identity)
                    available_identities.add(identity)
                else:
                    print(f"Feature extraction failed for {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Number of test identities: {len(available_identities)}, Identity list: {sorted(available_identities)}")
    return np.array(features_hu), np.array(features_lbp), np.array(features_hog), np.array(true_labels)

# Evaluate model
def evaluate_model(clf, scaler, features, true_labels, feature_type):
    if clf is None or scaler is None or len(features) == 0:
        print(f"Unable to evaluate {feature_type} model")
        return None, None
    
    features_scaled = scaler.transform(features)
    predictions = clf.predict(features_scaled)
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    return accuracy, cm

# Main execution
if __name__ == "__main__":
    # Define paths
    mask_base_dir = r"C:\Biometrics\data\testing\processed\mask"
    gray_base_dir = r"C:\Biometrics\data\testing\processed\gray"
    output_dir = r"C:\Biometrics\data\testing\processed"
    
    # Load models and scalers
    hu_model_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\hu_moments_svm.joblib"
    hu_scaler_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\hu_moments_scaler.joblib"
    lbp_model_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\lbp_svm.joblib"
    lbp_scaler_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\lbp_scaler.joblib"
    hog_model_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\hog_svm.joblib"
    hog_scaler_path = r"C:\Biometrics\data\testing\processed\body\identity_11\models\hog_scaler.joblib"
    
    hu_clf, hu_scaler = load_model_and_scaler(hu_model_path, hu_scaler_path)
    lbp_clf, lbp_scaler = load_model_and_scaler(lbp_model_path, lbp_scaler_path)
    hog_clf, hog_scaler = load_model_and_scaler(hog_model_path, hog_scaler_path)
    
    # Prepare test data
    features_hu, features_lbp, features_hog, true_labels = prepare_test_data(mask_base_dir, gray_base_dir)
    
    # Evaluate models
    accuracies = {}
    confusion_matrices = {}
    
    if hu_clf and hu_scaler and len(features_hu) > 0:
        accuracy_hu, cm_hu = evaluate_model(hu_clf, hu_scaler, features_hu, true_labels, "Hu Moments")
        accuracies["Hu Moments"] = accuracy_hu
        confusion_matrices["Hu Moments"] = cm_hu
    
    if lbp_clf and lbp_scaler and len(features_lbp) > 0:
        accuracy_lbp, cm_lbp = evaluate_model(lbp_clf, lbp_scaler, features_lbp, true_labels, "LBP")
        accuracies["LBP"] = accuracy_lbp
        confusion_matrices["LBP"] = cm_lbp
    
    if hog_clf and hog_scaler and len(features_hog) > 0:
        accuracy_hog, cm_hog = evaluate_model(hog_clf, hog_scaler, features_hog, true_labels, "HOG")
        accuracies["HOG"] = accuracy_hog
        confusion_matrices["HOG"] = cm_hog
    
    # Print results
    for feature_type, accuracy in accuracies.items():
        print(f"{feature_type} Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrices
    for feature_type, cm in confusion_matrices.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Identity')
        plt.ylabel('True Identity')
        plt.title(f'Confusion Matrix - {feature_type}')
        plt.show()