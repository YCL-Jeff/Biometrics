import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from collections import Counter

# Load landmark data
def load_landmarks(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

# Calculate distance
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Extract features (using only head_to_height_ratio)
def extract_features(landmarks_df, min_visibility=0.5, image_diagonal=1469):
    landmarks_df = landmarks_df[landmarks_df['visibility'] > min_visibility]
    
    def get_landmark_coords(lm_df, default_x=0, default_y=0):
        if lm_df.empty:
            return default_x, default_y
        return lm_df['x'].values[0], lm_df['y'].values[0]
    
    nose = landmarks_df[landmarks_df['landmark_id'] == 0]
    left_shoulder = landmarks_df[landmarks_df['landmark_id'] == 11]
    right_shoulder = landmarks_df[landmarks_df['landmark_id'] == 12]
    left_ankle = landmarks_df[landmarks_df['landmark_id'] == 27]
    right_ankle = landmarks_df[landmarks_df['landmark_id'] == 28]
    
    nose_x, nose_y = get_landmark_coords(nose)
    left_shoulder_x, left_shoulder_y = get_landmark_coords(left_shoulder)
    right_shoulder_x, right_shoulder_y = get_landmark_coords(right_shoulder)
    left_ankle_x, left_ankle_y = get_landmark_coords(left_ankle)
    right_ankle_x, right_ankle_y = get_landmark_coords(right_ankle)
    
    # Calculate height
    height_left = calculate_distance(nose_x, nose_y, left_ankle_x, left_ankle_y)
    height_right = calculate_distance(nose_x, nose_y, right_ankle_x, right_ankle_y)
    height = (height_left + height_right) / 2 if (height_left + height_right) > 0 else 1e-6
    
    # Calculate head to shoulder distance
    head_to_shoulder_left = calculate_distance(nose_x, nose_y, left_shoulder_x, left_shoulder_y)
    head_to_shoulder_right = calculate_distance(nose_x, nose_y, right_shoulder_x, right_shoulder_y)
    head_to_shoulder = (head_to_shoulder_left + head_to_shoulder_right) / 2 if (head_to_shoulder_left + head_to_shoulder_right) > 0 else 0
    
    # Calculate head_to_height_ratio
    head_to_height_ratio = head_to_shoulder / height if 0 <= head_to_shoulder / height <= 1 else 0
    
    return [head_to_height_ratio]  # Only return head_to_height_ratio

# Prepare training data
def prepare_training_data(base_dir):
    features = []
    labels = []
    available_identities = set()
    
    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('_landmarks.csv'):
                # Extract identity
                relative_path = os.path.relpath(root, base_dir)
                identity = relative_path.split('_')[-1]  # e.g., identity_XX
                file_path = os.path.join(root, filename)
                
                df = load_landmarks(file_path)
                if df is not None:
                    feature_vector = extract_features(df, min_visibility=0.5)
                    if feature_vector and not any(np.isnan(feature_vector)):
                        features.append(feature_vector)
                        labels.append(identity)
                        available_identities.add(identity)
    
    print(f"Number of available identities: {len(available_identities)}, Identity list: {sorted(available_identities)}")
    return np.array(features), np.array(labels)

# Train and evaluate model
def train_and_evaluate_model(features, labels, output_dir):
    if len(features) == 0 or len(labels) == 0:
        print("No features or labels available for training")
        return None, None
    
    # Count samples per class to determine max cv
    label_counts = Counter(labels)
    min_samples = min(label_counts.values())
    cv = max(2, min(5, min_samples - 1)) if min_samples > 1 else None
    if cv is None:
        print(f"Warning: Minimum samples per class ({min_samples}) is less than 2, skipping cross-validation")
    else:
        print(f"Using {cv}-fold cross-validation based on minimum samples per class: {min_samples}")
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Data augmentation
    features_expanded, labels_expanded = resample(features_scaled, labels, n_samples=1000, replace=True, random_state=42)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_expanded, labels_expanded, test_size=0.2, random_state=42, stratify=labels_expanded)
    
    # Train SVM model
    clf = SVC(kernel='rbf', C=0.5, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    # Cross-validation (if possible)
    if cv is not None:
        scores = cross_val_score(clf, features_scaled, labels, cv=cv)
        print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Predict
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Identity')
    plt.ylabel('True Identity')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Save model and scaler
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "identity_classifier.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    return clf, scaler

# Predict identity
def predict_identity(features, clf, scaler):
    features_scaled = scaler.transform(features)
    prediction = clf.predict(features_scaled)
    probability = clf.predict_proba(features_scaled).max()
    return prediction[0], probability

if __name__ == "__main__":
    base_dir = r"C:\Biometrics\data\results\landmarks_augmented"
    output_dir = r"C:\Biometrics\data\results"
    
    # Prepare training data
    features, labels = prepare_training_data(base_dir)
    
    # Train model
    clf, scaler = train_and_evaluate_model(features, labels, output_dir)
    
    # Example prediction (dynamically find the first file)
    if clf and scaler:
        identity_01_dir = r"C:\Biometrics\data\results\landmarks_augmented\identity_01"
        test_file = None
        for filename in os.listdir(identity_01_dir):
            if filename.endswith('_landmarks.csv'):
                test_file = os.path.join(identity_01_dir, filename)
                break
        
        if test_file:
            test_df = load_landmarks(test_file)
            if test_df is not None:
                test_features = extract_features(test_df, min_visibility=0.5)
                if test_features and not any(np.isnan(test_features)):
                    predicted_identity, confidence = predict_identity([test_features], clf, scaler)
                    print(f"Predicted identity: {predicted_identity}, Confidence score: {confidence:.4f}")
        else:
            print("Error: No landmarks.csv files found for identity_01")