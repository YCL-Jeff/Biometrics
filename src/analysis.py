from feature_extraction import process_folder, augment_image
from analysis import compute_distances, plot_histograms, compute_ccr, compute_eer, evaluate_at_eer
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
import cv2

def split_data(features):
    train_features = {}
    test_features = {}
    for subject_id, feats in features.items():
        if len(feats) >= 2:
            train_features[subject_id] = [feats[0]]
            test_features[subject_id] = [feats[1]]
    return train_features, test_features

def prepare_labels(features):
    return np.array([[int(k)] * len(v) for k, v in features.items()]).flatten()

def ensemble_predict(clf1_pred, dist_pred, weights=[0.5, 0.5]):
    """融合分類器和距離預測"""
    final_pred = np.round(weights[0] * clf1_pred + weights[1] * dist_pred).astype(int)
    return final_pred

def main():
    training_folder = r'C:\Biometrics\data\training'
    results_dir = os.path.join(os.path.dirname(training_folder), 'results')
    print(f"Processing folder: {training_folder}")
    
    if not os.path.exists(training_folder):
        print("Error: Training folder does not exist!")
        return

    print("Extracting features...")
    all_features = process_folder(training_folder)
    if not all_features:
        print("Error: No features extracted!")
        return
    print(f"Extracted features for {len(all_features)} subjects")

    # 資料增強
    augmented_features = {}
    for subject_id, feats in all_features.items():
        augmented_features[subject_id] = feats
        for feat in feats:
            image_path = [path for path in os.listdir(training_folder) if subject_id in path][0]
            image = cv2.imread(os.path.join(training_folder, image_path))
            for aug_img in augment_image(image):
                aug_feat = extract_features(os.path.join(training_folder, f"{subject_id}_aug.jpg"))
                if aug_feat:
                    augmented_features[subject_id].append(aug_feat)
    all_features = augmented_features

    train_features, test_features = split_data(all_features)
    if not train_features or not test_features:
        print("Error: No valid train/test data after splitting!")
        return
    print(f"Training subjects: {len(train_features)}, Test subjects: {len(test_features)}")

    print("Computing distances...")
    intra_distances, inter_distances = compute_distances(train_features, test_features)
    print(f"Intra-class distances: {len(intra_distances)}, Inter-class distances: {len(inter_distances)}")

    if not os.path.exists(os.path.join(results_dir, 'histograms')):
        os.makedirs(os.path.join(results_dir, 'histograms'))
    plot_histograms(intra_distances, inter_distances, os.path.join(results_dir, 'histograms', 'variation_histogram.png'))
    print("Histogram saved")

    ccr = compute_ccr(train_features, test_features)
    eer, eer_threshold = compute_eer(intra_distances, inter_distances)
    ccr_at_eer = evaluate_at_eer(train_features, test_features, eer_threshold)
    print(f"CCR: {ccr:.4f}")
    print(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")
    print(f"CCR at EER: {ccr_at_eer:.4f}")

    try:
        X_train = np.array([feat for feats in train_features.values() for feat in feats])
        y_train = prepare_labels(train_features)
        X_test = np.array([feat for feats in test_features.values() for feat in feats])
        y_test = prepare_labels(test_features)
        if X_train.size == 0 or X_test.size == 0:
            raise ValueError("Training or test data is empty!")
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 網格搜尋最佳化
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        clf.fit(X_train, y_train)
        print(f"Best parameters: {clf.best_params_}")
        y_pred_rf = clf.predict(X_test)
        ccr_rf = np.mean(y_pred_rf == y_test)
        print(f"CCR with Random Forest: {ccr_rf:.4f}")

        # 距離預測
        dist_pred = np.array([1 if dist < eer_threshold else 0 for dist in intra_distances + inter_distances][:len(y_test)])
        y_pred_ensemble = ensemble_predict(y_pred_rf, dist_pred, weights=[0.7, 0.3])
        ccr_ensemble = np.mean(y_pred_ensemble == y_test)
        print(f"CCR with Ensemble: {ccr_ensemble:.4f}")

    except Exception as e:
        print(f"Error in model training: {e}")

    print("Performing cross-validation...")
    all_images = [(sid, feat) for sid, feats in all_features.items() for feat in feats]
    X = np.array([img[1] for img in all_images])
    y = np.array([int(img[0]) for img in all_images])
    if len(X) < 2:
        print("Error: Insufficient data for cross-validation!")
        return
    X = scaler.fit_transform(X)
    kf = KFold(n_splits=min(5, len(all_features)), shuffle=True, random_state=42)
    ccrs, eers = [], []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"Cross-validation fold {fold}/{min(5, len(all_features))}")
        train_X, test_X = X[train_idx], X[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], 
                                     max_depth=clf.best_params_['max_depth'], random_state=42)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)
        ccr = np.mean(y_pred == test_y)
        ccrs.append(ccr)
        intra, inter = [], []
        for i in range(len(test_X)):
            for j in range(len(train_X)):
                dist = np.linalg.norm(test_X[i] - train_X[j])
                if test_y[i] == train_y[j]:
                    intra.append(dist)
                else:
                    inter.append(dist)
        if intra and inter:
            scores = intra + inter
            labels = [1] * len(intra) + [0] * len(inter)
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
            fnr = 1 - tpr
            eer = fpr[np.argmin(np.abs(fpr - fnr))] if fpr.size > 0 else 0.5
        else:
            eer = 0.5
        eers.append(eer)
    if ccrs and eers:
        print(f"Average CCR (Cross-validation): {np.mean(ccrs):.4f} ± {np.std(ccrs):.4f}")
        print(f"Average EER (Cross-validation): {np.mean(eers):.4f} ± {np.std(eers):.4f}")
    else:
        print("Error: No valid cross-validation results!")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"CCR: {ccr:.4f}\n")
        f.write(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}\n")
        f.write(f"CCR at EER: {ccr_at_eer:.4f}\n")
        f.write(f"CCR with Random Forest: {ccr_rf:.4f}\n")
        f.write(f"CCR with Ensemble: {ccr_ensemble:.4f}\n")
        f.write(f"Average CCR (Cross-validation): {np.mean(ccrs):.4f} ± {np.std(ccrs):.4f}\n" if ccrs else "Average CCR (Cross-validation): N/A\n")
        f.write(f"Average EER (Cross-validation): {np.mean(eers):.4f} ± {np.std(eers):.4f}\n" if eers else "Average EER (Cross-validation): N/A\n")
    print("Results saved to metrics.txt")

if __name__ == "__main__":
    main()