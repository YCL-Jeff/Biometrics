import cv2 as cv
import numpy as np
import os
import pandas as pd
from pathlib import Path

def build_gabor_filters(k_size=31, sigma=4.0, theta_list=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       lamda=10.0, gamma=0.5, psi=0):
    filters = []
    for theta in theta_list:
        kern = cv.getGaborKernel((k_size, k_size), sigma, theta, lamda, gamma, psi, ktype=cv.CV_32F)
        filters.append(kern)
    return filters

def apply_gabor_filters(image, filters):
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    responses = [cv.filter2D(gray, cv.CV_32F, kern) for kern in filters]
    return responses

def extract_gabor_features(responses):
    features = []
    for resp in responses:
        resp_flat = resp.flatten()
        non_zero_mask = resp_flat != 0
        if np.any(non_zero_mask):
            entropy = -np.sum(resp_flat[non_zero_mask] * np.log2(np.abs(resp_flat[non_zero_mask]) + 1e-10)) / np.sum(non_zero_mask)
        else:
            entropy = 0.0
        mean_val = np.mean(resp_flat)
        std_val = np.std(resp_flat)
        energy = np.sum(np.square(resp_flat))
        features.extend([mean_val, std_val, energy, entropy])
    return features

def parse_filename(filename):
    name = filename.split('.')[0]
    identity = name[5:8]
    view = name[-2:]
    return identity, view

def process_images_with_gabor(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    filters = build_gabor_filters()
    all_features = []
    all_image_names = []
    all_identities = []
    all_views = []
    for image_file in image_files:
        img_path = os.path.join(input_dir, image_file)
        img = cv.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {image_file}")
            continue
        responses = apply_gabor_filters(img, filters)
        features = extract_gabor_features(responses)
        all_features.append(features)
        identity, view = parse_filename(image_file)
        all_image_names.append(image_file)
        all_identities.append(identity)
        all_views.append(view)
        print(f"Processed {image_file} and extracted Gabor features")
    feature_names = [f'gabor_{i}_{stat}' for i in range(len(filters)) for stat in ['mean', 'std', 'energy', 'entropy']]
    df = pd.DataFrame(all_features, index=all_image_names, columns=feature_names)
    df['Identity'] = all_identities
    df['View'] = all_views
    output_path = os.path.join(output_dir, "gabor_features.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=True)
    print(f"Gabor features saved to {output_path}")

if __name__ == "__main__":
    input_dir = r"C:\Biometrics\data\results\body"
    output_dir = r"C:\Biometrics\data\results\analysis\gabor_features"
    process_images_with_gabor(input_dir, output_dir)