import numpy as np
import cv2 as cv
import os
import csv

def parse_filename(filename):
    """解析檔案名稱，提取身份ID和視角"""
    name = filename.split('.')[0]  # 移除副檔名
    identity = name[5:7]  # 提取身份ID (如 '50')
    view = name[-1:]      # 提取視角 (如 'f', 's')
    return identity, view

def apply_high_pass_filter(img, threshold=5):
    """Apply high-pass filter to remove values close to zero"""
    # Create a copy of the image
    filtered_img = img.copy()
    
    # Apply thresholding to filter out values close to zero
    # Any pixel value below the threshold will be set to zero
    filtered_img[filtered_img < threshold] = 0
    
    return filtered_img

def apply_stronger_high_pass_filter(img, lower_threshold=5, upper_threshold=250):
    """Apply a more aggressive high-pass filter that keeps only mid-high range values"""
    # Create a mask for values within our desired range
    mask = (img >= lower_threshold) & (img <= upper_threshold)
    
    # Create a new image with only the values in our desired range
    filtered_img = np.zeros_like(img)
    filtered_img[mask] = img[mask]
    
    return filtered_img

def apply_histogram_equalization(img, filter_type='none', lower_threshold=5, upper_threshold=250):
    """Apply histogram equalization to image with optional filtering"""
    # Split the image into channels
    b, g, r = cv.split(img)
    
    # Apply filtering if requested
    if filter_type == 'basic':
        b = apply_high_pass_filter(b, lower_threshold)
        g = apply_high_pass_filter(g, lower_threshold)
        r = apply_high_pass_filter(r, lower_threshold)
    elif filter_type == 'strong':
        b = apply_stronger_high_pass_filter(b, lower_threshold, upper_threshold)
        g = apply_stronger_high_pass_filter(g, lower_threshold, upper_threshold)
        r = apply_stronger_high_pass_filter(r, lower_threshold, upper_threshold)
    
    # Apply histogram equalization to each channel
    b_eq = cv.equalizeHist(b)
    g_eq = cv.equalizeHist(g)
    r_eq = cv.equalizeHist(r)
    
    # Merge the equalized channels
    img_eq = cv.merge((b_eq, g_eq, r_eq))
    
    return img_eq

def calculate_histogram(img, ignore_zeros=False):
    """Calculate histogram data for the image"""
    histograms = {}
    color = ('b', 'g', 'r')
    
    for i, col in enumerate(color):
        if ignore_zeros:
            # Create a mask to ignore zero values
            mask = img[:,:,i] > 0
            channel_data = img[:,:,i][mask]
            if len(channel_data) > 0:  # Check if there are any non-zero values
                hist, bins = np.histogram(channel_data, 256, [1, 256])  # Start from 1 to ignore zeros
                histograms[col] = hist
        else:
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            histograms[col] = hist.flatten()
    
    return histograms

def save_histogram_to_csv(histograms, filename, csv_dir, img_type, identity, view):
    """Save histogram data to CSV file"""
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"histogram_{img_type}.csv")
    
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        header = ['filename', 'identity', 'view', 'channel', 'bin', 'count']
        writer = csv.writer(csvfile)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(header)
        
        # Write histogram data
        for channel, hist_data in histograms.items():
            for bin_idx, count in enumerate(hist_data):
                writer.writerow([filename, identity, view, channel, bin_idx, int(count)])

def process_images(input_dir, csv_dir, lower_threshold=5, upper_threshold=250):
    """Process all images and save histogram data to CSV files"""
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Create metadata CSV file
    metadata_csv_path = os.path.join(csv_dir, "image_metadata.csv")
    with open(metadata_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Identity', 'View'])
        
        # Process each image
        for image_file in image_files:
            # Extract identity and view from filename
            try:
                identity, view = parse_filename(image_file)
                print(f"File: {image_file}, Identity: {identity}, View: {view}")
                csv_writer.writerow([image_file, identity, view])
            except Exception as e:
                print(f"Warning: Could not parse filename {image_file}: {e}")
                identity, view = "unknown", "unknown"
                csv_writer.writerow([image_file, identity, view])
            
            # Read image
            img_path = os.path.join(input_dir, image_file)
            img = cv.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read {image_file}")
                continue
            
            # Apply regular histogram equalization
            img_eq = apply_histogram_equalization(img, filter_type='none')
            
            # Apply histogram equalization with basic high-pass filter
            img_eq_basic = apply_histogram_equalization(img, filter_type='basic', lower_threshold=lower_threshold)
            
            # Apply histogram equalization with stronger high-pass filter
            img_eq_strong = apply_histogram_equalization(img, filter_type='strong', 
                                                       lower_threshold=lower_threshold,
                                                       upper_threshold=upper_threshold)
            
            # Calculate histograms
            print(f"Processing {image_file}...")
            
            # Original image histograms
            hist_original = calculate_histogram(img)
            hist_original_nozero = calculate_histogram(img, ignore_zeros=True)
            
            # Equalized image histograms
            hist_eq = calculate_histogram(img_eq)
            hist_eq_basic = calculate_histogram(img_eq_basic)
            hist_eq_strong = calculate_histogram(img_eq_strong)
            hist_eq_strong_nozero = calculate_histogram(img_eq_strong, ignore_zeros=True)
            
            # Save histograms to CSV
            save_histogram_to_csv(hist_original, image_file, csv_dir, "original", identity, view)
            save_histogram_to_csv(hist_original_nozero, image_file, csv_dir, "original_nozero", identity, view)
            save_histogram_to_csv(hist_eq, image_file, csv_dir, "equalized", identity, view)
            save_histogram_to_csv(hist_eq_basic, image_file, csv_dir, "equalized_basic", identity, view)
            save_histogram_to_csv(hist_eq_strong, image_file, csv_dir, "equalized_strong", identity, view)
            save_histogram_to_csv(hist_eq_strong_nozero, image_file, csv_dir, "equalized_strong_nozero", identity, view)
            
            print(f"Saved histogram data for {image_file}")

def main():
    # Define input and output paths
    input_dir = r"C:\Biometrics\data\results\body"
    csv_dir = r"C:\Biometrics\data\results\histograms_csv"
    
    # Define filter thresholds
    lower_threshold = 10  # Values below this will be filtered out
    upper_threshold = 245  # Values above this will be filtered out (for stronger filter)
    
    # Process images
    process_images(input_dir, csv_dir, lower_threshold, upper_threshold)
    print("Processing completed!")

if __name__ == "__main__":
    main()