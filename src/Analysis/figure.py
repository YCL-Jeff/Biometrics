import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 確保輸出目錄存在
output_dir = r"C:\Biometrics\data\results\analysis\plots"
os.makedirs(output_dir, exist_ok=True)

# 定義數據
body_corr_data = {
    'shoulder_to_height_ratio': [1.00, 0.06, -0.16],
    'shoulder_to_waist_ratio': [0.06, 1.00, -0.03],
    'ear_to_waist_to_waist_to_knee_ratio': [-0.16, -0.03, 1.00]
}
gabor_corr_data = {
    'gabor_0_mean': [1.00, 0.95, 0.96, -0.92, 1.00, 0.97, 0.98, -0.92, 1.00, 0.94, 0.96, -0.93, 1.00, 0.98, 0.98, -0.92],
    'gabor_0_std': [0.95, 1.00, 0.98, -0.86, 0.95, 0.98, 0.96, -0.86, 0.95, 0.97, 0.96, -0.87, 0.95, 0.98, 0.95, -0.86],
    'gabor_0_energy': [0.96, 0.98, 1.00, -0.91, 0.96, 0.97, 0.98, -0.91, 0.96, 0.95, 0.96, -0.91, 0.96, 0.97, 0.98, -0.90],
    'gabor_0_entropy': [-0.92, -0.86, -0.91, 1.00, -0.92, -0.87, -0.91, 1.00, -0.92, -0.81, -0.84, 1.00, -0.92, -0.89, -0.92, 1.00],
    'gabor_1_mean': [1.00, 0.95, 0.96, -0.92, 1.00, 0.97, 0.98, -0.92, 1.00, 0.94, 0.96, -0.93, 1.00, 0.98, 0.98, -0.92],
    'gabor_1_std': [0.97, 0.98, 0.97, -0.87, 0.97, 1.00, 0.98, -0.87, 0.97, 0.97, 0.96, -0.89, 0.97, 0.98, 0.97, -0.87],
    'gabor_1_energy': [0.98, 0.96, 0.98, -0.91, 0.98, 0.98, 1.00, -0.91, 0.98, 0.94, 0.96, -0.92, 0.98, 0.98, 1.00, -0.91],
    'gabor_1_entropy': [-0.92, -0.86, -0.91, 1.00, -0.92, -0.87, -0.91, 1.00, -0.92, -0.81, -0.84, 1.00, -0.92, -0.89, -0.92, 1.00],
    'gabor_2_mean': [1.00, 0.95, 0.96, -0.92, 1.00, 0.97, 0.98, -0.92, 1.00, 0.94, 0.96, -0.93, 1.00, 0.98, 0.98, -0.92],
    'gabor_2_std': [0.94, 0.97, 0.95, -0.81, 0.94, 0.97, 0.94, -0.81, 0.94, 1.00, 0.97, -0.89, 0.94, 0.98, 0.94, -0.81],
    'gabor_2_energy': [0.96, 0.96, 0.96, -0.84, 0.96, 0.96, 0.96, -0.84, 0.96, 0.97, 1.00, -0.91, 0.96, 0.97, 0.96, -0.84],
    'gabor_2_entropy': [-0.93, -0.87, -0.91, 1.00, -0.93, -0.89, -0.92, 1.00, -0.93, -0.89, -0.91, 1.00, -0.93, -0.89, -0.93, 1.00],
    'gabor_3_mean': [1.00, 0.95, 0.96, -0.92, 1.00, 0.97, 0.98, -0.92, 1.00, 0.94, 0.96, -0.93, 1.00, 0.98, 0.98, -0.92],
    'gabor_3_std': [0.98, 0.98, 0.97, -0.89, 0.98, 0.98, 0.98, -0.89, 0.98, 0.98, 0.97, -0.89, 0.98, 1.00, 0.98, -0.88],
    'gabor_3_energy': [0.98, 0.95, 0.98, -0.92, 0.98, 0.97, 1.00, -0.92, 0.98, 0.94, 0.96, -0.93, 0.98, 0.98, 1.00, -0.92],
    'gabor_3_entropy': [-0.92, -0.86, -0.90, 1.00, -0.92, -0.87, -0.91, 1.00, -0.92, -0.81, -0.84, 1.00, -0.92, -0.88, -0.92, 1.00]
}
hu_corr_data = {
    'h1': [1.00, 1.00, 0.29, 0.24, 0.15, 0.24, 0.45],
    'h2': [1.00, 1.00, 0.28, 0.22, 0.15, 0.24, 0.45],
    'h3': [0.29, 0.28, 1.00, 0.93, 0.36, 0.35, 0.16],
    'h4': [0.24, 0.22, 0.93, 1.00, 0.22, 0.23, 0.13],
    'h5': [0.15, 0.15, 0.36, 0.22, 1.00, 0.99, 0.18],
    'h6': [0.24, 0.24, 0.35, 0.23, 0.99, 1.00, 0.21],
    'h7': [0.45, 0.45, 0.16, 0.13, 0.18, 0.21, 1.00]
}

body_p_values = {
    'shoulder_to_height_ratio': 1.0000,
    'shoulder_to_waist_ratio': 0.5401,
    'ear_to_waist_to_waist_to_knee_ratio': 0.6681
}
gabor_p_values = {
    'gabor_0_mean': 0.9999,
    'gabor_0_std': 1.0000,
    'gabor_0_energy': 1.0000,
    'gabor_0_entropy': 0.6424,
    'gabor_1_mean': 0.9999,
    'gabor_1_std': 1.0000,
    'gabor_1_energy': 1.0000,
    'gabor_1_entropy': 0.6405,
    'gabor_2_mean': 0.9999,
    'gabor_2_std': 1.0000,
    'gabor_2_energy': 1.0000,
    'gabor_2_entropy': 0.7491,
    'gabor_3_mean': 0.9999,
    'gabor_3_std': 1.0000,
    'gabor_3_energy': 0.9999,
    'gabor_3_entropy': 0.6318
}
hu_p_values = {
    'h1': 1.0000,
    'h2': 1.0000,
    'h3': 0.0018,
    'h4': 0.0036,
    'h5': 0.4646,
    'h6': 0.6333,
    'h7': 0.9718
}

# 轉換為 DataFrame
body_corr_df = pd.DataFrame(body_corr_data, index=['shoulder_to_height_ratio', 'shoulder_to_waist_ratio', 'ear_to_waist_to_waist_to_knee_ratio'])
gabor_corr_df = pd.DataFrame(gabor_corr_data, index=list(gabor_corr_data.keys()))
hu_corr_df = pd.DataFrame(hu_corr_data, index=list(hu_corr_data.keys()))

# 繪製熱圖：Body Ratios Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(body_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Body Ratios Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'body_ratios_correlation_heatmap.png'))
plt.close()

# 繪製熱圖：Gabor Features Correlation
plt.figure(figsize=(12, 10))
sns.heatmap(gabor_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title('Gabor Features Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'gabor_features_correlation_heatmap.png'))
plt.close()

# 繪製熱圖：Hu Moments Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(hu_corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Hu Moments Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'hu_moments_correlation_heatmap.png'))
plt.close()

# 繪製條形圖：Body Ratios P-values
plt.figure(figsize=(8, 6))
body_p_df = pd.DataFrame.from_dict(body_p_values, orient='index', columns=['P-value'])
body_p_df.plot(kind='bar', legend=False)
plt.axhline(y=0.05, color='r', linestyle='--', label='P = 0.05')
plt.title('Body Ratios P-values')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'body_ratios_p_values_bar.png'))
plt.close()

# 繪製條形圖：Gabor Features P-values
plt.figure(figsize=(12, 6))
gabor_p_df = pd.DataFrame.from_dict(gabor_p_values, orient='index', columns=['P-value'])
gabor_p_df.plot(kind='bar', legend=False)
plt.axhline(y=0.05, color='r', linestyle='--', label='P = 0.05')
plt.title('Gabor Features P-values')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gabor_features_p_values_bar.png'))
plt.close()

# 繪製條形圖：Hu Moments P-values
plt.figure(figsize=(8, 6))
hu_p_df = pd.DataFrame.from_dict(hu_p_values, orient='index', columns=['P-value'])
hu_p_df.plot(kind='bar', legend=False)
plt.axhline(y=0.05, color='r', linestyle='--', label='P = 0.05')
plt.title('Hu Moments P-values')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hu_moments_p_values_bar.png'))
plt.close()

print(f"所有圖表已保存到 {output_dir}")