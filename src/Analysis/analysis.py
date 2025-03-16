import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 讀取數據
body_ratios_df = pd.read_csv(r"C:\Biometrics\data\results\analysis\body_ratios.csv", index_col=0)
gabor_features_df = pd.read_csv(r"C:\Biometrics\data\results\analysis\gabor_features\gabor_features.csv", index_col=0)
hu_moments_df = pd.read_csv(r"C:\Biometrics\data\results\analysis\hu_moments.csv", index_col=0)

# 清理數據
for df in [body_ratios_df, gabor_features_df, hu_moments_df]:
    df.iloc[:, :-2] = df.iloc[:, :-2].fillna(df.iloc[:, :-2].mean())
    df.iloc[:, :-2] = df.iloc[:, :-2].replace([np.inf, -np.inf], np.nan)
    df.iloc[:, :-2] = df.iloc[:, :-2].fillna(df.iloc[:, :-2].max())
    df.dropna(inplace=True)

# 內部一致性
def compare_within_class(df, identity):
    same_id_data = df[df['Identity'] == identity].iloc[:, :-2].values
    if len(same_id_data) < 2:
        return float('inf')
    distances = []
    for i in range(len(same_id_data)):
        for j in range(i + 1, len(same_id_data)):
            try:
                dist = euclidean(same_id_data[i], same_id_data[j])
                if np.isfinite(dist):
                    distances.append(dist)
            except ValueError:
                continue
    return np.mean(distances) if distances else float('inf')

identities = body_ratios_df['Identity'].unique()
within_body = {id: compare_within_class(body_ratios_df, id) for id in identities if id in body_ratios_df['Identity'].values}
within_gabor = {id: compare_within_class(gabor_features_df, id) for id in identities if id in gabor_features_df['Identity'].values}
within_hu = {id: compare_within_class(hu_moments_df, id) for id in identities if id in hu_moments_df['Identity'].values}

# 跨身份差異
def compare_between_class(df):
    unique_ids = df['Identity'].unique()
    if len(unique_ids) < 2:
        return 0
    distances = []
    for id1 in unique_ids:
        for id2 in unique_ids:
            if id1 != id2:
                id1_data = df[df['Identity'] == id1].iloc[:, :-2].mean().values
                id2_data = df[df['Identity'] == id2].iloc[:, :-2].mean().values
                try:
                    dist = euclidean(id1_data, id2_data)
                    if np.isfinite(dist):
                        distances.append(dist)
                except ValueError:
                    continue
    return np.mean(distances) if distances else 0

between_body = compare_between_class(body_ratios_df)
between_gabor = compare_between_class(gabor_features_df)
between_hu = compare_between_class(hu_moments_df)

# 信噪比
def calculate_snr(within_dict, between):
    snrs = {}
    for id, within in within_dict.items():
        snr = between / within if within > 0 and np.isfinite(within) else float('inf')
        snrs[id] = snr
    return snrs

snr_body = calculate_snr(within_body, between_body)
snr_gabor = calculate_snr(within_gabor, between_gabor)
snr_hu = calculate_snr(within_hu, between_hu)

# 特徵相關性
body_corr = body_ratios_df.iloc[:, :-2].corr().round(2)
gabor_corr = gabor_features_df.iloc[:, :-2].corr().round(2)
hu_corr = hu_moments_df.iloc[:, :-2].corr().round(2)

# ANOVA
def safe_f_oneway(df, col):
    groups = [group[col].values for name, group in df.groupby('Identity') if len(group) > 0]
    if len(groups) < 2:
        return np.nan
    return f_oneway(*groups).pvalue

body_p_values = {col: safe_f_oneway(body_ratios_df, col) for col in body_ratios_df.columns[:-2]}
gabor_p_values = {col: safe_f_oneway(gabor_features_df, col) for col in gabor_features_df.columns[:-2]}
hu_p_values = {col: safe_f_oneway(hu_moments_df, col) for col in hu_moments_df.columns[:-2]}

# 模擬分類性能
def evaluate_classification(df):
    X = df.iloc[:, :-2].values
    y = df['Identity'].values
    if len(X) > 1 and len(y) > 1 and not np.any(np.isnan(X)) and not np.any(np.isinf(X)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return accuracy_score(y_test, y_pred)
    return 0

accuracy_body = evaluate_classification(body_ratios_df)
accuracy_gabor = evaluate_classification(gabor_features_df)
accuracy_hu = evaluate_classification(hu_moments_df)

# 結合特徵
combined_df = pd.concat([body_ratios_df.iloc[:, :-2], gabor_features_df.iloc[:, :-2], hu_moments_df.iloc[:, :-2]], axis=1)
combined_df['Identity'] = body_ratios_df['Identity']
combined_df['View'] = body_ratios_df['View']
accuracy_combined = evaluate_classification(combined_df)

# 儲存結果到 DataFrame
results = pd.DataFrame({
    'Feature_Set': ['Body Ratios', 'Gabor Features', 'Hu Moments', 'Combined Features'],
    'Within': [
        np.mean([v for v in within_body.values() if np.isfinite(v)]),
        np.mean([v for v in within_gabor.values() if np.isfinite(v)]),
        np.mean([v for v in within_hu.values() if np.isfinite(v)]),
        np.nan
    ],
    'Between': [between_body, between_gabor, between_hu, np.nan],
    'SNR': [
        np.mean([v for v in snr_body.values() if np.isfinite(v)]),
        np.mean([v for v in snr_gabor.values() if np.isfinite(v)]),
        np.mean([v for v in snr_hu.values() if np.isfinite(v)]),
        np.nan
    ],
    'Accuracy (%)': [accuracy_body * 100, accuracy_gabor * 100, accuracy_hu * 100, accuracy_combined * 100]
})

# 儲存到 CSV 檔案
output_path = r"C:\Biometrics\data\results\analysis\analysis_results.csv"
results.to_csv(output_path, index=False)
print(f"分析結果已保存到: {output_path}")

# 打印結果
print("\nAnalysis Results:")
print(f"Body Ratios - Within: {results.loc[results['Feature_Set'] == 'Body Ratios', 'Within'].values[0]:.2f}, "
      f"Between: {results.loc[results['Feature_Set'] == 'Body Ratios', 'Between'].values[0]:.2f}, "
      f"SNR: {results.loc[results['Feature_Set'] == 'Body Ratios', 'SNR'].values[0]:.2f}, "
      f"Accuracy: {results.loc[results['Feature_Set'] == 'Body Ratios', 'Accuracy (%)'].values[0]:.2f}%")
print(f"Gabor Features - Within: {results.loc[results['Feature_Set'] == 'Gabor Features', 'Within'].values[0]:.2f}, "
      f"Between: {results.loc[results['Feature_Set'] == 'Gabor Features', 'Between'].values[0]:.2f}, "
      f"SNR: {results.loc[results['Feature_Set'] == 'Gabor Features', 'SNR'].values[0]:.2f}, "
      f"Accuracy: {results.loc[results['Feature_Set'] == 'Gabor Features', 'Accuracy (%)'].values[0]:.2f}%")
print(f"Hu Moments - Within: {results.loc[results['Feature_Set'] == 'Hu Moments', 'Within'].values[0]:.2f}, "
      f"Between: {results.loc[results['Feature_Set'] == 'Hu Moments', 'Between'].values[0]:.2f}, "
      f"SNR: {results.loc[results['Feature_Set'] == 'Hu Moments', 'SNR'].values[0]:.2f}, "
      f"Accuracy: {results.loc[results['Feature_Set'] == 'Hu Moments', 'Accuracy (%)'].values[0]:.2f}%")
print(f"Combined Features - Accuracy: {results.loc[results['Feature_Set'] == 'Combined Features', 'Accuracy (%)'].values[0]:.2f}%")

# 打印相關性和 P-value
print("\nBody Ratios Correlation:\n", body_corr)
print("\nGabor Features Correlation:\n", gabor_corr)
print("\nHu Moments Correlation:\n", hu_corr)
print("\nBody Ratios P-values:")
for col, p in body_p_values.items():
    print(f"{col}: {p:.4f}")
print("\nGabor Features P-values:")
for col, p in gabor_p_values.items():
    print(f"{col}: {p:.4f}")
print("\nHu Moments P-values:")
for col, p in hu_p_values.items():
    print(f"{col}: {p:.4f}")