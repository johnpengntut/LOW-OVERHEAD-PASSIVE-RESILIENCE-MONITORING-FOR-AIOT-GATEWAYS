# LOW-OVERHEAD-PASSIVE-RESILIENCE-MONITORING-FOR-AIOT-GATEWAYS
Dataset for the jornal paper
# eBPF-based End-to-End Timing Dataset for AIoT Edge Monitoring

This repository contains the dataset and preprocessing artifacts for our paper on lightweight, gateway-centric resilience monitoring for AIoT services using eBPF. 

The dataset captures kernel-level timing traces of an AIoT pipeline, aggregated into rolling windows to capture behavioral shifts, concept drifts, and system degradations without relying on active network probing.

## 📊 Dataset Overview

The dataset provides segment-level latency features extracted from an AIoT edge gateway. We primarily focus on two coupled segments:
* **T_de (Device-to-Edge):** Represents the wireless network transmission latency.
* **T_is (IoT Service):** Represents the edge computing queuing and RESTful processing latency.

The raw eBPF traces have been aggregated using a rolling window approach (**Window Size = 300s, Stride = 30s**) to capture both transient jitters and sustained degradations.

## 📁 Repository Structure

* `windows_W300_S30_features_with_split.csv`: The complete dataset containing all engineered features and the `split` column indicating whether the window belongs to the `train`, `val`, or `test` set.
* `train_windows.csv` / `val_windows.csv` / `test_windows.csv`: Subsets of the main dataset pre-split for machine learning models (e.g., Autoencoder, CUSUM).
* `feature_columns.txt`: A plain text list of the specific feature columns used for model training.
* `scaler_W300_S30.pkl`: A fitted `scikit-learn` StandardScaler object used to normalize the training data. Provided for exact reproducibility of our Autoencoder evaluation.
* `step2_meta.json`: Metadata logging the exact timestamps for the splits, total window counts, and the global p99 baseline reference values.
* `fig_counts_over_time_with_splits.png`: A visual timeline of the traffic counts and how the temporal splits (Train/Val/Test) were applied across normal and drift periods.

## 🛠️ Data Features

For each segment (`T_de` and `T_is`), the following statistical features are computed per window:
* **Basic Stats:** `mean`, `std`, `max`, `count`
* **Percentiles:** `p95`, `p99`, `q1`, `q3`, `iqr`
* **Tail Metrics:** `tail_ratio_p99_mean`, `tail_exceed_rate`, `winsorized_mean_ms_p95cap`

## 🚀 Quick Start (Python)

You can easily load the datasets and the scaler using `pandas` and `scikit-learn`:

```python
import pandas as pd
import pickle

# 1. Load the training data
df_train = pd.read_csv('train_windows.csv')

# 2. Extract only the model features
with open('feature_columns.txt', 'r') as f:
    features = [line.strip() for line in f if line.strip()]

X_train = df_train[features].values

# 3. Load the scaler and transform the data
with open('scaler_W300_S30.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
print(f"Loaded {X_train_scaled.shape[0]} windows with {X_train_scaled.shape[1]} features.")
