import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from src.config import config

#Loading Data
def load_all_data():
    X_list, y_list = [], []
    for root, _, files in os.walk(config.DATA_DIR):
        label = 0 if 'healthy' in root.lower() else 1
        for fname in files:
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, fname))
                if df.isnull().values.any() or len(df) < config.WINDOW_SIZE:
                    continue
                arr = df.values.astype(np.float32)
                for i in range(0, len(arr) - config.WINDOW_SIZE + 1, config.STRIDE):
                    X_list.append(arr[i:i + config.WINDOW_SIZE])
                    y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Loaded: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

#Scaling
def scale_X(X, scaler=RobustScaler(), should_fit=True):
    n, T, F = X.shape
    X_flat = X.reshape(n, T * F)
    return scaler.fit_transform(X_flat) if should_fit else scaler.transform(X_flat)

def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    class ProcessedData:
        def __init__(self, X, y, is_train):
            self.X = X 
            self.y = y
            self.X_scaled = scale_X(X, should_fit=is_train)
    return ProcessedData(X_train, y_train, True), ProcessedData(X_test, y_test, False)

def processed_data():
    X, y = load_all_data()
    return process_data(X, y)