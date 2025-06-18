import os 
import numpy as np
import pandas as pd

from glob import glob
import joblib

from src.config import config
from src.preprocessing.organizations import Organization

ORG_RAW_DATA = {
    'oakridge': 'data/raw/oakridge/excel',
}

def standardize_dataframe(df):
    # Reorder or select necessary columns
    expected_substrings = ['Time', '°C', '[C]']
    if expected_substrings:
        df = df[[col for col in df.columns if any(sub in str(col) for sub in expected_substrings)]]
    # df = df.dropna() 
    return df if len(df.columns) > 1 else None

def load_and_standardize(file_path):
    ext = os.path.splitext(file_path)[1]
    try:
        if ext in ['.csv', '.txt']:
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            print(f"Skipping unsupported file: {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return standardize_dataframe(df)

def save_pkl(df, output_path):
    joblib.dump(df, output_path, compress=3)  # compress=3 is a good speed-size balance

def process_org_data(org: Organization):
    os.makedirs(org.output_dir, exist_ok=True)

    for file_path in glob(os.path.join(org.input_dir, '*')):
        df = load_and_standardize(file_path)
        if df is not None:
            base = os.path.splitext(os.path.basename(file_path))[0]
            save_pkl(df, os.path.join(org.output_dir, base + '.pkl'))
