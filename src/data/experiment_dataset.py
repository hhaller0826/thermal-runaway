import os 
import torch
import torch.utils
import torch.utils.data
import random
from glob import glob 
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from src.config import config
from src.data import BatteryData

def load_train_test_split(preprocessed_dir = config.PROCESSED_DATA_DIR, configs = {}, test_size=0.2, random_state=42, dtype=None):
    """load data from the specified directory, filter based on the provided configs, and split into training & testing data.
    Parameters
    ----------
    configs : 'dict' or 'Dict(addict)'
    """
    configs = configs.get('train_test_split', configs) # open configs if needed
    data_list = extract_attributes_from_dir(preprocessed_dir, configs)
    assert len(data_list)>0, f'No data matches the provided configurations'

    random.seed(random_state)
    test_idx = int(test_size * len(data_list))
    random.shuffle(data_list)

    train_data = data_list[test_idx:]
    test_data = data_list[:test_idx]
    return train_data, test_data

def extract_attributes_from_dir(preprocessed_dir = config.PROCESSED_DATA_DIR, configs = {}):
    """load data from the specified directory, filter based on the provided configs, and split into features & labels.
    Parameters
    ----------
    configs : 'dict' or 'Dict(addict)'
    """
    pbar = tqdm(glob(os.path.join(preprocessed_dir, '*')), desc='Extracting attributes')
    battery_filters = configs.get('battery_filters', None) 

    data_list = []
    for filename in pbar:
        cell = BatteryData.load(filename)
        if battery_filters and any(getattr(cell, filter, None) in getattr(battery_filters, filter) for filter in battery_filters):
            continue

        data = extract_attributes_from_cell(cell, configs)
        data_list += data
    
    return data_list

def extract_attributes_from_cell(cell: BatteryData, configs = {}):
    """Return a list of feature, label objects"""
    label = torch.tensor(cell.is_healthy, dtype=torch.float32)
    timeseries_filters = configs.get('timeseries_filter', None)

    data_list = []
    for data in cell.timeseries_data:
        if timeseries_filters and getattr(data, 'description') and any(term in data.description for term in timeseries_filters):
            continue 
        
        # features.append(data.to_numpy())
        feature = torch.tensor(data.to_numpy(), dtype=torch.float32)
        data_list.append((feature, label))
    
    return data_list

class ExperimentDataset(torch.utils.data.Dataset):
    def __init__(self, data = []):
        self.data = data 

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def append_data(self, data):
        self.data += data 
    
    def add(self, feature, label):
        self.data.append((feature, label))