import os 
import torch
import numpy as np
import random
from glob import glob 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils.config import YamlHandler

from src.config import config
from src.data import BatteryData

def get_health_divided_train_test(healthy_test_size=0.2, unhealthy_test_size=1):
    healthy_train_cells, healthy_test_cells = get_divided_train_test(healthy_test_size)
    unhealthy_train_cells, unhealthy_test_cells = get_divided_train_test(unhealthy_test_size, 'unhealthy')

    train_cells = healthy_train_cells + unhealthy_train_cells
    test_cells = healthy_test_cells + unhealthy_test_cells
    return train_cells, test_cells

def get_divided_train_test(test_size=0.2, configfile='healthy'):
    configs = YamlHandler(f'configs/{configfile}.yaml').read_yaml()
    train_cells, test_cells = train_test_split_filenames(
        configs=configs,
        test_size=test_size
    )
    return train_cells, test_cells

def train_test_split_filenames(preprocessed_dir=config.PROCESSED_DATA_DIR, configs={}, test_size=0.2, random_state=42, dtype=None):
    configs = configs.get('train_test_split', configs) # open configs if needed

    pbar = tqdm(glob(os.path.join(preprocessed_dir, '*')), desc='Checking attributes')
    battery_filters = configs.get('battery_filters', None) 

    trainfiles, testfiles = [], []
    for filename in pbar:
        cell = BatteryData.load(filename)

        if battery_filters and any(getattr(cell, filter, None) in getattr(battery_filters, filter) for filter in battery_filters):
            continue
        
        testfiles.append(filename) if random.uniform(0, 1) < test_size else trainfiles.append(filename)

    return trainfiles, testfiles

def load_train_test_split(preprocessed_dir=config.PROCESSED_DATA_DIR, configs={}, test_size=0.2, random_state=42, dtype=None):
    """load data from the specified directory, filter based on the provided configs, and split into training & testing data.
    Parameters
    ----------
    configs : 'dict' or 'Dict(addict)'
    """
    configs = configs.get('train_test_split', configs) # open configs if needed
    features, labels = extract_attributes_from_dir(preprocessed_dir, configs)
    assert len(features)>0 and len(labels)>0, f'No data matches the provided configurations'

    if dtype in ('tensor', torch.tensor, torch.Tensor):
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=-1.)
        features[torch.isnan(features) | torch.isinf(features)] = 0. # TODO: is 0 the best call? should we use -1? 
        labels = torch.tensor(labels, dtype=torch.float32)
    
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def extract_attributes_from_dir(preprocessed_dir = config.PROCESSED_DATA_DIR, configs = {}):
    """load data from the specified directory, filter based on the provided configs, and split into features & labels.
    Parameters
    ----------
    configs : 'dict' or 'Dict(addict)'
    """
    pbar = tqdm(glob(os.path.join(preprocessed_dir, '*')), desc='Extracting attributes')
    battery_filters = configs.get('battery_filters', None) 

    features, labels = [], []
    for filename in pbar:
        cell = BatteryData.load(filename)

        if battery_filters and any(getattr(cell, filter, None) in getattr(battery_filters, filter) for filter in battery_filters):
            continue

        X_temp, y_temp = extract_attributes_from_cell(cell, configs)
        features += X_temp
        labels += y_temp
    
    return features, labels

def extract_attributes_from_cell(cell: BatteryData, configs = None):
    """Return a list of feature, label objects"""
    
    features = []
    for data in cell.timeseries_data:
        if configs and getattr(data, 'description') and any(term in data.description for term in configs.timeseries_filter):
            continue 
        
        # features.append(data.to_numpy())
        features.append(torch.tensor(data.to_numpy(), dtype=torch.float32))
    
    return features, [float(cell.is_healthy)] * len(features) 
