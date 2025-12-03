import torch
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from src.data import BatteryData
from src.data.train_test_split import get_divided_train_test

class BatteryDataset(Dataset):
    def __init__(self, cells, window_size=1000, window_skip=1, preload=False, index_tuple=(1)):
        """
        cells: list of filenames (paths to data)
        preload: if True, loads all data into memory at init (faster epoch time, more memory usage)

        index_tuple: which columns to extract. In a file that has time then temp, (1) will extract only temp, (0,1) will extract both.
        """
        self.cells = cells
        self.window_size = window_size
        self.window_skip = window_skip
        self.preload = preload
        self.index_tuple = index_tuple

        # Cached metadata
        self.index_map = []  # (file_idx, start_index)
        self.file_cache = {}  # Optional cache of loaded BatteryData objects

        # Build lookup table (maps global idx -> (file_idx, window_start))
        self.data = self._build_index() 

    def _build_index(self):
        """
        Precompute all window start indices per file and store them in index_map.
        This lets __getitem__ quickly locate which file/window to load.
        """
        # TODO: currently appending all of the sensors one after the other
        X = []
        for file_idx, filename in enumerate(tqdm(self.cells, desc="Building dataset index", leave=False)):
            try:
                # Load metadata once per file
                battery = BatteryData.load(filename)
                self.file_cache[file_idx] = None

                for data in battery.timeseries_data:
                    scaledx = data.to_numpy()[:, self.index_tuple]
                    if len(scaledx.shape) == 1: # only 1 column
                        temps = scaledx[~np.isnan(scaledx)] 
                    else:
                        temps = scaledx[~np.isnan(scaledx).any(axis=1)]

                    for start in range(0, len(temps) - self.window_size, self.window_skip):
                        self.index_map.append((file_idx, start))
                        if self.preload: X.append(temps[start:start + self.window_size])
            except Exception as e:
                print(f"Warning: could not load {filename}: {e}")

        if self.preload:
            X = np.array(X, dtype=np.float32)
            if type(self.index_tuple) is int:
                X = np.reshape(X, (X.shape[0], self.window_size, 1))
            else:
                X = np.reshape(X, (X.shape[0], self.window_size, len(self.index_tuple)))
            # X = torch.stack(X)
            return X
        
        return None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.preload:
            # Preloaded tensor data
            x = self.data[idx]
            # return torch.from_numpy(x)
            return x
        
        file_idx, start = self.index_map[idx]
        if self.file_cache[file_idx] is None:
            self.file_cache[file_idx] = BatteryData.load(self.cells[file_idx])
        battery = self.file_cache[file_idx]

        # Load corresponding file data (already in cache)
        for data in battery.timeseries_data:
            scaledx = data.to_numpy()[:, self.index_tuple]
            if len(scaledx.shape) == 1: # only 1 column
                temps = scaledx[~np.isnan(scaledx)] 
            else:
                temps = scaledx[~np.isnan(scaledx).any(axis=1)]

            # Validate bounds
            if start + self.window_size <= len(temps):
                x = temps[start:start + self.window_size]

                if len(scaledx.shape) == 1:
                    x = x.astype(np.float32).reshape(self.window_size, 1)
                else:
                    x = x.astype(np.float32).reshape(self.window_size, len(self.index_tuple))
                # return torch.from_numpy(x)
                return x

        # Safety fallback (should not occur)
        raise IndexError(f"Invalid index {idx} in dataset")

def get_divided_loaders(
    test_size=0.2,
    window_size=1000,
    window_skip=1,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    preload=False,
    healthy=True,
    index_tuple=(1)
):
    train_cells, test_cells = get_divided_train_test(test_size, 'healthy' if healthy else 'unhealthy')

    return help_get_divided_loaders(
        train_cells if test_size < 1.0 else None,
        test_cells if test_size > 0.0 else None,
        window_size,
        window_skip,
        batch_size,
        num_workers,
        preload,
        index_tuple
    )

def get_divided_loaders_from(
    filelist1 = "trained_models/healthy_train_cells.json",
    filelist2 = "trained_models/healthy_test_cells.json",
    window_size=1000,
    window_skip=1,
    batch_size=64,
    num_workers=4,
    preload=False,
    index_tuple=(1)
):
    with open(filelist1) as file:
        train_cells = json.load(file)

    with open(filelist2) as file:
        test_cells = json.load(file)

    return help_get_divided_loaders(
        train_cells,
        test_cells,
        window_size,
        window_skip,
        batch_size,
        num_workers,
        preload,
        index_tuple
    )

def help_get_divided_loaders(
        train_cells,
        test_cells,
        window_size=1000,
        window_skip=1,
        batch_size=64,
        num_workers=4,
        preload=False,
        index_tuple=(1)
):
    train_loader = DataLoader(
        BatteryDataset(train_cells, window_size, window_skip, preload=preload, index_tuple=index_tuple),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=pin_memory
    ) if train_cells else None

    test_loader = DataLoader(
        BatteryDataset(test_cells, window_size, window_skip, preload=preload, index_tuple=index_tuple),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory
    ) if test_cells else None

    return train_loader, test_loader