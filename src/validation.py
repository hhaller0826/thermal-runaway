import json
import glob
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data import BatteryData, BatteryDataset
from src.models import TransformerAutoencoder
from src.data.train_test_split import get_divided_train_test

# -----------------------------
# Plotting
# -----------------------------
SUBTITLE_FONT = {
    'size': 8,
    'style': 'italic'
}
def evaluate_and_plot(filename, model, device='mps', window_size=1000, window_skip=500, threshold=300, sampling_rate=1):
    """
    Evaluate the given file and plot the results over the file's temperature data. 
    """
    evaluations = evaluate_file_health(model, filename, device, window_size, window_skip, threshold)
    plot_evals_on_temp(filename, evaluations, window_size, window_skip, sampling_rate)

def plot_evals_on_temp(filename, evaluations, window_size, window_skip, sampling_rate, **kwargs):
    """
    Plots the temperature data in the given file with dots indicating whether the data was classified as healthy
    or unhealthy at various timesteps.
    """
    # Get temperature data
    bd = BatteryData.load(filename)
    temps = bd.timeseries_data[0].to_dict()['temperature_in_C']
    temps = np.array(list(temps))
    y = temps[~np.isnan(temps)]
    x = np.arange(len(y)) / sampling_rate

    # Get evaluation colors
    highlight_indices = []
    highlight_colors = []
    for i,e in enumerate(evaluations):
        highlight_indices.append(i*window_skip + window_size)
        if e: highlight_colors.append('red')
        else: highlight_colors.append('blue')

    # Plot the line
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Data', color='gray')

    # Overlay colored dots
    for idx, color in zip(highlight_indices, highlight_colors):
        if idx >= len(y): break
        plt.scatter(x[idx], y[idx], color=color, s=80, zorder=3, label=f'Index {idx}')

    # Make it nice
    plt.suptitle(f"{filename.split('/')[-1].split('.')[0]}")
    plt.title(f"window_size={window_size}; window_skip={window_skip}", fontdict=SUBTITLE_FONT)
    plt.xlabel("Time (s)")
    plt.ylabel("Temp (°C)")
    plt.grid(alpha=0.3)
    plt.show()


# -----------------------------
# Evaluating
# -----------------------------
def evaluate_file_health(filename, model, device="mps", window_size=1000, window_skip=500, threshold=300):
    """
    Evaluate the health of all windows in the given file. 
    """
    dataloader = DataLoader(
        BatteryDataset([filename], window_size, window_skip, preload=True, index_tuple=(1)),
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    return evaluate_health(model, dataloader, device, threshold)

def evaluate_health(model, dataloader, device, threshold):
    """
    Evaluate the health of all timeseries windows provided by the given dataloader. 
    """
    return [err>threshold for err in get_reconstruction_errors(model, dataloader, device)]

def store_reconstruction_errors(
        model_filename, 
        filenames, 
        device="mps", 
        window_size=20, 
        window_skip=1, 
        batch_size=64, 
        num_workers=4,
        preload=False,
        index_tuple=(1),
        save_to=None
    ):
    """
    Get the reconstruction error of all the timeseries windows in all of the provided files.
    If a 'save_to' filename is provided, store the output results there. 
    """
    run_info = {
        'model': model_filename,
        'window_size': window_size, 
        'window_skip': window_skip, 
        'batch_size': batch_size, 
        'num_workers': num_workers,
    }

    model = TransformerAutoencoder.load(model_filename)

    reconstruction_errors = {}
    pbar = tqdm(filenames, desc="Reconstructing files", total=len(filenames), leave=True)
    for file in pbar:
        dataloader = DataLoader(
            BatteryDataset([file], window_size, window_skip, preload=preload, index_tuple=index_tuple),
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers
        )

        errs = get_reconstruction_errors(model, dataloader, device, desc=f"{file.split('/')[-1]}")
        reconstruction_errors[file] = errs

    output_dict = {'info': run_info, 'data': reconstruction_errors}
    if save_to is not None:
        with open(save_to, 'w') as file:
            json.dump(output_dict, file, indent=4)
            print(f"Saved to {save_to}")
    
    return output_dict

def load_reconstruction_errors(path):
    with open(path, 'r') as file:
        output_dict = json.load(file)
    return output_dict

def get_reconstruction_errors(model, dataloader, device, desc=None):
    """
    Get the reconstruction error of all timeseries windows provided by the given dataloader.
    """    
    model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    errors = []

    pbar = tqdm(dataloader, desc=desc or 'Evaluate', total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch in pbar:
            if 'mps' in device: torch.mps.empty_cache()

            x = batch.to(device)
            recon = model(x)
            recon_cpu = recon.detach().to('cpu')
            del x, recon 

            per_elem = criterion(recon_cpu, batch) # (batch, seq_len, 1)
            per_sample_mse = per_elem.mean(dim=(1, 2)).cpu()  # (batch,)
            errors += per_sample_mse.tolist()
            
    return errors
