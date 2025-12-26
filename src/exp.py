# ALL MY JANK CODE THAT's RELEVANT RN

from src.data import BatteryData
import torch
import pickle
# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from tqdm import tqdm
from datetime import datetime
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.metrics import root_mean_squared_error
from src.data.train_test_split import *
from src.data import BatteryDataset
from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate_and_plot(filename, model, device='mps', window_size=1000, window_skip=500, threshold=500, sampling_rate=1):
    evaluations = evaluate_file_health(
        model,
        filename,
        device,
        window_size,
        window_skip,
        threshold
    )

    bd = BatteryData.load(filename)
    temps = bd.timeseries_data[0].to_dict()['temperature_in_C']
    temps = np.array(list(temps))
    y = temps[~np.isnan(temps)]
    x = np.arange(len(y)) / sampling_rate

    
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
    subtitle_font = {
        'size': 8,
        'style': 'italic'
    }
    plt.suptitle(f"Evaluations of {filename.split('/')[-1].split('.')[0]}")
    plt.title(f"window_size={window_size}; window_skip={window_skip}", fontdict=subtitle_font)
    plt.xlabel("Time (s)")
    plt.ylabel("Temp (°C)")
    plt.grid(alpha=0.3)
    plt.show()

def evaluate_file_health(model, filename, device, window_size, window_skip, threshold):
    dataloader = DataLoader(
        BatteryDataset([filename], window_size, window_skip, preload=True, index_tuple=(1)),
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    return evaluate_health(
        model,
        dataloader,
        device,
        threshold
    )


def evaluate_health(model, dataloader, device, threshold):
    if 'mps' in device: torch.mps.empty_cache()
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    evaluations = []

    pbar = tqdm(dataloader, desc='Evaluate', total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch in pbar:
            x = batch.to(device)
            recon = model(x)
            recon_cpu = recon.detach().to('cpu')
            del x, recon 

            per_elem = criterion(recon_cpu, batch) # (batch, seq_len, 1)
            per_sample_mse = per_elem.mean(dim=(1, 2)).cpu()  # (batch,)
            if per_sample_mse[0] > threshold:
                evaluations.append(1)
            else:
                evaluations.append(0)
            if 'mps' in device: torch.mps.empty_cache()
            
    return evaluations



def print_data_stats(data, name='TRAINING DATA'):
    print(f'\n==={name} STATS===')
    print(f'mean: \t{data.mean()} \nstd: \t{data.std()} \nmin: \t{data.min()}\nmax: \t{data.max()}')
    print(f'shape:\t {data.shape}')

def trained_model_name(name=None):
    dir = 'trained_models/'
    if name: return dir + name + '.pkl'
    return dir + f'trained_model_{len(os.listdir(dir))}.pkl'

def get_windows(cells, window_size=1000, window_skip=0):
    '''
    window_size: size of each segment of the timeseries data that the model will look at.
    window_skip: when processing full data file i don't think we need to split it into windows [(t,t+window_size), (t+1,t+1+window_size)...] 
        because that would mean that a file w 10,000 timesteps would have 9,900 windows. 
        Instead can do [(t,t+window_size), (t+window_skip,t+window_skip+window_size)...] so that file would have fewer than 9900 windows. 
    '''
    X = []
    for filename in tqdm(cells, desc='Getting windows'): 
        for data in BatteryData.load(filename).timeseries_data:
            scaledx = data.to_numpy()
            scaledx = scaledx[:, 1]
            temps = scaledx[~np.isnan(scaledx)]

            # TODO: Don't need to move window 1 time step at a time
            # Potential issue: we may want to change time window and slide based on size of time steps
            for i in range(len(temps) - window_size):
                X.append(temps[i:i+window_size])
                i += window_skip

    X = np.array(X, dtype=np.float32)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X


def build_lstm_autoencoder(input_shape):
    """
    Builds an LSTM autoencoder with TimeDistributed Batch Normalization.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).

    Returns:
        tf.keras.models.Model: The LSTM autoencoder model.
    """
    # Encoder
    encoder = [
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        TimeDistributed(BatchNormalization()),
        LSTM(64, activation='relu', return_sequences=False),
        BatchNormalization()
    ]

    # Decoder
    decoder = [
        RepeatVector(input_shape[0]),
        LSTM(64, return_sequences=True),
        TimeDistributed(BatchNormalization()),
        LSTM(128, return_sequences=True),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dense(input_shape[1]))
    ]

    # Autoencoder model
    autoencoder = Sequential(encoder + decoder)

    return autoencoder

def plot_results(data, preds, title='Error Rates over Time', xlabel='Timestep', ylabel='RMSE', ylim=55):
    x_values = range(data.shape[0])
    y_values = [root_mean_squared_error(data[i], preds[i]) for i in x_values]
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(top=ylim)
    plt.show()