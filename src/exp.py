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
from tqdm.auto import tqdm
from datetime import datetime
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.metrics import root_mean_squared_error
from src.data.train_test_split import *

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