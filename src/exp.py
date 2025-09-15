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
from datetime import datetime
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.metrics import root_mean_squared_error


def get_windows(cells, window_size=1000):
    '''
    ignore the mean & std stuff that was for testing reasons
    also got rid of scaling bc im now doing batch normalization during training
    '''
    data_arrays = []
    for filename in cells: 
        for data in BatteryData.load(filename).timeseries_data:
            data = data.to_numpy()[:, 1]
            data = data[~np.isnan(data)]
            data_arrays.append(data)

    all_train_data = np.concatenate(data_arrays, axis=0)
    mean = all_train_data.mean(axis=0)
    std = all_train_data.std(axis=0)

    X = []
    for filename in cells: 
        for data in BatteryData.load(filename).timeseries_data:
            scaledx = data.to_numpy()
            scaledx = scaledx[:, 1]
            temps = scaledx[~np.isnan(scaledx)]

            # TODO: Don't need to move window 1 time step at a time
            # Potential issue: we may want to change time window and slide based on size of time steps
            for i in range(len(temps) - window_size):
                X.append(temps[i:i+window_size])

    X = np.array(X, dtype=np.float32)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, mean, std


def build_lstm_autoencoder(input_shape):
    """
    Builds an LSTM autoencoder with TimeDistributed Batch Normalization.

    Args:
        input_shape (tuple): The shape of the input data (timesteps, features).

    Returns:
        tf.keras.models.Model: The LSTM autoencoder model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    encoded = LSTM(128, return_sequences=True)(inputs)
    encoded = TimeDistributed(BatchNormalization())(encoded)
    encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)
    encoded = BatchNormalization()(encoded)

    # Decoder
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, return_sequences=True)(decoded)
    decoded = TimeDistributed(BatchNormalization())(decoded)
    decoded = LSTM(128, return_sequences=True)(decoded)
    decoded = TimeDistributed(BatchNormalization())(decoded)
    decoded = TimeDistributed(Dense(input_shape[1]))(decoded) # Output layer matches input features

    # Autoencoder model
    autoencoder = Model(inputs, decoded)

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