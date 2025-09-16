from tensorflow import keras 
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization

class LSTMAutoencoder(keras.Sequential):
    def __init__(self, training_data):
        self.training_data = training_data
        super().__init__(self.encoder + self.decoder)

    @property
    def training_data(self): 
        return self._training_data

    @training_data.setter
    def training_data(self, data): 
        self._training_data = data

    @property
    def timewindow_len(self): 
        return self.training_data.shape[1]

    @property 
    def num_features(self): 
        return self.training_data.shape[2]
    
    @property
    def encoder(self):
        if getattr(self, '_encoder', None) is None:
            self._encoder = [
                Input(shape=(self.timewindow_len, self.num_features)),
                LSTM(128, return_sequences=True),
                TimeDistributed(BatchNormalization()),
                LSTM(64, activation='relu', return_sequences=False),
                BatchNormalization()
            ]
        return self._encoder
    
    @property
    def decoder(self):
        if getattr(self, '_decoder', None) is None:
            self._decoder = [
                RepeatVector(self.timewindow_len),
                LSTM(64, return_sequences=True),
                TimeDistributed(BatchNormalization()),
                LSTM(128, return_sequences=True),
                TimeDistributed(BatchNormalization()),
                TimeDistributed(Dense(self.num_features))
            ]
        return self._decoder