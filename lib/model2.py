# load dependacies
import pandas as pd
import numpy as np
import random as random
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, InputLayer, Input
from tensorflow.keras.optimizers import Adam
from globalvar import ENTIRE_NUMBER, TOT_NUMBER_OF_GTH
from model_base import BaseLSTM

SEQUENCE_LENGTH = 10  # 이전 몇 회차의 데이터를 보고 다음 회차를 예측할지 결정


class PredictLSTM2(BaseLSTM):
    """ PredictLSTM2 """
    def __init__(self, data_loader, verb, hid_dim, args):
        """ __init__ """
        super().__init__(version=2,
                         data_loader=data_loader,
                         verb=verb,
                         hid_dim=hid_dim,
                         args=args
                        )

    def training(self, num_epoch, num_batch):
        """ training """
        self._epoch = num_epoch
        self._batch = num_batch        
        history = self.model.fit(self.train_X, self.train_Y, epochs=num_epoch, batch_size=num_batch,)
        return history

    def create_model(self, train_X, hid_dim, args, verb="None"):
        """ create_model """
        model = Sequential()        
        model.add(LSTM(hid_dim,
                       input_shape=(train_X.shape[1], train_X.shape[2]),
                       activation="softmax",
                       return_sequences=True))
        # model.add(LSTM(hid_dim,
        #                return_sequences=True,
        #                activation="softmax",
        #                # recurrent_activation="tanh",
        #               ))
        model.add(LSTM(hid_dim,
                       return_sequences=False))
        model.add(Dense(45, activation='softmax'))
        model.add(Dense(45, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(f'PredictLSTM2.create_model {model}')
        return model

    def get_probability(self, previous_prob, selected=None):
        """ get_probability """
        return selected
