# load dependacies
import pandas as pd
import numpy as np
import random as random
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from globalvar import ENTIRE_NUMBER, TOT_NUMBER_OF_GTH
from model_base import BaseLSTM


class PredictLSTM1(BaseLSTM):
    """ PredictLSTM """
    def __init__(self, data_loader, verb, hid_dim, args):
        """ __init__ """
        super().__init__(version=1,
                         data_loader=data_loader,
                         verb=verb,
                         hid_dim=hid_dim,
                         args=args)

    def create_model(self, train_X, hid_dim, args, verb="None"):
        """ create_model """
        model = Sequential()
        model.add(LSTM(hid_dim, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(ENTIRE_NUMBER))
        model.compile(loss='mse', optimizer='adam')
        print(f'PredictLSTM.create_model {model}')
        return model
