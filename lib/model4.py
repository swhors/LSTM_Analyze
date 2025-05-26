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


class PredictLSTM4(BaseLSTM):
    """ PredictLSTM2 """
    def __init__(self, data_loader, verb, hid_dim, args):
        """ __init__ """
        super().__init__(version=2,
                         data_loader=data_loader,
                         verb=verb,
                         hid_dim=hid_dim,
                         args=args
                        )

    def training(self, num_epoch, num_batch, is_shuffle=True):
        return super().training(num_epoch=num_epoch,
                                num_batch=num_batch,
                                is_shuffle=is_shuffle)
        # """ training """
        # self._epoch = num_epoch
        # self._batch = num_batch
        # verbose=2 if self._verb == "verbose" else 0
        # history = self.model.fit(self.train_X,
        #                          self.train_Y,
        #                          epochs=num_epoch,
        #                          batch_size=num_batch,
        #                          verbose=verbose,
        #                         )
        # return history

    def create_model(self, train_X, hid_dim, args, verb="None"):
        """ create_model """
        if verb=="verbose":
            print(f'PredictLSTM4.create_model {args}')

        if 'lstm_units' in args[0]:
            lstm_units = args[0]['lstm_units']
        else:
            lstm_units = [(hid_dim, "softmax")]

        if 'learning_rate' in args[0]:
            learning_rate = args[0]['learning_rate']
        else:
            laarning_rate = 0.01

        if 'dropout' in args[0]:
            dropout = args[0]['dropout']
        else:
            dropout = 0

        if 'steps' in args[0]:
            steps = args[0]['steps']
        else:
            steps = 1

        if 'metrics' in args[0]:
            metrics = args[0]['metrics']
        else:
            metrics = ['accuracy']

        if 'last_lstm_return_sequences' in args[0]:
            last_lstm_return_sequences = args[0]['last_lstm_return_sequences']
        else:
            last_lstm_return_sequences = False

        if 'dense_units' in args[0]:
            dense_units = args[0]['dense_units']
        else:
            dense_units = []

        # if 'activations' in args[0]:
        #     activations = args[0]['activations']
        # else:
        #     activations = ['softmax']

        if 'dense_activation' in args[0]:
            dense_activation = args[0]['dense_activation']
        else:
            dense_activation = ['softmax']

        if 'output_dense_activation' in args[0]:
            output_dense_activation = args[0]['output_dense_activation']
        else:
            output_dense_activation = 'softmax'

        if 'loss' in args[0]:
            loss = args[0]['loss']
        else:
            loss = 'mse'

        if 'rand_seed' in args[0]:
            rand_seed = args[0]['rand_seed']
        else:
            rand_seed = 0

        return self.build_and_compile_lstm_model(
            seq_len=train_X.shape[1],
            n_features=train_X.shape[2],
            learning_rate=learning_rate,
            dropout=dropout,
            steps=steps,
            metrics=metrics,
            last_lstm_return_sequences=last_lstm_return_sequences,
            lstm_units=lstm_units,
            dense_units=dense_units,
            output_dense_activation=output_dense_activation,
            loss=loss,
            rand_seed=rand_seed
            )

    def get_probability(self, previous_prob, selected=None):
        """ get_probability """
        return selected
