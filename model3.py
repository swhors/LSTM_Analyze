# load dependacies
import pandas as pd
import numpy as np
import random as random
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, InputLayer, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from globalvar import ENTIRE_NUMBER, TOT_NUMBER_OF_GTH
from model_base import BaseLSTM

SEQUENCE_LENGTH = 10  # 이전 몇 회차의 데이터를 보고 다음 회차를 예측할지 결정


def create_model_cb(lstm_units=50,
                    dense_units=25,
                    train_X=[],
                    loss='binary_crossentropy',
                    activation='relu',
                    optimizer='adam',
                    learning_rate=0.001):
    print(f'lstm_units = {lstm_units}')
    print(f'dense_units = {dense_units}')
    print(f'loss = {loss}')
    print(f'activation = {activation}')
    print(f'optimizer = {optimizer}')
    print(f'learning_rate = {learning_rate}')
    model = Sequential([
        Input(shape=(train_X.shape[1],
                     train_X.shape[2])),
        LSTM(lstm_units, activation=activation,
             return_sequences=True), # 여러 LSTM 층을 쌓을 경우 True
        LSTM(int(lstm_units/2), activation=activation),
        Dense(dense_units, activation=activation),
        Dense(45) # 출력층: 6개 번호, 활성화 함수 없음 (회귀)
        ])
    optimizer = Adam(learning_rate=learning_rate)
    # model.compile(optimizer=optimizer, loss='mse') # 평균 제곱 오차 손실 함수
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(f'\n\t...\tCompleted to create model {model}')
    return model


class PredictLSTM3(BaseLSTM):
    """ PredictLSTM3 """
    def __init__(self, data_loader, verb, hid_dim, args):
        """ __init__ """
        super().__init__(version=2,
                         data_loader=data_loader,
                         verb=verb,
                         hid_dim=hid_dim,
                         args=args)
    
    def training(self, num_epoch, num_batch):
        """ training """
        self._epoch = num_epoch
        self._batch = num_batch        
        history = self.model.fit(self.train_X, self.train_Y, epochs=num_epoch, batch_size=num_batch,)
        return history
   
    def create_model_v1(self, train_X, hid_dim, verb="None"):
        model_kr = KerasRegressor(
            model=create_model_cb,
            lstm_units=50,
            dense_units=25,
            train_X=train_X,
            loss='binary_crossentropy',
            activation='relu',
            learning_rate=0.001,
            verbose=0 if verb is not "verbose" else 1# 학습 중 로그 출력 안 함
        )
        param_grid = {
            'lstm_units': [45],
            'dense_units': [45],
            'loss': ['binary_crossentropy'],
            'activation': ['relu', 'softmax', 'tanh', 'sigmoid'],
            'learning_rate': [0.001], #0.002, 0.005], # 학습률
            'batch_size': [1],
            'epochs': [50]              # 에포크 수
        }
        # param_grid = {
        #     'lstm_units': [90, 45, 20, 10],
        #     'dense_units': [90, 45, 20, 10],
        #     'loss': ['binary_crossentropy', 'mean_squared_error'],
        #     'activation': ['relu', 'softmax', 'tanh', 'sigmoid'],
        #     'learning_rate': [0.001, 0.002, 0.005], # 학습률
        #     'batch_size': [1, 3],
        #     'epochs': [20, 50, 100, 200]              # 에포크 수
        # }
        print("\nGridSearchCV를 시작합니다. 시간이 오래 걸릴 수 있습니다...")
        grid_search = GridSearchCV(estimator=model_kr,
                                   param_grid=param_grid,
                                   # cv=2, # 교차 검증 폴드 수 (데이터가 적으면 줄이세요)
                                   cv=2,
                                   scoring='neg_mean_squared_error', # 점수 기준 (낮을수록 좋음)
                                   verbose=0 if verb is not "verbose" else 1, # 진행 상황 출력
                                   n_jobs=-1) # 사용 가능한 모든 CPU 코어 사용
        print(f'grid_search step_1 = {grid_search}')
        grid_search.fit(train_X, self.train_Y)
        print(f'grid_search step_2 = {grid_search}')
        return grid_search.best_estimator_

    def create_model(self, train_X, hid_dim, args, verb="None"):
        return self.create_model_v1(train_X, hid_dim)
 
    def create_model_v0(self, train_X, hid_dim, verb="None"):
        """ create_model """
        model = Sequential()        
        model.add(LSTM(hid_dim,
                       input_shape=(train_X.shape[1], train_X.shape[2]),
                       return_sequences=True))
        model.add(LSTM(hid_dim,
                       return_sequences=True,
                       # activation="tanh",
                       activation="softmax",
                       recurrent_activation="softmax",
                      ))
        model.add(LSTM(hid_dim,
                       return_sequences=True,
                       activation="tanh",
                       # activation="softmax",
                       recurrent_activation="tanh",
                      ))
        model.add(LSTM(hid_dim,
                       return_sequences=False))
        model.add(Dense(45, activation='sigmoid'))
        # model.add(Dense(45, activation='sigmoid'))
        model.add(Dense(45, activation='softmax'))
        # self.model.compile(loss='mse', optimizer='adam')
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        print(f'PredictLSTM2.create_model {model}')
        return model

    def get_probability(self, previous_prob, selected=None):
        """ get_probability """
        return selected
