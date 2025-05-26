# load dependacies
import pandas as pd
import numpy as np
import random as random
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from lib.globalvar import ENTIRE_NUMBER, TOT_NUMBER_OF_GTH


def greed_method(cls_object, *args):
    """ greed_method """
    prediction_number_set = []
    for t in range(1, cls_object._trial + 1):
        yhat = cls_object.model.predict(cls_object.test_X) # [1x45] dim
        yhat_assigned = np.argsort(-yhat[:])
        prediction_number_set.append(yhat_assigned[0][:6] + 1)
    return prediction_number_set


def sampling_method(cls_object, *args):
    """ sampling_method """
    prediction_number_set = []
    if len(args) == 0:
        use_pre = True
    else:
        use_pre = args[0]
    yhat = cls_object.model.predict(cls_object.test_X) # [1x45] dim
    probability = cls_object.get_probability(list(yhat[0])) # use the output as prob. desity dist.
    for t in range(1, cls_object._trial+1):
        selected = np.random.choice(ENTIRE_NUMBER, size=6, replace=False, p=probability)
        prediction_number_set.append([int(i) + 1 for i in selected])
        if cls_object.verb == 'verbose':
            print("predicted set of numbers at {}th trial :".format(t), selected + 1)
        if use_pre:
            probabality = cls_object.get_probability(probability, selected)
    return prediction_number_set


predict_funcs={"greed": greed_method,
               "sampling": sampling_method}


class BaseLSTM:
    """ BaseLSTM"""
    def __init__(self, version, data_loader, verb, hid_dim, args):
        """ __init__ """
        self._version = version
        self.train_X = data_loader.train_X
        self.test_X = data_loader.test_X
        self.train_Y = data_loader.train_Y
        self.test_Y = data_loader.test_Y
        self.n_hours = data_loader.window_prev
        self._hid_dim = hid_dim
        self._verb = verb
        self.model = self.create_model(self.train_X, hid_dim, args, verb=verb)
        self._epoch = 0
        self._batch = 0
        self._trial = 0

    @property
    def verb(self) -> str:
        return self._verb

    @verb.setter
    def verb(self, value: str):
        self._verb = value

    def create_model(self, train_X, hid_dim, args, verb="None"):
        """ create_model """
        pass

    def save(self):
        """ save_model """
        from datetime import datetime
        file_path = f'./lstm{self._version}_{self._hid_dim}_{self._epoch}_{self._batch}_{datetime.now()}.h5'
        self.model.save(file_path)
        print(f'BaseLSTM.save_model')

    def training(self, num_epoch, num_batch, is_shuffle=True, steps_per_epoch=1):
        """ training """
        # no validation currently.
        self._epoch = num_epoch
        self._batch = num_batch
        verbose=2 if self._verb == "verbose" else 0
        print(f'verbose={verbose}')
        print(f'model={self.model}')
        print(f'num_epoch={num_epoch}/{type(num_epoch)}')
        # history = self.model.fit(self.train_X,
        #                          self.train_Y,
        #                          epochs = num_epoch,
        #                          batch_size = num_batch,
        #                          verbose=verbose,
        #                          shuffle=is_shuffle)
        return self.model.fit(
            self.train_X,
            self.train_Y,
            batch_size=num_batch,
            steps_per_epoch=steps_per_epoch,
            validation_data=(self.test_X, self.test_Y),
            epochs=num_epoch,
            verbose=verbose,
            shuffle=is_shuffle
            )
        """
        # use_multiprocessing=True,
        # workers=8,
        # callbacks=callbacks,
        """

    def predict_numbers(self, mode2, trial=20, use_pre = True):
        """ predict_numbers """
        self._trial = trial
        if mode2 in predict_funcs:
            return predict_funcs[mode2](self, [use_pre])
        print(f'Error : illegal mode2 values. [{mode2}]')
        return []

    def get_probability(self, previous_prob, selected=None):
        """ get_probability """
        pdf = [0 if i < 0 else i for i in previous_prob]
        pdf_sum = sum(pdf)
        pdf_norm = [i/pdf_sum for i in pdf] #normalize for make it as a pdf form.
        pdf_norm /= sum(pdf_norm)
        return pdf_norm

    def predict_randomely(self, trial):
        """ predict_randomely """
        prediction_number_set = []
        for t in range(1,trial+1):
            selected = np.random.choice(ENTIRE_NUMBER, size=6, replace=False)
            prediction_number_set.append([int(i) + 1 for i in selected])
        return prediction_number_set
        
    def evaluate(self, overall_prediction):
        gth = np.argsort(-self.test_Y[:])
        gth = gth[:,:TOT_NUMBER_OF_GTH]   # considering bonus number, entire number is 7, not 6
        print('-----------evaluation ----------')
        print('Winning numbers :', gth[0])
        trial = 1
        all_count = 0
        for pred_set in overall_prediction:
            count = 0
            for i in pred_set:
                if (i-1) in gth:
                    count += 1          
                    all_count += 1
            if self.verb == 'verbose':                       
                print('{}th predicted:'.format(trial), pred_set)            
                print('{}th trial: {} out of 6 is correct !!'.format(trial, count))
            
            trial += 1 
        print(f'all_count {all_count}')
        print('Precision:{}%'.format(100*all_count/(6*(trial-1))))

    def build_and_compile_lstm_model(
        self,
        seq_len: int,
        n_features: int,
        learning_rate: float,
        dropout: float,
        steps: int,
        metrics: str,
        lstm_units: list,
        last_lstm_return_sequences: bool=False,
        return_state=False,
        dense_units: list=None,
        output_dense_activation: str=None,
        loss: str=None,
        stateful=False,
        lstm_model="LSTM",
        rand_seed: int=0):
        """
        LSTM 네트워크를 생성한 결과를 반환한다.
        :param seq_len: Length of sequences. (Look back window size)
        :param n_features: Number of features. It requires for model input shape.
        :param lstm_units: Number of cells each LSTM layers. ex:[(54, "softmax"), ]
        :param learning_rate: Learning rate.
        :param dropout: Dropout rate.
        :param steps: Length to predict.
        :param metrics: Model loss function metric.
        :param single_output: Whether 'yhat' is a multiple value or a single value.
        :param last_lstm_return_sequences: Last LSTM's `return_sequences`. Allow when `single_output=False` only.
        :param dense_units: list of number and activation  for cells each Dense layers. exam: [(12, 'relu'), .. ]
        :param loss: Loss, example = "MSE" or "binary_crossentropy"
        :param lstm_model: str, example = "LSTM" or "Bidirectional"
        """
        # lstm_class={"LSTM": LSTM, "Bidirectional": Bidirectional}
        # if lstm_model not in lstm_class:
        #     cur_lstm_class = Sequential
        # else:
        #     cur_lstm_class = lstm_class[lstm_model]
        if rand_seed > 0:
            tf.random.set_seed(rand_seed)
        model = Sequential()
        model.add(Input(shape=(seq_len, n_features)))
        return_sequences = True
        for i, units in enumerate(lstm_units, start=1):
            # LSTM -> ... -> LSTM -> Dense(steps)
            print(f'lstm unit : {i}, {units}')
            if i == len(lstm_units):
                return_sequences = last_lstm_return_sequences
            if lstm_model=="LSTM":
                model.add(
                    LSTM(units=units[0],
                         activation=units[1],
                         stateful=stateful,
                         return_sequences=return_sequences,
                         return_state=return_state
                         ))
            else:
                k_init = tf.keras.initializers.Constant(value=0.1)
                b_init = tf.keras.initializers.Constant(value=0)
                r_init = tf.keras.initializers.Constant(value=0.1)
                model.add(
                    Bidirectional(
                        LSTM(units=units[0],
                             activation=units[1],
                             stateful=stateful,
                             return_sequences=return_sequences,
                             return_state=return_state,
                             kernel_initializer=k_init,
                             bias_initializer=b_init,
                             recurrent_initializer=r_init
                             ))
                )
        if len(dense_units) > 0 and last_lstm_return_sequences:
            model.add(Flatten())
        for unit in dense_units:
            model.add(Dense(units=unit[0], activation=unit[1]))
        if dropout > 0:
            model.add(Dropout(rate=dropout))
        if output_dense_activation is not None:
            model.add(Dense(units=steps))
        else:
            model.add(Dense(units=steps, activation=output_dense_activation))
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        print(f'loss = {loss}')
        if len(metrics) == 0:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

