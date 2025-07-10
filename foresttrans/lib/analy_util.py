"""
Functions
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization
from IPython.display import Markdown


def draw_markdown(rows: list):
    """ draw_markdown """
    from IPython.display import Markdown
    # Define a variable
    # Display the variable's value in Markdown
    table_header = "|title|content|"
    table_sep = "|--|--|"
    table_rows = [table_header,
                  table_sep,
                  ]
    for row in rows:
        table_rows.append(f"|{row[0]}|{row[1]}|")
    table_rows_str = "\n".join(table_rows)
    Markdown(table_rows_str)
    for row in rows:
        print(f'{row[0]:12s} = {row[1]}')


def draw_graph(X, title):
    """ draw_graph """
    print(X)
    y = [i for i in range(len(X))]
    plt.figure(figsize=(12, 6))
    plt.plot(y, X, label='Nums')
    plt.xlabel('rounds')
    plt.ylabel('selected')
    plt.title(title)
    plt.legend()
    plt.show()


def draw_scatter(Y: list,
                 title,
                 show_line=True,
                 show_dot=True,
                 show_cols=[0],
                 ext_datas=[],
                 show_label=True,
                 helper_line_nums=[],
                 marker_size=3,
                 fig_size=(12,6),
                 save_fig=False,
                 show_time=False
                ):
    """ draw_scatter """
    plt.figure(figsize=fig_size)
    colors=['b', 'g', 'r', 'c', 'm', 'y']
    if show_line == False and show_dot == False:
        show_line = True
    last = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    add_helper_cnt = 1 if len(ext_datas) > 0 else 0
    # draw helper line
    for helper_line_num in helper_line_nums:
        helper_line = [helper_line_num for i in range(len(Y[0]) + add_helper_cnt)]
        X = [i for i in range(len(Y[0]) + add_helper_cnt)]
        color = "tab:gray" if helper_line_num % 2 == 0 else 'tab:cyan'
        plt.plot(X, helper_line, color=color)

    for show_col in show_cols:
        if show_col < len(Y):
            if show_label:
                label = f'Nums{show_col}'            
            else:
                label = ""
            X = [i for i in range(len(Y[show_col]))]
            if show_line:
                plt.plot(X, Y[show_col], label=label, color=colors[show_col])
                label = ""
            if show_dot:
                plt.plot(X, Y[show_col], marker='o',
                         markersize=marker_size,
                         color = colors[show_col], label=label)
        last[show_col].append(Y[show_col][-1:][0])
    if len(ext_datas) > 0:
        for ext in ext_datas:
            if ext[0] < 6 and ext[0] in last:
                show_col = ext[0]
                last[show_col].append(ext[1])
                loc = np.arange(len(Y[0])-1, len(Y[0]) + len(last[show_col])-1)
                if show_line:
                    plt.plot(loc, last[show_col], color='k')
                if show_dot:
                    plt.plot(loc, last[show_col], marker='o', markersize=marker_size, color='k')
    plt.xlabel('rounds')
    plt.ylabel('selected')
    plt.title(title)
    if show_label:
        plt.legend()
    if save_fig:
        if show_time:
            plt.savefig(f'img/{title}_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.png', bbox_inches='tight')
        else:
            plt.savefig(f'img/{title}.png', bbox_inches='tight')
    plt.show()


def get_pre_bef(nums, wanted_num):
    results = []
    for pos in range(len(nums)):
        parts = []
        if nums[pos] == wanted_num:
            if pos > 0 and pos < (len(nums) - 1 ):
                parts.append(nums[pos-1])
                parts.append(nums[pos])
                parts.append(nums[pos+1])
            if pos == 0 and pos < (len(nums) - 1 ):
                parts.append(-1)
                parts.append(nums[pos])
                parts.append(nums[pos+1])
            if pos >= (len(nums) - 1 ) and pos > 0:
                parts.append(nums[pos-1])
                parts.append(nums[pos])
                parts.append(-1)
            results.append(parts)
    return results
            

def draw_average(title,
                 my_list,
                 length,
                 selected,
                 helper_line_nums=[10,20,30,40],
                 show_diff=False,
                 save_fig=False):
    print(f'{"_"*40}')
    averages = []
    metrics = []
    show_cols = [0]
    for i in range(length):
        averages.append(float(0))
    for i in range(len(my_list) - length + 1):
        item_part = my_list[i: i+length]
        average = np.mean(item_part)
        averages.append(float(average))
    metrics.append(averages)
    to_ordered = averages.copy()
    unique_ordered = list(set(to_ordered))
    unique_ordered.sort()
    draw_markdown([('최소값',f'{unique_ordered[1:6]}'),
                   ('최대값',f'{unique_ordered[-5:]}'),
                   ('마지막',f'{averages[-5:]}')
                  ])
    if show_diff:
        diff = [0]
        diff += [averages[i] - averages[i-1] for i in range(1, len(averages), 1)]
        metrics.append(diff)
        show_cols.append(1)
    draw_scatter(metrics,
                 f'{title}_average_{length}',
                 True,
                 True,
                 show_cols=show_cols,
                 helper_line_nums=helper_line_nums,
                 save_fig=save_fig)


def get_frequency(datas):
    # datas = my_list.copy()
    # datas.sort()
    ordered_dict = {}
    for key in datas:
        if key not in ordered_dict:
            ordered_dict[key] = 1
        else:
            ordered_dict[key] += 1
    ordered_dict_sorted = sorted(ordered_dict.items(), key=lambda item: item[1])
    return ordered_dict_sorted


def get_information(selected,
                    num_data,
                    ext_data,
                    wanted_data_length,
                    helper_line_nums_1,
                    helper_line_nums_2,
                    show_diff=False,
                    lengths=[5,8,10,13,15],
                    start_pos_0=0,
                    start_pos_1=0,
                    save_fig=False):
    length = 8
    if start_pos_0 > len(num_data):
        start_pos_0 = 0
    if start_pos_1 > len(num_data):
        start_pos_1 = 0
    title = f'Line_{selected+1}_{ext_data}'
    draw_scatter([num_data[start_pos_0:]],
                 title,
                 True,
                 True,
                 [0],
                 [(0,ext_data)],
                 helper_line_nums=helper_line_nums_1,
                 save_fig=save_fig)
    my_list = num_data[start_pos_1:] + [ext_data]
    last_num = num_data[-1:][0]
    ordered = my_list.copy()
    ordered.sort()
    ordered = list(set(ordered))
    freq_parts = get_frequency(my_list)
    freq_all = get_frequency(num_data)
    pre_bef = get_pre_bef(num_data, last_num)
    draw_markdown([
        ('최소 값',f'{ordered[:wanted_data_length]}'),
        ('최대 값',f'{ordered[-wanted_data_length:]}'),
        ('사용자 선택',f'{ext_data}'),
        ('마지막 값',f'{last_num}'),
        ('마지막 값들',f'{num_data[-wanted_data_length:]}'),
        ('자주 나오는 값(부분)',f'{freq_parts[-wanted_data_length:]}'),
        ('자주 나오는 값(전체)',f'{freq_all[-wanted_data_length:]}'),
        ('마지막 수의 앞과 뒤',f'{pre_bef}')
    ]
    )
    for length in lengths:
        draw_average(title=title,
                     my_list=my_list,
                     length=length,
                     selected=selected,
                     helper_line_nums=helper_line_nums_2,
                     show_diff=show_diff,
                     save_fig=save_fig)


class DataScaling():
    def __init__(self):
        # self._scaler_cls = MinMaxScaler()
        self._scaler_cls = StandardScaler()

    def fit_transform(self, data):
        return self._scaler_cls.fit_transform(data)

    def inverse_transform(self, datas: list):
        inversed_data = []
        cnt = 1
        for data in datas:
            # print(data[0])
            cnt += 1
            if data[1] == True:
                inversed = self._scaler_cls.inverse_transform(data[0].reshape(-1, 1))
            else:
                inversed = self._scaler_cls.inverse_transform(data[0])
            inversed_data.append(inversed)
        return inversed_data


def analyze_v1(random_state,
               row_data,
               layer_count=3,
               activation='sigmoid',
               units=50,
               epochs=50,
               batch_size=1,
               draw_graph=True,
               sequence_length=10,
               train_ratio=0.85,
               verbose=0):
    # 1. Prepare the Data
    # Create sample time series data
    df = pd.DataFrame(zip(row_data), columns=['matched_cnts'])
    if verbose > 1:
        print('df', df)
    if verbose > 1:
        print(f'Current Step #02 [now = {datetime.now()}]')

    data = df['matched_cnts'].values.reshape(-1, 1)

    # Normalize the data
    scaler = DataScaling()
    scaled_data = scaler.fit_transform(data)
    if verbose > 1:
        print(f'Current Step #04 [now = {datetime.now()}]')

    # Define sequence length (timesteps)

    # Create sequences for training
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length, 0])
        y.append(scaled_data[i + sequence_length, 0])
    X, y = np.array(X), np.array(y)
    if verbose > 1:
        print(f'Current Step #06 [now = {datetime.now()}]')

    # Reshape X for LSTM input (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    if verbose > 1:
        print(f'Current Step #08 [now = {datetime.now()}]')

    # Split data into training and testing sets
    train_size = int(len(X) * train_ratio)

    X_train, X_test, X_last = X[:train_size], X[train_size:], np.array([[y[-1:]]])
    y_train, y_test = y[:train_size], y[train_size:]
    if verbose > 1:
        print(f'Current Step #10 [now = {datetime.now()}]')

    tf.random.set_seed(random_state)

    # 2. Build the LSTM Model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    for i in range(layer_count):
        model.add(LSTM(units=units, return_sequences=True, activation=activation))
    model.add(LSTM(units=units, activation=activation))
    model.add(Dense(units=1)) # Output layer for predicting a single value
    if verbose > 1:
        print(f'Current Step #12 [now = {datetime.now()}]')

    # 3. Compile and Train the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    if verbose > 1:
        print(f'Current Step #14 [now = {datetime.now()}]')
    # 4. Make Predictions
    train_predict = model.predict(X_train, verbose=verbose)
    test_predict = model.predict(X_test, verbose=verbose)
    last_predict = model.predict(X_last, verbose=verbose)
    if verbose > 0:
        print(f'Current Step #16 [now = {datetime.now()}]')

    # Inverse transform predictions to original scale
    scaled_data1 = scaled_data
    wanted_datas = [(train_predict, False),
                    (test_predict, False),
                    (last_predict, True),
                    (y_train, True),
                    (y_test, True),
                    (scaled_data, False)]
    inversed_datas = scaler.inverse_transform(wanted_datas)
    train_predict = inversed_datas[0]
    test_predict = inversed_datas[1]
    last_predict = inversed_datas[2]
    y_train_original = inversed_datas[3]
    y_test_original = inversed_datas[4]
    scaled_data = inversed_datas[5]

    if verbose > 1:
        print(f'Current Step #18 [now = {datetime.now()}]')

    # 5. Evaluate the Model (Optional, but recommended)
    from sklearn.metrics import mean_squared_error
    train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict))

    if verbose > 0:
        print(f'Current Step #20 [now = {datetime.now()}]')
        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        print(f'random_state = {random_state}')
        print(f'last_prediction = {last_predict}')
        print(f'data_max_val = {scaled_data.max()}')
    # You can also visualize the results
    if draw_graph:
        # import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(scaled_data, label='Original Data')
        plt.plot(np.arange(sequence_length,
                           sequence_length + len(train_predict)),
                 train_predict,
                 label='Train Predictions')
        plt.plot(np.arange(sequence_length + len(train_predict),
                           sequence_length + len(train_predict) + len(test_predict)),
                 test_predict,
                 label='Test Predictions')
        plt.plot(np.arange(sequence_length + len(train_predict) + len(test_predict),
                       sequence_length + len(train_predict) + len(test_predict) + len(last_predict)),
                 last_predict,
                 label='Last Predictions')
        plt.xlabel(f'{random_state}\'s Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    return scaled_data.max(), last_predict, random_state


def data_prepare(row_data, train_ratio=0.85, sequence_length=10):
    ## data preparation
    print(f'Current Step #06 [now = {datetime.now()}]')
    df = pd.DataFrame(zip(row_data), columns=['matched_cnts'])
    data = df['matched_cnts'].values.reshape(-1, 1)
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print(f'Current Step #04 [now = {datetime.now()}]')
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length, 0])
        y.append(scaled_data[i + sequence_length, 0])
    X, y = np.array(X), np.array(y)
    # Reshape X for LSTM input (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # Split data into training and testing sets
    train_size = int(len(X) * train_ratio)
    X_train, X_test, X_last = X[:train_size], X[train_size:], np.array([[y[-1:]]])
    y_train, y_test = y[:train_size], y[train_size:]
    print(f'Current Step #08 [now = {datetime.now()}]')    
    return X_train, X_test, X_last, y_train, y_test, scaler


def search_model(X_train,
                 X_test,
                 X_last,
                 y_train,
                 y_test,
                 n_epochs=50,
                 max_trial=50,
                 random_state=355314,
                 title="model_cur",
                 verbose=0):
    n_input = X_train.shape[1]
    print(f'Current Step #10 [now = {datetime.now()}]')
    tf.random.set_seed(random_state)
    def build_model(hp):
        # print(f'Current Step #20 [now = {datetime.now()}]')    
        model = Sequential()
        model.add(Input(shape=(n_input, 1)))    
        model.add(LSTM(units=hp.Int('units',
                                    min_value=32,
                                    max_value=512,
                                    step=32),
                       return_sequences=True,
                       activation='tanh'))    
        model.add(Dense(units=hp.Int('units',
                                     min_value=32,
                                     max_value=512,
                                     step=32),
                        activation='tanh'))
        model.add(Dense(1))
        # print(f'Current Step #22 [now = {datetime.now()}]')    
        model.compile(loss='mse', metrics=['mse'], optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])))
        # print(f'Current Step #24 [now = {datetime.now()}]')    
        return model
    
    print(f'Current Step #12 [now = {datetime.now()}]')    
    bayesian_opt_tuner = BayesianOptimization(
        build_model,
        objective='mse',
        max_trials=max_trial,
        # executions_per_trial=1,
        directory=os.path.normpath(f'./models/{title}'),
        project_name='kerastuner_bayesian_poc',
        overwrite=True)
    print(f'Current Step #14 [now = {datetime.now()}]')    
    bayesian_opt_tuner.search(X_train,
                              y_train,
                              epochs=n_epochs,
                              # validation_data=(X_test, y_test),
                              validation_split=0.2,
                              verbose=verbose)
    print(f'Current Step #16 [now = {datetime.now()}]')    
    bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=1)
    model = bayes_opt_model_best_model[0]
    if verbose > -1:
        print(model.summary())
    print(f'Current Step #18 [now = {datetime.now()}]')    
    return model


print(f'define this code. [{datetime.now()}]')