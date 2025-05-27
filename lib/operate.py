"""
operate.py
"""

# load dependacies
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from tensorflow.keras.layers import LSTM

from lib.globalvar import *
from lib.data_loader import DataLoader
from lib.model1 import PredictLSTM1
from lib.model2 import PredictLSTM2
from lib.model3 import PredictLSTM3
from lib.model4 import PredictLSTM4

from lib.util_pred import print_data
from lib.util_pred import flat_data_with_sum, flat_data, get_frequency
from lib.util_pred import save_model, import_mode
from lib.util_pred import get_random_in_list
from lib.util_pred import print_data_with_sort
from lib.util_pred import get_sorted_n_values
from lib.util_pred import dict_key_count
from lib.util_pred import print_list
from lib.util_pred import print_dict_list
from lib.util_pred import print_title
from lib.activation import ActivationOutput, RecurrentActivation
from datetime import datetime


def create_model(id, model_type, model_info, dataset, hid_dim, verbose=True, epoch=50):
    """ create_model """
    model_classes = {'lstm1': PredictLSTM1,
                 'lstm2': PredictLSTM2,
                 'lstm3': PredictLSTM3,
                 'lstm4': PredictLSTM4}
    model_class = model_classes[model_type]
    verb = "verbose" if verbose else "None"
    title = f"model{id+1}"
    if verbose:
        print(f'create_model.{title} start to create model. {datetime.now()}')
    model = model_class(model_id=id,
                        data_loader=dataset,
                        hid_dim=hid_dim,
                        verb=verb,
                        args=[model_info])
    if verbose:
        print(f'create_model.{title} comleted to generate model. {datetime.now()}')
    start_time = datetime.now()
    hist = model.training(num_epoch=epoch, num_batch=1)
    end_time = datetime.now()
    elapsed = end_time - start_time
    if verbose:
        print(f'create_model.{title} elapsed = {elapsed}')
        print(f'create_model.{title} comleted to train model. {datetime.now()}')
    return model


def create_model_v2(id, model_type, lstm_info, dataset, hid_dim=45, verbose=True):
    """ create_model """
    model_classes = {'lstm1': PredictLSTM1,
                 'lstm2': PredictLSTM2,
                 'lstm3': PredictLSTM3,
                 'lstm4': PredictLSTM4}
    model_class = model_classes[model_type]
    verb = "verbose" if verbose else "None"
    title = f"model{id+1}"
    if verbose:
        print(f'create_model_v2.{title} start to create model. {datetime.now()}')
    model = model_class(model_id=id,
                        data_loader=dataset,
                        hid_dim = hid_dim,
                        verb = verb,
                        args=[lstm_info["model"]])
    if verbose:
        print(f'create_model_v2.{title} comleted to generate model. {datetime.now()}')
        print(f'create_model_v2.{title} start to train model. {datetime.now()}')
        print(f'create_model_v2.{title}   epochs = {lstm_info["train"]["epochs"]} {datetime.now()}')
    start_time = datetime.now()
    hist = model.training(num_epoch=lstm_info["train"]["epochs"],
                          num_batch=lstm_info["train"]["batch"],
                          is_shuffle=lstm_info["train"]["is_shuffle"]
                         )
    end_time = datetime.now()
    elapsed = end_time - start_time
    if verbose:
        print(f'create_model.{title} elapsed = {elapsed}')
        print(f'create_model.{title} comleted to train model. {datetime.now()}')
    return model


def create_model_v1(id, dataset, epoch=50, verbose=True):
    """ create_model_v1 """
    model = create_model(id=id,
                         model_type="lstm4",
                         layers=layers[id],
                         dataset=dataset,
                         hid_dim=45,
                         epoch=epoch,
                         verbose=verbose)
    return (model, layers[id])


def get_predicted(title, model, mode, use_pre, gen_num, last, verbose=True):
    """
    get_predicted

    param: gen_num -> 한 번에 몇 개의 데이터를 생성할 지에 대한 변수. 예) 5: 5개의 번호 열
    """
    if verbose:
        print_title(title=title)
        print(f'start predicting. {datetime.now()}')
    predictions = model.predict_numbers(mode2=mode, trial=gen_num, use_pre=use_pre)
    selected_five = get_random_in_list(predictions, 5)
    if verbose:
        print_data(title='wanted_data',
                   data_set=last)
        print('-----')
        print_list(title="selected_five", datas=selected_five)
    cnt, checked = dict_key_count(target=last[0], datas=predictions)
    if verbose:
        print(f'checked cnt = {cnt}')
        print_list(title="checked", datas=checked)
        print(f'end predicting. {datetime.now()}')
    return cnt, selected_five, checked, predictions
