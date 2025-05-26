# load dependacies
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from lib.globalvar import *
from lib.data import DataLoader
from lib.model import PredictLSTM
from lib.model2 import PredictLSTM2

import argparse
import matplotlib.pyplot as plt

model_classes = {'lstm': PredictLSTM,
                 'lstm2': PredictLSTM2}


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run prediction..")
    
    parser.add_argument('--window', type = int, default = 1 , help = "time stamps")
    parser.add_argument('--data_dir', nargs='?', default='./selectd1.csv')
    
    parser.add_argument('--mode', nargs='?', default='back-test', help = "back-test or predict")
    parser.add_argument('--mode2', nargs='?', default='sampling', help = "greed or sampling")
    parser.add_argument('--verb', nargs='?', default='not_verbose', help = "verbose or not_verb")
    
    parser.add_argument('--trial', type = int, default='100', help = "how much trials to generate")
    parser.add_argument('--training_length', type=float, default = 0.9)
    parser.add_argument('--epoch', type=int, default = 3)
    parser.add_argument('--batch', type=int, default = 1)
    parser.add_argument('--model', nargs='?', default='lstm2', help = "lstm or lstm2")
    
    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    dataset = DataLoader(data_dir=args.data_dir,
                         training_length=args.training_length,
                         window_prev=args.window,
                         mode=args.mode)
    epoch = args.epoch
    batch = args.batch
    model_class = model_classes[args.model]
    predict = model_class(dataset, hid_dim = 32, verb = args.verb)
    hist = predict.training(num_epoch = epoch, num_batch = batch)
    prediction_number_set = predict.predict_numbers(args.mode2, args.trial)
    random_pred_set = predict.predict_randomely(args.trial)
    
    if args.mode == 'back-test':
        predict.evaluate(prediction_number_set)
        print("---------Random baseline-------------")
        predict.evaluate(random_pred_set)
    print('predicted')
    cnt = 1
    for pre in random_pred_set:
        p_temp = [ int(i) + 1 for i in pre ]
        p_temp.sort()
        print(f'{cnt} : {p_temp}')
        cnt += 1

    cnt = 1
    for pre in prediction_number_set:
        p_temp = [ int(i) + 1 for i in pre ]
        p_temp.sort()
        print(f'{cnt} : {p_temp}')
        cnt += 1

    print('previios > 5 12 24 26 39 42 +20')
    
    
    # dataset.test_X = np.array([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    #      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]])
