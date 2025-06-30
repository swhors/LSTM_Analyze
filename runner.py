import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
from datetime import datetime
import argparse
from foresttrans.lib.rf_data_util import load_data_by_db, create_randomforest_db, insert_randomforest_db, main_predict
from foresttrans.lib.rf_data_util import predict_and_test, list_to_dict, rf_prediction_test, rf_predict, print_predicts
from foresttrans.lib.rf_data_util import main, main_process, draw_data_graph

import os

# 불필요한 경고 메시지를 숨깁니다.
warnings.filterwarnings('ignore')

print(f'completed to load library. [{datetime.now()}]')


def get_args():
    parser = argparse.ArgumentParser(description='foresttrans argparser')
    parser.add_argument('--n_estimators',
                        type=int,
                        help='n_estimators')
    parser.add_argument('--data_lengths', 
                        type=int,
                        help='data_lengths')
    parser.add_argument('--random_state_gap',
                        metavar='random_state_gap',
                        type=int,
                        default=1,
                        help='random_state_gap, default=1')
    parser.add_argument('--random_state_begin',
                        type=int,
                        default=2,
                        help='random_state_begin, default=2')
    parser.add_argument('--random_state_end',
                        type=int,
                        default=1,
                        help='random_state_end, default=1')
    parser.add_argument('--last_rounds_begin',
                        type=int,
                        default=1001,
                        help='last_rounds_begin')
    parser.add_argument('--last_rounds_end',
                        type=int,
                        default=1000,
                        help='last_rounds_end, last_rounds_end < last_rounds_begin')
    parser.add_argument('--db_file_path',
                        default="../db/metrics.db",
                        help='db file location')
    parser.add_argument('--version', metavar='version', default="T_00_00",
                        help='version')
    parser.add_argument('--is_tuning', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = get_args()
    print(f'PYTHONPATH         : {os.environ["PYTHONPATH"]}')
    print(f'Working Path       : {os.getcwd()}')
    print(f'n_estimators       : {args.n_estimators}')
    print(f'data_lengths       : {args.data_lengths}')
    print(f'random_state_gap   : {args.random_state_gap}')
    print(f'random_state_begin : {args.random_state_begin}')
    print(f'random_state_end   : {args.random_state_end}')
    print(f'last_rounds_begin  : {args.last_rounds_begin}')
    print(f'last_rounds_end    : {args.last_rounds_end}')
    print(f'db_path            : {args.db_file_path}')
    print(f'is_tuning          : {args.is_tuning}')

    run_this = False

    if args.last_rounds_begin > args.last_rounds_end and args.random_state_begin < args.random_state_end:
        db_path=os.getcwd() + args.db_file_path

        version=args.version
        # !echo "delete from rndforest where version=\"{version}\"" | sqlite3 ../db/metrics.db
        parameters = {
            "n_estimators": [args.n_estimators],
            "data_lengths": [args.data_lengths],
            "random_state_gap": args.random_state_gap,
            "random_state_begin": args.random_state_begin,
            "random_state_end": args.random_state_end,
            "last_rounds": [i for i in range(args.last_rounds_begin, args.last_rounds_end, -1)],
            "db_file_path": db_path,
            "write_db_file_path": db_path
            }
        result_sets = main(parameters=parameters,
                           version=version,
                           sum_min=0,
                           sum_max=-1,
                           trial=1,
                           is_tuning=args.is_tuning,
                           write_to_db=True)