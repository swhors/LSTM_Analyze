"""
small_predict.py
"""
from lib.util_pred import print_list
from lib.util_pred import print_dict_list
from lib.util_pred import print_title
from lib.operate import get_predicted
from datetime import datetime


def predict_small(round, trial, last, models, max_model_cnt, mode2, verbose=True):
    """ predict_small """
    # all (SMALL): predict #2
    
    matched_cnts_small = [0 for i in range(max_model_cnt)]
    selected_fives_small = [None for i in range(max_model_cnt)]
    matched_list_small = [None for i in range(max_model_cnt)]
    predicted_all_small = [None for i in range(max_model_cnt)]

    for i in range(max_model_cnt):
        title = f"history #{i+1}"
        model_num = i
        if models[model_num] is None:
            print(f"model #{model_num+1} is None")
        else:
            models[model_num][0].verb = "None"
            matched_cnts_small[model_num], \
            selected_fives_small[model_num], \
            matched_list_small[model_num], \
            predicted_all_small[model_num] = get_predicted(
                title=title,
                model=models[model_num][0],
                mode=mode2,
                use_pre=False,
                last=last,
                verbose=verbose,
                trial=trial
            )
    print(f'predicted time : round={round} date={datetime.now()}')
    return matched_cnts_small, matched_list_small


def print_predict_small(matched_cnts_small, matched_list_small, max_model_cnt):
    """ print_predict_small """
    # all (SMALL): print predicted    
    print(f'print time : {datetime.now()}')
    print_dict_list(title="matched_cnt", datas=matched_cnts_small)
    for i in range(max_model_cnt):
        print_list(title=f"matched_list[{i+1}]", datas=matched_list_small[i])
