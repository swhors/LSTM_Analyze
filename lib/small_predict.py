"""
small_predict.py
"""
from lib.util_pred import print_list
from lib.util_pred import print_dict_list
from lib.util_pred import print_title
from lib.operate import get_predicted
from datetime import datetime
from lib.db_operate import conn_db, close_db
from lib.db_operate import init_metric, insert_metric


def predict_in_round(round, gen_num, last, models, max_model_cnt, mode2, verbose=True):
    """ predict_small """
    # all (SMALL): predict #2
    
    matched_cnts_small = [0 for i in range(len(models))]
    selected_fives_small = [None for i in range(len(models))]
    matched_list_small = [None for i in range(len(models))]
    predicted_all_small = [None for i in range(len(models))]

    i = 0
    for model in models:
        title = f"history #{model[0].model_id}"
        model_num = i
        model[0].verb = "None"
        matched_cnts_small[model_num], \
        selected_fives_small[model_num], \
        matched_list_small[model_num], \
        predicted_all_small[model_num] = get_predicted(
            title=title,
            model=model[0],
            mode=mode2,
            use_pre=False,
            last=last,
            verbose=verbose,
            gen_num=gen_num
            )
        i += 1
    if verbose:
        print(f'predicted time : round={round} date={datetime.now()}')
    return matched_cnts_small, matched_list_small


def predict_small(test_id,
                  version,
                  models,
                  write_to_db,
                  round_limit,
                  last,
                  gen_num,
                  db_file_name,
                  max_model_cnt,
                  mode2,
                  verbose=False):
    """ predict_metric """
    need_db_init = False
    conn = None
    if write_to_db:
        if verbose:
            print(f'connect to db [{db_file_name}]')
        conn = conn_db(db_file_name)
        init_metric(conn, drop_table=need_db_init)
        if verbose:
            print(f'success to connect db [{conn}]')
    for round in range(round_limit+1):
        matched_cnts_all = []
        matched_list = []
        date = str(datetime.now())
        matched_cnts_small, matched_list_small = predict_in_round(round=round,
                                                                  gen_num=gen_num,
                                                                  last=last,
                                                                  models=models,
                                                                  max_model_cnt=max_model_cnt,
                                                                  mode2=mode2,
                                                                  verbose=verbose
                                                                  )
        matched_cnts_all.append(matched_cnts_small)
        matched_list.append(matched_list_small)

        if write_to_db:
            if verbose:
                print(f'db connection handle __0 =[{conn}]')
            insert_metric(conn=conn,
                          test_id=test_id,
                          version=version,
                          round=round,
                          date=date,
                          model_datas=matched_list,
                          verbose=verbose)
    if write_to_db:
        close_db(conn=conn)


def print_predict_small(matched_cnts_small, matched_list_small, max_model_cnt):
    """ print_predict_small """
    # all (SMALL): print predicted    
    print(f'print time : {datetime.now()}')
    print_dict_list(title="matched_cnt", datas=matched_cnts_small)
    for i in range(max_model_cnt):
        print_list(title=f"matched_list[{i+1}]", datas=matched_list_small[i])
