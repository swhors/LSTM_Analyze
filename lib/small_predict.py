"""
small_predict.py
"""
import pickle
import time
from lib.util_pred import print_list
from lib.util_pred import print_dict_list
from lib.util_pred import print_title
from datetime import datetime
from lib.db_operate import conn_db, close_db
from lib.db_operate import init_metric
from lib.db_operate import init_models
from lib.db_operate import insert_metric
from lib.db_operate import insert_model
from lib.operate import create_model, create_model_v1, create_model_v2
from lib.operate import get_predicted


def predict_in_round(round, gen_num, last, models, max_model_cnt, mode2, verbose=0):
    """ predict_small """
    # all (SMALL): predict #2
    
    matched_cnts_small = [0 for i in range(len(models))]
    selected_fives_small = [None for i in range(len(models))]
    matched_list_small = [None for i in range(len(models))]
    predicted_all_small = [None for i in range(len(models))]

    i = 0
    for model in models:
        title = f"history #{model[0].model_id}"
        model[0].verb = "None" if verbose == 0 else "verbose"
        matched_cnts_small[i], \
        selected_fives_small[i], \
        matched_list_small[i], \
        predicted_all_small[i] = get_predicted(
            title=title,
            model=model[0],
            mode=mode2,
            use_pre=False,
            last=last,
            verbose=verbose,
            gen_num=gen_num
            )
        i += 1
    if verbose > 0:
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
                  verbose=0):
    """ predict_metric """
    need_db_init = False
    conn = None
    verb_str = True if verbose > 0 else False
    if write_to_db:
        if verbose>0:
            print(f'connect to db [{db_file_name}]')
        conn = conn_db(db_file_name)
        init_metric(conn, drop_table=need_db_init)
        if verbose>0:
            print(f'success to connect db [{conn}]')
    matched_cnts_all = []
    matched_list = []
    model_ids = [model[0].model_id for model in models]

    for round in range(round_limit+1):
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
            if verbose>0:
                print(f'db connection handle __0 =[{conn}]')
            insert_metric(conn=conn,
                          test_id=test_id,
                          version=version,
                          round=round,
                          model_ids=model_ids,
                          date=date,
                          model_datas=[matched_list_small],
                          verbose=verb_str)
    if write_to_db:
        close_db(conn=conn)


def print_predict_small(matched_cnts_small, matched_list_small, max_model_cnt):
    """ print_predict_small """
    # all (SMALL): print predicted    
    print(f'print time : {datetime.now()}')
    print_dict_list(title="matched_cnt", datas=matched_cnts_small)
    for i in range(max_model_cnt):
        print_list(title=f"matched_list[{i+1}]", datas=matched_list_small[i])


def save_model(test_id, version, models, verbose=False):
    """ save_model """
    conn = conn_db('./db/metrics.db')
    init_models(conn)
    model_id = 1
    date = datetime.now()
    for model in models:
        model_bin = pickle.dumps(model[0].model)
        insert_model(conn=conn,
                     test_id=test_id,
                     version=version,
                     model_id=model_id,
                     date=date,
                     model=model_bin,
                     verbose=verbose
                    )
        model_id += 1
    close_db(conn)


def gen_multi_model(test_id, lstm_args, version, verbose=0):
    """ gen_multi_model """
    models = []
    print(f'start to train all models. {datetime.now()}')
    # for i in range(max_model_cnt):
    for i in lstm_args.keys():
        print(f'{i}\'s training. status=start {datetime.now()}')
        model = create_model_v2(id=i,
                                model_type="lstm4",
                                lstm_info=lstm_args[i],
                                dataset=lstm_args[i]["dataset"],
                                verbose=verbose
                                )
        models.append((model, lstm_args[i]))
        print(f'{i}\'s training. status=end {datetime.now()}')
    print(f'completed to train all models. {datetime.now()}')
    return models


def build_and_predict(args: dict, lstm_args, sleep_time=0, verbose=0):
    """ build_and_predict """
    if 'test_id' not in args:
        return
    test_id = args['test_id']
    version = args['version']
    # max_model_cnt = args['max_model_cnt']
    # max_model_cnt = len(lstm_args)
    write_to_db = args['write_to_db']
    last = args['last']
    db_file_name = args['db_file_name']
    if 'mode2' not in args:
        mode2 = "sampling"
    else:
        mode2 = args['mode2']
    if 'gen_num' not in args:
        gen_num = 5
    else:
        gen_num = args['gen_num']

    if 'round_limit' not in args:
        round_limit = 40
    else:
        round_limit = args['round_limit']
    print(f'{"*"*20}\nworking model_version = {version}  {datetime.now()}\n{"*"*20}')
    verb = True if verbose > 0 else False
    models = gen_multi_model(test_id=test_id,
                             lstm_args=lstm_args,
                             version=version,
                             verbose=verbose
                            )
    if sleep_time > 0:
        time.sleep(sleep_time)
    print(f'train to create model({version}). {datetime.now()}')
    predict_small(test_id=test_id,
                  version=version,
                  models=models,
                  write_to_db=write_to_db,
                  round_limit=round_limit,
                  last=last,
                  gen_num=gen_num,
                  db_file_name=db_file_name,
                  mode2=mode2,
                  max_model_cnt=len(models),
                  verbose=verbose
                  )
    save_model(test_id=test_id, version=version, models=models, verbose=verb)
