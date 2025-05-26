"""
metrics_loader.py

sqlite3 db에서 매트릭 데이터를 로드하는 함수

"""
from lib.db_operate import conn_db, close_db
from lib.db_operate import select_metric, delete_all_metric, insert_metric


def row_2_metrics(datas):
    """ row_2_metrics """
    ids = []
    versions = []
    rounds = []
    trials = []
    dates = []
    model_ids = []
    metrics = []
    checked = []
    checked_size = []
    for data in datas:
        ids.append(data[0])
        versions.append(data[1])
        rounds.append(data[2])
        model_ids.append(data[3])
        trials.append(data[4])
        dates.append(data[5])
        metrics.append(data[6].split(','))
        checked.append(data[7].split(','))
        checked_size.append(data[8])
    return ids, versions, rounds, trials, dates, model_ids, metrics, checked, checked_size


def row_by_model_id(datas):
    """ row_by_model_id """
    model_datas = {}
    for data in datas:
        if data[3] not in model_datas:
            model_datas[data[3]] = [data]
        else:
            model_datas[data[3]].append(data)
    return model_datas


def load_data_by_version(version: int, verbose=False):
    """ load_data_by_version """
    conn = conn_db('./db/metrics.db')
    need_db_init = False
    datas = select_metric(conn=conn, version=version, limit=0)
    model_datas = row_by_model_id(datas)
    view_model_datas = False
    
    if verbose:
        print(type(model_datas))
        print(model_datas)

    model_metrics = {}
    max_model_id = 5
    for id in range(1, max_model_id + 1):
        ids, versions, rounds, trials, dates, model_ids, metrics, checked, checked_size = row_2_metrics(model_datas[id])
        model_metrics[id] = {}
        model_metrics[id]['id'] = ids
        model_metrics[id]['rounds'] = rounds
        model_metrics[id]['trials'] = trials
        model_metrics[id]['dates'] = dates
        model_metrics[id]['metrics'] = metrics
        model_metrics[id]['checked'] = checked
        model_metrics[id]['checked_size'] = checked_size
    if verbose:
        print(model_metrics[1])
    close_db(conn)
    return model_metrics


def load_data_all(verbose=False):
    """ load_data_all """
    conn = conn_db('./db/metrics.db')
    datas = select_metric(conn=conn, version=version, limit=0)
    ids, versions, rounds, trials, dates, model_ids, metrics, checked, checked_size = row_2_metrics(datas)
    if verbose:
        print(data)
        print(cnt, metrics[cnt], checked_size[cnt])
    close_db(conn)
    return ids, versions, rounds, trials, dates, model_ids, metrics, checked, checked_size

