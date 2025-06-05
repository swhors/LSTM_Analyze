"""
metric_analyze.py
"""
from contextlib import closing
from enum import Enum


class MatchedGetType(Enum):
    """ MatchedGetType """
    DATA_ONLY = 0
    LEN_ONLY = 1
    DATA_AND_LEN = 2


def load_data_by_version(test_id, model_ids):
    """ load_data """
    from lib.db_operate import select_metric, conn_db, close_db
    from datetime import datetime
    conn = conn_db("./db/metrics.db")
    metrics_col = []
    cnt = 1
    for id in model_ids:
        metric_dict = {'id': id, 'metric': {}}
        print(f'read_db.0_{cnt} (model_id={id}, test_id={test_id})')
        cnt += 1
        metrics = select_metric(conn=conn,
                                test_id=test_id,
                                model_id=id,
                                matched_size=3,
                                matched_size_cond=">=",
                                verbose=False)
        for metric in metrics:
            test_id = metric[1]
            version = metric[2]
            round = metric[3]
            matched_len = metric[9]
            if matched_len >= 3:
                if version not in metric_dict['metric']:
                    metric_dict['metric'][version] = [round]
                else:
                    if round not in metric_dict['metric'][version]:
                        metric_dict['metric'][version].append(round)
        metrics_col.append(metric_dict)
    close_db(conn)
    print(f'read_db.10 complete (model_id={id})')
    return metrics_col


def load_data(test_id, model_ids):
    """ load_data """
    from lib.db_operate import select_metric, conn_db, close_db
    from datetime import datetime
    conn = conn_db("./db/metrics.db")
    metrics_col = []
    cnt = 1
    for id in model_ids:
        metric_dict = {'id': id, 'metric': {}}
        print(f'read_db.0_{cnt} (model_id={id})')
        cnt += 1
        metrics = select_metric(conn=conn, test_id=test_id, model_id=id, verbose=False)
        for metric in metrics:
            test_id = metric[1]
            version = metric[2]
            round = metric[3]
            matched_len = metric[9]
            if matched_len >= 3:
                if round not in metric_dict['metric']:
                    metric_dict['metric'][round] = [version]
                else:
                    if version not in metric_dict['metric'][round]:
                        metric_dict['metric'][round].append(version)
        metrics_col.append(metric_dict)
    close_db(conn)
    print(f'read_db.10 complete (model_id={id})')
    return metrics_col


def load_data_matched_freq(test_id, model_ids):
    """ load_data """
    from lib.db_operate import select_metric, conn_db, close_db
    from datetime import datetime
    with closing(conn_db("./db/metrics.db")) as conn:
        metrics_col = []
        cnt = 1
        for id in model_ids:
            metric_dict = {'id': id, 'metric': {}}
            print(f'read_db.0_{cnt} (model_id={id})')
            cnt += 1
            metrics = select_metric(conn=conn,
                                    test_id=test_id,
                                    model_id=id,
                                    matched_size=3,
                                    matched_size_cond=">=",
                                    verbose=False)
            for metric in metrics:
                test_id = metric[1]
                version = metric[2]
                round = metric[3]
                matched = metric[8]
                matched_len = metric[9]
                if matched_len >= 3:
                    if matched not in metric_dict['metric']:
                        metric_dict['metric'][round] = [version]
                    else:
                        if version not in metric_dict['metric'][round]:
                            metric_dict['metric'][round].append(version)
            metrics_col.append(metric_dict)
        print(f'read_db.10 complete (model_id={id})')
        return metrics_col


def load_all_matched_info(test_ids: list, model_ids:list, get_type=MatchedGetType.LEN_ONLY):
    """ load_all_matched """
    from lib.db_operate import conn_db, close_db
    from lib.db_operate import select_metric_all_matched_size
    from datetime import datetime
    with closing(conn_db("./db/metrics.db")) as conn:
        metrics_col = []
        cnt = 1
        for test_id in test_ids:
            metric_dict = {'test_id': test_id, 'metric': {}}
            for model_id in model_ids:            
                print(f'load_all_matched.read_db.01_{cnt:02} (model_id={id}, test_id={test_id})')
                cnt += 1
                metrics = select_metric_all_matched_size(conn=conn,
                                                         test_id=test_id,
                                                         model_id=model_id,
                                                         verbose=False)
                for metric in metrics:
                    if model_id not in metric_dict['metric']:
                        metric_dict['metric'][model_id] = [metric]
                    else:
                        metric_dict['metric'][model_id].append(metric)
            metrics_col.append(metric_dict)
        print(f'load_data_matched_freq_v2.read_db.10 complete (model_id={id})')
        return metrics_col


def load_data_matched_freq_v2(test_ids: list, model_ids: list):
    """ load_data """
    from lib.db_operate import select_metric, conn_db, close_db
    from datetime import datetime
    with closing(conn_db("./db/metrics.db")) as conn:
        metrics_col = []
        cnt = 1
        for id in model_ids:
            metric_dict = {'id': id, 'metric': {}}
            for test_id in test_ids:            
                print(f'load_data_matched_freq_v2.read_db.01_{cnt:02} (model_id={id}, test_id={test_id})')
                cnt += 1
                metrics = select_metric(conn=conn,
                                        test_id=test_id,
                                        model_id=id,
                                        matched_size=3,
                                        matched_size_cond=">=",
                                        verbose=False)
                for metric in metrics:
                    test_id = metric[1]
                    version = metric[2]
                    round = metric[3]
                    matched = metric[8]
                    matched_len = metric[9]
                    if matched_len >= 3:
                        key = f'{version:02}:{round:02}'
                        if key not in metric_dict['metric']:
                            metric_dict['metric'][key] = [test_id]
                        else:
                            if test_id not in metric_dict['metric'][key]:
                                metric_dict['metric'][key].append(test_id)
            metrics_col.append(metric_dict)
        print(f'load_data_matched_freq_v2.read_db.10 complete (model_id={id})')
        return metrics_col


def draw_metric(metrics_col, x_label, y_label, width=0, height=0):
    """ draw_metric """
    from lib.graph_util import draw_data_graph
    for metrics in metrics_col:
        model_id = metrics['id']
        keys = list(metrics['metric'].keys())
        values = list(metrics['metric'].values())
        values_len = [len(l) for l in values]
        draw_data_graph(title=f"frequency of model no {model_id}",
                        x_data=keys,
                        y_data=values_len,
                        x_label=x_label,
                        y_label=y_label)


def draw_metric_v2(test_ids, model_ids, x_y_datas, x_label, y_label, is_bar=True, width=0, height=0):
    """ draw_metric """
    from lib.graph_util import draw_datas_graph
    draw_datas_graph(title=f"matched_count of {test_ids} / {model_ids}",
                     x_y_datas=x_y_datas,
                     x_label=x_label,
                     y_label=y_label,
                     is_bar=is_bar
                     )


def draw_graph(test_id, model_ids, width=0, height=0, del_res=True):
    metrics_col = load_data(test_id=test_id, model_ids=model_ids)
    draw_metric(metrics_col=metrics_col,
                x_label="round",
                y_label="frequency",
                width=width,
                height=height
               )

    del(metrics_col)

    import time
    time.sleep(1)

    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"CPU 시간: {usage.ru_utime} 초")
    print(f"메모리 사용량: {usage.ru_maxrss} KB")


def draw_graph_by_version(test_id, model_ids, width=0, height=0, del_res=True):
    metrics_col = load_data_by_version(test_id=test_id, model_ids=model_ids)
    draw_metric(metrics_col=metrics_col,
                x_label="version",
                y_label="frequency",
                width=width,
                height=height
               )

    del(metrics_col)

    import time
    time.sleep(1)

    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"CPU 시간: {usage.ru_utime} 초")
    print(f"메모리 사용량: {usage.ru_maxrss} KB")


def draw_graph_matched_size(test_ids, model_ids, from_pos=-1, to_pos=-1, width=0, height=0, del_res=True):
    metrics_col = load_all_matched_info(test_ids=test_ids,
                                        model_ids=model_ids)
    x_y_datas = []
    for metric in metrics_col:
        for model_id in metric['metric']:
            y_data = [y[0] for y in metric['metric'][model_id]]
            x_data = [i for i in range(len(y_data))]
            if from_pos>0 and to_pos>from_pos:
                y_data = y_data[from_pos:to_pos]
                x_data = x_data[from_pos:to_pos]
            else:
                if from_pos>0 and from_pos<len(y_data):
                    y_data = y_data[from_pos:]
                    x_data = x_data[from_pos:]
                if to_pos>0 and to_pos<len(y_data):
                    y_data = y_data[:to_pos]
                    x_data = x_data[:to_pos]
            x_y_datas.append((x_data, y_data))
    draw_metric_v2(test_ids=test_ids,
                   model_ids=model_ids,
                   x_y_datas=x_y_datas,
                   x_label="round",
                   y_label="matched count",
                   width=0,
                   height=0,
                   is_bar=False)


def get_matched_case_by_round(test_ids, model_ids, min_val, del_res=True):
    metrics_col = load_all_matched_info(test_ids=test_ids,
                                        model_ids=model_ids)
    matched_cnts = {}
    for metric in metrics_col:
        for model_id in metric['metric']:
            round = 0
            for m in metric['metric'][model_id]:
                if m[0] >= min_val:
                    if round not in matched_cnts:
                        matched_cnts[round] = [metric['test_id']]
                    else:
                        matched_cnts[round].append(metric['test_id'])
                round += 1
    return matched_cnts
    # draw_metric_v2(test_ids=test_ids,
    #                model_ids=model_ids,
    #                x_y_datas=x_y_datas,
    #                x_label="round",
    #                y_label="matched count",
    #                width=0,
    #                height=0,
    #                is_bar=False)
