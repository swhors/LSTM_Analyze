"""
db_operate.py

db에서 데이터를 조회하는 기능을 정의합니다.
swhors@naver.com
"""
import sqlite3
from contextlib import closing


def check_conn(func):
    """ check_conn """
    def inner(*args, **kwargs):
        if len(args) > 0:
            conn = args[0]
        else:
            conn = None
        if kwargs is not None and "conn" in kwargs:
            conn = kwargs['conn']
        if conn is None:
            print(f"error: connection is not established.[func={func}]")
            return
        else:
            return func(*args, **kwargs)
    return inner


def conn_db(db_name):
    """ conn_db """
    # Connect to a database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    return conn


@check_conn
def close_db(conn):
    """ close_db """
    conn.close()


@check_conn
def drop_table(conn, table_name):
    """ drop_table """
    if len(table_name) <= 0:
        print('Error : drop_table table_name is empty.')
        return
    with closing(conn.cursor()) as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")


def init_table(conn, table_name, columns, drop_table=False, verbose=False):
    """ init_results """
    with closing(conn.cursor()) as cursor:
        if verbose:
            print(f'init_table.0 table_name={table_name}')
            print(f'init_table.1 columns={columns}')
            print(f'init_table.2 drop_table={drop_table}')
        if drop_table:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        query = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(columns)})'
        if verbose:
            print(f'init_table.00 query={query}')
        cursor.execute(query)    


@check_conn
def init_results(conn, drop_table=True):
    """ init_results """
    table_name="results"
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "round INTEGER",
               "metric TEXT"]
    init_table(conn=conn, table_name=table_name, columns=columns, drop_table=drop_table)


@check_conn
def init_metric(conn, drop_table=True):
    """ init_metric """
    table_name = "metrics"
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "test_id TEXT",
               "version INTEGER",
               "round INTEGER",
               "model_id INTEGER",
               "trial INTEGER",
               "date DATETIME",
               "metric TEXT",
               "matched TEXT",
               "matched_size INTEGER"]
    init_table(conn=conn, table_name=table_name, columns=columns, drop_table=drop_table)


@check_conn
def init_models(conn, drop_table=True):
    """ init_models """
    table_name = "models"
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "test_id TEXT",
               "version INTEGER",
               "model_id INTEGER",
               "date DATETIME",
               "model BLOB"]
    init_table(conn=conn, table_name=table_name, columns=columns, drop_table=drop_table)


def select_operate(conn, table_name, columns, where, order, limit, verbose=False):
    """ select_metric """
    with closing(conn.cursor()) as cursor:
        if len(columns) == 0:
            query = f'SELECT * FROM {table_name}'
        else:
            query = f'SELECT ({",".join(columns)}) FROM {table_name}'
        query = query + ("" if len(where) == 0 else f' where {" and ".join(where)}')
        query = query + ("" if len(order) == 0 else f' order by {" ".join(order)}')
        query = query + ("" if limit == 0 else f" limit {limit}")
        if verbose:
            print(query)
        cursor.execute(query)
        rows = cursor.fetchall()
        datas = []
        for row in rows:
            if verbose:
                print(f'{type(row)}, {row}')
            datas.append(row)
        return datas


def select_metric_by_cols(conn,
                          columns=[],
                          test_id="",
                          version=-1,
                          model_id=-1,
                          limit=0,
                          matched_size=-1,
                          matched_size_cond=">",
                          verbose=False):
    """ select_metric_by_cols """
    # Retrieve data from the table
    where = []
    if len(test_id) > 0:
        where.append(f"test_id=\'{test_id}\'")
    if version >= 0:
        where.append(f"version={version}")
    if model_id > 0:
        where.append(f"model_id={model_id}")
    if matched_size > 0:
        where.append(f"model_id{matched_size_cond}{matched_size}")
    return select_operate(conn=conn,
                          table_name="metrics",
                          columns=columns,
                          where=where,
                          order=[],
                          limit=limit,
                          verbose=verbose
                         )


@check_conn
def select_metric_all_matched(conn,
                              test_id,
                              version=-1,
                              limit=0,
                              model_id=-1,
                              matched_size=-1,
                              matched_size_cond=">",
                              verbose=False):
    """ select_metric """
    return select_metric_by_cols(conn=conn,
                                 columns=['matched'],
                                 test_id=test_id,
                                 version=version,
                                 model_id=model_id,
                                 limit=0,
                                 matched_size=matched_size,
                                 matched_size_cond=matched_size_cond,
                                 verbose=verbose)


@check_conn
def select_metric_all_matched_size(conn,
                                   test_id,
                                   version=-1,
                                   limit=0,
                                   model_id=-1,
                                   matched_size=-1,
                                   matched_size_cond=">",
                                   verbose=False):
    """ select_metric """
    return select_metric_by_cols(conn=conn,
                                 columns=['matched_size'],
                                 test_id=test_id,
                                 version=version,
                                 model_id=model_id,
                                 limit=0,
                                 matched_size=matched_size,
                                 matched_size_cond=matched_size_cond,
                                 verbose=verbose)


@check_conn
def select_metric(conn,
                  test_id="",
                  version=-1,
                  model_id=-1,
                  limit=0,
                  matched_size=-1,
                  matched_size_cond=">",
                  verbose=False):
    """ select_metric """
    return select_metric_by_cols(conn=conn,
                                 test_id=test_id,
                                 version=version,
                                 model_id=model_id,
                                 limit=0,
                                 matched_size=matched_size,
                                 matched_size_cond=matched_size_cond,
                                 verbose=verbose)


@check_conn
def select_model(conn, test_id="", version=-1, model_id=-1, limit=0, verbose=False):
    """ select_metric """
    # Retrieve data from the table
    where = []
    if len(test_id) > 0:
        where.append(f"test_id=\'{test_id}\'")
    if version >= 0:
        where.append(f"version={version}")
    if model_id > 0:
        where.append(f"model_id={model_id}")
    return select_operate(conn=conn,
                          table_name="models",
                          columns=[],
                          where=where,
                          order=[],
                          limit=limit,
                          verbose=verbose)

@check_conn
def delete_all_metric(conn):
    """ delete_all_metric """
    if conn is None:
        print("Error: need to connect db.")
        return
    query = "DELETE * from metrics"
    with closing(conn.cursor()) as cursor:
        cursor.execute(query)
        conn.commit()


@check_conn
def insert_operate(conn, table, columns, values, auto_commit=True, verbose=False):
    """ insert_operate """
    column_val=["?" for i in range(len(columns))]
    query = f"INSERT INTO {table} ({','.join(columns)}) "\
            f"VALUES ({','.join(column_val)})"
    with closing(conn.cursor()) as cursor:
        if verbose:
            print(f'query={query}')
            print('-'*30)
        cursor.execute(query, values)
        if auto_commit:
            conn.commit()


def insert_model(conn, test_id, version, model_id, date, model, verbose=False):
    """ insert_model """
    table = "models"
    col_dicts = {"test_id": test_id,
                 "version": version,
                 "model_id": model_id,
                 "date": date,
                 "model": model}
    insert_operate(conn=conn,
                   table=table,
                   columns=list(col_dicts.keys()),
                   values=list(col_dicts.values()),
                   auto_commit=True,
                   verbose=verbose
                   )


def insert_metric(conn, test_id, version, round, date, model_ids, model_datas, verbose=False):
    """ insert_metric """
    table = "metrics"
    for model_data in model_datas:
        model_id_cnt = 0
        for metrics in model_data:
            trial = 1
            model_id = model_ids[model_id_cnt]
            for metric, checked in metrics:
                checked_len = len(checked)
                col_dicts = {"test_id": test_id,
                             "version": version,
                             "round": round,
                             "model_id": model_id,
                             "trial": trial,
                             "date": date,
                             "metric": ",".join([str(m) for m in metric]),
                             "matched": ",".join([str(c) for c in checked]),
                             "matched_size": checked_len}
                if verbose:
                    print(f'col_dicts={col_dicts}')
                    print('-'*30)
                insert_operate(conn=conn,
                               table=table,
                               columns=list(col_dicts.keys()),
                               values=list(col_dicts.values()),
                               auto_commit=False,
                               verbose=verbose)
                trial += 1
            model_id_cnt += 1
    conn.commit()