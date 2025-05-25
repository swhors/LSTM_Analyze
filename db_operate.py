"""
db_operate.py
"""
import sqlite3
from contextlib import closing


def conn_db(db_name):
    """ conn_db """
    # Connect to a database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)
    return conn


def close_db(conn):
    """ close_db """
    if conn is None:
        print("Error: need to connect db.")
        return
    # Close the connection
    conn.close()


def drop_table(conn, table_name):
    """ drop_table """
    if len(table_name) <= 0:
        print('Error : drop_table table_name is empty.')
        return
    with closing(conn.cursor()) as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")


def init_table(conn, table_name, columns, drop_table=False, verbose=False):
    """ init_results """
    if conn is None:
        print("Error: need to connect db.")
        return
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


def init_results(conn, drop_table=True):
    table_name="results"
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "round INTEGER",
               "metric TEXT"]
    init_table(conn=conn, table_name=table_name, columns=columns, drop_table=drop_table)


def init_metric(conn, drop_table=True):
    table_name = "metrics"
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "version INTEGER",
               "round INTEGER",
               "model_id INTEGER",
               "trial INTEGER",
               "date DATETIME",
               "metric TEXT",
               "matched TEXT",
               "matched_size INTEGER"]
    init_table(conn=conn, table_name=table_name, columns=columns, drop_table=drop_table)


def select_metric(conn, version=-1, model_id=-1, limit=0, verbose=False):
    """ select_metric """
    if conn is None:
        print("Error: need to connect db.")
        return
    # Retrieve data from the table
    where_clause = []
    if version >= 0:
        where_clause.append(f"version={version}")
    if model_id > 0:
        where_clause.append(f"model_id={model_id}")
    cursor = conn.cursor()
    query = "SELECT * FROM metrics"
    query = query + ("" if len(where_clause) == 0 else f' where {" and ".join(where_clause)}')
    query = query + ("" if limit ==0 else f" limit {limit}")
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


def delete_all_metric(conn):
    """ delete_all_metric """
    if conn is None:
        print("Error: need to connect db.")
        return
    query = "DELETE * from metrics"
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()


def insert_metric(conn, version, round, date, model_datas, verbose=False):
    """ insert_metric """
    if conn is None:
        print("Error: need to connect db.")
        return
    query = "INSERT INTO metrics (version, round, model_id, trial, date, metric, matched, matched_size) "\
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    cursor = conn.cursor()
    for model_data in model_datas:
        model_id = 1
        for metrics in model_data:
            trial = 1
            for metric, checked in metrics:
                checked_len = len(checked)
                if verbose:
                    print(f'{version}, {round}, {model_id}, {trial}, {date}, {metric}, {checked}, {checked_len}')
                    print('-'*30)
                cursor.execute(query,
                               (version,
                                round,
                                model_id,
                                trial,
                                date,
                                ",".join([str(m) for m in metric]),
                                ",".join([str(c) for c in checked]),
                                checked_len
                               )
                              )
                trial += 1
            model_id += 1
    conn.commit()
