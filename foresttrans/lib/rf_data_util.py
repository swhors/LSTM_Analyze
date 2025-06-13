"""
util.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
from datetime import datetime
from contextlib import closing
import sqlite3
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


def load_data_by_db(db_file_path, last_round, length, reverse=False):
    import os
    with closing(sqlite3.connect(db_file_path)) as conn:
        with closing(conn.cursor()) as cur:
            results = []
            first_bonus = 0
            query = f'select * from results where round>={last_round-length} and round <={last_round} '
            query += "order by round desc" if reverse else "order by round asc"
            datas = cur.execute(query).fetchall()
            for data in datas:
                if first_bonus == 0:
                    first_bonus = data[2]
                results.append((data[1], [int(i) for i in data[2].split(',')], data[3]))
            return results, first_bonus


def create_randomforest_db(db_file_path, verbose=1):
    columns = ["id INTEGER PRIMARY KEY AUTOINCREMENT",
               "n_estimator INTEGER",
               "data_length INTEGER",
               "random_state INTEGER",
               "rounds TEXT",
               "matched_cnts TEXT",
               "sum_val INTEGER",
               "version TEXT"]
    # (n_estimator, data_length, random_state, predict_lens[n_estimator][data_length][random_state], sum_val)
    query = f'CREATE TABLE IF NOT EXISTS rndforest ({",".join(columns)})'
    if verbose > 0:
        print('query=', query)
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db_file_path)) as conn:
        conn.execute(query)


"""
(25, 400, 2400, [[1176, 1175, 1174, 1173], [1, 1, 3, 2]], 7)
"""
def insert_randomforest_db(db_file_path, version, db_datas, auto_commit=True, verbose=0):
    """ insert_randomforest_db """
    if version == 0:
        version = str(datetime.now().timestamp())
    with closing(sqlite3.connect(db_file_path)) as conn:
        col_dicts = {"n_estimator": db_datas[0],
                     "data_length": db_datas[1],
                     "random_state": db_datas[2],
                     "rounds": ",".join([str(i) for i in db_datas[3][0]]),
                     "matched_cnts": ",".join([str(i) for i in db_datas[3][1]]),
                     "sum_val": db_datas[4],
                     "version": version
                     }
        columns = col_dicts.keys()
        values=list(col_dicts.values())
        column_val=["?" for i in range(len(columns))]
        query = f"INSERT INTO rndforest ({','.join(columns)}) "\
                f"VALUES ({','.join(column_val)})"
        with closing(conn.cursor()) as cursor:
            if verbose > 0:
                print(f'query={query}')
                print('-'*30)
            cursor.execute(query, values)
        if auto_commit:
            conn.commit()


def rf_predict(train_X, n_estimators=100, random_state=350, trial=5, verbose=0):
    """ rf_predict """
    # 데이터를 Pandas DataFrame으로 변환합니다.
    df = pd.DataFrame(train_X, columns=['회차', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6'])
    df = df.sort_values(by='회차').reset_index(drop=True)

    if verbose > 0:
        print("--- 로또 당첨 번호 이력 ---")
        print(df)

    # 기계 학습을 위한 데이터 준비
    # 각 회차의 당첨 번호(X)와 바로 다음 회차의 당첨 번호(y)를 사용합니다.
    X = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']]
    y = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].shift(-1)

    # 마지막 행은 다음 회차 데이터가 없으므로 제거합니다.
    X = X[:-1]
    y = y.dropna()
    
    if verbose > 0:
        print("\n--- 훈련 데이터 (X) ---")
        print(X.head())
        print("\n--- 정답 데이터 (y) ---")
        print(y.head())

    # 머신러닝 모델 선택 및 훈련
    # 랜덤 포레스트 회귀 모델을 사용합니다.
    # the random_state parameter is used to control the randomness of the algorithm, ensuring reproducibility of results. 
    # the n_estimators parameter specifies the number of decision trees in the forest. 
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state) # n_estimators: 만들 트리의 개수
    model.fit(X, y)

    # 예측할 회차의 이전 회차 데이터 (1173회차)
    last_draw = df.iloc[-1][['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values.reshape(1, -1)

    if verbose > 0:
        print(f"\n--- {df.iloc[-1]['회차']}회차 데이터로 다음 회차 예측 ---")
        print(last_draw)

    # 다음 번호 예측
    predicted_numbers_set = []
    for i in range(trial):
        predicted_numbers_float = model.predict(last_draw)
        # 예측된 번호 처리
        # 1. 소수점을 반올림하여 정수로 만듭니다.
        # 2. 1~45 사이의 값으로 보정합니다.
        # 3. 중복된 번호를 제거하고 6개를 선택합니다.
        predicted_numbers = set()
        for num in predicted_numbers_float[0]:
            # 반올림하여 정수로 변환
            int_num = int(round(num))
            # 1보다 작으면 1로, 45보다 크면 45로 보정
            if int_num < 1:
                int_num = 1
            elif int_num > 45:
                int_num = 45
            predicted_numbers.add(int_num)
        # 중복 제거 후 6개가 안되면, 부족한 만큼 다른 번호로 채웁니다.
        # (여기서는 가장 빈도가 높은 번호들 중 예측되지 않은 번호를 추가하는 방식을 사용)
        if len(predicted_numbers) < 6:
            all_numbers = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values.flatten()
            counts = pd.Series(all_numbers).value_counts()
    
            extra_needed = 6 - len(predicted_numbers)
            for num in counts.index:
                if extra_needed == 0:
                    break
                if num not in predicted_numbers:
                    predicted_numbers.add(num)
                    extra_needed -= 1
        # 최종 예측 번호를 정렬하여 출력
        final_prediction = sorted(list(predicted_numbers))[:6]
        predicted_numbers_set.append(final_prediction)
    return predicted_numbers_set


def rf_prediction_test(round_nums,
                       actual_numbers,
                       bonus,
                       predicted_numbers_set,
                       n_estimators=500,
                       random_state=500,
                       only_one=True,
                       verbose=0):
    if verbose>0:
        print("======================================")
        print(f"\n--- 참고: {round_nums}회 실제 번호 ---")
        print(f"번호: {actual_numbers}, 보너스: {bonus}")

    matched_len = 0

    for final_prediction in predicted_numbers_set:
        # 예측 결과와 실제 결과 비교
        matching_numbers = set(final_prediction).intersection(set(actual_numbers))
        if verbose > 0:
            print(f"\n--- 예측과 실제 결과 비교 ---")
            print(f"일치하는 번호: {sorted(list(matching_numbers))}")
            print(f"일치 개수: {len(matching_numbers)}")
            print(F"예측 번호: {final_prediction}")
            print(f"n_estimators: {n_estimators}")
            print(f"random_state: {random_state}")
        matched_len = len(matching_numbers)
        if only_one:
            break
    return matched_len


def list_to_dict(item_lists):
    """ list_to_dict """
    dicts = {}
    for item in item_lists:
        dicts[item[0]] = (item[1], item[2])
    return dicts


def predict_and_test(round, bonus, h_data, n_estimators, random_state, trial=5, with_data=False, verbose=0):
    """ predict_and_test """
    train_X = []
    test_X = []
    for i in range(len(h_data) - 1):
        train_X.append(h_data[i][1])
    test_X = h_data[len(h_data) - 1][1]
    predicted_numbers_set = rf_predict(train_X=train_X,
                                       n_estimators=n_estimators,
                                       random_state=random_state,
                                       trial=trial,
                                       verbose=verbose)
    predict_len1=rf_prediction_test(round,
                                    test_X,
                                    bonus,
                                    predicted_numbers_set,
                                    n_estimators,
                                    random_state,
                                    verbose)
    if with_data:
        return predicted_numbers_set, predict_len1
    return predict_len1


def main_predict(n_estimator,
                 last_round,
                 data_length,
                 random_state,
                 db_file_path,
                 trial,
                 verbose=0):
    """ main_predict """
    predict_lens = {}
    hist_data, first_bonus = load_data_by_db(db_file_path=db_file_path, last_round=last_round, length=data_length)
    for train_x in hist_data:
        train_x[1].insert(0, train_x[0])
    predicted_numbers_set, _ = predict_and_test(round=last_round,
                                                bonus=first_bonus,
                                                h_data=hist_data,
                                                n_estimators=n_estimator,
                                                random_state=random_state,
                                                trial=trial,
                                                with_data=True,
                                                verbose=verbose)
    return predicted_numbers_set


### Prepare and train, predict
def main_process(version,
                 n_estimators,
                 last_rounds,
                 data_lengths,
                 random_state_gap=100,
                 random_state_begin=0,
                 random_state_end=300000,
                 db_file_path='../db/metrics.db',
                 write_to_db=False,
                 write_db_file_path='../db/metrics.db',
                 trial=5,
                 verbose=0):
    """ main_process """
    predict_lens = {}
    for n_estimator in n_estimators:
        predict_lens[n_estimator] = {}
        if n_estimator > 1:
            for last_round in last_rounds:
                cnt = 0
                for data_length in data_lengths:
                    hist_data, first_bonus = load_data_by_db(db_file_path=db_file_path, last_round=last_round, length=data_length)
                    for train_x in hist_data:
                        train_x[1].insert(0, train_x[0])
                    if data_length not in predict_lens[n_estimator]:
                        predict_lens[n_estimator][data_length] = {}
                    for random_state in range(random_state_begin, random_state_end, random_state_gap):
                        predict_len = predict_and_test(last_round,
                                                       first_bonus,
                                                       hist_data,
                                                       n_estimator,
                                                       random_state,
                                                       trial,
                                                       verbose=0)
                        if n_estimator not in predict_lens:
                            predict_lens[n_estimator] = {}
                        if random_state not in predict_lens[n_estimator][data_length]:
                            predict_lens[n_estimator][data_length][random_state] = [[last_round], [predict_len]]
                        else:
                            predict_lens[n_estimator][data_length][random_state][0].append(last_round)
                            predict_lens[n_estimator][data_length][random_state][1].append(predict_len)
                        cnt += 1
                        if cnt % 1000 == 0:
                            print(f'proceed [now={datetime.now()}, n_estimator={n_estimator}, cnt={cnt}, '\
                                  f'last_round={last_round}, random_state={random_state}, data_length={data_length}]')
    return predict_lens


def print_predicts(predict_lens, sum_min=-1, sum_max=-1, version="T_00_00", write_to_file=False, write_to_db=False, verbose=0):
    result_set = []
    for n_estimator in predict_lens:
        for data_length in predict_lens[n_estimator]:
            for random_state in predict_lens[n_estimator][data_length]:
                sum_val = sum(predict_lens[n_estimator][data_length][random_state][1])
                if sum_min == -1 and sum_max == -1:
                    if verbose > 0:
                        print(n_estimator,
                              data_length,
                              random_state,
                              predict_lens[n_estimator][data_length][random_state])
                    result_set.append((n_estimator,
                                       data_length,
                                       random_state,
                                       predict_lens[n_estimator][data_length][random_state],
                                       sum_val))
                else:
                    if sum_min > 0 and sum_max > 0 and sum_val >= sum_min and sum_val <= sum_max:
                        if verbose > 0:
                            print(n_estimator,
                                  data_length,
                                  random_state,
                                  predict_lens[n_estimator][data_length][random_state])
                        result_set.append((n_estimator,
                                           data_length,
                                           random_state,
                                           predict_lens[n_estimator][data_length][random_state],
                                           sum_val))
                    else:
                        if sum_min > 0 and sum_val >= sum_min:
                            if verbose > 0:
                                print(n_estimator,
                                      data_length,
                                      random_state,
                                      predict_lens[n_estimator][data_length][random_state])
                            result_set.append((n_estimator,
                                               data_length,
                                               random_state,
                                               predict_lens[n_estimator][data_length][random_state],
                                               sum_val))
                        elif sum_max > 0 and sum_val <= sum_max:
                            if verbose > 0:
                                print(n_estimator,
                                      data_length,
                                      random_state,
                                      predict_lens[n_estimator][data_length][random_state])
                            result_set.append((n_estimator,
                                               data_length,
                                               random_state,
                                               predict_lens[n_estimator][data_length][random_state],
                                               sum_val))
                if write_to_file:
                    with closing(open("metric_temp.txt", "at")) as fd:
                        fd.write(f'{n_estimator},{data_length},{random_state},')
                        fd.write(f'{"/".join([str(i) for i in predict_lens[n_estimator][data_length][random_state][0]])},')
                        fd.write(f'{"/".join([str(i) for i in predict_lens[n_estimator][data_length][random_state][0]])},')
                        fd.write(f'{sum_val}\n')
                if write_to_db:
                    db_datas = (n_estimator,
                                data_length,
                                random_state,
                                predict_lens[n_estimator][data_length][random_state],
                                sum_val)
                    insert_randomforest_db('../db/metrics.db',
                                           version=version,
                                           db_datas=db_datas,
                                           verbose=verbose)    
    return result_set


def main(parameters, version, sum_min=-1, sum_max=-1, write_to_file=False, write_to_db=False, trial=5, verbose=0):
    """ main """
    print(f'start   [now={datetime.now()}]')
    db_file_path = '../db/metrics.db' if 'db_file_path' not in parameters else parameters["db_file_path"]
    write_db_file_path = '../db/metrics.db' if 'write_db_file_path' not in parameters else parameters["write_db_file_path"]
    predict_lens = main_process(version=version,
                                n_estimators=parameters["n_estimators"],
                                last_rounds=parameters["last_rounds"],
                                data_lengths=parameters["data_lengths"],
                                random_state_gap=parameters["random_state_gap"],
                                random_state_begin=parameters["random_state_begin"],
                                random_state_end=parameters["random_state_end"],
                                trial=trial,
                                db_file_path=db_file_path,
                                write_db_file_path=write_db_file_path,
                                verbose=verbose)
    print(f'completed [now={datetime.now()}]')
    print(f'start to read and write data: [now={datetime.now()}]')
    result_set = print_predicts(predict_lens=predict_lens,
                                sum_min=sum_min,
                                sum_max=sum_max,
                                version=version,
                                write_to_file=write_to_file,
                                write_to_db=write_to_db)
    print(f'complete to read and data: [now={datetime.now()}]')
    return result_set


def draw_data_graph(title,
                    info,
                    x_data,
                    y_data,
                    x_label,
                    y_label,
                    width=0,
                    height=0,
                    file_name="",
                    is_bar=True,
                    verb=False):
    """ draw_data_graph """
    if is_bar:
        plt.bar(x_data, y_data)
    else:
        plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(info.keys()) > 0:
        title +="\n"
        for key in info:
            title += (key + ":" + str(info[key]) + "\n")
    plt.title(title)
    if len(file_name) > 0:
        plt.legend(loc=0)
        plt.savefig(file_name)
    if width > 0 and height > 0:
        plt.figure(figsize=(width,height))
    plt.show()

