#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 113300 --random_state_end 114001 --last_rounds_begin 1001 --last_rounds_end 1000 --db_file_path "/db/metrics.db" --version="T_01_10"

#     "last_rounds": [i for i in range(1175, 770, -1)],

#{'n_estimator': '10', 'data_length': '100', 'random_state': '114116', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 16, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '114287', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 10, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '114905', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 15, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '116469', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 11, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '117668', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 15, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '118407', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 15, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '118991', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 15, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '119179', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 6, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '116702', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 18, #'version': 'T_01_95'}
#{'n_estimator': '10', 'data_length': '100', 'random_state': '116725', 'last_round': '1177', 'first_round': ['701'], 'round_length': 477, 'over_3_cnt': 16, #'version': 'T_01_95'}


# export START=114114 + 5
# export START=114285
# export START=114903
# export START=116467
# export START=117666
# export START=118405
# export START=118989
# export START=119177
# export START=116700
export START=116723
export GAP=5
export LAST_ROUND=1177

export VERSION="T_01_94"

PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin $START --random_state_end $((START + GAP)) --last_rounds_begin $LAST_ROUND --last_rounds_end 770 --db_file_path "/db/metrics.db" --version $VERSION
