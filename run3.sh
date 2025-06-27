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

export GAP=5
export LAST_ROUND=1177
export N_ESTIMATOR=10
export VERSION="T_01_93"
export DATA_LENGTH=100

STARTS=(114114 114285 114903 116467 117666 118405 118989 119177 116700 116723)
for START in "${STARTS[@]}"; do
  echo "$START"
PYTHONPATH=$PWD python runner.py --n_estimators $N_ESTIMATOR --data_lengths $DATA_LENGTH --random_state_gap 1 --random_state_begin $START --random_state_end $((START + GAP)) --last_rounds_begin $LAST_ROUND --last_rounds_end 770 --db_file_path "/db/metrics.db" --version $VERSION

done
