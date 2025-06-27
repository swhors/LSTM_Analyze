#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 113300 --random_state_end 114001 --last_rounds_begin 1001 --last_rounds_end 1000 --db_file_path "/db/metrics.db" --version="T_01_10"

#     "last_rounds": [i for i in range(1175, 770, -1)],

#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 114001 --random_state_end 114003 --last_rounds_begin 1175 --last_rounds_end 770 --db_file_path "/db/metrics.db" --version="T_01_10"
#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 114003 --random_state_end 114004 --last_rounds_begin 1175 --last_rounds_end 770 --db_file_path "/db/metrics.db" --version="T_01_10"
#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 114004 --random_state_end 114005 --last_rounds_begin 1175 --last_rounds_end 770 --db_file_path "/db/metrics.db" --version="T_01_10"

# export START=3700 + 50
# export START=116700 + 30
# export START=113700 + 100
# export START=113600 + 100
# export START=113800 + 100
# export START=113200 + 100
# export START=114000 + 100

# export START=116800 + 100
export START=116700
export GAP=100

export LAST_ROUND_BEGIN=1177
export LAST_ROUND_END=770
export N_ESTIMATOR=45
export DATA_LENGTH=100

export VERSION="T_01_15"

PYTHONPATH=$PWD python runner.py --n_estimators $N_ESTIMATOR --data_lengths $DATA_LENGTH --random_state_gap 1 --random_state_begin $START --random_state_end $((START + GAP)) --last_rounds_begin $LAST_ROUND_BEGIN --last_rounds_end $LAST_ROUND_END --db_file_path "/db/metrics.db" --version $VERSION

