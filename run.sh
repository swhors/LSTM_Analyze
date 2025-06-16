#PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 113300 --random_state_end 114001 --last_rounds_begin 1001 --last_rounds_end 1000 --db_file_path "/db/metrics.db" --version="T_01_10"

#     "last_rounds": [i for i in range(1175, 770, -1)],

PYTHONPATH=$PWD python runner.py --n_estimators 100 --data_lengths 100 --random_state_gap 1 --random_state_begin 114001 --random_state_end 114003 --last_rounds_begin 1175 --last_rounds_end 770 --db_file_path "/db/metrics.db" --version="T_01_10"
