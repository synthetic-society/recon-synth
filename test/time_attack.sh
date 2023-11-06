#!/bin/bash
REPS=32
N_PROCS=32
DATA_DIR=results/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/priv_game

# prepare experiment
python3 gen_raw_data.py --data_name acs --n_users 1000 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --randomize
for N_ROWS in 1000000 100000 10000 100 10
do
python3 gen_synth_data.py --data_name acs --synth_model NonPrivate --n_rows $N_ROWS --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR 
done

for N_QUERIES in 10 100 1000 10000
do
    echo "gen_queries,$N_QUERIES" | tee -a log_time.txt
    /usr/bin/time -a -o log_time.txt -f "%e" python3 gen_queries.py --data_name acs --query_type simple --k 4 --n_queries $N_QUERIES --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --regen
    for N_ROWS in 10 100 1000 10000 100000
    do
        echo "process_queries,$N_QUERIES,$N_ROWS" | tee -a log_time.txt
        /usr/bin/time -a -o log_time.txt -f "%e" python3 process_queries.py --data_name acs --synth_model NonPrivate --n_rows $N_ROWS --scale_type cond --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --query_type simple --k 4

        echo "attack,$N_QUERIES,$N_ROWS" | tee -a log_time.txt
        /usr/bin/time -a -o log_time.txt -f "%e" python3 run_attack.py --data_name acs --synth_model NonPrivate --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 4
    done
done

cd $SCRIPT_DIR