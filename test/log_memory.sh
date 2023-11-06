#!/bin/bash
REPS=32
N_PROCS=32
DATA_DIR=results/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/priv_game

python3 process_queries.py --data_name acs --synth_model NonPrivate --n_rows 1000000 --scale_type cond --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --query_type simple --k 4 &
python3 run_attack.py --data_name acs --synth_model NonPrivate --n_rows 1000000 --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 4 --n_queries 100000 &
for SECONDS in $(seq 0 1 3600)
do
    free -m | grep Mem: | sed 's/Mem://g' | awk '{print $2}' >> $DATA_DIR/memory_usage.txt
    sleep 1
done

cd $SCRIPT_DIR