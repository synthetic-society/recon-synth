#!/bin/bash
REPS=32
N_PROCS=32
DATA_DIR=results/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/priv_game

echo "[$(date +%F_%T)] Preparing experiments..."
for DATA_NAME in acs
do
    echo "[$(date +%F_%T)] $DATA_NAME"
    
    # generate raw dataset and select target user to attack
    echo "[$(date +%F_%T)] Generating raw datasets..."
    python3 gen_raw_data.py --data_name $DATA_NAME --n_users 1000 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --randomize

    # generate 3-way queries for utility
    echo "[$(date +%F_%T)] Generating random sample of 3-way queries (for utility)..."
    python3 gen_queries.py --data_name $DATA_NAME --query_type any --k 3 --n_queries 100 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR

    # generate 3-way queries for our attack
    echo "[$(date +%F_%T)] Generating all 3-way queries involving secret bit (for Adv_recon)..."
    python3 gen_queries.py --data_name $DATA_NAME --query_type simple --k 3 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR
done
echo "[$(date +%F_%T)] Complete"

cd $SCRIPT_DIR