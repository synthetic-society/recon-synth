#!/bin/bash
REPS=500
N_PROCS=32
DATA_DIR=results/

echo "[$(date +%F_%T)] Preparing extra experiments (generating 2-way and 4-way queries)..."
for DATA_NAME in acs fire
do
    echo "[$(date +%F_%T)] $DATA_NAME"
    
    # generate all 2-way queries for our attack
    python3 gen_queries.py --data_name $DATA_NAME --query_type simple --k 2 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR

    # generate 100K random 4-way queries for our attack
    python3 gen_queries.py --data_name $DATA_NAME --query_type simple --k 4 --n_queries 100000 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR
done
echo "[$(date +%F_%T)] Complete"