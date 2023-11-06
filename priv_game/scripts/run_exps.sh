REPS=500
N_PROCS=32
DATA_DIR=results/

echo "[$(date +%F_%T)] Running recon, dcr & infer attacks..."
for DATA_NAME in acs fire
do
    echo "[$(date +%F_%T)] $DATA_NAME"

    for SYNTH_MODEL in NonPrivate RAP_2Kiters BayNet_3parents CTGAN IndHist PrivBayes_3parents_100eps PrivBayes_3parents_10eps PrivBayes_3parents_1eps RAP_2Kiters_100eps RAP_2Kiters_10eps RAP_2Kiters_1eps
    do
        echo "[$(date +%F_%T)]  => $SYNTH_MODEL"
        for N_ROWS in 1000000 100000 10000 1000 100 10
        do
            echo "[$(date +%F_%T)]    => $N_ROWS"
            echo "[$(date +%F_%T)]        => Sampling synthetic data..."
            python3 gen_synth_data.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR

            # utility
            echo "[$(date +%F_%T)]        => Getting utility..."
            python3 process_queries.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type normal --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --query_type any --k 3

            # (pre-processing) our attack with 3-way queries
            echo "[$(date +%F_%T)]        => Pre-processing recon attack..."
            python3 process_queries.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type normal --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --query_type simple --k 3
            python3 process_queries.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --query_type simple --k 3

            # run our attack with 3-way queries
            echo "[$(date +%F_%T)]        => Running recon attack..."
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type normal --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 3
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 3

            # infer attack
            echo "[$(date +%F_%T)]        => Running infer attack..."
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --attack_name infer --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR

            # DCR attack
            echo "[$(date +%F_%T)]        => Running dcr attack..."
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --attack_name dcr --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR
        done
    done
done

python3 analyze_privacy_utility.py --all --reps $REPS --data_dir $DATA_DIR --n_procs $N_PROCS
echo "[$(date +%F_%T)] Complete"
