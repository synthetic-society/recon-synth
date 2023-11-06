REPS=32
N_PROCS=32
DATA_DIR=results/

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/priv_game

echo "[$(date +%F_%T)] Running recon attack in different configurations..."
for DATA_NAME in acs
do
    echo "[$(date +%F_%T)] $DATA_NAME"

    echo "[$(date +%F_%T)]  => 2 way queries"
    for SYNTH_MODEL in NonPrivate
    do
        echo "[$(date +%F_%T)]    => $SYNTH_MODEL"
        for N_ROWS in 10 100 1000 10000 100000 1000000
        do
            echo "[$(date +%F_%T)]      => $N_ROWS"
            # (pre-processing) our attack with 2-way queries
            python3 process_queries.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 2

            # run our attack with 2-way queries
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 2
        done
    done

    echo "[$(date +%F_%T)]  => 4 way queries"
    for SYNTH_MODEL in NonPrivate
    do
        echo "[$(date +%F_%T)]    => $SYNTH_MODEL"
        for N_ROWS in 10 100 1000 10000 100000 1000000
        do
            echo "[$(date +%F_%T)]      => $N_ROWS"
            # (pre-processing) our attack with 4-way queries
            python3 process_queries.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 4

            # run our attack with 4-way queries
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 4
        done
    done

    # combine 2, 3 and 4-way queries
    python3 combine_queries.py --data_name $DATA_NAME --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR

    echo "[$(date +%F_%T)]  => 2 + 3 + 4 way queries"
    for SYNTH_MODEL in NonPrivate
    do
        echo "[$(date +%F_%T)]    => $SYNTH_MODEL"
        for N_ROWS in 10 100 1000 10000 100000 1000000
        do
            echo "[$(date +%F_%T)]      => $N_ROWS"
            # run our attack with 2+3+4-way queries
            python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 234
        done
    done

    # run increasing queries
    echo "[$(date +%F_%T)]  => Increasing queries"
    for SYNTH_MODEL in NonPrivate
    do
        echo "[$(date +%F_%T)]    => $SYNTH_MODEL"
        for N_ROWS in 10 100 1000 10000 100000 1000000
        do
            echo "[$(date +%F_%T)]    => $N_ROWS"
            for K in 2 3 4 234
            do
                echo "[$(date +%F_%T)]    => $K"
                for N_QUERIES in 1 10 100 1000 10000
                do
                    echo "[$(date +%F_%T)]      => $N_QUERIES"
                    python3 run_attack.py --data_name $DATA_NAME --synth_model $SYNTH_MODEL --n_rows $N_ROWS --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k $K --n_queries $N_QUERIES
                done
            done
        done
    done
done

python3 analyze_privacy_utility.py --all --reps $REPS --data_dir $DATA_DIR --n_procs $N_PROCS
echo "[$(date +%F_%T)] Complete"

cd $SCRIPT_DIR