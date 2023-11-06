"""
Step 5: Run attack (Adv_recon, Adv_infer, or Adv_dcr) against generated synthetic data
For each repetition of privacy game (rep):
    - load previously generated raw and synthetic data
    - load previously generated queries
    - load noisy result of queries
    - run attack
"""
import click
import os
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Process, Array, Queue
from threading import Thread
import psutil
from time import sleep
import lzma

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *
from attacks import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def single_rep(rep, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, data_dir):
    # run attack against a single repetition of privacy game
    # setup dirs
    rep_dir = f'{data_dir}/rep_{rep}'
    query_subdir = f'simple/{k}way'
    query_dir = f'{rep_dir}/queries/{query_subdir}/'
    synth_df_dir = f'{rep_dir}/{synth_model}/{n_rows_names[n_rows]}'
    synth_result_dir = f'{synth_df_dir}/{query_subdir}_{scale_type}/'
    os.makedirs(synth_result_dir, exist_ok=True)

    # load raw dataset and target user
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')
    u = int(np.genfromtxt(f'{rep_dir}/user.csv'))

    if attack_name == 'infer':
        # Adv_infer
        synth_df = pd.read_csv(f'{synth_df_dir}/synth_df.csv.gz', compression='gzip')

        # train classifier to map quasi-ids -> secret bit on synthetic data
        cols_X = list(synth_df.columns)
        cols_X.remove(secret_bit)
        X_train = synth_df[cols_X].to_numpy()
        y_train = synth_df[secret_bit].to_numpy()

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # run inference on target user to get secret bit
        X_test = df[cols_X].to_numpy()
        est_secret_bits = clf.predict(X_test)
        sol = clf.predict_proba(X_test)
        feasible = True
    elif attack_name == 'dcr':
        # Adv_dcr
        synth_df = pd.read_csv(f'{synth_df_dir}/synth_df.csv.gz', compression='gzip')

        # DCR only predicts for a single target user but fill secret bits with 0s for ease of analysis in
        # analyze_privacy_utility.py
        est_secret_bits = np.zeros(len(df))
        sol = np.zeros(len(df))
        feasible = True

        # quasi-ids + secret bit 0
        user_df_0 = df.iloc[[u]].copy(deep = True)
        user_df_0.iloc[0, user_df_0.columns.get_loc(secret_bit)] = 0

        # quasi-ids + secret bit 1
        user_df_1 = df.iloc[[u]].copy(deep = True)
        user_df_1.iloc[0, user_df_1.columns.get_loc(secret_bit)] = 1

        user_0 = user_df_0.to_numpy()[0]
        user_1 = user_df_1.to_numpy()[0]
        synth = synth_df.to_numpy()

        min_dist_to_0 = np.min(np.sqrt(np.sum(np.square(user_0 - synth), axis=1)))
        min_dist_to_1 = np.min(np.sqrt(np.sum(np.square(user_1 - synth), axis=1)))

        if min_dist_to_0 + min_dist_to_1 == 0:
            # special case partial record with secret bit 0 and partial record with secret bit 1 present
            
            # count number of records with partial record and secret bit 0
            intersection = pd.merge(user_df_0, synth_df, how='inner', on=list(df.columns))
            n_0 = len(intersection)

            # count number of records with partial record and secret bit 1
            intersection = pd.merge(user_df_1, synth_df, how='inner', on=list(df.columns))
            n_1 = len(intersection)

            est_secret_bits[u] = 0 if n_0 > n_1 else 1
            sol[u] = est_secret_bits[u]
        else:
            est_secret_bits[u] = 0 if min_dist_to_0 < min_dist_to_1 else 1
            sol[u] = min_dist_to_0 / (min_dist_to_1 + min_dist_to_0)
    else: 
        # Adv_recon (our attack)

        # split real dataset into attribute matrix
        attrs, _ = process_data(df, secret_bit)

        # load queries
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)

        # load noisy answers
        synth_result = np.load(f'{synth_result_dir}/synth_result.npz')['arr_0']
        
        # only use queries that have target value 1
        target_vals = np.array([target_val for (_, _, target_val) in queries])
        queries = [(attr_inds, attr_vals, target_val) for (attr_inds, attr_vals, target_val) in queries if target_val == 1]
        synth_result = synth_result[np.nonzero(target_vals)[0]]

        # clip queries to n_queries
        if n_queries > 0:
            queries = queries[:n_queries]
            synth_result = synth_result[:n_queries]

        # get query matrix for queries
        A = simple_kway(queries, attrs)

        # minimize L1 error
        est_secret_bits, sol, feasible = query_attack(A, synth_result, 1)
            
    # save results
    np.savez_compressed(f'{synth_result_dir}/est_secret_bits_{n_queries}_{attack_name}.npz', est_secret_bits)
    np.savez_compressed(f'{synth_result_dir}/sol_{n_queries}_{attack_name}.npz', sol)

    success = 100 if df.iloc[u, df.columns.get_loc(secret_bit)] == est_secret_bits[u] else 0
    error = 0 if feasible else 100

    return success, error

def worker(proc, function, args, rep_queue, results, errors, completed_queue):
    # set processors to use
    p = psutil.Process()
    p.cpu_affinity([proc])

    while True:
        sleep(4 * proc / 32) # prevent process dead-locking
        next_rep = rep_queue.get() 
        if next_rep is None:
            break
        results[next_rep], errors[next_rep], = function(next_rep, *args)
        completed_queue.put(None)

def track_progress_fn(completed_queue, total):
    with tqdm(total=total, leave=False) as pbar:
        curr_num = 0
        while True:
            completed_queue.get()
            pbar.update(1)
            curr_num += 1

            if curr_num == total:
                break

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--synth_model', default='BayNet_3parents', type=str, help='synthetic model to fit (BayNet_Xparents, RAP_Xiters, RAP_Xiters_NN, CTGAN, NonPrivate, Real, GaussianCopula, TVAE, CopulaGAN)')
@click.option('--n_rows', type=int, default=1000, help='number of rows of synthetic data')
@click.option('--k', type=int, default=3, help='k-way marginals')
@click.option('--scale_type', default='cond', help='scale to adjust synthetic result by. normal => size of synthetic dataset. cond => number of users selected by quasi-ids', type=click.Choice(['normal', 'cond'], case_sensitive=False))
@click.option('--n_queries', type=int, default=-1, help='number of queries to use to run attack (-1 uses all queries)')
@click.option('--attack_name', default='recon', help='attack to run', type=click.Choice(['recon', 'infer', 'dcr']))
@click.option('--secret_bit', type=str, default=None, help='secret bit to reconstruct')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start running attack from')
@click.option('--reps', type=int, default=100, help='number of repetitions to attack')
@click.option('--n_procs', type=int, default=1, help='number of processes to use to run the attack')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
def run_attack(data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit,
    start_rep_idx, reps, n_procs, data_dir):
    data_dir = f'{data_dir}/{data_name}/reps'
    secret_bit = secret_bit if secret_bit is not None else get_default_secret_bit(data_name)
    accs = Array('i', range(start_rep_idx + reps))
    errors = Array('i', range(start_rep_idx + reps))
    rep_queue = Queue()
    for i in range(reps):
        rep_queue.put(start_rep_idx + i)
    for i in range(n_procs):
        rep_queue.put(None) # signal end of reps to each processor
    completed_queue = Queue()
    track_progress_thread = Thread(target=track_progress_fn, args=(completed_queue,reps,), daemon=True)
    track_progress_thread.start()

    processes = []
    for proc in range(n_procs):
        p = Process(target=worker, args=(proc, single_rep,
            (synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, data_dir),
            rep_queue, accs, errors, completed_queue))
        p.start()
        processes.append(p)
    
    track_progress_thread.join()

    for p in processes:
        p.kill()
    
    acc = 0
    error = 0
    for i in range(start_rep_idx, start_rep_idx + reps):
        acc += accs[i] / reps
        error += errors[i] / reps
    # print(f'Accuracy: {acc}%\tFeasible: {100 - error}%')

if __name__ == '__main__':
    run_attack()
