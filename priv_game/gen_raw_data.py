"""
Step 1: Generate raw data for privacy game
For each repetition of privacy game (rep):
    - sample random raw dataset (D)
    - choose random target user (u)
    - flip that target user's bit randomly (D')
    - save raw data and index of user in raw data (D' and u)
"""
import click
import os
import concurrent.futures
from tqdm import tqdm
import numpy as np

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def single_rep(rep, data_name, secret_bit, n_users, data_dir, seed, randomize, unique_quasi, balance):
    # prepare raw dataset for a single repetition of privacy game

    # setup dirs
    np.random.seed(seed)
    rep_dir = f'{data_dir}/rep_{rep}'
    os.makedirs(rep_dir, exist_ok=True)

    while True:
        # sample raw dataset
        df = load_data(data_name, n_users, secret_bit, randomize=randomize, unique_quasi=unique_quasi,
            balance=balance)
        df.to_csv(f'{rep_dir}/orig_df.csv.gz', index=False, compression='gzip')

        # re-index dataframe
        df['index'] = np.arange(len(df))

        # filter records with unique quasi id to choose target user
        subset = list(df.columns)
        subset.remove(secret_bit)
        subset.remove('index')
        unique_records = df.drop_duplicates(subset=subset, keep=False)

        if len(unique_records) > 0:
            # target user w/ unique quasi id can be selected
            break

        # no target user can be sampled, resample raw dataset

    # choose random user with unique quasi id
    u = unique_records.iloc[np.random.randint(len(unique_records)), unique_records.columns.get_loc('index')]
    df = df.drop(['index'], axis=1)

    # flip bit of user randomly
    new_bit = np.random.randint(2)
    df.iloc[u, df.columns.get_loc(secret_bit)] = new_bit

    # save raw data
    df.to_csv(f'{rep_dir}/df.csv.gz', compression='gzip', index=False)
    np.savetxt(f'{rep_dir}/user.csv', np.array([u]), fmt='%d')

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--secret_bit', type=str, default=None, help='secret bit to reconstruct')
@click.option('--n_users', type=int, default=1000, help='trim dataset to n samples')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start generating raw dataset from')
@click.option('--reps', type=int, default=100, help='number of repetitions')
@click.option('--n_procs', type=int, default=1, help='number of processes to use')
@click.option('--data_dir', type=str, default='results/', help='directory to save generated data to')
@click.option('--randomize', is_flag=True, help='randomize selection of raw datasets')
@click.option('--unique_quasi', is_flag=True, help='only use records with unique quasi-ids')
@click.option('--balance', is_flag=True, help='balance target dataset around secret bit')
def gen_raw_data(data_name, secret_bit, n_users, start_rep_idx, reps, n_procs, data_dir, randomize, unique_quasi,
    balance):
    data_dir = f'{data_dir}/{data_name}/reps'
    secret_bit = secret_bit if secret_bit is not None else get_default_secret_bit(data_name)

    # parallelize reps across n_procs
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor, \
        tqdm(total=reps, desc='reps') as pbar:
        seeds = np.random.randint(0, 2147483647, size=reps)
        futures = [executor.submit(single_rep, start_rep_idx + rep, data_name, secret_bit, n_users, 
            data_dir, seeds[rep], randomize, unique_quasi, balance) for rep in range(reps)]
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)

if __name__ == '__main__':
    gen_raw_data()