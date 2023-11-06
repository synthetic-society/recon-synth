"""
Combine 2, 3 and 4-way queries for extra experiments
For each repetition of privacy game (rep):
    - load previously generated queries, true and noisy answers for each dataset, SDG and synthetic data size
    - concatenate queries, true and noisy answers
    - save queries and true and noisy answers
"""

import pickle
import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
import click
import lzma

def combine_queries_single(data_name, rep, synth_model, synth_size, result_dir):
    rep_dir = f'{result_dir}/{data_name}/reps/rep_{rep}'
    if synth_model is None:
        # raw dataset
        query_dir = f'{rep_dir}/queries/simple/'
        query_dir_suffix = ''
        result_prefix = ''
    else:
        # synthetic dataset
        query_dir = f'{rep_dir}/{synth_model}/{synth_size}/simple/'
        query_dir_suffix = '_cond'
        result_prefix = 'synth_'

    n_user = []
    result = []

    if synth_model is None:
        queries = []
    
    for k in [2, 3, 4]:
        # load answers
        n_user.append(np.load(f'{query_dir}/{k}way{query_dir_suffix}/{result_prefix}n_user.npz')['arr_0'])
        result.append(np.load(f'{query_dir}/{k}way{query_dir_suffix}/{result_prefix}result.npz')['arr_0'])

        if synth_model is None:
            # concatenate queries as well
            with lzma.open(f'{query_dir}/{k}way/queries.pkl.xz', 'rb') as f:
                curr_queries = pickle.load(f)
            queries.extend(curr_queries)
    n_user = np.concatenate(n_user)
    result = np.concatenate(result)

    query_dir = f'{query_dir}/234way{query_dir_suffix}/'
    os.makedirs(query_dir, exist_ok=True)

    # save queries
    if synth_model is None:
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'wb') as f:
            pickle.dump(queries, f)
    np.savez_compressed(f'{query_dir}/{result_prefix}result.npz', result)
    np.savez_compressed(f'{query_dir}/{result_prefix}n_user.npz', n_user)

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--reps', type=int, default=100, help='number of repetitions')
@click.option('--n_procs', type=int, default=1, help='number of processes to use')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
def combine_queries(data_name, reps, n_procs, data_dir):
    # combine 2, 3, and 4 way queries in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
        for synth_model in [None, 'NonPrivate', 'RAP_2Kiters', 'BayNet_3parents', 'CTGAN', 'IndHist']:
            for synth_size in ['10', '100', '1K', '10K', '100K', '1M']:
                if synth_model is None and synth_size != '10':
                    continue

                futures = []
                for rep in range(reps):
                    futures.append(executor.submit(combine_queries_single, data_name, rep, synth_model, synth_size, data_dir))
                
                with tqdm(total=len(futures), leave=False, desc=f'({data_name}, {synth_model}, {synth_size}): ') as pbar:
                    for _ in concurrent.futures.as_completed(futures):
                        pbar.update(1)

if __name__ == '__main__':
    combine_queries()