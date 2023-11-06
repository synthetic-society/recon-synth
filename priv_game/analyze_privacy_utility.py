"""
Analyze privacy and utility of synthetic data after running full pipeline of attacks
and output dataframes containing summarized results
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import click
import pickle
import os
import concurrent.futures
from sklearn.metrics import roc_auc_score
import math
import lzma

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def df_utility_exists(df, data_name, synth_model, n_rows):
    return df is not None and len(df[
        (df['data_name'] == data_name) & 
        (df['synth_model'] == synth_model) &
        (df['synth_size'] == n_rows)
    ]) > 0

def get_query_utility(data_name, synth_model, n_rows, k, start_rep_idx, reps, data_dir, df):
    # calculate utility of synthetic data (3-TVD and MRE_{>10})
    if df_utility_exists(df, data_name, synth_model, n_rows):
        return None, {}

    data_dir = f'{data_dir}/{data_name}/reps'
    
    full_results = dict()
    full_results['avg_tvd'] = []
    full_results['rel_mean'] = []

    for i in tqdm(range(start_rep_idx, start_rep_idx + reps), desc='TVD', leave=False):
        # setup dirs
        rep_dir = f'{data_dir}/rep_{i}/'
        df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')
        query_subdir = f'any/{k}way'
        query_dir = f'{rep_dir}/queries/{query_subdir}/'
        synth_df_dir = f'{rep_dir}/{synth_model}/{n_rows_names[n_rows]}/'
        synth_result_dir = f'{synth_df_dir}/{query_subdir}_normal/'

        if not os.path.exists(f'{synth_result_dir}/synth_result.npz'):
            return None, {}

        # load queries and true and noisy answers
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)
        result = np.load(f'{query_dir}/result.npz')['arr_0']
        synth_result = np.load(f'{synth_result_dir}/synth_result.npz')['arr_0']

        # calculate MRE_{>10}
        curr_rel_error = np.abs((result - synth_result) / result)[result > 10]
        full_results['rel_mean'].append(curr_rel_error.mean())
        
        # calculate 3-TVD
        marginal_inds = dict()
        for i, (attr_comb, _) in enumerate(queries):
            if attr_comb in marginal_inds:
                marginal_inds[attr_comb].append(i)
            else:
                marginal_inds[attr_comb] = [i]
        
        tvds = []
        for query_inds in marginal_inds.values():
            # calculate probability distributions of marginal
            curr_result = result[query_inds] 
            if curr_result.sum() != 0:
                curr_result /= curr_result.sum()

            curr_synth_result = synth_result[query_inds]
            if curr_synth_result.sum() != 0:
                curr_synth_result /= curr_synth_result.sum()

            curr_tvd = 0.5 * np.sum(np.abs(curr_result - curr_synth_result))
            tvds.append(curr_tvd)
    
        full_results['avg_tvd'].append(np.mean(tvds))

    avg_results = {
        'data_name': data_name,
        'synth_model': synth_model,
        'synth_size': n_rows,
    }

    # calculate mean and standard deviation across reps
    for key, vals in full_results.items():
        avg_results[key] = np.mean(vals)
        avg_results[f'{key}_std'] = np.std(vals)
    
    return 'utility', avg_results

def get_attack_score_single(i, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, data_dir):
    # get expected secret bit, estimated secret bit and score for attack on single repetition of privacy game
    rep_dir = f'{data_dir}/rep_{i}/'
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')
    user = int(np.genfromtxt(f'{rep_dir}/user.csv'))

    query_subdir = f'simple/{k}way_{scale_type}'
    synth_df_dir = f'{rep_dir}/{synth_model}/{n_rows_names[n_rows]}/'
    synth_result_dir = f'{synth_df_dir}/{query_subdir}/'

    synth_result_path = f'{synth_result_dir}/est_secret_bits_{n_queries}_{attack_name}.npz'
    synth_score_path = f'{synth_result_dir}/sol_{n_queries}_{attack_name}.npz'

    if not os.path.exists(synth_result_path):
        raise Exception('Attack not run yet')

    exp_secret_bit = df.iloc[user, df.columns.get_loc(secret_bit)]
    est_secret_bits = np.load(synth_result_path)['arr_0']
    sol = np.load(synth_score_path)['arr_0']
    if len(sol.shape) == 2:
        sol = sol[:, 0]
    est_secret_bit = est_secret_bits[user]

    return exp_secret_bit, est_secret_bit, sol[user]

def df_privacy_exists(df, data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name):
    return df is not None and len(df[
        (df['data_name'] == data_name) & 
        (df['synth_model'] == synth_model) &
        (df['synth_size'] == n_rows) &
        (df['k'] == k) &
        (df['scale_type'] == scale_type) &
        (df['n_queries'] == n_queries) &
        (df['attack_name'] == attack_name)
    ]) > 0

def get_attack_acc(data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, start_rep_idx, reps, data_dir, df):
    if df_privacy_exists(df, data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name):
        return None, {}

    data_dir = f'{data_dir}/{data_name}/reps'

    successes = []
    y_true = []
    y_score = []
    try:
        for i in tqdm(range(start_rep_idx, start_rep_idx + reps), desc=attack_name, leave=False):
            exp_secret_bit, est_secret_bit, score = get_attack_score_single(i, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, data_dir) 

            if exp_secret_bit == est_secret_bit:
                successes.append(1)
            else:
                successes.append(0)
        
            y_true.append(exp_secret_bit)
            y_score.append(score)

        successes = np.array(successes)
        auroc = roc_auc_score(y_true, y_score)

        return 'privacy', {
            'data_name': data_name,
            'synth_model': synth_model,
            'synth_size': n_rows,
            'k': k,
            'scale_type': scale_type,
            'n_queries': n_queries,
            'attack_name': attack_name,
            'acc': successes.mean(),
            'auroc': auroc
        }
    except Exception:
        return None, {}

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--synth_model', default='BayNet_3parents', type=str, help='synthetic model to fit (BayNet_Xparents, RAP_Xiters, RAP_Xiters_NN, CTGAN, NonPrivate, Real, GaussianCopula, TVAE, CopulaGAN)')
@click.option('--n_rows', type=int, default=1000, help='number of rows of synthetic data')
@click.option('--k', type=int, default=3, help='k-way marginals')
@click.option('--scale_type', default='normal', help='scale to adjust synthetic result by. normal => size of synthetic dataset. cond => number of users selected by attributes', type=click.Choice(['normal', 'cond'], case_sensitive=False))
@click.option('--n_queries', type=int, default=-1, help='number of queries to use to run attack (-1 uses all queries)')
@click.option('--secret_bit', type=str, default=None, help='secret bit to reconstruct')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start running attack from')
@click.option('--reps', type=int, default=100, help='number of repetitions to attack')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
@click.option('--all', is_flag=True, show_default=True, default=False, help="Generate all stats")
@click.option('--n_procs', type=int, default=1, help='number of processes to use')
def analyze_attack(data_name, synth_model, n_rows, k, scale_type, n_queries, secret_bit, start_rep_idx, reps, data_dir, all, n_procs):
    if all:
        # get results for all datasets, SDG, synthetic data size, and attacks
        records_privacy = []
        records_utility = []

        if os.path.exists(f'{data_dir}/results_privacy.csv'):
            df_privacy = pd.read_csv(f'{data_dir}/results_privacy.csv')
        else:
            df_privacy = None

        if os.path.exists(f'{data_dir}/results_utility.csv'):
            df_utility = pd.read_csv(f'{data_dir}/results_utility.csv')
        else:
            df_utility = None

        # traverse all datasets, SDGs, synthetic data sizes, and attacks
        data_names = os.listdir(data_dir)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
            futures = []
            # traverse all datasets
            for data_name in data_names:
                if data_name not in ['acs', 'fire']:
                    continue

                secret_bit = get_default_secret_bit(data_name)
                reps_dir = f'{data_dir}/{data_name}/reps'
                rep_dir = f'{reps_dir}/rep_{reps - 1}'
                # traverse all SDGs
                for synth_model in os.listdir(rep_dir):
                    if synth_model == 'queries' or synth_model == 'tmp' or os.path.isfile(f'{rep_dir}/{synth_model}'):
                        continue

                    n_rows = {v: k for k, v in n_rows_names.items()}

                    # traverse all synthetic data sizes
                    for n_row_name in os.listdir(f'{rep_dir}/{synth_model}/'):
                        if n_row_name.startswith('model'):
                            continue

                        # traverse all recon attacks using different k-way queries
                        for k in [2, 3, 4, 234]:
                            for scale_type in ['cond', 'normal']:
                                synth_result_dir = f'{rep_dir}/{synth_model}/{n_row_name}/simple/{k}way_{scale_type}'
                                if not os.path.exists(synth_result_dir):
                                    continue

                                n_queriess = [int(filename.split('_')[1]) for filename in os.listdir(synth_result_dir) if filename.startswith('sol_') and filename.endswith('_recon.npz')]
                                for n_queries in n_queriess:
                                    futures.append(executor.submit(get_attack_acc, data_name, synth_model, n_rows[n_row_name], k, scale_type, n_queries, 'recon', secret_bit, start_rep_idx, start_rep_idx + reps, data_dir, df_privacy))

                        # traverse dcr attack
                        futures.append(executor.submit(get_attack_acc, data_name, synth_model, n_rows[n_row_name], 3, 'cond', -1, 'dcr', secret_bit, start_rep_idx, start_rep_idx + reps, data_dir, df_privacy))

                        # traverse infer attack
                        futures.append(executor.submit(get_attack_acc, data_name, synth_model, n_rows[n_row_name], 3, 'cond', -1, 'infer', secret_bit, start_rep_idx, start_rep_idx + reps, data_dir, df_privacy))

                        # get utility for current dataset, SDG, and synthetic data size
                        futures.append(executor.submit(get_query_utility, data_name, synth_model, n_rows[n_row_name], 3, start_rep_idx, start_rep_idx + reps, data_dir, df_utility))
                        
            for future in concurrent.futures.as_completed(futures):
                df_name, record = future.result()

                if df_name == 'privacy':
                    records_privacy.append(record)
                elif df_name == 'utility':
                    records_utility.append(record)
        
        new_results_privacy = pd.DataFrame.from_records(records_privacy)
        if df_privacy is None:
            df_privacy = new_results_privacy
        else:
            df_privacy = pd.concat([df_privacy, new_results_privacy])
        df_privacy = df_privacy.sort_values(by=['data_name', 'synth_model', 'synth_size', 'k', 'scale_type', 'attack_name', 'n_queries'])
        df_privacy.to_csv(f'{data_dir}/results_privacy.csv', index=False)

        new_results_utility = pd.DataFrame.from_records(records_utility)
        if df_utility is None:
            df_utility = new_results_utility
        else:
            df_utility = pd.concat([df_utility, new_results_utility])
        df_utility = df_utility.sort_values(by=['data_name', 'synth_model', 'synth_size'])
        df_utility.to_csv(f'{data_dir}/results_utility.csv', index=False)
    else:
        # get results for a single dataset, SDG, and synthetic data size
        secret_bit = secret_bit if secret_bit is not None else get_default_secret_bit(data_name)
        _, record_recon = get_attack_acc(data_name, synth_model, n_rows, k, scale_type, n_queries, 'recon', secret_bit, start_rep_idx, reps, data_dir, None)
        _, record_infer = get_attack_acc(data_name, synth_model, n_rows, 3, 'cond', -1, 'infer', secret_bit, start_rep_idx, reps, data_dir, None)
        _, record_dcr = get_attack_acc(data_name, synth_model, n_rows, 3, 'cond', -1, 'dcr', secret_bit, start_rep_idx, reps, data_dir, None)
        _, record_utility = get_query_utility(data_name, synth_model, n_rows, 3, start_rep_idx, start_rep_idx + reps, data_dir, None)

        print('Privacy (Adv_recon)')
        print('-------')
        print(f'Attack Accuracy: {record_recon["acc"] * 100}%')
        print(f'AUROC: {record_recon["auroc"] * 100}%')
        print()

        print('Privacy (Adv_dcr)')
        print('-------')
        print(f'(Overall) Attack Accuracy: {record_dcr["acc"] * 100}%')
        print(f'AUROC: {record_dcr["auroc"] * 100}%')
        print()

        print('Privacy (Adv_infer)')
        print('-------')
        print(f'(Overall) Attack Accuracy: {record_infer["acc"] * 100}%')
        print(f'AUROC: {record_infer["auroc"] * 100}%')
        print()

        print('Utility')
        print('-------')
        print(f'Average 3-TVD: {record_utility["avg_tvd"]}')
        print(f'Mean Relative Error MRE_{{_10}}: {record_utility["rel_mean"]}')

if __name__ == '__main__':
    analyze_attack()