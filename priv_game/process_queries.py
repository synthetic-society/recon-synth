"""
Step 4: Evaluate noisy answers to generated queries from synthetic data
For each repetition of privacy game (rep):
    - load previously generated unique queries
    - load previously generated raw and synthetic data
    - save queries and their results
"""
import click
from tqdm import tqdm
import pickle
import os
from multiprocessing import Process
import psutil
import lzma

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *
from attacks import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def process_batch(proc_start, proc_end, rep_start, rep_end, synth_model_name, n_rows, query_type, k,
    scale_type, secret_bit, data_dir):
    # process batch of reps at one go

    # set processors to use
    p = psutil.Process()
    p.cpu_affinity(list(range(proc_start, proc_end)))

    for rep in tqdm(range(rep_start, rep_end), leave=False, desc='reps_batch'):
        # setup dirs
        rep_dir = f'{data_dir}/rep_{rep}'
        query_subdir = f'{query_type}/{k}way'
        query_dir = f'{rep_dir}/queries/{query_subdir}/'
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)
        n_user = np.load(f'{query_dir}/n_user.npz')['arr_0']
        synth_df_dir = f'{rep_dir}/{synth_model_name}/{n_rows_names[n_rows]}'

        # load raw dataset
        df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')

        if query_type == 'any':
            attrs = df.to_numpy()
        else:
            # split real dataset into attribute matrix
            attrs, _ = process_data(df, secret_bit)

        synth_df = pd.read_csv(f'{synth_df_dir}/synth_df.csv.gz', compression='gzip')

        if query_type == 'any':
            synth_attrs = synth_df.to_numpy()
            synth_secret_bits = np.zeros(len(synth_attrs)) # dummy vector to satisfy type
        else:
            # split synthetic dataset into attribute matrix
            synth_attrs, synth_secret_bits = process_data(synth_df, secret_bit)
        
        # extract queries for rep in batches
        synth_n_user, synth_result = get_result(synth_attrs, synth_secret_bits, queries, query_type)

        if scale_type == 'normal':
            scale = len(attrs) / len(synth_attrs)
        elif scale_type == 'cond':
            # optionally scale results wrt quasi-ids from raw dataset
            scale = np.where(synth_n_user > 0, n_user / synth_n_user, len(attrs) / len(synth_attrs))
        synth_result *= scale

        # save query results
        synth_result_dir = f'{synth_df_dir}/{query_subdir}_{scale_type}/'
        os.makedirs(synth_result_dir, exist_ok=True)
        np.savez_compressed(f'{synth_result_dir}/synth_n_user.npz', synth_n_user)
        np.savez_compressed(f'{synth_result_dir}/synth_result.npz', synth_result)

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--synth_model', default='BayNet_3parents', type=str, help='synthetic model to fit (BayNet_Xparents, RAP_Xiters, RAP_Xiters_NN, CTGAN, NonPrivate, Real, GaussianCopula, TVAE, CopulaGAN)')
@click.option('--n_rows', type=int, default=1000, help='number of rows of synthetic data')
@click.option('--query_type', default='simple', help='type of queries (simple or range based k-way marginal queries involving secret bit or any k-way marginal queries)', type=click.Choice(['simple', 'range', 'any'], case_sensitive=False))
@click.option('--k', type=int, default=3, help='k-way marginals')
@click.option('--scale_type', default='cond', help='scale to adjust synthetic result by. normal => size of synthetic dataset. cond => number of users selected by attributes', type=click.Choice(['normal', 'cond'], case_sensitive=False))
@click.option('--secret_bit', type=str, default=None, help='secret bit to reconstruct')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start processing from')
@click.option('--reps', type=int, default=100, help='number of repetitions')
@click.option('--n_procs', type=int, default=1, help='number of processors to use')
@click.option('--single_procs', type=int, default=1, help='number of processors to use to process queries for a single synthetic dataset')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
def process_queries(data_name, synth_model, n_rows, query_type, k, scale_type, secret_bit, start_rep_idx, reps, n_procs, single_procs, data_dir):
    data_dir = f'{data_dir}/{data_name}/reps'
    secret_bit = secret_bit if secret_bit is not None else get_default_secret_bit(data_name)
    num_processes = n_procs // single_procs
    batch_size = reps // num_processes

    # evaluate noisy answers to generated queries on synthetic data in parallel
    with tqdm(total=num_processes, desc='procs', leave=False) as pbar:
        processes = []
        for rep_start in range(start_rep_idx, start_rep_idx+reps, batch_size):
            rep_end = min(rep_start + batch_size, start_rep_idx + reps)

            proc_start = len(processes) * single_procs
            proc_end = (len(processes) + 1) * single_procs
            p = Process(target=process_batch, args=(proc_start, proc_end, rep_start, rep_end,
                synth_model, n_rows, query_type, k, scale_type, secret_bit, data_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            pbar.update(1)

if __name__ == '__main__':
    process_queries()