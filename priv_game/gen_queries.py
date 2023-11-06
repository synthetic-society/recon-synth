"""
Step 3: Generate simple and any k-way queries to be used in utility and privacy evaluations
For each repetition of privacy game (rep):
    - load previously generated raw data
    - randomly sample a pool of queries or iterate through all possible queries
    - evaluate true answers for queries on raw data
    - save queries and true answers
"""
import click
import os
from tqdm import tqdm
import pickle
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

def single_rep(proc, rep, query_type, k, n_queries, secret_bit, data_dir, regen):
    # set processors to use
    p = psutil.Process()
    p.cpu_affinity([2 * proc, 2 * proc + 1])

    # setup dirs
    rep_dir = f'{data_dir}/rep_{rep}'
    query_dir = f'{rep_dir}/queries/{query_type}/{k}way/'
    os.makedirs(query_dir, exist_ok=True)

    # load raw dataset
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')

    if query_type == 'any':
        attrs = df.to_numpy()
        secret_bits = np.zeros(len(attrs)) # dummy vector to satisfy type
    else:
        # split real dataset into attribute matrix
        attrs, secret_bits = process_data(df, secret_bit)

    if os.path.exists(f'{query_dir}/queries.pkl.xz') and not regen:
        # load generated queries if exists
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)
    else:
        # generate k-way queries from scratch
        queries = gen_kway(attrs, n_queries, query_type, k, select_unique=False)

    # evaluate true answers to queries on raw dataset
    n_user, result = get_result(attrs, secret_bits, queries, query_type)

    # save queries and true answers
    with lzma.open(f'{query_dir}/queries.pkl.xz', 'wb') as f:
        pickle.dump(queries, f)
    np.savez_compressed(f'{query_dir}/result.npz', result)
    np.savez_compressed(f'{query_dir}/n_user.npz', n_user)

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--query_type', default='simple', help='type of queries (simple k-way marginal queries involving secret bit or any k-way marginal queries)', type=click.Choice(['simple', 'any']))
@click.option('--k', type=int, default=3, help='k-way marginals')
@click.option('--n_queries', type=int, default=-1, help='number of queries to generate (-1 generates all possible k-way queries)')
@click.option('--secret_bit', type=str, default=None, help='secret bit to reconstruct')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start generating queries from')
@click.option('--reps', type=int, default=100, help='number of repetitions')
@click.option('--n_procs', type=int, default=2, help='number of processes to use')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
@click.option("--regen", is_flag=True, show_default=True, default=False, help="Regenerate queries")
def gen_queries(data_name, query_type, k, n_queries, secret_bit, start_rep_idx, reps, n_procs, data_dir, regen):
    data_dir = f'{data_dir}/{data_name}/reps'
    secret_bit = secret_bit if secret_bit is not None else get_default_secret_bit(data_name)
    n_procs = n_procs // 2

    # generate random queries and evaluate true answers in parallel
    for big_rep in tqdm(range(start_rep_idx, start_rep_idx + reps, n_procs), desc='big_reps', leave=False):
        processes = []
        rem_batch_size = min(n_procs, start_rep_idx + reps - big_rep)
        for small_rep in range(rem_batch_size):
            rep = big_rep + small_rep

            p = Process(target=single_rep,
                args=(small_rep, rep, query_type, k, n_queries, secret_bit, data_dir, regen))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    gen_queries()