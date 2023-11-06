"""
Step 2: Generate synthetic data from raw datasets
For each repetition of privacy game (rep):
    - load previously generated raw data
    - fit synthetic data
    - save synthetic data
"""
import click
import os
from tqdm import tqdm
from sdv.tabular import GaussianCopula, CTGAN, TVAE, CopulaGAN
from multiprocessing import Process
import psutil
import json
import shutil

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *
from generative_models import BayNet, NonPrivate, IndHist, RAP

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def single_rep_sample(rep, data_dir, synth_model_name, proc, n_rows, domain, device):
    # sample synthetic dataset from fitted generative model for a single repetition of privacy game

    # set processors to use
    p = psutil.Process()
    p.cpu_affinity([proc])

    # setup dirs
    rep_dir = f'{data_dir}/rep_{rep}'
    synth_df_dir = f'{rep_dir}/{synth_model_name}/{n_rows_names[n_rows]}'
    os.makedirs(synth_df_dir, exist_ok=True)

    # load raw dataset
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')

    if os.path.exists(f'{rep_dir}/{synth_model_name}/1M/synth_df.csv.gz'):
        # larger synthetic data generated, simply sample smaller synthetic dataset from larger one
        large_synth_df = pd.read_csv(f'{rep_dir}/{synth_model_name}/1M/synth_df.csv.gz', compression='gzip')
        synth_df = large_synth_df.sample(n_rows)
    else:
        # change current working directory for parallelization of GaussianCopula
        cwd = os.getcwd()
        os.chdir(rep_dir) 

        # choose synthetic data model
        if synth_model_name == 'GaussianCopula':
            synth_model = GaussianCopula()
        elif synth_model_name == 'CTGAN':
            synth_model = CTGAN()
        elif synth_model_name == 'CopulaGAN':
            synth_model = CopulaGAN()
        elif synth_model_name == 'TVAE':
            synth_model = TVAE()
        elif 'BayNet' in synth_model_name:
            n_parents = int(synth_model_name.split('_')[1].split('parents')[0])
            synth_model = BayNet(n_parents=n_parents, curr_id='model', tmp_dir=synth_model_name)
        elif 'PrivBayes' in synth_model_name:
            n_parents = int(synth_model_name.split('_')[1].split('parents')[0])
            eps = float(synth_model_name.split('_')[2].split('eps')[0])
            synth_model = BayNet(n_parents=n_parents, curr_id='model', epsilon=eps, tmp_dir=synth_model_name)
        elif 'RAP' in synth_model_name:
            n_iters = int(synth_model_name.split('_')[1].split('Kiters')[0]) * 1000
            eps = 0
            if 'eps' in synth_model_name:
                eps = float(synth_model_name.split('_')[2].split('eps')[0])
            synth_model = RAP(T=n_iters, epsilon=eps, domain=domain, device=device)

            # fit model on the dataset
            synth_model.fit(df)

            # remove `save` folder
            shutil.rmtree('save')

            # save fitted model
            synth_model.save(f'{synth_model_name}/model.pkl')
        elif synth_model_name == 'IndHist':
            synth_model = IndHist(curr_id='model', tmp_dir=synth_model_name)
        elif synth_model_name == 'NonPrivate':
            synth_model = NonPrivate(replace=True)
            synth_model.fit(df)
        elif synth_model_name == 'Real':
            synth_model = NonPrivate(replace=False)
            synth_model.fit(df)
        else:
            raise Exception(f'Fitting function for model {synth_model_name} not implemented') 
        
        # load fitted model
        if 'Bay' in synth_model_name or 'IndHist' in synth_model_name:
            synth_model.description_file = f'{synth_model_name}/{synth_model.curr_id}.json' 
        elif synth_model_name != 'NonPrivate' and synth_model_name != 'Real' and 'RAP' not in synth_model_name:
            synth_model = synth_model.load(f'{synth_model_name}/model.pkl')

        # sample synthetic data from model
        synth_dfs = []
        batch_size = 10000
        for i in tqdm(range(0, n_rows, batch_size), leave=False, desc='synth rows'):
            rem_rows = min(batch_size, n_rows - i)
            if 'Bay' in synth_model_name or 'IndHist' in synth_model_name:
                synth_dfs.append(synth_model.sample(num_rows=rem_rows, remove_desc=False))
            else:
                synth_dfs.append(synth_model.sample(num_rows=rem_rows))
        synth_df = pd.concat(synth_dfs)

        # reset current working directory
        os.chdir(cwd)
    
    # save synthetic dataset
    synth_df.to_csv(f'{synth_df_dir}/synth_df.csv.gz', index=False, compression='gzip')

def single_rep_fit(rep, data_dir, synth_model_name, proc_start, proc_end):
    # fit generative model for a single repetition of privacy game

    # set processors to use
    p = psutil.Process()
    p.cpu_affinity(list(range(proc_start, proc_end)))

    # setup dirs
    rep_dir = f'{data_dir}/rep_{rep}'

    if os.path.exists(f'{rep_dir}/{synth_model_name}/1M/synth_df.csv.gz'):
        # nothing to fit, return
        return

    # load dataset
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')

    # change current working directory for parallelization of GaussianCopula
    cwd = os.getcwd()
    os.chdir(rep_dir) 

    os.makedirs(synth_model_name, exist_ok=True)

    # choose synthetic data model
    if synth_model_name == 'GaussianCopula':
        synth_model = GaussianCopula()
    elif synth_model_name == 'CTGAN':
        synth_model = CTGAN()
    elif synth_model_name == 'CopulaGAN':
        synth_model = CopulaGAN()
    elif synth_model_name == 'TVAE':
        synth_model = TVAE()
    elif 'BayNet' in synth_model_name:
        n_parents = int(synth_model_name.split('_')[1].split('parents')[0])
        synth_model = BayNet(n_parents=n_parents, curr_id='model', tmp_dir=synth_model_name)
    elif 'PrivBayes' in synth_model_name:
        n_parents = int(synth_model_name.split('_')[1].split('parents')[0])
        eps = float(synth_model_name.split('_')[2].split('eps')[0])
        synth_model = BayNet(n_parents=n_parents, curr_id='model', epsilon=eps, tmp_dir=synth_model_name)
    elif synth_model_name == 'IndHist':
        synth_model = IndHist(curr_id='model', tmp_dir=synth_model_name)
    else:
        # easier to fit & generate synthetic data for RAP together, hence skip
        # reset current working directory
        os.chdir(cwd)

        # nothing to fit, return
        return

    # fit model on the dataset
    synth_model.fit(df)

    # save fitted model
    if 'Bay' not in synth_model_name and 'IndHist' not in synth_model_name:
        synth_model.save(f'{synth_model_name}/model.pkl')

    # reset current working directory
    os.chdir(cwd)

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--synth_model', default='BayNet_3parents', type=str, help='synthetic model to fit (BayNet_Xparents, PrivBayes_Xparents_Yeps, RAP_Xiters, RAP_Xiters_NN, CTGAN, TVAE, NonPrivate, Real, GaussianCopula, CopulaGAN)')
@click.option('--n_rows', type=int, default=1000, help='number of rows of synthetic data')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start generating synthetic data from')
@click.option('--reps', type=int, default=100, help='number of repetitions')
@click.option('--n_procs', type=int, default=1, help='total number of processes to use')
@click.option('--use_gpus', is_flag=True, show_default=True, default=False, help='Use GPUs for fitting (only for RAP --- ensure n_procs == # GPUs)')
@click.option('--single_procs', type=int, default=1, help='number of processes to use to fit a single synthetic dataset')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
def gen_synth_data(data_name, synth_model, n_rows, start_rep_idx, reps, n_procs, use_gpus, single_procs, data_dir):
    data_dir = f'{data_dir}/{data_name}/reps'
    batch_size = n_procs // single_procs

    with open(f'../datasets/domain/{data_name}-domain.json', 'r') as f:
        domain = json.load(f)

    # fit generative models in parallel
    for big_rep in tqdm(range(start_rep_idx, start_rep_idx + reps, n_procs), desc='big_reps (fit)', leave=False):
        processes = []
        rem_batch_size = min(batch_size, start_rep_idx + reps - big_rep)
        for small_rep in range(rem_batch_size):
            rep = big_rep + small_rep
            start = small_rep * single_procs
            end = (small_rep + 1) * single_procs

            p = Process(target=single_rep_fit,
                args=(rep,data_dir,synth_model,start,end))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
    # generate synthetic data from fitted generative models in parallel
    for big_rep in tqdm(range(start_rep_idx, start_rep_idx + reps, n_procs), desc='big_reps (gen)', leave=False):
        processes = []
        rem_batch_size = min(n_procs, start_rep_idx + reps - big_rep)
        gpu_id = 0
        for small_rep in range(rem_batch_size):
            rep = big_rep + small_rep

            device = f'cuda:{gpu_id}' if use_gpus else 'cpu'
            p = Process(target=single_rep_sample,
                args=(rep,data_dir,synth_model,small_rep,n_rows,domain,device))
            p.start()
            processes.append(p)

            gpu_id += 1

        for p in processes:
            p.join()


if __name__ == '__main__':
    gen_synth_data()
