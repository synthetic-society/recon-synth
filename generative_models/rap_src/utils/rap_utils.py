import pickle
import numpy as np

from .data import Dataset, Domain, get_all_workloads
from .general import get_min_dtype
from ..qm import KWayMarginalQMTorch, KWayMarginalSetQMTorch
from ..syndata import FixedGenerator, NeuralNetworkGenerator

def wrap_data(df, domain):
    # wrap data in Dataset and Domain objects
    domain = Domain(domain.keys(), domain.values())
    # for saving memory
    dtype = get_min_dtype(sum(domain.config.values()))
    df = df.astype(dtype)
    data = Dataset(df, domain)

    return data

def get_qm(data, marginal, device):
    workloads = get_all_workloads(data, marginal)
    query_manager = KWayMarginalQMTorch(data, workloads, verbose=False, device=device)
    return query_manager

def get_model(query_manager, K, device, seed):
    model_dict = {}
    model_dict['G'] = FixedGenerator(query_manager, K=K, device=device, init_seed=seed)
    model_dict['lr'] = 1e-1
    model_dict['eta_min'] = None

    return model_dict