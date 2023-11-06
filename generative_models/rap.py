"""
Class that encapsulates the RAP model from dp-query-release repo
https://github.com/terranceliu/dp-query-release
"""

import torch

from .rap_src.algo.nondp import IterativeAlgoNonDP
from .rap_src.algo.rap_softmax import IterAlgoRAPSoftmax
from .rap_src.utils.general import get_data_onehot
from .rap_src.utils import get_per_round_budget_zCDP
from .rap_src.utils.rap_utils import wrap_data, get_qm, get_model

class RAP():
    def __init__(self, T=2000, epsilon=0, delta=1e-6, domain=None, device='cpu'):
        """
        Initialize RAP SDG

        Parameters
        ----------
        T: int
            number of iterations to train for
        epsilon: float
            privacy parameter, epsilon (to satisfy (eps, delta)-DP)
        delta: float
            privacy parameter, delta (to satisfy (eps, delta)-DP)
        domain: dict[str, int]
            domain of data, col -> number of unique values
        device: str
            device to train model on
        """
        self.epsilon = epsilon
        self.delta = delta
        self.domain = domain
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # hyperparameters for RAP (most are default)
        self.marginal = 3
        self.K = 1000
        self.T = T
        self.max_idxs = 1024
        self.samples_per_round = 2
        self.alpha = 0.5
        self.seed = 0

    def fit(self, df):
        """
        Fit SDG model to training dataset

        Parameters
        ----------
        df: pd.DataFrame
            training data
        """
        data = wrap_data(df, self.domain) 
        query_manager = get_qm(data, self.marginal, self.device)
        model_dict = get_model(query_manager, self.K, self.device, self.seed)
        self.model = model_dict['G']

        if self.epsilon == 0:
            # no-DP
            algo = IterativeAlgoNonDP(model_dict['G'], self.T, seed=self.seed, lr=model_dict['lr'],
                eta_min=model_dict['eta_min'], max_idxs=self.max_idxs, max_iters=1, sample_by_error=True, log_freq=1,
                verbose=False)
        else:
            # (eps, delta)-DP
            eps0, rho = get_per_round_budget_zCDP(self.epsilon, self.delta, self.T * self.samples_per_round, alpha=self.alpha)
            algo = IterAlgoRAPSoftmax(model_dict['G'], self.T, eps0, seed=self.seed,
                samples_per_round=self.samples_per_round, lr=model_dict['lr'], max_iters=1, max_idxs=self.max_idxs,
                verbose=False)

        true_answers = query_manager.get_answers(data).to(self.device)
        algo.fit(true_answers)
    
    def save(self, filename):
        """
        Save SDG model for reuse

        Parameters
        ----------
        filename: str
            path to save model to
        """
        torch.save(self.model.generator.state_dict(), filename)
    
    def load(self, filename):
        """
        Load SDG model from save state

        Parameters
        ----------
        filename: str
            path to load model from
        """
        self.model.generator.load_state_dict(torch.load(filename))
    
    def sample(self, num_rows):
        """
        Sample records from trained SDG model

        Parameters
        ----------
        num_rows: int
            number of synthetic records to generate

        Returns
        -------
        df: pd.DataFrame
            synthetic dataset
        """
        return self.model.get_syndata(num_samples=num_rows).df
