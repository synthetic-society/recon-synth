"""
Class that encapsulates BayNet model from DataSynthesizer
https://github.com/DataResponsibly/DataSynthesizer
"""
from DataSynthesizer.DataDescriber import DataDescriber

import os
import uuid
import contextlib
import numpy as np
import json
import pandas as pd

class BayNet():
    def __init__(self, n_parents = 2, tmp_dir='tmp/', epsilon=0, curr_id=None):
        """
        Initialize BayNet/PrivBayes SDG

        Parameters
        ----------
        n_parents: int
            degree of bayesian network
        tmp_dir: str
            temporary directory to store temporary model files to
        epsilon: float
            privacy parameter, epsilon (to satisfy eps-DP)
        curr_id: str
            (random) id to uniquely identify this model
        """
        self.n_parents = n_parents
        self.tmp_dir = tmp_dir
        self.epsilon = epsilon

        if curr_id is None:
            # sample random uuid
            self.curr_id = uuid.uuid4().hex[:8]
        else:
            self.curr_id = curr_id

    def fit(self, df):
        """
        Fit SDG model to training dataset

        Parameters
        ----------
        df: pd.DataFrame
            training data
        """
        # create temporary directory to store dataset files
        os.makedirs(self.tmp_dir, exist_ok=True)

        # input dataset
        input_data = f'{self.tmp_dir}/df_{self.curr_id}.csv'

        # save dataframe
        df.to_csv(input_data, index=False)

        # location of two output files
        self.description_file = f'{self.tmp_dir}/{self.curr_id}.json'

        # An attribute is categorical if its domain size is less than this threshold.
        # set so native-country (41 unique values) is categorical
        threshold_value = 42

        # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
        degree_of_bayesian_network = self.n_parents

        # fit model to dataset
        describer = DataDescriber(category_threshold=threshold_value)
        # block print statements
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                                    epsilon=self.epsilon, 
                                                                    k=degree_of_bayesian_network)
        describer.save_dataset_description_to_file(self.description_file)

        # delete temporary file
        os.remove(input_data)

    def sample(self, num_rows, remove_desc=True, seed=None):
        """
        Sample records from trained SDG model

        Parameters
        ----------
        num_rows: int
            number of synthetic records to generate
        remove_desc: bool
            remove temporary model files necessary to generate synthetic data
        seed: int
            seed the random generator

        Returns
        -------
        df: pd.DataFrame
            synthetic dataset
        """
        if seed is None:
            # generate random seed
            seed = np.random.randint(0, 2147483647)

        with open(self.description_file) as f:
            model = json.load(f)
        
        visit_order = model['bayesian_network']
        root_attr = visit_order[0][1][0]
        cond_probs = BayNet.parse_cond_probs(model['conditional_probabilities'])

        samples = []
        for _ in range(num_rows):
            sample = dict()

            # sample root attribute from its 1-way marginal
            root_attr_dist = cond_probs[root_attr]
            sample[root_attr] = np.random.choice(len(root_attr_dist), p=root_attr_dist)
            
            for curr_attr, curr_parents in visit_order:
                # sample subsequent attributes from conditional distributions
                parent_vals = tuple([sample[parent] for parent in curr_parents])
                attr_dist = cond_probs[curr_attr][parent_vals]
                sample[curr_attr] = np.random.choice(len(attr_dist), p=attr_dist)
            
            samples.append(sample)
        
        synth_df = pd.DataFrame.from_records(samples)
        for col in synth_df.columns:
            synth_df[col] = synth_df[col].apply(lambda x: model['attribute_description'][col]['distribution_bins'][x])
        
        if remove_desc:
            # delete temporary file
            os.remove(self.description_file)

        synth_df = synth_df[model['meta']['all_attributes']]
        return synth_df

    @staticmethod
    def parse_cond_probs(cond_probs):
        '''
        Convert conditional probabilities from dict child -> dict '[parent values]' -> [probs]
        to dict child -> tuple (parent values) -> [probs]
        '''
        new_cond_probs = dict()
        for attr, cond_prob in cond_probs.items():
            if isinstance(cond_prob, list):
                new_cond_probs[attr] = cond_prob
                continue

            new_cond_prob = dict()
            for parent_vals, probs in cond_prob.items():
                new_cond_prob[tuple(eval(parent_vals))] = probs
            new_cond_probs[attr] = new_cond_prob
        
        return new_cond_probs
