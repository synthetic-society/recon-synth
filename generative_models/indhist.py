"""
Class that encapsulates IndHist model from DataSynthesizer
https://github.com/DataResponsibly/DataSynthesizer
"""
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

import os
import uuid
import contextlib
import numpy as np

class IndHist():
    def __init__(self, tmp_dir='tmp/', curr_id=None):
        """
        Initialize IndHist SDG

        Parameters
        ----------
        tmp_dir: str
            temporary directory to store temporary model files to
        curr_id: str
            (random) id to uniquely identify this model
        """
        self.tmp_dir = tmp_dir

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

        # fit model to dataset
        describer = DataDescriber(category_threshold=threshold_value)
        # block print statements
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data)
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

        # generate data from model
        generator = DataGenerator()
        generator.generate_dataset_in_independent_mode(num_rows, self.description_file, seed)

        if remove_desc:
            # delete temporary file
            os.remove(self.description_file)

        return generator.synthetic_dataset
