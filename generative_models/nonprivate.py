"""
Class that encapsulates Non-Private Synthetic Data Generation algorithm
that simply samples with replacement from the dataset provided
"""

class NonPrivate():
    def __init__(self, replace=True):
        """
        Initialize NonPrivate SDG

        Parameters
        ----------
        replace: bool
            sample with replacement
        """
        self.replace = replace

    def fit(self, df):
        """
        Fit SDG to training dataset

        Parameters
        ----------
        df: pd.DataFrame
            training data
        """
        self.df = df
    
    def sample(self, num_rows):
        """
        Sample records from training dataset (optinally w/ replacement)

        Parameters
        ----------
        num_rows: int
            number of synthetic records to generate

        Returns
        -------
        df: pd.DataFrame
            synthetic dataset
        """
        return self.df.sample(n=num_rows, replace=self.replace) 
