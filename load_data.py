"""
Utility functions to load datasets
"""
import pandas as pd
import os

# convert synthetic data sizes to human-readable format
n_rows_names = {
    10: '10',
    100: '100',
    1000: '1K',
    10000: '10K',
    100000: '100K',
    1000000: '1M'
}

def get_default_secret_bit(data_name):
    """
    Generate default secret bit for dataset

    Parameters
    ----------
    data_name: str
        name of dataset ("acs" | "fire")

    Returns
    ------
    secret_bit: str
        default secret bit associated with dataset ("SEX" | "ALS Unit")
    """
    if data_name == 'acs':
        return 'SEX'
    elif data_name == 'fire':
        return 'ALS Unit'
    else:
        raise Exception(f'ERROR: {data_name} not configured in get_default_secret_bit')

def process_data(df, secret_bit):
    """
    Convert pandas DataFrame into numeric attribute matrix and vector of secret bits

    Parameters
    ----------
    df: pd.DataFrame
        dataset
    secret_bit: str
        name of secret attribute

    Returns
    ------
    attrs: np.ndarray
        numeric attribute n x d matrix
    secret_bits: np.ndarray
        n array of secret attributes for each user
    """
    secret_bits = df[secret_bit].to_numpy()
    attrs = df.drop([secret_bit], axis=1).to_numpy()

    return attrs, secret_bits

def load_data(data_name, n_df, secret_bit, randomize=False, unique_quasi=False, balance=False):
    """
    Load dataset as DataFrame, cleaning it (if necessary) to convert categorical attributes into numerical attributes
    and sample a raw dataset to be used in privacy game

    Parameters
    ----------
    data_name: str
        name of dataset ("acs" | "fire")
    n_df: int
        number of rows to sample for raw dataset
    secret_bit: str
        name of secret attribute
    randomize: bool
        randomize selection of rows
    unique_quasi: bool
        only choose rows with unique quasi-ids
    balance: bool
        balance raw dataset with equal number of '0's and '1's in the secret bit

    Returns
    ------
    raw_df: pd.DataFrame
        raw dataset to be used in privacy game
    """
    curr_dir = os.path.dirname(__file__)
    if data_name == 'acs':
        df = pd.read_csv(f'{curr_dir}/datasets/acs.csv')
    elif data_name == 'fire':
        df = pd.read_csv(f'{curr_dir}/datasets/fire.csv')
    else:
        raise Exception(f'ERROR: {data_name} not configured in load_data')
    
    if unique_quasi:
        subset = list(df.columns)
        subset.remove(secret_bit)
        df = df.drop_duplicates(subset=subset, keep=False)

    if balance:
        # choose balanced dataset with n_df samples
        df_0 = df[df[secret_bit] == 0]
        df_1 = df[df[secret_bit] == 1]

        if randomize:
            df = pd.concat([df_0.sample(n_df // 2), df_1.sample(n_df // 2)])
        else:
            df = pd.concat([df_0[:n_df // 2], df_1[:n_df // 2]])
    else:
        if randomize:
            df = df.sample(n_df)
        else:
            df = df.sample(n_df, random_state=0)

    return df