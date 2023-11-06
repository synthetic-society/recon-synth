"""
Utility functions to select k-way marginal queries involving any combination of attributes
"""
import numpy as np
from itertools import combinations, product

def get_result_any_kway(df_matrix, queries):
    """
    Get result for any k-way queries
    
    Parameters
    ----------
    df_matrix: np.ndarray
        n x (d + 1) array of attributes and secret bits for each user
    queries: list[(list[int], list[int], int)]
        list of queries encoded in the form (attribute ids, attribute values, secret attribute value)
    
    Returns
    -------
    results: np.ndarray
        unnecessary return so that this function is the same type as `get_result_simple_kway`
        q array of results for queries
    results: np.ndarray
        q array of results for queries
    """
    # convert queries to query matrix
    A = any_kway(queries, df_matrix)
    result = np.sum(A, axis=1)
    return result, result

def gen_all_any_kway(df_matrix, k):
    """
    Generate all k-way marginal queries

    Parameters
    ----------
    df_matrix: np.ndarray
        n x (d + 1) array of attributes and secret bits for each user
    k: int
        total number of attributes selected by query
    
    Returns
    -------
    queries: list[(list[int], list[int])]
        list of queries encoded in the form ([a1, a2, ..., ak], [v1, v2, ..., vk])
        which represents a1 = v1 ^ a2 = v2 ^ ... ^ ak = vk
    """
    _, d = df_matrix.shape

    # gather unique values for each attribute
    uniq_vals = []
    for attr_ind in range(d):
        uniq_vals.append(np.unique(df_matrix[:, attr_ind]))

    # cycle through all combinations of attributes
    # attributes cannot be repeated
    queries = []
    attr_indss = combinations(range(d), k - 1)
    for attr_inds in attr_indss:
        curr_uniq_vals = [uniq_vals[attr] for attr in attr_inds]
        attr_valss = product(*curr_uniq_vals)
        for attr_vals in attr_valss:
            queries.append((attr_inds, attr_vals)) 
    
    return queries

def gen_rand_any_kway(df_matrix, n_queries, k):
    """
    Generate random k-way marginal queries

    Parameters
    ----------
    df_matrix: np.ndarray
        n x (d + 1) array of attributes and secret bits for each user
    n_queries: int
        number of random queries to generate
    k: int
        total number of attributes selected by query
    
    Returns
    -------
    queries: list[(list[int], list[int])]
        list of queries encoded in the form ([a1, a2, ..., ak], [v1, v2, ..., vk])
        which represents a1 = v1 ^ a2 = v2 ^ ... ^ ak = vk
    """
    _, d = df_matrix.shape

    queries = []

    # gather unique values for each attribute
    uniq_vals = []
    for attr_ind in range(d):
        uniq_vals.append(np.unique(df_matrix[:, attr_ind]))

    for _ in range(n_queries):
        # select random set of attributes (attributes cannot be repeated)
        attr_inds = tuple(np.random.choice(np.arange(d), size=k, replace=False))

        # cycle through all values of attributes
        curr_uniq_vals = [uniq_vals[attr] for attr in attr_inds]
        attr_valss = product(*curr_uniq_vals)
        for attr_vals in attr_valss:
            queries.append((attr_inds, attr_vals)) 
    
    return queries

def any_kway(queries, df_matrix):
    """
    Convert any k-way marginal queries to subset queries over dataset

    Parameters
    ----------
    queries: list[(list[int], list[int])]
        list of queries encoded in the form (attribute ids, attribute values)
    df_matrix: np.ndarray
        n x (d + 1) array of attributes and secret bits for each user
    
    Returns
    -------
    A: np.ndarray
        q x n query matrix
    """
    n = len(queries[0][0])
    A = np.zeros(shape=(len(queries), len(df_matrix)))
    for i, (attr_inds, vals) in enumerate(queries):
        boolss = []
        for attr_ind, val in zip(attr_inds, vals):
            boolss.append(df_matrix[:, attr_ind] == val)

        final_bools = boolss[0]
        for j in range(n - 1):
            final_bools = final_bools & boolss[j + 1]

        A[i] = np.where(final_bools, 1, 0)
    
    return A