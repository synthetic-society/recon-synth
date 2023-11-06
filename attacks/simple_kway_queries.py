"""
Utility functions to select simple k-way marginal queries involving secret attribute
"""
import numpy as np
from itertools import combinations, product

def get_result_simple_kway(attrs, secret_bits, queries):
    """
    Get result for simple k-way queries
    
    Parameters
    ----------
    attrs: np.ndarray
        n x d array of attributes for each user
    secret_bits: np.ndarray
        n array of secret attributes for each user
    queries: list[(list[int], list[int], int)]
        list of queries encoded in the form (attribute ids, attribute values, secret attribute value)
    
    Returns
    -------
    n_user: np.ndarray
        q array of number of users selected by each query (based on attrs only)
    results: np.ndarray
        q array of results for queries
    """
    # convert queries to query matrix over attrs
    A = simple_kway(queries, attrs)
    n_users = np.sum(A, axis=1)
    target_vals = np.array([target_val for (_, _, target_val) in queries])

    result_target1 = np.array(A.dot(secret_bits), dtype=np.float64)
    result_target0 = n_users - result_target1

    result = np.where(target_vals == 0, result_target0, result_target1)

    return n_users, result

def gen_all_simple_kway(attrs, k):
    """
    Generate all simple k-way marginal queries
    (k - 1) attributes + 1 secret

    Parameters
    ----------
    attrs: np.ndarray
        n x d array of attributes for each user
    k: int
        total number of attributes selected by query
    
    Returns
    -------
    queries: list[(list[int], list[int], int)]
        list of queries encoded in the form ([a1, a2, ...], [v1, v2, ...], t)
        which represents a1 = v1 ^ a2 = v2 ^ ... ^ s = t
    """
    _, d = attrs.shape

    # gather unique values for each attribute
    uniq_vals = []
    for attr_ind in range(d):
        uniq_vals.append(np.unique(attrs[:, attr_ind]))

    # cycle through all combinations of attributes
    # attributes cannot be repeated
    queries = []
    attr_indss = combinations(range(d), k - 1)
    for attr_inds in attr_indss:
        curr_uniq_vals = [uniq_vals[attr] for attr in attr_inds]
        attr_valss = product(*curr_uniq_vals)
        for attr_vals in attr_valss:
            for target_val in [0, 1]:
                queries.append((attr_inds, attr_vals, target_val)) 
    
    return queries

def gen_rand_simple_kway(attrs, n_queries, k):
    """
    Generate random simple k-way marginal queries
    (k - 1) attributes + 1 target

    Parameters
    ----------
    attrs: np.ndarray
        n x d array of attributes for each user
    n_queries: int
        number of random queries to generate
    k: int
        total number of attributes selected by query
    
    Returns
    -------
    queries: list[(list[int], list[int], int)]
        list of queries encoded in the form ([a1, a2, ...], [v1, v2, ...], t)
        which represents a1 = v1 ^ a2 = v2 ^ ... ^ s = t
    """
    _, d = attrs.shape

    queries = []

    # choose random attributes
    attr_indss = np.random.randint(d, size=(n_queries, k - 1))

    # gather unique values for each attribute
    uniq_vals = []
    for attr_ind in range(d):
        uniq_vals.append(np.unique(attrs[:, attr_ind]))

    for attr_inds in attr_indss:
        # choose random unique value
        attr_vals = []
        for attr_ind in attr_inds:
            attr_val = np.random.choice(uniq_vals[attr_ind])
            attr_vals.append(attr_val)

        # choose random unique value for target
        # target_val = np.random.randint(2)
        target_val = 1
        queries.append((tuple(attr_inds), tuple(attr_vals), target_val))
    
    return queries

def simple_kway(queries, attrs):
    """
    Convert simple k-way marginal queries to subset queries over attribute matrix

    Parameters
    ----------
    queries: list[(list[int], list[int], int)]
        list of queries encoded in the form (attribute ids, attribute values, secret attribute value)
    attrs: np.ndarray
        n x d array of attributes for each user
    
    Returns
    -------
    A: np.ndarray
        q x n query matrix
    """
    A = np.zeros(shape=(len(queries), len(attrs)))
    for i, (attr_inds, vals, _) in enumerate(queries):
        boolss = []
        for attr_ind, val in zip(attr_inds, vals):
            boolss.append(attrs[:, attr_ind] == val)

        final_bools = boolss[0]
        for curr_bools in boolss[1:]:
            final_bools = final_bools & curr_bools

        A[i] = np.where(final_bools, 1, 0)
    
    return A