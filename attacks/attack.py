"""
Utility functions to run attacks on queries and using classifier
"""
from .solvers import l1_solve

def query_attack(A, noisy_result, procs):
    """
    Run our attack (Adv_recon)

    Parameters
    ----------
    A: np.ndarray
        q x n query matrix
    noisy_result: np.ndarray
        q length vector of (noisy) results to queries
    procs: int
        number of processors to use
    
    Returns
    -------
    est_secret_bits: np.ndarray
        n length {0, 1} vector of estimated secret bits 
    sol: np.ndarray
        n length [0, 1] vector of "scores"
    success: bool
        whether Gurobi successfully solved the LP
    """
    # formulate & solve LP
    return l1_solve(A, noisy_result, procs)

def classifier_attack(predict_attribute_matrix, train_attribute_matrix, train_secret_bits, clf):
    """
    Run attribute inference attack by Stadler et al. (Adv_infer)
    https://www.usenix.org/system/files/sec22summer_stadler.pdf

    Parameters
    ----------
    predict_attribute_matrix: np.ndarray
        n x d matrix of quasi-ids of users from target dataset
    train_attribute_matrix: np.ndarray
        n x d matrix of quasi-ids of users from synthetic dataset
    train_secret_bits: np.ndarray
        n length vector of secret bits of users from synthetic dataset
    clf: RandomForestClassifier | LogisticRegression
        classifier used to predict secret bits from quasi-ids
    
    Returns
    -------
    est_secret_bits: np.ndarray
        n length {0, 1} vector of estimated secret bits
    sol: np.ndarray
        n length [0, 1] vector of "scores"
    """
    clf.fit(train_attribute_matrix, train_secret_bits)

    return clf.predict(predict_attribute_matrix), clf.predict_proba(predict_attribute_matrix)[:, 0]