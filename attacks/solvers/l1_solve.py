"""
Utility function to solve L1 minimization problem using Gurobi solver
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def l1_solve(A, noisy_result, procs=1):
    """
    Solve linear program that minimizes L1 norm of error
    i.e. min_c ||A @ c - noisy_result||_1 s.t. 0 <= c <= 1
    by solving the dual linear program
    TODO: insert citation for conversion to dual LP
    
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
        n length {0, 1} vector of estimated 
    sol: np.ndarray
        n length [0, 1] vector of "scores"
    success: bool
        whether Gurobi successfully solved the LP
    """
    m, n = A.shape

    # prepare linear program
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', procs)
        env.start()
        with gp.Model(env=env) as model:
            # create vars
            x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
            e1 = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="e1")
            e2 = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0, name="e2")

            c1 = np.ones(m)
            c2 = np.ones(m)
            model.setObjective(c1 @ e1 - c2 @ e2, GRB.MINIMIZE)
            
            A_x = A.copy()
            model.addConstr(A_x @ x - e1 - e2 == noisy_result)

            # solve linear program
            model.optimize()
            if model.status == 2:
                # success
                sol = np.array(model.X)[:n]
            else:
                # failure, solution is random
                sol = np.random.randint(0, 2, size=n)
            est_secret_bits = np.where(sol >= 0.5, 1, 0)

            return est_secret_bits, sol, model.status == 2
