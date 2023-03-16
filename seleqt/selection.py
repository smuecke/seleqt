import numpy as np

from .stats import mutual_information, pairwise_mutual_information, is_discrete


def feature_selection_qubo(redundancy, importance, alpha=0.5, importance_threshold=1e-8, threshold_penalty=None):
    # check if arrays have the right shape
    error_msg = 'Shapes of redundancy and/or importance array do not match; expected (n, n) and (n,)'
    try:
        u_, v_ = redundancy.shape
        n,     = importance.shape
    except ValueError:
        raise ValueError(error_msg)
    if (u_ != v_) or (n != u_):
        raise ValueError(error_msg)
    assert np.all(redundancy >= 0), 'Redundancy contains negative values'
    assert np.all(importance >= 0), 'Importance contains negative values'
    assert np.all(np.isclose(np.diag(redundancy), 0)), 'Redundancy has non-zero diagonal'
    # use redundancy and importance values verbatim
    Q = (1-alpha)*redundancy - alpha*np.diag(importance)
    # set (scaled) importance values below given threshold to a
    # small penalty value to avoid randomness in solution; by default,
    # use absolute maximum value of QUBO matrix as penalty value.
    pen = np.linalg.norm(Q, np.infty) if threshold_penalty is None else threshold_penalty
    diag = np.diag(Q).copy()
    diag[diag>-importance_threshold] = pen
    np.fill_diagonal(Q, diag)
    return Q


def redundancy(x):
    assert is_discrete(x), 'data must be discrete (i.e., a subtype of integer)'
    red = pairwise_mutual_information(x)
    np.fill_diagonal(red, 0)
    return red

def importance(x, y):
    assert is_discrete(x) and is_discrete(y), 'data must be discrete (i.e., a subtype of integer)'
    return mutual_information(x, y[:, np.newaxis]).reshape(-1)


def qfs(redundancy, importance, k: int, importance_threshold=1e-8, threshold_penalty=None, precision_limit=1e-4, qubo_solver=None):
    """
    Perform binary search to find an alpha value such that the QUBO
    solution has the desired number of features.
    """
    n = importance.size
    assert 0 < k < n, f'k must be between 1 and {n-1}'
    if qubo_solver is None:
        raise ValueError('Please specify a QUBO solver (solve_qubo=...)')
    
    def get_min_x(alpha):
        Q = feature_selection_qubo(redundancy, importance, alpha, importance_threshold, threshold_penalty)
        return qubo_solver(Q)

    # perform QFS algorithm
    left, right = 0.0, 1.0
    while (right-left) > precision_limit:
        mid = (right+left)/2
        min_x = get_min_x(mid)
        k_ = min_x.sum()
        if k_ == k:
            return mid, min_x
        elif k_ < k:
            left = mid
        else:
            right = mid
    mid = (right+left)/2
    min_x = get_min_x(mid)
    return mid, min_x