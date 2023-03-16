import itertools

import numpy as np


def entropy(x):
    n = x.size
    unique = np.unique(x)
    entr = 0
    for u in unique:
        count = (x == u).sum()
        entr += count*np.log2(count)
    return -entr/n + np.log2(n)


def joint_entropy(x, y):
    n = x.size
    _, counts = np.unique(np.vstack((x, y)).T, return_counts=True, axis=0)
    probs = counts / n
    return -(probs*np.log2(probs)).sum()


def mutual_information(xs, ys):
    n, d = xs.shape 
    m, f = ys.shape
    assert n == m

    entr_xs = np.array([ entropy(xs[:,i]) for i in range(d) ])
    entr_ys = np.array([ entropy(ys[:,i]) for i in range(f) ])
    mutinf = np.zeros((d, f))
    for i, j in itertools.product(range(d), range(f)):
        v = joint_entropy(xs[:, i], ys[:, j])
        mutinf[i, j] = entr_xs[i] + entr_ys[j] - v
    return mutinf


def pairwise_mutual_information(xs):
    _, d = xs.shape
    entr_xs = np.array([ entropy(xs[:,i]) for i in range(d) ])
    mutinf = np.zeros((d, d))
    for i, j in itertools.combinations_with_replacement(range(d), r=2):
        if i == j:
            mutinf[i, i] = entr_xs[i]
        else:
            v = joint_entropy(xs[:, i], xs[:, j])
            mutinf[i,j] = entr_xs[i] + entr_xs[j] - v
    return mutinf + np.triu(mutinf, 1).T


def is_discrete(data):
    return np.issubdtype(data.dtype, np.integer)


def discretize(xs, bins=10, method='equal', share_bins=True):
    get_bins = {
        'equal':    lambda x: np.linspace(x.min(), x.max(), bins, endpoint=False),
        'quantile': lambda x: np.quantile(x, np.linspace(0, 1, bins, endpoint=False))
    }[method.lower()]

    xs_ = np.empty_like(xs).astype(np.int64)
    if share_bins or xs.ndim == 1:
        bins_ = get_bins(xs)
    if xs.ndim == 1:
        xs_[:] = np.digitize(xs, bins_)-1
    else:
        for i in range(xs.shape[1]):
            if not share_bins:
                bins_ = get_bins(xs[:,i])
            xs_[:,i] = np.digitize(xs[:,i], bins_)-1
    return xs_