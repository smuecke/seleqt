# `seleqt` - Quantum Feature Selection

This package contains a minimal implementation of the Quantum Feature Selection (QFS) algorithm described in [Mücke et al., 2023](https://link.springer.com/article/10.1007/s42484-023-00099-z).
Given a labeled dataset, this package provides methods to

* discretize the data
* compute redundancy and importance values based on mutual information
* generate feature selection QUBO instances for a given α
* find an α such that a FS QUBO has exactly k bits in its minimizing binary vector


# Installation

This package can be installed directly via `pip` from the PyPi repository:
```
pip3 install seleqt
```


# Usage

Data and labels must be given as `numpy` arrays:
The data `x` must have shape `(n, d)`, where `n` is the number of samples/observations and `d` is the number of features.
The labels `y` must have shape `(n,)`.
Both `x` and `y` must be **discrete**, i.e., their type must be a subtype of integer.
Optimally, their values should correspond to indices (`0`,`1`,...,`B-1`) of a finite number of bins `B`.

```python
from seleqt import *
```

## Discretizing

This package provides a method `discretize` that takes real-valued data and discretizes it by computing bins and returning the corresponding bin indices.
Its parameters are
* `x`: Input data
* `bins`: Number of bins (default 10)
* `method`: Method to compute bin edges, must be either `"equal"` or `"quantile"`.
    For `"equal"`, the value range is divided into `bins` equally sized bins.
    For `"quantile"`, the b/`bins`-quantiles for b in 0,...,`bins` are computed and used as bin edges. Default is `"equal"`.
* `share_bins`: Boolean that indicates if the bins should be computed over all values together vs. all columns separately. Default is `True`.

```python
x_ = discretize(x, bins=20, method='quantile')
```


## Redundancy and Importance

The methods `redundancy()` and `importance()` compute what their names suggest.
As inputs, `redundancy()` expects just `x`, and `importance()` expects both `x` and `y`.
All inputs must be discrete.
The result of `redundancy()` is a `numpy` array of shape `(d, d)`, and `importance()` returns a shape `(d,)` array.

```python
red = redundancy(x_)
imp = importance(x_, y)
```


## Create QUBO Instance

Given redundancy and importance, the method `feature_selection_qubo()` creates a QUBO parameter matrix according to Eq. 18 in the paper.
It takes the following arguments:
* `redundancy`: `numpy.ndarray` of shape `(d, d)` containing pairwise redundancy between the features
* `importance`: `numpy.ndarray` of shape `(d,)` containing importance for each feature
* `alpha`: float between 0 and 1; weighting of importance against redundancy. Corresponds to α in Eq. 18.
* `importance_threshold`: non-negative float; minimal meaningful importance value. Corresponds to ε in Eq. 18.
* `threshold_penalty`: non-negative float; penalty value swapped in for importance when it is below `importance_threshold`. Corresponds to μ in Eq. 18. Default is the 

The return value is a `numpy.ndarray` of shape `(d, d)` containing the QUBO parameters as an upper-triangular matrix.

```python
Q = feature_selection_qubo(red, imp, alpha=0.7)
```


## Perform QFS Algorithm

To perform Alg. 1 from the paper, you can use `qfs()`, which takes the following arguments:
* `redundancy` and `importance`: Same as for `feature_selection_qubo()`
* `k`: int, with 0 < `k` < `d`; target number of features to select.
* `importance_threshold` and `threshold_penalty`: Same as for `feature_selection_qubo()`
* `precision_limit`: non-negative float; smallest allowed search interval for binary search. Default is `1e-8`.
* `qubo_solver`: Callable function that takes a `(d, d)` QUBO parameter matrix and outputs a binary vector `x` that minimizes `x @ Q @ x`.
Corresponds to the QUBO oracle used in Alg. 1.

The result in a tuple `(alpha, min_x)` containing the optimal α value and the minimizing binary vector, which has `k` ones.

The following example uses the light-weight package [`qubolite`](https://github.com/smuecke/qubolite) to implement the QUBO oracle.
Note that the brute-force solver is only feasible for dimensions up to about 30.

```python
from qubolite         import qubo
from qubolite.solving import brute_force

def brute_force_solver(q):
    return brute_force(qubo(q))

alpha, min_x = qfs(red, imp, 5, qubo_solver=brute_force_solver)
```

# Citation

If you use this package in your scientific work, please cite the following article:

```
Mücke, S., Heese, R., Müller, S. et al. Feature selection on quantum computers. Quantum Mach. Intell. 5, 11 (2023). https://doi.org/10.1007/s42484-023-00099-z
```

Bibtex:
```
@article{muecke2023,
    title={Feature selection on quantum computers},
    author={M{\"u}cke, Sascha and Heese, Raoul and M{\"u}ller, Sabine and Wolter, Moritz and Piatkowski, Nico},
    journal={Quantum Machine Intelligence},
    volume={5},
    number={1},
    pages={11},
    year={2023},
    publisher={Springer},
    doi={10.1007/s42484-023-00099-z}
}
```
