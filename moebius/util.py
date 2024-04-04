import numpy as np
from . import olr_universal

def make_diagonal(matrix):
    for i, dim in enumerate(matrix):
        A = np.array(dim)
        matrix[i] = np.tril(A) + np.triu(A.T, 1)

    return matrix

def print_gmm_params_fancy(w: list, means: list, cov: list, decimal_places: int = 2):
    w = np.around(w, decimal_places)
    means = np.around(means, decimal_places)
    cov = np.around(cov, decimal_places)
    o = olr_universal(w, means, cov)

    print('\033[92m ---  weights   --- \x1b[0m')
    print(w)
    print('\033[92m ---   means    --- \x1b[0m')
    print(means)
    print('\033[92m --- covariance --- \x1b[0m')
    print(cov)
    print('\033[94m ---    OLR     --- \x1b[0m')
    print(o)
