import numpy as np

def make_diagonal(matrix):
    for i, dim in enumerate(matrix):
        A = np.array(dim)
        matrix[i] = np.tril(A) + np.triu(A.T, 1)

    return matrix
