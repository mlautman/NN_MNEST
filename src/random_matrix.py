from __future__ import division
# from math import sqrt
import numpy as np
from sklearn.preprocessing import normalize


def normalize_matrix(
        M,
        norm='l2',
        axis=1
):
    return np.matrix(
        normalize(
            np.array(M),
            axis=axis,
            norm=norm)
    )


# def normalize(M, norm='karth'):
#     if norm == 'cols_sum':
#         for i in range(M.shape[1]):
#             M[:, i] /= M[:, i].sum()
#     elif norm == 'l1_row':
#         for i in range(M.shape[0]):
#             M[i, :] /= M[i, :].sum()

#     elif norm == 'l2_r':
#         for i in range

#     return M


def generate_random_matrix(size):
    return np.matrix(
        np.random.randn(size[0], size[1])
    )


def generate_random_matrix_interval(size, mean=0., interval_width=2.):
    return np.matrix(
        np.random.random(size) - (0.5 - mean)
    ) * interval_width
