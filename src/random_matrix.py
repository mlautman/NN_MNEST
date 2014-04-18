from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
# from sklearn import linear_model


def generate_random_norm_matrix(
    rows,
    cols,
    norm='l1',
    axis=1
):
    return np.matrix(
        normalize(
            np.random.randn(
                rows,
                cols
            ),
            axis=axis,
            norm=norm))


def generate_random_matrix(rows, cols):
    return np.matrix(np.random.randn(rows, cols))
