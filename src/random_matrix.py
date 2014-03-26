from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
# from sklearn import linear_model


def generate_random_matrix(rows, cols, norm='l2'):
    return normalize(np.random.randn(rows, cols), axis=1, norm=norm)
