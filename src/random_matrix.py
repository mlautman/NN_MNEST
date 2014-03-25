from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import linear_model
from src.random_matrix import generate_random_matrix


def generate_random_matrix(rows, cols, norm='l2'):
    return normalize(np.random.randn(rows, cols), axis=1, norm=norm)


d = 784
r = int(d/2)
k = 2*d

sigma = np.matrix(generate_random_matrix(d, r))
W_T_0 = np.matrix(generate_random_matrix(k, r))

y_t = np.array(x_t)*sigma
