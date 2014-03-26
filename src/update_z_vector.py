from __future__ import division
import numpy as np
from math import sqrt


def _initial_z_t(z_length):
    return np.array([0]*z_length)


def _initial_mew():
    return 1.


def _update_mew(mew_o):
    return (1 + sqrt(1+4*mew_o**2))/2


def _gamma(mew_n, mew_o):
    return (mew_o - 1)/mew_n


def _update_z_t(W, z_t, z_t_o, y_t, mew_o):
    return


def solve_for_z_t(W_T, y_t):
    return y_t*np.linalg.pinv(W_T)


class Image_t:

    def __init__(self, label, x_t, sigma, W_T_0, z_length, lambda1):
        self.z_t_n = None
        self.lambda1 = None
        self.y_t = np.matrix(x_t * sigma)
        self.mew_o = None
        self.mew_n = _initial_mew()
        self.z_t_o = _initial_z_t(z_length)
        self.z_t_n = solve_for_z_t(W_T_0, self.y_t)

    def compute_gamma(self):
        self.mew_o = self.mew_n
        self.mew_n = _update_mew(self.mew_o)
        return _gamma(self.mew_n, self.mew_o)

    def _S_lambda(self, u_t):
        for i, v in enumerate(u_t):
            if v > 0 and v > self.lambda1:
                u_t[i] = v - self.lambda1*v
            elif v < 0 and v < -self.lambda1:
                u_t[i] = v + self.lambda1*v
        return u_t

    def _hangman(self, u_t, W_T):
        return self._S_lambda(u_t + (self.y_t - u_t*W_T)*np.linalg.pinv(W_T))

    def update_z_t(self, W_T):
        z_t_o_tmp = self.z_t_o
        gamma = _gamma(self.mew_n, self.mew_o)
        self.z_t_o = self.z_t_n
        self.z_t_n = self._hangman(self.z_t_n * (1 + gamma) - z_t_o_tmp, W_T)
