from __future__ import division
import numpy as np
from math import sqrt
from src.show_images import show_image_np


# Things we want to output while we run the algo
#   sum of get_z_delta for each run (sequential)
#   image of get_z_delta
#   Image animation as gif.

def _initial_z_t(z_shape):
    return np.matrix([0]*z_shape[0]*z_shape[1])


def _initial_mew():
    return 1


def _update_mew(mew_o):
    return (1 + sqrt(1+4*mew_o**2))/2


def _gamma(mew_n, mew_o):
    return (mew_o - 1)/mew_n


def solve_for_z_t(W_T, y_t):
    return y_t*np.linalg.pinv(W_T)


def toggle(a):
    return 1 if a == 0 else 0


class Image_t:

    def __init__(self, label, X, Y, Z, lambda1=.1):
        self.label = label
        self.lambda1 = lambda1
        self.X = X
        self.Y = Y

        self.Z = Z
        self.Z_o = Z
        self.mew_o = [None]*X.shape[0]
        self.mew_n = [_initial_mew()]*X.shape[0]

    def update_lambda(self, lambda1):
        self.lambda1 = lambda1

    def _compute_gamma(self, index):
        self.mew_o = self.mew_n
        self.mew_n = _update_mew(self.mew_n)
        return _gamma(self.mew_n, self.mew_o)

    def _S_lambda(self, u):
        u = u[0, :]/max(u.max(), abs(u.min()))
        for i, v in enumerate(u.tolist()[0]):
            if (v > 0) and (v > self.lambda1):
                u[0, i] = (v - self.lambda1 * v)
            elif (v < 0) and (v < -self.lambda1):
                u[0, i] = (v - self.lambda1 * v)
            else:
                u[0, i] = 0
        return u

    def _hangman(self, u, M):
        return self._S_lambda(
            u - (self.y - u * M)*np.linalg.pinv(M)
            )

    def update_z_t(self, W, S):
        M = W * S
        Z_n = self.Z

        z_o_i = toggle(self.z_new)
        gamma = self._compute_gamma()
        self.z[z_o_i] = self._hangman(
            self.z[z_n_i] + gamma * (self.z[z_n_i] - self.z[z_o_i]),
            M
            )
        self.z_new = toggle(self.z_new)

    def get_z(self):
        return self.z[self.z_new]

    def get_z_delta(self):
        return self.z[self.z_new] - self.z[toggle(self.z_new)]

    def estimate_x(self, W):
        return self.get_z() * W

    def show_x_estimate(self, W):
        show_image_np(self.estimate_x(W))

    def get_z_t_delta(self):
        return self.z[self.z_new] - self.z[self.z_new]
