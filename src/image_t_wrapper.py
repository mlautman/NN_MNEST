from __future__ import division
import numpy as np
from math import sqrt
from src.show_images import show_image_np


def _initial_z_t(z_length):
    return np.array([0]*z_length)


def _initial_mew():
    return 1.


def _update_mew(mew_o):
    return (1 + sqrt(1+4*mew_o**2))/2


def _gamma(mew_n, mew_o):
    return (mew_o - 1)/mew_n


def solve_for_z_t(W_T, y_t):
    return y_t*np.linalg.pinv(W_T)


def toggle(a):
    return 1 if a == 0 else 0


class Image_t:

    def __init__(self, label, x, y, z, lambda1=.1):
        self.label = label
        self.x = x
        self.y = y
        self.z = [z, _initial_z_t(z.size)]
        self.z_new = 0
        self.mew_o = None
        self.mew_n = _initial_mew()

    def _compute_gamma(self):
        self.mew_o = self.mew_n
        self.mew_n = _update_mew(self.mew_n)
        return _gamma(self.mew_n, self.mew_o)

    def _S_lambda(self, u):
        for i, v in enumerate(u):
            if v > 0 and v > self.lambda1:
                u[i] = v - self.lambda1 * v
            elif v < 0 and v < -self.lambda1:
                u[i] = v + self.lambda1 * v
        return u

    def _hangman(self, u, W_T):
        return self._S_lambda(
            u + (self.y - u * W_T) * W_T.transpose()
            )

    def update_z_t(self, W_T):
        z_n_i = self.z_new
        z_o_i = toggle(self.z_new)
        gamma = self._compute_gamma()
        self.z[z_o_i] = self._hangman(
            self.z[z_n_i] + gamma * (self.z[z_n_i] - self.z[z_o_i]),
            W_T
            )
        self.z_new = toggle(self.z_new)

    def get_z(self):
        return self.z[self.z_new]

    def get_z_delta(self):
        return self.z[self.z_new] - self.z[toggle(self.z_new)]

    def estimate_x(self, W, sigma):
        return self.get_z() * W * np.pinv(sigma)

    def show_x_estimate(self, W, sigma):
        show_image_np(self.estimate_x(W, sigma))

    def get_z_t_delta(self):
        return self.z[self.z_new] - self.z[self.z_new]
