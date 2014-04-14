from __future__ import division
import numpy as np
from numpy import sign
from math import sqrt
from src.show_images import show_image_np
from src.show_images import plot_list


def _initial_z_t(z_shape):
    return np.matrix([0]*z_shape)


def _initial_mew():
    return 1


def gamma(mew):
    return (mew[1]-1)/mew[0]


def _S_lambda(u, sparsity):
    print u.shape
    print u
    u = u[0, :]/max(u.max(), abs(u.min()))
    print u.shape
    for i, v in enumerate(u.tolist()[0]):
        if (v > 0) and (v > sparsity):
            u[0, i] = (v - sparsity * v)
        elif (v < 0) and (v < -sparsity):
            u[0, i] = (v - sparsity * v)
        else:
            u[0, i] = 0
    return u


def _hangman(u, y, M, sparsity):
    return _S_lambda(
        u + (y - u * M) * np.linalg.pinv(M),#.transpose(),
        sparsity
        )


class Image_t(object):
    def __init__(self, label, x, W, S, sparse=.1, z_targ=None):
        self.label = label
        self._x = x
        self._y = x * S
        self._z = _initial_z_t(W.shape[0])
        self._z_p = _initial_z_t(W.shape[0])
        self._z_pp = _initial_z_t(W.shape[0])
        self.sparse = sparse
        self._z_target = z_targ
        self.mew = [_initial_mew(), _initial_mew()]
        self.z_delta_hist = []

    @property
    def x(self):
        """
        returns the internal value stored as x
        x is a np.array representation of the image
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def z(self):
        """
        x = W * z

        Returns: the internal value stored as z

        z is the Neural Nets's output vector.
        z must be optimized through successive
        calls to self.update(M)

        """
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    @property
    def z_targ(self):
        """
        """
        return self._z_targ

    @z_targ.setter
    def z_targ(self, value):
        self._z_targ = value

    def _update_mew(self):
        self.mew[1] = self.mew[0]
        self.mew[0] = (1 + sqrt(1+4*self.mew[1]**2))/2

    def update(self, M):
        self._update_mew()
        self._z_pp = self._z_p
        self._z_p = self._z
        self._z = _hangman(
            self._z_p + gamma(self.mew) * (self._z_p - self._z_pp),
            self.y,
            M,
            self.sparse
            )
        self.z_delta_hist.append(self.z_delta_mag())

    def z_delta(self):
        return self._z - self._z_p

    def z_delta_mag(self):
        z_d = self.z_delta()
        return np.multiply(z_d, z_d).sum()

    def x_est(self, W):
        return self._z * W

    def show_x_est(self, W):
        show_image_np(self.x_est(W))

    def get_z_err(self):
        return self._z

    def get_z_err_mag(self):
        z_err = self.z_targ - self._z
        return sqrt(np.multiply(z_err, z_err).sum())

    def plot_z_deltas(self):
        plot_list(self.z_delta_hist)
