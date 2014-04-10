from __future__ import division
import numpy as np
# from math import sqrt
# from src.show_images import show_image_np


def _initial_z_t(z_shape):
    return np.matrix([0]*z_shape[0]*z_shape[1])


class Image_t(object):
    def __init__(self, label, x, W, S, sparse=.1, z_targ=None):
        self.label = label
        self._x = x
        self._y = W * x
        self._z = _initial_z_t()
        self._sparse = sparse
        self._z_target = z_targ

    def update():
        return

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
