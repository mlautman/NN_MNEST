from __future__ import division
import numpy as np
from math import sqrt
from src.show_images import show_image_np
from src.show_images import plot_list


def _initial_z_t(z_len):
    return np.matrix([1] * z_len)


class Image_t(object):
    def __init__(
        self,
        label,
        x,
        W,
        S,
        sparse=.01,
        track_learning=True,
        z_targ=None
    ):
        self.label = label
        self._x = x
        self._y = x * S
        self._z = _initial_z_t(W.shape[0])
        self._z_p = _initial_z_t(W.shape[0])
        self._z_pp = _initial_z_t(W.shape[0])
        self.sparse = sparse
        self.track_learning = track_learning
        self.z_targ = z_targ
        self._delta_hist = []
        self._err_hist = []


    """
    setters and getters
    """
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
        """
        y = x * S

        y is an intermediate between x and z.
        y is a random projection of x into a higher dimension
        """
        return self._y
    @y.setter
    def y(self, value):
        self._y = value


    @property
    def z(self):
        """
        x = z * W

        Returns: the internal value stored as z

        z is the Neural Nets's output vector.
        z must be optimized through successive
        calls to self.update(M)

        """
        return self._z
    @z.setter
    def z(self, value):
        self._z = value


    """
    Meat
    """

    def x_est(self, W):
        return self._z * W


    def show_x_est(self, W):
        show_image_np(self.x_est(W))

    def err(self):
        return self.z_targ - self._z

    def delta(self):
        return self._z - self._z_p


    def delta_mag(self):
        z_d = self.delta()
        return sqrt(np.multiply(z_d, z_d).sum() / z_d.size)


    def err_mag(self):
        z_err = self.err()
        return sqrt(np.multiply(z_err, z_err).sum() / z_err.size)


    def update(self, z_updater, M):
        """
        updates z
        """
        self._z_pp = self._z_p
        self._z_p = self._z
        self._z = z_updater.update_z(self, M)
        """
        for tracking learning over time
        """
        if self.track_learning:
            self._delta_hist.append(self.delta_mag())
            if not self.z_targ is None:
                self._err_hist.append(
                    self.err_mag()
                )


    """
    Visualizations
    """

    def plot_deltas(self):
        plot_list(self._delta_hist)

    def plot_errs(self):
        if not self.z_targ is None:
            plot_list(self._err_hist)
