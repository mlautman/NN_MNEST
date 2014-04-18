import numpy as np
from math import sqrt
from random_matrix import generate_random_norm_matrix
from src.show_images import plot_list



class W_T(object):
    def __init__(
            self,
            dimensions,
            norm='l1',
            axis=1,
            learn_rate=.005,
            track_learning=True,
            W_targ=None
    ):
        """
        Initialize a W matrix to project z's back to x's.
        The dimensions of W should be (x_len, z_len)
        """
        self.dimensions = dimensions
        self.norm = norm
        self.axis = axis
        self.a = learn_rate
        self.track_learning = track_learning
        self.W_targ = W_targ
        self._delta_hist = []
        self._err_hist = []

        self._W = generate_random_norm_matrix(
            dimensions[0],
            dimensions[1],
            norm=norm,
            axis=axis
        )
        self._W_P = self.W


    @property
    def W(self):
        return self._W
    @W.setter
    def W(self, value):
        self._W = value


    def err(self):
        return self._W - self.W_targ

    def delta(self):
        return self._W - self._W_P

    def delta_mag(self):
        W_delta = self.delta()
        return sqrt(np.multiply(W_delta, W_delta).sum())

    def err_mag(self):
        W_err = self.err()
        return sqrt(np.multiply(W_err, W_err).sum())


    def update(self, images):
        """
        X.shape = (n , d)
        Y.shape = (n , r)
        Z.shape = (n , k)
        S.shape = (d , r)
        W.shape = (k , d)
        """
        for i in images:
            self._W_P = self._W

            self.W = (
                self.W - self.a * (
                    i.z.transpose() * (i.z * self.W - i.x)
                )
            )

            if self.track_learning:
                self._delta_hist.append(self.delta_mag())
                if not self.W_targ is None:
                    self._err_hist.append(
                        self.err_mag()
                    )


    def compute_error(self):
        """
        compute_error computes the error between the ideal W_targ and
        the current W value.
        The error function used is:
            sqrt(sum( (w(i,j) - w_targ(i,j))^2, for all i's and j's))
        """
        if not self.W_targ is None:
            W_err = self.W - self.W_targ
            return sqrt(np.multiply(W_err, W_err).sum())



    """
    Visualizations
    """

    def plot_deltas(self):
        plot_list(self._delta_hist)

    def plot_errs(self):
        plot_list(self._err_hist)
