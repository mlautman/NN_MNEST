import numpy as np
from math import sqrt
from random_matrix import generate_random_matrix
from src.show_images import plot_list



class W_T(object):
    def __init__(
            self,
            dimensions,
            norm='l1',
            axis=1,
            learn_rate=.01,
            track_learning=True
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
        self.z_delta_hist = []
        self.z_err_hist = []

        self._W = generate_random_matrix(
            dimensions[0],
            dimensions[1],
            norm=norm,
            axis=axis
        )


    @property
    def W(self):
        return self._W
    @W.setter
    def W(self, value):
        self._W = value



    def ridge_update_W(self, images, targ=None):
        """
        X.shape = (n , d)
        Y.shape = (n , r)
        Z.shape = (n , k)
        S.shape = (d , r)
        W.shape = (k , d)
        """
        for i in images:
            self.W = (
                self.W - self.a * (i.z.transpose() * (i.z * self.W - i.x))
            )
        if self.track_learning:
            self._delta_hist.append(self.z_delta_mag())
            if not targ is None:
                self._err_hist.append(self.get_err_mag(targ))


    def compute_error(self, W_targ):
        """
        compute_error computes the error between the ideal W_targ and
        the current W value.
        The error function used is:
            sqrt(sum( (w(i,j) - w_targ(i,j))^2, for all i's and j's))
        """
        W_err = self.W - W_targ
        return sqrt(np.multiply(W_err, W_err).sum())



    """
    Visualizations
    """

    def plot_z_deltas(self):
        plot_list(self._delta_hist)

    def plot_z_errs(self):
        plot_list(self._err_hist)
