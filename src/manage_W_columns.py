import numpy as np
from random_matrix import generate_random_matrix


class W_T(object):
    def __init__(self, dimensions, norm='l1', axis=0, learn_rate=.01):
        self.dimensions = dimensions
        self.norm = norm
        self.axis = axis
        self.a = learn_rate
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


    def ridge_update_W(self, images):
        """
        X.shape = (n , d)
        Y.shape = (n , r)
        Z.shape = (n , k)
        S.shape = (d , r)
        W.shape = (k , d)
        """
        for i in images:
            self.W = (
                self.W - self.a * ((self.W * i.z - i.x) * i.z.transpose())
            )

    def compute_error(self, W_targ):
        """
        compute_error computes the error between the ideal W_targ and
        the current W value.
        The error function used is:
            sum( (w(i,j) - w_targ(i,j))^2, for all i's and j's)
        """
        W_err = self.W - W_targ
        return np.multiply(W_err, W_err).sum()
