from math import sqrt
# import numpy as np


def _initial_mew():
    return 1


class z_updater(object):
    def __init__(self):
        self.mews = [_initial_mew(), _initial_mew()]

    def _gamma(self):
        return (self.mews[1] - 1) / self.mews[0]


    def _S_lambda(self, u, sparsity):
        # print u.shape
        # print u
        u = u[0, :] / max(u.max(), abs(u.min()))
        # print u.shape
        for i, v in enumerate(u.tolist()[0]):
            if (v > 0) and (v > sparsity):
                u[0, i] = (v - sparsity * v)
            elif (v < 0) and (v < -sparsity):
                u[0, i] = (v - sparsity * v)
            else:
                u[0, i] = 0
        return u


    def _hangman(self, u, y, M, sparsity):
        # print (y - u * M)
        return self._S_lambda(
            # u + (y - u * M) * np.linalg.pinv(M),
            u + (y - u * M) * M.transpose(),
            sparsity
        )

    def update_mew(self):
        self.mews[1] = self.mews[0]
        self.mews[0] = (1 + sqrt(1 + 4 * self.mews[1] ** 2)) / 2


    def update_z(self, z, M):
        return self._hangman(
            z._z_p + self._gamma() * (z._z_p - z._z_pp),
            z.y,
            M,
            z.sparse
        )
