from scipy import optimize as opt


def square_e(M):
    for i in range(len(M)):
        for j in range(len(M[0, :])):
            M[i, j] = M[i, j]**2

    return M


class W_T:
    def __init__(self, W, S, alpha):
        self.W_o = W
        self.W = W
        self.S = S
        self.alpha = alpha

    @property
    def W(self):
        return self.W

    def _optimize_fn(self, Z, Y):
        return (
            square_e(Y - Z * self.S * self.W).sum() +
            self.alpha*square_e(self.W).sum()
            )

    def ridge_update_W(self, Z, Y):
        """
        X.shape = (n , d)
        Y.shape = (n , r)
        Z.shape = (n , k)
        S.shape = (d , r)
        W.shape = (k , d)
        """
        self.W = opt.fmin(self._optimize_fn, self.W, args=(Z, Y), maxiter=2,)
        print self.W.shape
