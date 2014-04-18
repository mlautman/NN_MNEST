class karthik_z_update_fn(opject)


def gamma(mew):
    return (mew[1] - 1) / mew[0]


def _S_lambda(u, sparsity):
    print u.shape
    print u
    u = u[0, :] / max(u.max(), abs(u.min()))
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
        u + (y - u * M) * np.linalg.pinv(M),
        # u + (y - u * M) * M.transpose(),
        sparsity
    )

hangman(
            self._z_p + gamma(self.mew) * (self._z_p - self._z_pp),
            self.y,
            M,
            self.sparse
        )