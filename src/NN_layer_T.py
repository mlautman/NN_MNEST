

class NN_layer_T(object):
    def __init__(
            self,
            W=None
    ):
        """
        init is called with option of passing in the projection matrix W
        If W is passed in, then there is no need to train the W and the Z
        matrices. If None is passed in then the user will have to pass in
        Data to fill out the resulting
        """
        self.W = W

    def train(
            X,
            Z=None,
            W=None,
            max_iter=10,
            converg=.0001,
            chunk_size=None,
            W_learning_speed=.001,
            Z_sparsity=0.1,
            y_len=None,
            z_len=None,
    ):
        return

    def compute_Z(X):
        """
        Given an input of X computes the Z matrix
        TODO
        """
        return
