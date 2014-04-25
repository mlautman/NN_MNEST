import csv
import sys
import time

import numpy as np
from math import sqrt
from src.Image_t import Image_t
from src.csv_lib import columns
from src.random_matrix import normalize_matrix
from src.random_matrix import generate_random_matrix
from src.random_matrix import generate_random_matrix_interval
from src.z_updater import z_updater
from src.W_T import W_T


def S_lambda(u, sparsity):
    for i, v in enumerate(u.tolist()[0]):
        if (v > 0) and (v > sparsity):
            u[0, i] = (v - sparsity * v)
        elif (v < 0) and (v < -sparsity):
            u[0, i] = (v - sparsity * v)
        else:
            u[0, i] = 0
    return u


def slist_2_ilist(slist):
    return [int(a) for a in slist]


def slist_2_npmatrix(slist):
    return np.matrix([int(a) / 255. - .5 for a in slist])


def prep_data(file, prep_data_fn=None, header=[0]):
    """
    based on the prep_data_fn, this function reads in each
    line except the ones in the header and then stores
    the rows with matching label columns, in a dictionary
    mapping label to a list with the data in it
    (ASSUMED TO BE THE FIRST COLUMN)
    """
    if prep_data_fn is None:
        def prep_data_fn(x):
            return x

    images = []
    labels = []
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for i, row in enumerate(reader):
            if i > 100:
                return labels, images
            images += [prep_data_fn(row[1:])]
            labels += [row[0]]
    return labels, images


def generate_images_t(labels, X, W, S, z_len, max_from_each=(sys.maxint - 1)):
    # W_inv = np.linalg.pinv(W)
    all_images = [None] * len(X)
    for i, x in enumerate(X):
        all_images[i] = Image_t(labels[i], x, W, S)
    return all_images


def generate_fake_data(x_len, y_len, z_len, num_images=100):
    W_targ = normalize_matrix(
        generate_random_matrix((z_len, x_len)),
        norm='l2',
        axis=1
    )

    S = generate_random_matrix((x_len, y_len)) / sqrt(x_len * y_len)

    Z = []
    X = []
    Y = []
    for i in range(num_images):
        z = generate_random_matrix((1, z_len))
        z = z
        x = z * W_targ
        y = z * W_targ * S
        Z.append(z)
        X.append(x)
        Y.append(y)

    images = [None] * len(X)
    for i, x in enumerate(X):
        images[i] = Image_t(
            0,
            x,
            W_targ,
            S,
            sparse=0.0,
            z_targ=Z[i]
        )

    W = W_T((z_len, x_len), W_targ=W_targ)
    W.W = W_targ
    z_up = z_updater()

    return images, W, W_targ, S, z_up, X, Y, Z


def test_z():
    x_len = 28 ** 2
    y_len = int(x_len * 2)
    z_len = int(x_len)
    images, W, z_up, X, Y, Z = generate_fake_data(x_len, y_len, z_len)




def main_test_z(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(10 * x_len)
    z_len = int(3 * x_len)
    images, W, W_targ, S, z_up, X, Y, Z = generate_fake_data(
        x_len, y_len, z_len, num_images=2)

    # print W.W_targ == W_targ
    W.W_targ = W_targ
    z0 = images[0]
    # z0.z = z0.z_targ
    # # print z0.y - z0.z_targ * (W.W * S)
    # z0._z_pp = z0.z_targ
    # z0._z_p = z0.z_targ #+ generate_random_matrix_interval(
    # #     (z0.z.shape),
    # #     mean=0.,
    # #     interval_width=.1
    # # )
    M = (W.W_targ * S)
    for i in range(200):
        z0.update(z_up, M)
        z_up.update_mew()

    z0.plot_errs()

    return images, z0, z_up, Z, Y, X, W, S


def main_test_W(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(x_len * 4)
    z_len = int(3 * x_len)
    W_targ = normalize_matrix(
        generate_random_matrix((z_len, x_len))
    )
    S = generate_random_matrix((x_len, y_len)) / sqrt(x_len * y_len)
    Z = []
    X = []
    Y = []
    for i in range(100):
        z = generate_random_matrix((1, z_len))
        x = z * W_targ
        y = z * W_targ * S
        Z.append(z)
        X.append(x)
        Y.append(y)

    images = [
        Image_t(
            0,
            x,
            W_targ,
            S,
            sparse=0,
            z_targ=Z[i]
        ) for i, x in enumerate(X)]

    W = W_T((z_len, x_len), W_targ=W_targ)
    z_up = z_updater()
    for im in images:
        im._z = im.z_targ
        im._z_p = im.z_targ
        im._z_p_p = im.z_targ

    for i in range(10):
        W.update(images)

    return images, z_up, Z, Y, X, W_targ, W, S


def cycle_learn(images, W, S, z_up, inc=5):
    num_images = len(images)
    for i in range(0, num_images - 1, inc):
        M = W.W * S
        for j in range(i, min(num_images, i + inc)):
            images[j].update(z_up, M)
        W.update(images[j:j + inc])
    z_up.update_mew()


# def main_test_both(fname='./mnest_train.csv'):
#     x_len = 784
#     y_len = int(x_len / 2)
#     z_len = int(2 * x_len)


#     W = W_T((z_len, x_len), W_targ=W_targ)
#     z_up = z_updater()

#     for i in range(4):
#         cycle_learn(images, W, S, z_up)

#     return images, z_up, Z, Y, X, W_targ, W, S


def main(fname='./mnest_train.csv'):
    t0 = time.time()
    x_len = columns(fname) - 1
    y_len = int(x_len * 10)
    z_len = int(3 * x_len)
    W = W_T((z_len, x_len))
    S = generate_random_matrix((x_len, y_len)) / sqrt(x_len * y_len)
    print time.time() - t0
    labels, X = prep_data(fname, prep_data_fn=slist_2_npmatrix)
    print time.time()-t0
    images = generate_images_t(labels, X, W.W, S, z_len, max_from_each=100)
    print time.time() - t0
    z_up = z_updater()

    return images, z_up, W, S


if __name__ == '__main__':
    main(sys.argv(1))
