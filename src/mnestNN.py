import csv
import sys
import time

import numpy as np

from src.Image_t import Image_t
from src.csv_lib import columns
from src.random_matrix import generate_random_norm_matrix
from src.random_matrix import generate_random_matrix
from src.z_updater import z_updater
from src.W_T import W_T


def slist_2_ilist(slist):
    return [int(a) for a in slist]


def slist_2_npmatrix(slist):
    return np.matrix([int(a) / 255. - .5 for a in slist])
import csv
import sys
import time

import numpy as np

from src.Image_t import Image_t
from src.csv_lib import columns
from src.random_matrix import generate_random_norm_matrix
from src.random_matrix import generate_random_matrix
from src.z_updater import z_updater
from src.W_T import W_T


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

    images = [None] * 10
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 100:
                return images
            if i not in header:
                if images[int(row[0])] is None:
                    images[int(row[0])] = [prep_data_fn(row[1:])]
                else:
                    images[int(row[0])] += [prep_data_fn(row[1:])]
    return images


def generate_images_t(X, W, S, z_len, max_from_each=(sys.maxint - 1)):
    # W_inv = np.linalg.pinv(W)
    all_images = [None] * len(X)
    for label, x_label in enumerate(X):
        x_for_label = [None] * len(x_label)
        for i, x in enumerate(x_label):
            x_for_label[i] = Image_t(label, x, W, S)
        all_images[label] = x_for_label
    return all_images





def main_test_z(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W = generate_random_norm_matrix(z_len, x_len)
    S = generate_random_norm_matrix(x_len, y_len)
    Z = []
    X = []
    Y = []
    for i in range(100):
        z = generate_random_matrix(1, z_len)
        x = z * W
        y = z * W * S
        Z.append(z)
        X.append(x)
        Y.append(y)

    images = [
        Image_t(
            0,
            x,
            W,
            S,
            sparse=0,
            z_targ=Z[i]
        ) for i, x in enumerate(X)]

    z_up = z_updater()
    z_up0 = z_updater()
    z0 = images[0]
    M = W * S
    for i in range(500):
        z0.update(z_up0, M)
        z_up0.update_mew()

    z0.plot_z_errs()

    return images, z0, z_up, z_up0, Z, Y, X, W, S


def main_test_W(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W_targ = generate_random_norm_matrix(z_len, x_len)
    S = generate_random_norm_matrix(x_len, y_len)
    Z = []
    X = []
    Y = []
    for i in range(100):
        z = generate_random_matrix(1, z_len)
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

    for i in range(300):
        W.update(im)

    return images, z_up, Z, Y, X, W_targ, W, S


def cycle_learn(images, W, S, z_up, inc=5):
    num_images = len(images)
    for i in range(0, num_images - 1, inc):
        M = W.W * S
        for j in range(i, min(num_images, i + inc)):
            images[j].update(z_up, M)
        W.update(images[j:j + inc])
    z_up.update_mew()


def main_test_both(fname='./mnest_train.csv'):
    x_len = 784
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W_targ = generate_random_norm_matrix(z_len, x_len)
    S = generate_random_norm_matrix(x_len, y_len)
    Z = []
    X = []
    Y = []
    for i in range(25):
        z = generate_random_matrix(1, z_len)
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

    for i in range(4):
        cycle_learn(images, W, S, z_up)

    return images, z_up, Z, Y, X, W_targ, W, S


def main(fname='./mnest_train.csv'):
    t0 = time.clock()
    x_len = columns(fname) - 1
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W = W_T((z_len, x_len))
    S = generate_random_norm_matrix(x_len, y_len)
    print time.clock() - t0
    X = prep_data(fname, prep_data_fn=slist_2_npmatrix)
    print time.clock() - t0
    images = generate_images_t(X, W.W, S, z_len, max_from_each=100)
    print time.clock() - t0

    z_up = z_updater()

    return images, z_up, W, S


if __name__ == '__main__':
    main_test_z(sys.argv(1))



def main_test_z(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W = generate_random_norm_matrix(z_len, x_len)
    S = generate_random_norm_matrix(x_len, y_len)
    Z = []
    X = []
    Y = []
    for i in range(10):
        z = generate_random_matrix(1, z_len)
        x = z * W
        y = z * W * S
        Z.append(z)
        X.append(x)
        Y.append(y)

    images = [
        Image_t(
            0,
            x,
            W,
            S,
            sparse=0,
            z_targ=Z[i]
        ) for i, x in enumerate(X)]

    z_up = z_updater()
    z_up0 = z_updater()
    z0 = images[0]
    for i in range(500):
        z0.update(z_up0, W * S)
        z_up0.update_mew()

    z0.plot_z_errs()
    return images, z0, z_up, z_up0, Z, Y, X, W, S


def main_test_W(fname='./mnest_train.csv'):
    x_len = 100
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W_targ = generate_random_norm_matrix(z_len, x_len)
    S = generate_random_norm_matrix(x_len, y_len)
    Z = []
    X = []
    Y = []
    for i in range(10):
        z = generate_random_matrix(1, z_len)
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

    return images, z_up, Z, Y, X, W_targ, W, S


def cycle_learn(images, W, S, z_up, inc=10):
    num_images = len(images)
    for i in range(0, num_images - 1, inc):
        M = W.W * S
        for j in range(i, min(num_images, i + inc)):
            images[j].update(z_up, M)
        W.update(images[j:j + inc])
    z_up.update_mew()


def main(fname='./mnest_train.csv'):
    t0 = time.clock()
    x_len = columns(fname) - 1
    y_len = int(x_len / 2)
    z_len = int(2 * x_len)
    W = W_T((z_len, x_len))
    S = generate_random_norm_matrix(x_len, y_len)
    print time.clock() - t0
    X = prep_data(fname, prep_data_fn=slist_2_npmatrix)
    print time.clock() - t0
    images = generate_images_t(X, W.W, S, z_len, max_from_each=1000)
    print time.clock() - t0

    z_up = z_updater()

    return images, z_up, W, S


if __name__ == '__main__':
    main_test_z(sys.argv(1))
