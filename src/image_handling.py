from __future__ import division

import sys
import csv
import numpy as np
import pandas as pd
from collections import Counter
from src.random_matrix import generate_random_matrix
from src.image_t_wrapper import Image_t


def dict_2_dataframe(dictionary):
    """
    takes in a dictionary object of form
    dictionary must be of the form below!!!
    {label:list of data with matching label}
    """
    li = [[label, im] for label, allim in dictionary.items() for im in allim]
    df = pd.DataFrame(li)
    df.columns = ['label', 'image']
    return df


def label_counter(fname):
    """
    counts the number of instances of rows in a csv
    that have the same
    """
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        # we initialize a counter for each type of image
        # that way we know how many of each of them we have
        c = Counter()
        for i, row in enumerate(reader):
            if int(row[0]) in c:
                c[int(row[0])] += 1
            else:
                c[int(row[0])] = 1
    return c


def columns(fname):
    """
    counts the number columns in the csv
    """
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        return len(reader.next())


# X = prep_data(fname, slist_2_npmatrix)

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

    images = [None]*10
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i not in header:
                if images[int(row[0])] is None:
                    images[int(row[0])] = [prep_data_fn(row[1:])]
                else:
                    images[int(row[0])] += [prep_data_fn(row[1:])]
    return images


def slist_2_ilist(slist):
    return [int(a) for a in slist]


def slist_2_npmatrix(slist):
    return np.matrix([int(a) for a in slist])


def generate_images_t(X_all, W, sigma, z_len, max_from_each=(sys.maxint-1)):
    W_inv = np.linalg.pinv(W)
    all_images = [None]*len(X_all)
    for label, X in enumerate(X_all):
        these_images = [None]*min(len(X), max_from_each)
        for x_i, x in enumerate(X):
            if x_i < max_from_each:
                y = x*sigma
                z = y * W_inv
                # mew = 1.0
                # z_toggle_index = 0
                # image = [
                #     label,  # label
                #     x,      # x
                #     y,      # = x * sigma
                #     [       # z_t =[z_t_a, z_t_b, mew, ...]
                #         y * W_inv,
                #         np.matrix([0] * z_len),
                #         mew,
                #         z_toggle_index
                #         ]
                #     ]
                image_t = Image_t(label, x, y, z)
                these_images[x_i] = image_t

        all_images[label] = these_images
    return all_images


def main(fname):

    x_len = columns(fname)-1
    y_len = int(x_len/2)
    z_len = int(2*x_len)
    sigma = np.matrix(generate_random_matrix(x_len, y_len))
    W = np.matrix(generate_random_matrix(z_len, y_len))
    X = prep_data(fname, prep_data_fn=slist_2_npmatrix)
    images_t = generate_images_t(X, W, sigma, z_len)

    return images_t


if __name__ == '__main__':
    main(sys.argv(1))
