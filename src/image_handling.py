import sys
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def show_image_v(image, rlen=28):
    """
    shows you your image
        parameters:
            image = image stretched into a vector
        notes:
            BLOCKS UNTIL YOU CLOSE OUT IMAGE
    """
    b = [image[(i-1)*rlen:(i)*rlen] for i in range(1, len(image)/rlen)]
    plt.imshow(b)
    plt.show()


def show_image_m(image):
    """
    shows you your image
        parameters:
            image = image as a matrix
        notes:
            BLOCKS UNTIL YOU CLOSE OUT IMAGE
    """
    plt.imshow(image)
    plt.show()


def put_data_in_dict(file, prep_data_fn=None, header=[0]):
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

    im = {}
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i not in header:
                if row[0] in im:
                    im[row[0]].append(prep_data_fn(row[1:]))
                else:
                    im[row[0]] = [prep_data_fn(row[1:])]
    return im


def slist_2_ilist(slist):
    return [int(a) for a in slist]


def slist_2_nplist(slist):
    return np.array([int(a) for a in slist])


def main(fname):
    cl = put_data_in_dict(fname, slist_2_ilist)
    return cl

if __name__ == '__main__':
    main(sys.argv(1))
