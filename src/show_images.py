import numpy as np
from matplotlib import pyplot as plt


def show_image_v(image, rlen=28):
    """
    shows you your image
        parameters:
            image = image stretched into a vector
        notes:
            BLOCKS UNTIL YOU CLOSE OUT IMAGE
    """
    b = [image[(i-1)*rlen:(i)*rlen] for i in range(1, int(len(image)/rlen))]
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


def show_image_np(np_image, rlen=28):
    plt.imshow(np.resize(np_image, (int(np_image.size/rlen), rlen)))
    plt.show()


def plot_list(list):
    plt.plot(list)
    plt.show()
