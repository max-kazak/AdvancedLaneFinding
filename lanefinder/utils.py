"""
This module contains general application functions.
"""

import numpy as np
import matplotlib.pyplot as plt


def fig2data(fig):
    """
    Code sourced from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

    return buf


def plot_for_img(img):
    """
    Prepares matplotlib figure and plot the same size as the input image.
    It's convenient when you want to put image in the figure and plot something on top of it.

    :param img: input image
    :return: fig, plot
    """
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plot = fig.add_subplot(111)
    plot.axis('off')
    return fig, plot


def mask_to_3ch(mask, r=True, g=True, b=True):
    """
    Convert binary mask to 3 channel image. (for visualization purposes)

    :param mask: binary 1-channel mask
    :param r: fill red channel with 255 where mask==1
    :param g: fill green channel with 255 where mask==1
    :param b: fill blue channel with 255 where mask==1
    :return: mask image in BGR format
    """
    mask_3ch = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    if b:
        mask_3ch[:, :, 0] = mask * 255
    if g:
        mask_3ch[:, :, 1] = mask * 255
    if r:
        mask_3ch[:, :, 2] = mask * 255
    return mask_3ch


def polyval(fit, arg):
    """
    Calculate polynomial value at arg. x=f(y)

    :param fit: polynomial parameters
    :param arg: numerical or array of y values
    :return: x
    """
    return fit[0]*arg**2 + fit[1]*arg + fit[2]
