# Aim of this script is to load a trained LeNet5 network
# Visualize the weights in the first convolutional layer
# I tried in vain to get inspiration from various links:
#   http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
#   https://github.com/grishasergei/conviz

from scipy import misc
import os
import time
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
from math import sqrt
import python3_utils

PLOT_DIR = './out/plots'
def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, 0o777)
    #python3_utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

# First let's load meta graph and restore weights
graph = tf.get_default_graph()
saver = tf.train.import_meta_graph('./results/4/checkpoint-9999.meta')

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./results/4/'))

    retrieved_W = graph.get_tensor_by_name("conv1/weights:0")

    # get weights of all convolutional layers
    # no need for feed dictionary here
    conv_weights = sess.run([tf.get_collection('conv1/weights')])


    plot_conv_weights(conv_weights, 'conv{}'.format(1))