from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF, get_logger
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

# FOR VISUALIZING TRAJECTORIES--------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import make_blobs
mpl.use('TkAgg')
# ------------------------------------------------------------------------


def plot_open():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax2.set_xlim([-5, 5])
    #ax2.set_ylim([-5, 5])
    plt.show(block=False)
    return fig, ax1, ax2


def calculate_trajectories_output(rollout_trajectories):
    # reshape tensor from (32, 11, 9) to (32, 1, 2)
    rt_dim1, rt_dim2, rt_dim3 = rollout_trajectories.shape
    end_rollout_trajectories = tf.reshape(rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))

    # find car position (1, 1, 2)
    car_position = tf.reshape(rollout_trajectories[0, 0, 5:7], (1, 1, 2))

    # get relative positions of the endpoints
    relative_rollout_trajectories = end_rollout_trajectories - car_position
    final = np.reshape(relative_rollout_trajectories, (32, 2))

    return final


def calculate_trajectories_input(rollout_trajectories):
    # reshape tensor from (32, 11, 9) to (32, 1, 2)
    rt_dim1, rt_dim2, rt_dim3 = rollout_trajectories.shape
    end_rollout_trajectories = tf.concat([rollout_trajectories[:, 0, 2:3], rollout_trajectories[:, 0, 8:9]], axis=-1)
    end_rollout_trajectories = tf.expand_dims(end_rollout_trajectories, axis=1)

    # find car position (1, 1, 2)
    car_position = tf.reshape(rollout_trajectories[0, 0, 5:7], (1, 1, 2))

    # get relative positions of the endpoints
    relative_rollout_trajectories = end_rollout_trajectories - car_position
    final = np.reshape(relative_rollout_trajectories, (32, 2))

    return final


def plot_update(ax1, ax2, fig, rollout_trajectories):
    #plt.clf()
    ax1.clear()
    ax2.clear()

    reshaped_end = calculate_trajectories_output(rollout_trajectories)
    reshaped_input = calculate_trajectories_input(rollout_trajectories)

    # clustering
    clustered = linkage(reshaped_end, method='ward')
    labels = fcluster(clustered, t=3.5, criterion='distance')
    k = len(np.unique(labels))

    # VISUALS:
    # fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(k):
        ax2.scatter(reshaped_end[labels == i + 1, 0], reshaped_end[labels == i + 1, 1], c=colors[i])
    # ax.scatter(relative_rollout_trajectories[:, 0, 0], relative_rollout_trajectories[:, 0, 1])
    ax2.scatter(0, 0, s=100, c='black')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])

    # clustering
    clustered = linkage(reshaped_input, method='ward')
    labels = fcluster(clustered, t=3.5, criterion='distance')
    k = len(np.unique(labels))

    # VISUALS:
    # fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(k):
        ax1.scatter(reshaped_input[labels == i + 1, 0], reshaped_input[labels == i + 1, 1], c=colors[i])
    # ax.scatter(relative_rollout_trajectories[:, 0, 0], relative_rollout_trajectories[:, 0, 1])
    ax1.scatter(0, 0, s=100, c='black')
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-10, 10])


    plt.draw()
    plt.pause(0.002)
    #plt.waitforbuttonpress()
    #plt.show(block=False)


def plot_close():
    plt.close()
