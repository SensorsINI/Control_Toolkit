import numpy as np
import tensorflow as tf

# FOR VISUALIZING TRAJECTORIES--------------------------------------------
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

# Use TkAgg for Windows and MACOSX for Mac
if platform.system() == 'Darwin':
    mpl.use('MACOSX')
else:
    mpl.use('TkAgg')
# ------------------------------------------------------------------------


class TrajectoriesVizualizer:
    def __init__(self):
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.plot_open()

    def plot_open(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        plt.show(block=False)

    def calculate_output(self, rollout_trajectories):
        # reshape tensor from (32, 11, 9) to (32, 1, 2)
        rt_dim1, rt_dim2, rt_dim3 = rollout_trajectories.shape
        end_rollout_trajectories = tf.reshape(rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))

        # find car position (1, 1, 2)
        car_position = tf.reshape(rollout_trajectories[0, 0, 5:7], (1, 1, 2))

        # get relative positions of the endpoints
        relative_rollout_trajectories = end_rollout_trajectories - car_position
        final = np.reshape(relative_rollout_trajectories, (32, 2))

        return final

    def calculate_input(self, inputs):
        # INPUT: Qn -> (32, 10, 2)
        # reshape tensor from (32, 10, 2) to (32, 20)
        dim1, dim2, dim3 = inputs.shape
        reshaped_inputs = tf.reshape(inputs, (dim1, dim2 * dim3))

        # run TSNE
        final = TSNE(n_components=2).fit_transform(reshaped_inputs)

        return final

    def plot_set(self, data, axis, over_label, x_label, y_label, cluster, x_lim=None, y_lim=None):
        # If clustering enabled, cluster. If not, simply plot the data.
        if cluster:
            # clustering
            clustered = linkage(data, method='ward')
            labels = fcluster(clustered, t=3.5, criterion='distance')
            k = len(np.unique(labels))

            # VISUALS:
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for i in range(k):
                axis.scatter(data[labels == i + 1, 0], data[labels == i + 1, 1], c=colors[i])
        else:
            axis.scatter(data[:, 0], data[:, 1], c='r')

        # Origin point
        axis.scatter(0, 0, s=100, c='black')

        # If limits given set scale, if not automatic scale
        if x_lim is None:
            axis.set_xlim(auto=True)
        else:
            axis.set_xlim(x_lim)
        if y_lim is None:
            axis.set_ylim(auto=True)
        else:
            axis.set_ylim(y_lim)

        # Set the titles accordingly
        axis.set_title(over_label)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

        # Make square, keep proportions
        axis.set_aspect('equal', adjustable='box')

    def plot_update(self, rollout_trajectories):
        self.ax1.clear()
        self.ax2.clear()

        # Calculate the plot data
        reshaped_output = self.calculate_output(rollout_trajectories)
        reshaped_input = self.calculate_input(rollout_trajectories)

        # Set plots
        self.plot_set(reshaped_input, self.ax1, 'Input Space', 'X-LABEL', 'Y-LABEL',
                      False, [-250, 250], [-250, 250])
        self.plot_set(reshaped_output, self.ax2, 'Output Space', 'x Position', 'y Position',
                      True, [-5, 5], [-5, 5])

        # Spacing of plots
        self.fig.tight_layout(pad=1.0)

        plt.draw()

        # CHOOSE BETWEEN THE TWO MODES
        plt.pause(0.002)
        #plt.waitforbuttonpress()

    def plot_close(self):
        plt.close()
