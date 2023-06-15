import numpy as np
import tensorflow as tf

# FOR VISUALIZING TRAJECTORIES--------------------------------------------
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

from scipy.ndimage import rotate

# Use TkAgg for Windows and MACOSX for Mac
if platform.system() == 'Darwin':
    mpl.use('MACOSX')
else:
    mpl.use('TkAgg')
# ------------------------------------------------------------------------


def rotate_points(points, angle):
    """
    Function that takes our trajectories as a (... x 2) tensor and rotates it by given angle.

    :param points:  trajectories (... x 2)
    :param angle:   rotation angle (rad)
    :return:        rotated trajectories (... x 2)
    """
    # Compute sin and cos of the angle
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Apply rotation matrix
    rotated_points = np.empty_like(points)
    rotated_points[:, 0] = points[:, 0] * cos_angle - points[:, 1] * sin_angle
    rotated_points[:, 1] = points[:, 0] * sin_angle + points[:, 1] * cos_angle

    return rotated_points


class TrajectoryVisualizer:
    """
    A class that helps visualize input and output spaces with the help of matplotlib.
    """
    def __init__(self):
        """
        Initialize the TrajectoryVisualizer class.
        """
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.plot_open()

    def plot_open(self):
        """
        Open a fig with two plots.
        """
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        plt.show(block=False)

    def calculate_output(self, rollout_trajectories):
        """

        :param rollout_trajectories:
        :return:
        """
        # reshape tensor from (32, 11, 9) to (32, 1, 2)
        rt_dim1, rt_dim2, rt_dim3 = rollout_trajectories.shape
        end_rollout_trajectories = tf.reshape(rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))

        # TESTING AREA vvvvvvvvvvvv
        if False:
            selected_indices = [0, 1, 2, 5, 6]
            reshaped_rollout_trajectories = tf.gather(rollout_trajectories, selected_indices, axis=2)
            reshaped_rollout_trajectories = tf.reshape(reshaped_rollout_trajectories,
                                                   (rt_dim1, rt_dim2 * len(selected_indices)))
            return PCA(n_components=2).fit_transform(reshaped_rollout_trajectories)
        # TESTING AREA ^^^^^^^^^^^^

        # find car position (1, 1, 2)
        car_position = tf.reshape(rollout_trajectories[0, 0, 5:7], (1, 1, 2))

        # get relative positions of the endpoints
        relative_rollout_trajectories = end_rollout_trajectories - car_position

        # reshape
        final = np.reshape(relative_rollout_trajectories, (32, 2))

        # normalize angle with yaw angle (3rd state)
        rotated_final = rotate_points(final, np.pi/2 - rollout_trajectories[0, 0, 2])

        return rotated_final

    def calculate_input(self, inputs):
        # INPUT: Qn -> (32, 10, 2)
        # reshape tensor from (32, 10, 2) to (32, 20)
        dim1, dim2, dim3 = inputs.shape
        reshaped_inputs = tf.reshape(inputs, (dim1, dim2 * dim3))

        # run TSNE -> set window limits to -250/250
        # final = TSNE(n_components=2).fit_transform(reshaped_inputs)

        # run PCA -> set window limits to -10/10
        final = PCA(n_components=2).fit_transform(reshaped_inputs)

        return final

    def plot_set(self, data, axis, over_label, x_label, y_label, cluster, x_lim=None, y_lim=None, unop_data=None):
        # Scatter unoptimized data if applicable.
        if unop_data is not None:
            axis.scatter(unop_data[:, 0], unop_data[:, 1], c='k')

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
        axis.scatter(0, 0, s=100, c='purple')

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

    def plot_update(self, rollout_trajectories, Qn, unop_rt=None, unop_q=None):
        self.ax1.clear()
        self.ax2.clear()

        # Calculate the plot data
        reshaped_input = self.calculate_input(Qn)
        reshaped_output = self.calculate_output(rollout_trajectories)

        reshaped_input_unop = None if unop_rt is None else self.calculate_input(unop_q)
        reshaped_output_unop = None if unop_q is None else self.calculate_output(unop_rt)

        # Set plots
        self.plot_set(reshaped_input, self.ax1, 'Input Space', 'X-LABEL', 'Y-LABEL',
                      False, [-10, 10], [-10, 10], reshaped_input_unop)
        self.plot_set(reshaped_output, self.ax2, 'Output Space', '<-- Left | Right -->', 'Forward -->',
                      False, [-5, 5], [-5, 5], reshaped_output_unop)

        # Spacing of plots
        self.fig.tight_layout(pad=1.0)

        plt.draw()

        # CHOOSE BETWEEN THE TWO MODES
        plt.pause(0.002)
        #plt.waitforbuttonpress()

    def plot_close(self):
        plt.close()
