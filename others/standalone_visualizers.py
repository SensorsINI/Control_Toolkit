import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_interp_pdf_cdf_vis(x_val, x_min, x_max, y_val, num_interp_pts):
    # Convert TensorFlow tensors to NumPy arrays for indexing
    x_val = x_val.numpy()
    y_val = y_val.numpy()

    # Sort the x and y
    x_sorted_indices = np.argsort(x_val)
    x_sorted = x_val[x_sorted_indices]
    y_sorted = y_val[x_sorted_indices]

    # linear interpolation
    x_interp = np.linspace(x_min, x_max, num_interp_pts)
    y_interp = np.interp(x_interp, x_sorted, y_sorted)

    # pdf
    x = x_interp
    y_pdf = y_interp / np.sum(y_interp)

    # cdf
    y_cdf = np.cumsum(y_pdf)

    return x, y_pdf, y_cdf


# VISUALIZE COLOR CODED TRAJECTORIES-----------------------------------------
# TO ACCESS LIDAR AND WAYPOINTS:
# self.cost_function.cost_function.variable_parameters.lidar_points # (40, 2)
# self.cost_function.cost_function.variable_parameters.next_waypoints  # (15, 7)
def visualize_color_coded_trajectories(trajectories, weights, unop_trajectories=None, kpf_trajectories=None,
                                       lidar_points=None, waypoints=None):
    # Create a color map for the weights (blue to red)
    cmap = plt.cm.get_cmap('RdYlBu')

    # Create a figure and axis
    fig, ax = plt.subplots()

    if unop_trajectories is not None and kpf_trajectories is not None:
        unop_trajectories = unop_trajectories[:, :, 5:7]
        kpf_trajectories = kpf_trajectories[:, :, 5:7]

    # Flatten the trajectories and weights arrays for correct association with colors
    trajectories = trajectories[:, :, 5:7]
    flat_trajectories = trajectories.numpy().reshape(-1, 2)
    flat_weights = np.repeat(weights, trajectories.shape[1])

    # Plot all points with a single scatter plot to apply the colormap correctly
    scatter = ax.scatter(flat_trajectories[:, 0], flat_trajectories[:, 1], s=2, cmap=cmap, c=flat_weights,
                         vmin=np.min(weights), vmax=np.max(weights))

    norm_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    if unop_trajectories is not None:
        # unoptimized rollout trajectories
        for i in range(unop_trajectories.shape[0]):
            ax.plot(unop_trajectories[i, :, 0],
                    unop_trajectories[i, :, 1],
                    color='black',
                    alpha=0.2)

    # color coded rollout trajectories
    for i in range(trajectories.shape[0]):
        ax.plot(trajectories[i, :, 0],
                trajectories[i, :, 1],
                color=cmap(norm_weights)[i],
                alpha=0.5)

    if kpf_trajectories is not None:
        # newly proposed trajectories
        for i in range(kpf_trajectories.shape[0]):
            ax.plot(kpf_trajectories[i, :, 0],
                    kpf_trajectories[i, :, 1],
                    color='fuchsia',
                    alpha=0.5,
                    lw=1)

    # Plot the starting point of the first trajectory (black point)
    ax.scatter(trajectories[0, 0, 0], trajectories[0, 0, 1], s=100, color='purple')

    if lidar_points is not None:
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c='black', marker='P', s=20)

    if waypoints is not None:
        ax.scatter(waypoints[:, 1], waypoints[:, 2], c='green', marker='D', s=20)

    if unop_trajectories is not None and kpf_trajectories is not None:
        combined_trajectories = np.concatenate((trajectories, unop_trajectories, kpf_trajectories))
    else:
        combined_trajectories = trajectories

    ax.set_xlim(np.min(combined_trajectories[:, :, 0]), np.max(combined_trajectories[:, :, 0]))
    ax.set_ylim(np.min(combined_trajectories[:, :, 1]), np.max(combined_trajectories[:, :, 1]))

    # Set plot title and color bar
    ax.set_title('Trajectories with Color-Coded Points Based on Weight')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Weight')

    # Show the plot
    plt.show(block=False)

    plt.waitforbuttonpress()
    # plt.pause(2)
    plt.close()


# VISUALIZE DISTRIBUTION WITH WEIGHTS
def visualize_control_input_distributions(action_low, action_high, weights, mpc_horizon, num_interp_pts, Qn,
                                          Q_kpf=None):
    # Define the limits for the 2D space visualization
    ac_min, ac_max = action_low[0], action_high[0]
    tc_min, tc_max = action_low[1], action_high[1]
    y_min = 0
    y_max = np.max(weights)

    num_timesteps = mpc_horizon

    # Create a figure and two subplots (one for AC control input and one for TC control input)
    fig, (ax_ac, ax_tc) = plt.subplots(1, 2, figsize=(10, 5))

    for timestep in range(num_timesteps):
        # Set the x-axis limits and labels for both subplots
        ax_ac.set_xlim(ac_min, ac_max)
        ax_ac.set_xlabel('Angular Control (AC)')
        ax_tc.set_xlim(tc_min, tc_max)
        ax_tc.set_xlabel('Translational Control (TC)')

        # Set the y-axis limits and labels for both subplots
        ax_ac.set_ylim(y_min, y_max)
        ax_ac.set_ylabel('Weight')
        ax_tc.set_ylim(y_min, y_max)
        ax_tc.set_ylabel('Weight')

        # Get control inputs and weights for the current timestep
        control_inputs = Qn[:, timestep, :]

        # Extract AC and TC control inputs and corresponding weights
        ac_control_inputs = control_inputs[:, 0]
        tc_control_inputs = control_inputs[:, 1]

        # Update the bar plots for AC and TC control inputs with corresponding weights on the y-axes
        ax_ac.bar(ac_control_inputs, weights, width=(ac_max - ac_min) / 200, label=f'Timestep {timestep + 1}',
                  align='center', alpha=0.5)
        ax_ac.scatter(ac_control_inputs, weights, color='black', s=10, zorder=2)

        ax_tc.bar(tc_control_inputs, weights, width=(tc_max - tc_min) / 200, label=f'Timestep {timestep + 1}',
                  align='center', alpha=0.5)
        ax_tc.scatter(tc_control_inputs, weights, color='black', s=10, zorder=2)

        if Q_kpf is not None:
            y = tf.ones(shape=(Q_kpf.shape[0],)) * ((y_max - y_min) * 0.1 + y_min)
            ax_ac.scatter(Q_kpf[:, timestep, 0], y, color='purple', s=10, zorder=2)
            ax_tc.scatter(Q_kpf[:, timestep, 1], y, color='purple', s=10, zorder=2)

        (
            ac_ci_interp,
            ac_pdf_interp,
            ac_cdf_interp
        ) = get_interp_pdf_cdf_vis(ac_control_inputs, ac_min, ac_max, weights, num_interp_pts)

        (
            tc_ci_interp,
            tc_pdf_interp,
            tc_cdf_interp
        ) = get_interp_pdf_cdf_vis(tc_control_inputs, tc_min, tc_max, weights, num_interp_pts)

        # plot pdf
        ac_pdf_image = ac_pdf_interp * (y_max - y_min) + y_min
        tc_pdf_image = tc_pdf_interp * (y_max - y_min) + y_min
        ax_ac.plot(ac_ci_interp, ac_pdf_image, color='green', label='INT. PDF')
        ax_tc.plot(tc_ci_interp, tc_pdf_image, color='green', label='INT. PDF')

        # normalize and fit the cdf
        ac_cdf_image = ac_cdf_interp / ac_cdf_interp[-1] * (y_max - y_min) + y_min
        tc_cdf_image = tc_cdf_interp / tc_cdf_interp[-1] * (y_max - y_min) + y_min

        # plot cdf
        ax_ac.plot(ac_ci_interp, ac_cdf_image, color='red', label='CDF')
        ax_tc.plot(tc_ci_interp, tc_cdf_image, color='red', label='CDF')

        # Update the plot titles for both subplots
        ax_ac.set_title(f'AC Control Input vs. Weight (Timestep {timestep + 1})')
        ax_tc.set_title(f'TC Control Input vs. Weight (Timestep {timestep + 1})')

        # Show the plot
        plt.waitforbuttonpress()

        # Clear the subplots for the next timestep
        ax_ac.clear()
        ax_tc.clear()

    # Close the figure after all timesteps are shown
    plt.close()

    """x_min, x_max = self.action_low[0], self.action_high[0]
    y_min, y_max = self.action_low[1], self.action_high[1]
    num_timesteps = self.mpc_horizon

    # Create a color map for the weights (blue to red)
    cmap = plt.cm.get_cmap('RdYlBu')

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Angular Control')
    ax.set_ylabel('Translational Control')

    # Set plot title and color bar
    ax.set_title('Timestep: 0')

    # Plot empty scatter points for initialization
    sc = ax.scatter([], [], c=[], cmap=cmap, vmin=np.min(self.kpf_weights),
                    vmax=np.max(self.kpf_weights))

    # Create the color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Weight (More similar - red, Less similar - blue)')

    for timestep in range(num_timesteps):
        # Get control inputs and weights for the current timestep
        control_inputs = Qn[:, timestep, :]
        weights = self.kpf_weights

        # Update the scatter plot data with control_inputs and colors
        sc.set_offsets(control_inputs)
        sc.set_array(weights)

        # Update the plot title
        ax.set_title(f'Timestep: {timestep + 1}')

        # Show the plot
        plt.waitforbuttonpress()  # Add a pause to show the plot for a short time

    # Close the figure after all timesteps are shown
    plt.close(fig)"""
