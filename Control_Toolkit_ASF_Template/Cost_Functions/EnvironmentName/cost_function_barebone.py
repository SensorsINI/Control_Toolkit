"""
Every environment, be it a simulated or physical one, can be controlled using different cost functions.
A cost function is specific to an environment, which is why each cost function module is grouped by environment name within Control_Toolkit_ASF/Cost_Functions.
"""
import os
from Control_Toolkit.Cost_Functions import cost_function_base

from others.globals_and_utils import load_config

from CartPole.cartpole_model import TrackHalfLength
from CartPole.state_utilities import ANGLE_IDX, POSITION_IDX

# TODO: Load constants from the cost config file, like this:
config = load_config(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))

# TODO: Rename parent folder from EnvironmentName to the actual name of you environment
# TODO: Load constants like this:
# dd_weight = config["EnvironmentName"]["default"]["dd_weight"]
# cc_weight = config["EnvironmentName"]["default"]["cc_weight"]
# ep_weight = config["EnvironmentName"]["default"]["ep_weight"]
# ccrc_weight = config["EnvironmentName"]["default"]["ccrc_weight"]
# R = config["EnvironmentName"]["default"]["R"]


class cost_function_barebone(cost_function_base):
    """This class can contain arbitrary helper functions to compute the cost of a trajectory or inputs."""
    # Example: Cost for difference from upright position
    # def E_pot_cost(self, angle):
    #     """Compute penalty for not balancing pole upright (penalize large angles)"""
    #     return self.controller.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2
    # Example: Actuation cost
    # def CC_cost(self, u):
    #     return R * self.lib.sum(u**2, 2)

    # final stage cost
    def get_terminal_cost(self, s):
        terminal_state = s[:, -1, :]  # Terminal state has shape (batch_size, state_dim)
        # TODO: Compute terminal cost
        # return terminal_cost

    # all stage costs together
    def get_stage_cost(self, s, u, u_prev):
        # Shape of input s: (batch_size, horizon, state_dim)
        # TODO: Compute stage cost
        # return stage_cost
        pass

    # total cost of the trajectory
    def get_trajectory_cost(self, s_hor, u, u_prev=None):
        # TODO: Sum of stage costs + terminal cost already implemented in base class
        # but could be overwritten here, e.g. sum with exponentially decaying weights
        # stage_cost = self.get_stage_cost(s_hor[:, 1:, :], u, u_prev)
        # total_cost = self.lib.sum(gamma*stage_cost, 1)
        # return total_cost
        pass
