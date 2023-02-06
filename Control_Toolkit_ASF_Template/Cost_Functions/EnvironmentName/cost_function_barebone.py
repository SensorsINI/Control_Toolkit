"""
Every environment, be it a simulated or physical one, can be controlled using different cost functions.
A cost function is specific to an environment, which is why each cost function module is grouped by environment name within Control_Toolkit_ASF/Cost_Functions.
"""
import os
from SI_Toolkit.computation_library import TensorType
import yaml
from Control_Toolkit.Cost_Functions import cost_function_base


# TODO: Load constants from the cost config file, like this:
config = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"), Loader=yaml.FullLoader)

# TODO: Rename parent folder from EnvironmentName to the actual name of you environment
# TODO: Load constants like this:
# dd_weight = config["EnvironmentName"]["default"]["dd_weight"]
# cc_weight = config["EnvironmentName"]["default"]["cc_weight"]
# ep_weight = config["EnvironmentName"]["default"]["ep_weight"]
# ccrc_weight = config["EnvironmentName"]["default"]["ccrc_weight"]
# R = config["EnvironmentName"]["default"]["R"]


class cost_function_barebone(cost_function_base):
    """This class can contain arbitrary helper functions to compute the cost of a trajectory or inputs."""
    MAX_COST = 0.0  # Define maximum value the cost can take. Used for shifting
    
    # Example: Cost for difference from upright position
    # def _E_pot_cost(self, angle):
    #     """Compute penalty for not balancing pole upright (penalize large angles)"""
    #     return self.controller.target_equilibrium * 0.25 * (1.0 - self.lib.cos(angle)) ** 2
    # Example: Actuation cost
    # def _CC_cost(self, u):
    #     return R * self.lib.sum(u**2, 2)

    # final stage cost
    def get_terminal_cost(self, terminal_states: TensorType):
        # Terminal state has shape [batch_size, num_states]
        # TODO: Compute terminal cost here
        # return terminal_cost
        pass

    # all stage costs together
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        # Shape of states: [batch_size, mpc_horizon, num_states]
        # TODO: Compute stage cost
        # return stage_cost
        pass

    # total cost of the trajectory
    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None):
        # TODO: Sum of stage costs + terminal cost already implemented in base class
        # but could be overwritten here, e.g. sum with exponentially decaying weights
        # stage_cost = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)
        # gamma = np.array([0.99 ** i for i in range(state_horizon.shape[1])])
        # total_cost = self.lib.sum(gamma*stage_cost, 1)
        # return total_cost
        pass
