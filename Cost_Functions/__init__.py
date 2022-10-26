from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary, TensorType
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import get_logger

logger = get_logger(__name__)


class cost_function_base:
    # Default: Class supports all libs to compute costs
    supported_computation_libraries = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}
    
    def __init__(self, controller: template_controller, ComputationLib: "type[ComputationLibrary]") -> None:
        self.controller = controller
        self.set_computation_library(ComputationLib)
    
    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        """Compute a batch of terminal costs for a batch of terminal states.

        :param terminal_states: Has shape [batch_size, num_states]
        :type terminal_states: TensorType
        :return: The terminal costs. Has shape [batch_size]
        :rtype: TensorType
        """
        raise NotImplementedError("To be implemented in subclass.")

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        """Compute all stage costs of a batch of states and contol inputs.
        One "stage" is one step in the MPC horizon.

        :param states: Has shape [batch_size, mpc_horizon, num_states]
        :type states: TensorType
        :param inputs: Has shape [batch_size, mpc_horizon, num_control_inputs]
        :type inputs: TensorType
        :param previous_input: The actually most recently applied control input
        :type previous_input: TensorType
        :return: The stage costs. Has shape [batch_size, mpc_horizon]
        :rtype: TensorType
        """
        raise NotImplementedError("To be implemented in subclass.")

    def get_trajectory_cost(
        self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None
    ) -> TensorType:
        """Helper function which computes a batch of the summed cost of a trajectory.
        Can be overwritten in a subclass, e.g. if weighted sum is required.
        The batch dimension is used to compute for multiple rollouts in parallel.

        :param state_horizon: Has shape [batch_size, mpc_horizon+1, num_states]
        :type state_horizon: TensorType
        :param inputs: A batch of control inputs. Has shape [batch_size, mpc_horizon, num_control_inputs]
        :type inputs: TensorType
        :param previous_input: The most recent actually applied control, defaults to None
        :type previous_input: TensorType, optional
        :return: The summed cost of the trajectory. Has shape [batch_size].
        :rtype: TensorType
        """
        stage_cost = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        total_cost = self.lib.sum(stage_cost, 1)  # Sum across the MPC horizon dimension
        total_cost = total_cost + self.get_terminal_cost(state_horizon[:, -1, :])
        return total_cost

    def set_computation_library(self, ComputationLib: "type[ComputationLibrary]"):
        assert isinstance(ComputationLib, type), "Need to set a library of type[ComputationLibrary]"
        if not ComputationLib in self.supported_computation_libraries:
            raise ValueError(f"The cost function {self.__class__.__name__} does not support {ComputationLib.__name__}")
        self.lib = ComputationLib
