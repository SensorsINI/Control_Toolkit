from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary, TensorType
from Control_Toolkit.others.globals_and_utils import get_logger
from types import SimpleNamespace

logger = get_logger(__name__)


class cost_function_base:
    # Default: Class supports all libs to compute costs
    supported_computation_libraries = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}
    # Define default values used for cost normalization
    MIN_COST = -1.0
    MAX_COST = 0.0
    COST_RANGE = MAX_COST - MIN_COST
    
    def __init__(self, variable_parameters: SimpleNamespace, ComputationLib: "type[ComputationLibrary]") -> None:
        self.variable_parameters = variable_parameters
        self.set_computation_library(ComputationLib)

        self.batch_size = None
        self.horizon = None


    def configure(
            self,
            batch_size: int,
            horizon: int,
    ):
        self.batch_size = batch_size
        self.horizon = horizon

    
    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        """Compute a batch of terminal costs for a batch of terminal states.

        :param terminal_states: Has shape [batch_size, num_states]
        :type terminal_states: TensorType
        :return: The terminal costs. Has shape [batch_size]
        :rtype: TensorType
        """
        # Default behavior: Return a zero cost scalar per sample of batch
        return self.lib.zeros_like(terminal_states)[:,:1]  # Shape: (batch_size x 1)

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        """Compute all stage costs of a batch of states and contol inputs.
        One "stage" is one step in the MPC horizon.
        Stage costs are shifted so that they are <= 0. Reason: reward = -cost is then >= 0 and therefore easier to interpret.

        :param states: Has shape [batch_size, mpc_horizon, num_states]
        :type states: TensorType
        :param inputs: Has shape [batch_size, mpc_horizon, num_control_inputs]
        :type inputs: TensorType
        :param previous_input: The actually most recently applied control input
        :type previous_input: TensorType
        :return: The stage costs. Has shape [batch_size, mpc_horizon]
        :rtype: TensorType
        """
        stage_costs = self._get_stage_cost(states, inputs, previous_input)  # Select all but last state of the horizon
        return stage_costs - self.MAX_COST
        # Could also normalize to [-1, 0]:
        # (stage_costs - self.MIN_COST) / self.COST_RANGE - 1 
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
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
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        terminal_cost = self.lib.reshape(self.get_terminal_cost(state_horizon[:, -1, :]), (-1, 1))
        total_cost = self.lib.mean(self.lib.concat([stage_costs, terminal_cost], 1), 1)  # Average across the MPC horizon dimension
        return total_cost

    def set_computation_library(self, ComputationLib: "type[ComputationLibrary]"):
        assert isinstance(ComputationLib, type), "Need to set a library of type[ComputationLibrary]"
        if not ComputationLib in self.supported_computation_libraries:
            raise ValueError(f"The cost function {self.__class__.__name__} does not support {ComputationLib.__name__}")
        self.lib = ComputationLib
