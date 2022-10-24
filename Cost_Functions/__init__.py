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
    
    def get_terminal_cost(self, s_hor: TensorType):
        raise NotImplementedError()

    def get_stage_cost(self, s: TensorType, u: TensorType, u_prev: TensorType):
        raise NotImplementedError()

    def get_trajectory_cost(
        self, s_hor: TensorType, u: TensorType, u_prev: TensorType = None
    ):
        # Helper function which computes the summed cost of a trajectory
        # Can be overwritten in a subclass, e.g. if weighted sum is required
        stage_cost = self.get_stage_cost(s_hor[:, 1:, :], u, u_prev)
        total_cost = self.lib.sum(stage_cost, 1)
        total_cost = total_cost + self.get_terminal_cost(s_hor)
        return total_cost

    def set_computation_library(self, ComputationLib: "type[ComputationLibrary]"):
        assert isinstance(ComputationLib, type), "Need to set a library of type[ComputationLibrary]"
        if not ComputationLib in self.supported_computation_libraries:
            raise ValueError(f"The cost function {self.__class__.__name__} does not support {ComputationLib}")
        self.lib = ComputationLib
