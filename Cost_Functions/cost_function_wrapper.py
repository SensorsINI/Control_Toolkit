from importlib import import_module
import os
from SI_Toolkit.computation_library import TensorType
import yaml
from copy import deepcopy as dcp
from types import MappingProxyType
from Control_Toolkit.Controllers import template_controller

from Control_Toolkit.Cost_Functions import cost_function_base

# cost_function config
cost_function_config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)


class CostFunctionWrapper:
    """
    Wrapper class for cost functions.
    It allows the creation of an instance with deferred specification which specific class containing cost functions is to be used.
    
    Usage:
    1) Instantiate this wrapper in controller.
    2) Call `configure` with a reference to the controller instance and the name of the cost function, once this is known to the controller.
    
    You can do both steps 
    """
    def __init__(self):
        self.cost_function = None
        self.cost_function_name_default: str = cost_function_config[
            "cost_function_name_default"
        ]

    def configure(
        self,
        controller: template_controller,
        cost_function_specification: str=None,
    ):
        """Configure this wrapper as part of a controller.

        :param controller: Reference to the controller instance this wrapper is used by
        :type controller: template_controller
        :param cost_function_specification: Name of cost function class within Control_Toolkit_ASF.Cost_Functions.<<Environment_Name>> package. If not specified, the 'default' one is used.
        :type cost_function_specification: str, optional
        """
        environment_name = controller.environment_name
        computation_library = controller.computation_library  # Use library dictated by controller
        
        # Set cost function attributes from given specification. Resort to defaults if required.
        self.update_cost_function_name_from_specification(
            environment_name, cost_function_specification
        )

        cost_function_module = import_module(
            f"Control_Toolkit_ASF.Cost_Functions.{environment_name}.{self.cost_function_name}"
        )
        self.cost_function: cost_function_base = getattr(
            cost_function_module, self.cost_function_name
        )(controller, computation_library)

    def update_cost_function_name_from_specification(
        self, environment_name: str, cost_function_specification: str = None
    ):
        if cost_function_specification is None:
            self.cost_function_name = self.cost_function_name_default.replace("-", "_")
        elif isinstance(cost_function_specification, str):
            self.cost_function_name = cost_function_specification.replace("-", "_")
        else:
            raise ValueError(
                f"Cannot interpret cost function specification {cost_function_specification}."
            )

    def get_terminal_cost(self, terminal_states: TensorType):
        """Refer to :func:`the base cost function <Control_Toolkit.Cost_Functions.cost_function_base.get_terminal_cost>`"""
        return self.cost_function.get_terminal_cost(terminal_states)

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType):
        """Refer to :func:`the base cost function <Control_Toolkit.Cost_Functions.cost_function_base.get_stage_cost>`"""
        return self.cost_function.get_stage_cost(states, inputs, previous_input)

    def get_trajectory_cost(
        self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None
    ):
        """Refer to :func:`the base cost function <Control_Toolkit.Cost_Functions.cost_function_base.get_trajectory_cost>`"""
        return self.cost_function.get_trajectory_cost(state_horizon, inputs, previous_input)

    def copy(self):
        """
        Makes a copy of a cost_function, specification get preserved, except configuration
        The cost function needs to be reconfigured, however the specification needs not to be provided.
        """
        cost_function_copy = CostFunctionWrapper()
        cost_function_copy.cost_function_name = self.cost_function_name
        return cost_function_copy
