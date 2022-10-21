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
    def __init__(self):
        self.cost_function = None
        self.cost_function_name_default: str = cost_function_config[
            "cost_function_name_default"
        ]

    def configure(
        self,
        controller: template_controller,
        cost_function_specification=None,
    ):
        environment_name = controller.environment_name
        computation_library = controller.computation_library
        
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

    def get_terminal_cost(self, s_hor: TensorType):
        return self.cost_function.get_terminal_cost(s_hor)

    def get_stage_cost(self, s: TensorType, u: TensorType, u_prev: TensorType):
        return self.cost_function.get_stage_cost(s, u, u_prev)

    def get_trajectory_cost(
        self, s_hor: TensorType, u: TensorType, u_prev: TensorType = None,
    ):
        return self.cost_function.get_trajectory_cost(s_hor, u, u_prev)

    def copy(self):
        """
        Makes a copy of a cost_function, specification get preserved, except configuration
        The cost function needs to be reconfigured, however the specification needs not to be provided.
        """
        cost_function_copy = CostFunctionWrapper()
        cost_function_copy.cost_function_name = self.cost_function_name
        return cost_function_copy
