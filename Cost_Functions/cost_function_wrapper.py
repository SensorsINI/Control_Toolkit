from importlib import import_module
import os

from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import TensorType
import yaml
from copy import deepcopy as dcp
from types import MappingProxyType
from Control_Toolkit.Controllers import template_controller

from Control_Toolkit.Cost_Functions import cost_function_base
from others.globals_and_utils import load_or_reload_config_if_modified

log=get_logger(__name__)

# cost_function config
(cost_function_config,_) = load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml"))


class CostFunctionWrapper:
    def __init__(self):
        self.cost_function = None
        self.cost_function_name_default: str = cost_function_config[
            "cost_function_name_default"
        ]
        log.info(f'default cost function name is {self.cost_function_name_default}')

    def configure(
        self,
        controller: template_controller,
        cost_function_specification=None,
        config:dict=None
    ):
        """
        Configures the cost function. TODO This lazy constructor is needed why?

        :param controller: the controller that uses this cost function
        :param cost_function_specification: the string name of the cost function class, to construct the class
        :param config: the config dict() that holds all the configuration values

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
        log.info(f'configured controller {controller.__class__} with cost function {self.cost_function.__class__}')

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
        self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None, config:dict=None
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
