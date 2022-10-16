from importlib import import_module
from typing import Any
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary, TensorFlowLibrary

from Control_Toolkit.others.globals_and_utils import get_controller, import_controller_by_name
from Control_Toolkit_ASF.Cost_Functions import cost_function_base


def build_predictor(*args):
    raise NotImplementedError()


class Planner(EnvironmentBatched):
    def __init__(self,
        config_predictor: dict[str, Any],
        config_cost_function: dict[str, Any],
        config_controller: dict[str, Any]
    ) -> None:
        """Parse config data to instantiate predictor and cost function, then create a controller with them."""
        # Build predictor
        predictor = build_predictor(**config_predictor)
        
        # Create cost function
        cost_function_name: str = config_cost_function["cost_function_name"]
        cost_function_name = cost_function_name.replace("-", "_")
        cost_function_module = import_module(f"Control_Toolkit_ASF.Cost_Functions.{cost_function_name}")
        self.cost_function: cost_function_base = getattr(cost_function_module, cost_function_name)()
        
        # Create controller
        controller_name = config_controller["controller_name"]
        Controller = import_controller_by_name(controller_name)
        
        if controller_name[-2:] == "tf":
            self.cost_function.set_computation_library(TensorFlowLibrary)
        else:
            self.cost_function.set_computation_library(NumpyLibrary)
        
        if Controller is None:
            self.controller = None
        else:
            self.controller: template_controller = Controller(
                predictor=predictor,
                cost_function=self.cost_function,
                action_space=self.action_space,
                observation_space=self.observation_space,
                **config_controller[controller_name],
            )
        