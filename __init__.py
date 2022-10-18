import os
from importlib import import_module

import numpy as np
import yaml
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.environment import (EnvironmentBatched,
                                                NumpyLibrary, PyTorchLibrary,
                                                TensorFlowLibrary)
from Control_Toolkit.others.globals_and_utils import import_controller_by_name


config_controller = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)


class Planner:
    def __init__(self, controller_name: str, environment_name: str, action_space: Box, observation_space: Box) -> None:
        """Parse config data to instantiate predictor and cost function, then create a controller with them."""
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Determine the library used by the environment
        if controller_name.endswith("tf"):
            ComputationLib = TensorFlowLibrary
        elif controller_name.endswith("pytorch"):
            ComputationLib = PyTorchLibrary
        else:
            ComputationLib = NumpyLibrary
        
        
        # Create cost function
        cost_function_name: str = config_cost_function["cost_function_name"]
        cost_function_module = import_module(f"Control_Toolkit_ASF.Cost_Functions.{environment_name}.{cost_function_name}")
        self.cost_function: cost_function_base = getattr(cost_function_module, cost_function_name)(ComputationLib)
        
        # Create controller
        Controller = import_controller_by_name(controller_name)
        
        if Controller is None:
            self.controller = None
        else:
            self.controller: template_controller = Controller(
                cost_function=self.cost_function,
                action_space=self.action_space,
                observation_space=self.observation_space,
                **config_controller[controller_name],
            )
    
    def step(self, s: np.ndarray, time=None):
        return self.controller.step(s, time)

    def controller_reset(self):
        self.controller.controller_reset()