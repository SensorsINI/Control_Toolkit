import os
from importlib import import_module
from typing import Optional
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import yaml
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box

from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.computation_library import (NumpyLibrary, PyTorchLibrary,
                                                TensorFlowLibrary, TensorType)
from Control_Toolkit.others.globals_and_utils import get_logger, import_optimizer_by_name


config_optimizer = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")), Loader=yaml.FullLoader)
config_controller = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)


logger = get_logger(__name__)


class controller_mpc_tf(template_controller):
    _computation_library = TensorFlowLibrary
    _has_optimizer = True
    
    def configure(self, optimizer_name: Optional[str]=None):
        if optimizer_name in {None, ""}:
            optimizer_name = str(config_controller["mpc-tf"]["optimizer"])
            logger.info(f"Using default optimizer {optimizer_name} specified in config file")
        
        # Create cost function
        cost_function_name: str = config_cost_function["cost_function_name"]
        cost_function_module = import_module(f"Control_Toolkit_ASF.Cost_Functions.{self.environment_name}.{cost_function_name}")
        self.cost_function: cost_function_base = getattr(cost_function_module, cost_function_name)(self, self.computation_library)
        
        # Create predictor
        self.predictor = PredictorWrapper()
        
        # MPC Controller always has an optimizer
        Optimizer = import_optimizer_by_name(optimizer_name)
        self.optimizer: template_optimizer = Optimizer(
            controller=self,
            predictor=self.predictor,
            cost_function=self.cost_function,
            action_space=self.action_space,
            observation_space=self.observation_space,
            optimizer_logging=self.controller_logging,
            **config_optimizer[optimizer_name],
        )
        
    def step(self, s: np.ndarray, time=None, updated_attributes: dict[str, TensorType]={}):
        self.update_attributes(updated_attributes)
        return self.optimizer.step(s, time)

    def controller_reset(self):
        self.optimizer.optimizer_reset()
        