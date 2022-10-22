import os
from typing import Optional
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import yaml
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper

from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.computation_library import (NumpyLibrary, PyTorchLibrary,
                                                TensorFlowLibrary, TensorType)
from Control_Toolkit.others.globals_and_utils import get_logger, import_optimizer_by_name


config_optimizer = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")), Loader=yaml.FullLoader)
config_controller = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)


computation_library = str(config_controller["mpc"]["computation_library"])
cost_function_specification = dict(config_controller["mpc"]).get("cost_function_specification", None)
logger = get_logger(__name__)


class controller_mpc(template_controller):
    _has_optimizer = True
    # Assign computation library
    if "tensorflow" in computation_library.lower():
        _computation_library = TensorFlowLibrary
    elif "pytorch" in computation_library.lower():
        _computation_library = PyTorchLibrary
    elif "numpy" in computation_library.lower():
        _computation_library = NumpyLibrary
    else:
        logger.warning(f"Found unknown spec {computation_library} for computation library in MPC controller. Using default numpy.")
        _computation_library = NumpyLibrary
    
    def configure(self, optimizer_name: Optional[str]=None):
        if optimizer_name in {None, ""}:
            optimizer_name = str(config_controller["mpc"]["optimizer"])
            logger.info(f"Using default optimizer {optimizer_name} specified in config file")
        
        # Create cost function
        self.cost_function = CostFunctionWrapper()
        self.cost_function.configure(self, cost_function_specification=cost_function_specification)
        
        # Create predictor
        self.predictor = PredictorWrapper()
        
        # MPC Controller always has an optimizer
        Optimizer = import_optimizer_by_name(optimizer_name)
        self.optimizer: template_optimizer = Optimizer(
            predictor=self.predictor,
            cost_function=self.cost_function,
            num_states=self.num_states,
            num_control_inputs=self.num_control_inputs,
            control_limits=self.control_limits,
            optimizer_logging=self.controller_logging,
            **config_optimizer[optimizer_name],
        )
        
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)
        u = self.optimizer.step(s, time)
        self.update_logs(self.optimizer.logging_values)
        return u

    def controller_reset(self):
        self.optimizer.optimizer_reset()
        