import os
from typing import Optional
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import yaml
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper

from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.others.globals_and_utils import get_logger, import_optimizer_by_name


config_optimizers = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)
logger = get_logger(__name__)


class controller_mpc(template_controller):
    _has_optimizer = True
    
    def configure(self, optimizer_name: Optional[str]=None, predictor_specification: Optional[str]=None):
        if optimizer_name in {None, ""}:
            optimizer_name = str(self.config_controller["optimizer"])
            logger.info(f"Using default optimizer {optimizer_name} specified in config file")
        config_optimizer = config_optimizers[optimizer_name]
        
        # Create cost function
        cost_function_specification = self.config_controller.get("cost_function_specification", None)
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
            computation_library=self.computation_library,
            **config_optimizer,
        )
        # Some optimizers require additional controller parameters (e.g. predictor_specification or dt) to be fully configured.
        # Do this here. If the optimizer does not require any additional parameters, it will ignore them.
        self.optimizer.configure(dt=self.config_controller["dt"], predictor_specification=predictor_specification)
        
        self.predictor.configure(
            batch_size=self.optimizer.num_rollouts,
            horizon=self.optimizer.mpc_horizon,
            dt=self.config_controller["dt"],
            computation_library=self.computation_library,
            predictor_specification=predictor_specification
        )
        
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)
        u = self.optimizer.step(s, time)
        self.update_logs(self.optimizer.logging_values)
        return u

    def controller_reset(self):
        self.optimizer.optimizer_reset()
        