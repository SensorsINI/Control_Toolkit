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

from torch import inference_mode


config_optimizers = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)
logger = get_logger(__name__)


class controller_mpc(template_controller):
    _has_optimizer = True
    
    def configure(self, optimizer_name: Optional[str]=None, predictor_specification: Optional[str]=None):
        if optimizer_name in {None, ""}:
            optimizer_name = str(self.config_controller["optimizer"])
            logger.info(f"Using optimizer {optimizer_name} specified in controller config file")
        if predictor_specification in {None, ""}:
            predictor_specification: Optional[str] = self.config_controller.get("predictor_specification", None)
            logger.info(f"Using predictor {predictor_specification} specified in controller config file")
        
        config_optimizer = config_optimizers[optimizer_name]
        
        # Create cost function
        cost_function_specification = self.config_controller.get("cost_function_specification", None)
        self.cost_function = CostFunctionWrapper()
        
        # Create predictor
        self.predictor = PredictorWrapper()

        # The logic of initializing and configuring:
        # predictor to be configured needs num_rollout (=batch size) and mpc horizon
        # which is an attribute of optimizer
        # -> predictor configure goes after initializing optimizer
        # optimizer to be fully functional needs num_states and num_control_inputs
        # which is an attribute of system model, hence predictor
        # -> optimizer configure goes after
        # Hence splitting __init__ method into propoer init and configure
        # Allows to solve the chicken-and-egg problem.

        # MPC Controller always has an optimizer
        Optimizer = import_optimizer_by_name(optimizer_name)
        self.optimizer: template_optimizer = Optimizer(
            predictor=self.predictor,
            cost_function=self.cost_function,
            control_limits=self.control_limits,
            optimizer_logging=self.controller_logging,
            computation_library=self.computation_library,
            calculate_optimal_trajectory=self.config_controller.get('calculate_optimal_trajectory'),
            **config_optimizer,
        )

        self.predictor.configure(
            batch_size=self.optimizer.num_rollouts,
            horizon=self.optimizer.mpc_horizon,
            dt=self.config_controller["dt"],
            computation_library=self.computation_library,
            variable_parameters=self.variable_parameters,
            predictor_specification=predictor_specification,
        )

        self.cost_function.configure(
            batch_size=self.optimizer.num_rollouts,
            horizon=self.optimizer.mpc_horizon,
            variable_parameters=self.variable_parameters,
            environment_name=self.environment_name,
            computation_library=self.computation_library,
            cost_function_specification=cost_function_specification
        )

        self.optimizer.configure(
            dt=self.config_controller["dt"],
            predictor_specification=predictor_specification,
            num_states=self.predictor.num_states,
            num_control_inputs=self.predictor.num_control_inputs,
        )

        if self.lib.lib == 'Pytorch':
            self.step = inference_mode()(self.step)
        else:
            self.step = self.step

        
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)
        u = self.optimizer.step(s, time)
        self.update_logs(self.optimizer.logging_values)
        return u

    def controller_reset(self):
        self.optimizer.optimizer_reset()
        