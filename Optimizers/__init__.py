from typing import Tuple

import numpy as np
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.others.globals_and_utils import create_rng
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


class template_optimizer:
    def __init__(
            self,
            predictor: PredictorWrapper,
            cost_function: CostFunctionWrapper,
            num_states: int,
            num_control_inputs: int,
            control_limits: Tuple[np.ndarray, np.ndarray],
            optimizer_logging: bool,
            seed: int,
            num_rollouts: int,
            mpc_horizon: int,
            predictor_specification: str,
        ) -> None:
        self.num_rollouts = num_rollouts
        self.mpc_horizon = mpc_horizon
        self.cost_function = cost_function
        self.u = 0.0
        
        # Configure predictor
        self.predictor = predictor
        self.predictor.configure(
            batch_size=self.num_rollouts, horizon=self.mpc_horizon, predictor_specification=predictor_specification
        )
        
        self.num_states = num_states
        self.num_control_inputs = num_control_inputs
        self.action_low, self.action_high = control_limits
        
        # Initialize random sampler
        self.rng = create_rng(self.__class__.__name__, seed, use_tf=True)
        
        self.logging_values = {}  # Can store trajectories and other things we want to log
        self.optimizer_logging = optimizer_logging
    
    def step(self, s: np.ndarray, time=None):
        raise NotImplementedError()
    
    def optimizer_reset(self):
        raise NotImplementedError()