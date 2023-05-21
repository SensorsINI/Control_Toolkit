from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

import numpy as np
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.others.globals_and_utils import create_rng
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


class template_optimizer:
    supported_computation_libraries = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}
    
    def __init__(
            self,
            predictor: PredictorWrapper,
            cost_function: CostFunctionWrapper,
            control_limits: "Tuple[np.ndarray, np.ndarray]",
            optimizer_logging: bool,
            seed: int,
            num_rollouts: int,
            mpc_horizon: int,
            computation_library: "type[ComputationLibrary]",
        ) -> None:

        # Check if the computation_library passed is compatible with this optimizer
        if computation_library not in self.supported_computation_libraries:
            raise ValueError(f"The optimizer {self.__class__.__name__} does not support {computation_library.__name__}")

        self.lib = computation_library

        self.num_rollouts = num_rollouts
        self.mpc_horizon = mpc_horizon
        self.cost_function = cost_function
        self.u = 0.0
        
        # Configure predictor
        self.predictor = predictor
        
        self.num_states = None
        self.num_control_inputs = None
        self.action_low, self.action_high = control_limits
        self.action_low = self.lib.to_tensor(self.action_low, self.lib.float32)
        self.action_high = self.lib.to_tensor(self.action_high, self.lib.float32)
        
        # Initialize random sampler
        self.rng = create_rng(self.__class__.__name__, seed, computation_library=computation_library)
        
        self.logging_values = {}  # Can store trajectories and other things we want to log
        self.optimizer_logging = optimizer_logging
    
    def configure(self,
                  num_states: int,
                  num_control_inputs: int,
                  default_configure: bool = True,
                  ) -> None:

        self.num_states = num_states
        self.num_control_inputs = num_control_inputs

        if default_configure:
            self.optimizer_reset()

        """Pass any additional arguments from the controller to the optimizer."""
    
    def step(self, s: np.ndarray, time=None):
        raise NotImplementedError("Implement this function in a subclass.")
    
    def optimizer_reset(self):
        raise NotImplementedError("Implement this function in a subclass.")

    @property
    def optimizer_name(self):
        name = self.__class__.__name__
        if name != "template_optimizer":
            return name.replace("optimizer_", "").replace("_", "-").lower()
        else:
            raise AttributeError()
