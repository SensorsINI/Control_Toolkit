from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import TensorType
import numpy as np
from gym.spaces.box import Box
from Control_Toolkit.Controllers import template_controller

from Control_Toolkit.others.globals_and_utils import create_rng
from Control_Toolkit_ASF.Cost_Functions import cost_function_base


class template_optimizer:
    def __init__(self, controller: template_controller, predictor: PredictorWrapper, cost_function: cost_function_base, predictor_specification: str, action_space: Box, observation_space: Box, seed: int, num_rollouts: int, mpc_horizon: int, optimizer_logging: bool) -> None:
        self.num_rollouts = num_rollouts
        self.mpc_horizon = mpc_horizon
        self.cost_function = cost_function
        self.u = 0.0
        
        # Configure predictor
        self.predictor = predictor
        self.predictor.configure(
            batch_size=self.num_rollouts, horizon=self.mpc_horizon, predictor_specification=predictor_specification
        )
        
        # Infer number of states and actions and their limits from the action and observation spaces
        self.action_space = action_space
        self.observation_space = observation_space
        assert len(action_space.shape) == 1, "Only vector action space currently supported"
        self.num_control_inputs = action_space.shape[0]
        if observation_space is not None:
            assert len(observation_space.shape) == 1, "Only vector observation space currently supported"
            self.num_states = observation_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # Initialize random sampler
        self.rng = create_rng(self.__class__.__name__, seed, use_tf=True)
        
        self.controller = controller  # Reference to controller instance using this optimizer
        self.optimizer_logging = optimizer_logging
    
    def send_logs_to_controller(self, logging_values: dict[str, TensorType]):
        # Very basic observer pattern: If logging is enabled, notify controller of logging values
        self.controller.update_logs(logging_values)
    
    def step(self, s: np.ndarray, time=None):
        raise NotImplementedError()
    
    def optimizer_reset(self):
        raise NotImplementedError()