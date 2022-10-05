from abc import ABC, abstractmethod

import numpy as np
from Control_Toolkit.Cost_Functions import cost_function_default
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import create_rng
from gym.spaces.box import Box
from SI_Toolkit.Predictors import predictor

"""
For a controller to be found and imported by CartPoleGUI/DataGenerator it must:
1. Be in Controller folder
2. Have a name starting with "controller_"
3. The name of the controller class must be the same as the name of the file.
4. It must have __init__ and step methods

We recommend you derive it from the provided template.
See the provided examples of controllers to gain more insight.
"""


class template_controller(ABC):
    _controller_name: str = ""

    def __init__(
        self,
        predictor: predictor,
        cost_function: cost_function_default,
        seed: int,
        action_space: Box,
        observation_space: Box,
        mpc_horizon: int,
        num_rollouts: int,
        controller_logging: bool,
    ):
        self.predictor = predictor
        self.cost_function = cost_function
        
        # Environment-related parameters
        assert len(action_space.shape) == 1, "Only vector action space currently supported"
        assert len(observation_space.shape) == 1, "Only vector observation space currently supported"
        self.num_control_inputs = action_space.shape[0]
        self.num_states = observation_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        
        # MPC parameters
        self.num_rollouts = num_rollouts
        self.mpc_horizon = mpc_horizon
        
        # Initialize random sampler
        self.rng = create_rng(self.__class__.__name__, seed, use_tf=True)
        
        # Reset
        self.u = 0.0

        # Logging-related
        self.controller_logging = controller_logging
        self.save_vars = [
            "Q_logged",
            "J_logged",
            "realized_cost_logged",
            "s_logged",
            "u_logged",
            "trajectory_ages_logged",
        ]
        self.logs = {s: [] for s in self.save_vars}
        for v in self.save_vars:
            setattr(self, v, None)
    
    @abstractmethod
    def step(self, s: np.ndarray, time=None):
        Q = None  # This line is not obligatory. ;-) Just to indicate that Q must me defined and returned
        pass
        return Q  # normed control input in the range [-1,1]

    # Optionally: A method called after an experiment.
    # May be used to print some statistics about controller performance (e.g. number of iter. to converge)
    def controller_report(self):
        raise NotImplementedError

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        raise NotImplementedError
    
    @property
    def controller_name(self):
        name = self.__class__.__name__
        if name != "template_controller":
            return name.replace("controller_", "").replace("_", "-").lower()
        else:
            raise AttributeError()
    
    def get_outputs(self) -> "dict[str, np.ndarray]":
        """Retrieve a dictionary of controller outputs. These could be saved traces of input plans or the like.

        :return: A dictionary of numpy arrays
        :rtype: dict[str, np.ndarray]
        """
        return {
            k: np.stack(v, axis=0) if len(v) > 0 else None for k, v in self.logs.items()
        }
        
    def update_logs(self) -> None:
        if self.controller_logging:
            for name, var in zip(
                self.save_vars, [getattr(self, var_name, None) for var_name in self.save_vars]
            ):
                if var is not None:
                    self.logs[name].append(
                        var.numpy().copy() if hasattr(var, "numpy") else var.copy()
                    )
