from abc import ABC, abstractmethod

import os
from typing import Tuple
import numpy as np
import yaml
from SI_Toolkit.computation_library import ComputationLibrary, TensorType

config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml")), Loader=yaml.FullLoader)

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
    _computation_library: "type[ComputationLibrary]" = None  # Define this in your controller class
    _has_optimizer = False
    
    def __init__(
        self,
        environment_name: str,
        num_states: int,
        num_control_inputs: int,
        control_limits: Tuple[np.ndarray, np.ndarray],
        initial_environment_attributes: "dict[str, TensorType]",
    ):
        self.config_controllers = yaml.load(
            open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")),
            Loader=yaml.FullLoader
        )
        self.config_controller = self.config_controllers[self.controller_name]

        # Environment-related parameters
        self.environment_name = environment_name
        self.initial_environment_attributes = initial_environment_attributes
        
        self.num_states = num_states
        self.num_control_inputs = num_control_inputs
        self.control_limits = control_limits
        self.action_low, self.action_high = self.control_limits
        
        # Set properties like target positions on this controller
        for property, new_value in initial_environment_attributes.items():
            setattr(self, property, self.computation_library.to_variable(new_value, self.computation_library.float32))
                
        # Initialize control variable
        self.u = 0.0

        # Logging-related
        self.controller_logging = self.config_controllers[self.controller_name]["controller_logging"]
        self.save_vars = [
            "Q_logged",
            "J_logged",
            "s_logged",
            "u_logged",
            "realized_cost_logged",
            "trajectory_ages_logged",
            "rollout_trajectories_logged",
        ]
        self.logs: "dict[str, list[TensorType]]" = {s: [] for s in self.save_vars}
    
    def configure(self, **kwargs):
        # In your controller, implement any additional initialization steps here
        pass
    
    def update_attributes(self, updated_attributes: "dict[str, TensorType]"):
        for property, new_value in updated_attributes.items():
            self.computation_library.assign(getattr(self, property), new_value)
    
    @abstractmethod
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
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
    
    @property
    def controller_data_for_csv(self):
        return {}

    @property
    def computation_library(self):
        if self._computation_library == None:
            raise NotImplementedError("Controller class needs to specify its computation library")
        return self._computation_library
    
    @property
    def has_optimizer(self):
        return self._has_optimizer
    
    def get_outputs(self) -> "dict[str, np.ndarray]":
        """Retrieve a dictionary of controller outputs. These could be saved traces of input plans or the like.
        The values for each control iteration are stacked along the first axis.

        :return: A dictionary of numpy arrays
        :rtype: dict[str, np.ndarray]
        """
        return {
            name: np.stack(v, axis=0) if len(v) > 0 else None for name, v in self.logs.items()
        }
        
    def update_logs(self, logging_values: "dict[str, TensorType]") -> None:
        if self.controller_logging:
            for name, var in zip(
                self.save_vars, [logging_values.get(var_name, None) for var_name in self.save_vars]
            ):
                if var is not None:
                    self.logs[name].append(
                        var.numpy().copy() if hasattr(var, "numpy") else var.copy()
                    )
