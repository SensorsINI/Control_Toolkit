import os
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import yaml
from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import (ComputationLibrary, NumpyLibrary,
                                            PyTorchLibrary, TensorFlowLibrary,
                                            TensorType)
from others.globals_and_utils import load_or_reload_config_if_modified

config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml")), Loader=yaml.FullLoader)
logger = get_logger(__name__)

"""
For a controller to be found and imported by CartPoleGUI/DataGenerator it must:
1. Be in Controllers folder
2. Have a name starting with "controller_"
3. The name of the controller class must be the same as the name of the file.
4. It must have __init__ and step methods

We recommend you derive it from the provided template.
See the provided examples of controllers to gain more insight.
"""

class template_controller(ABC):
    _has_optimizer = False
    # Define the computation library in your controller class or in the controller's configuration:
    _computation_library: ComputationLibrary = None
    
    def __init__(
        self,
        dt: float,
        environment_name: str,
        num_states: int,
        num_control_inputs: int,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        initial_environment_attributes: "dict[str, TensorType]",
    ):
        # Load controller config and select the entry for the current controller
        (config_controllers,_) = load_or_reload_config_if_modified(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")) # ignore the _ changes return since this is initial call
        # self.controller_name is inferred from the class name, which is the class being instantiated
        # Example: If you create a controller_mpc, this controller_template.__init__ will be called
        # but the class name will be controller_mpc, not template_controller.
        config_key=self.controller_name
        self.config_controller = config_controllers[config_key]
        # add timestep .dt to all controllers here
        self.config_controller["dt"] = dt
        
        # Set computation library
        computation_library_name = str(self.config_controller.get("computation_library", ""))
        
        if computation_library_name:
            # Assign computation library from config
            logger.info(f"Found library {computation_library_name} for MPC controller.")
            if "tensorflow" in computation_library_name.lower():
                self._computation_library = TensorFlowLibrary
            elif "pytorch" in computation_library_name.lower():
                self._computation_library = PyTorchLibrary
            elif "numpy" in computation_library_name.lower():
                self._computation_library = NumpyLibrary
            else:
                raise ValueError(f"Computation library {computation_library_name} could not be interpreted.")
        else:
            # Try using default computation library set as class attribute
            if not issubclass(self.computation_library, ComputationLibrary):
                raise ValueError(f"{self.__class__.__name__} does not have a default computation library set. You have to define one in this controller's config.")
            else:
                logger.info(f"No computation library specified in controller config. Using default {self.computation_library} for class.")
        self.lib = self.computation_library  # Shortcut to make easy using functions from computation library, this is also used by CompileAdaptive to recognize library

        # Environment-related parameters
        self.environment_name = environment_name
        self.initial_environment_attributes = initial_environment_attributes
        
        self.num_states = num_states
        self.num_control_inputs = num_control_inputs
        self.control_limits = control_limits
        self.action_low, self.action_high = self.control_limits

        # todo these are special for cartpole but we would need a base cartpole controller class to put them there
        # self.target_position=None
        # self.target_equilibrium=None
        
        # Set properties like target positions on this controller
        for property, new_value in initial_environment_attributes.items():
            setattr(self, property, self.computation_library.to_variable(new_value, self.computation_library.float32))

        # set all controller config numerical values as float variables in the computation space, e.g. tensorflow, so they can be updqted during runtime
        for property, value in self.config_controller.items():
            if value is float or value is int:
                setattr(self, property, self.computation_library.to_variable(value, self.computation_library.float32))

        # Initialize control variable
        self.u = 0.0

        # Logging-related
        self.controller_logging = self.config_controller["controller_logging"]
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
    
    @abstractmethod
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, Union[TensorType,float]]" = dict()):
        """
        Execute one timestep of control.

        :param s: the state array, dimensions are TODO add dimension to help users
        :param time: the time in seconds
        :param updated_attributes: dict of updated attributes

        :returns: the next control action u e.g. a normed control input in the range [-1,1] TODO is this correct?

        """
        ### Any computations in order to retrieve the current control. Such as:
        ## If the environment's target positions etc. change, copy the new attributes over to this controller so the cost function knows about it:
        # self.update_attributes(updated_attributes)
        ## Use some sort of optimization procedure to get your control, e.g.
        # u = self.optimizer.step(s, time)
        ## Use the following call to populate the self.logs dictionary with savevars, such as:
        # self.update_logs(self.optimizer.logging_values)
        # return u  # e.g. a normed control input in the range [-1,1]
        pass
        return None

    # Optionally: A method called after an experiment.
    # May be used to print some statistics about controller performance (e.g. number of iter. to converge)
    def controller_report(self):
        logger.info("No controller report implemented for this controller. Stopping without report.")
        pass

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        raise NotImplementedError
    
    @property # decorates the controller so it has the field .controller_name that gets its short name which is the key in the .yml config file
    def controller_name(self):
        """ Generates standard name for this controller, but use this method like it were a field.

        :returns: the short name which is the key to the controller in the config_controller.yml file, e.g. 'cartpole_mppi'

        """
        name = self.__class__.__name__
        if name != "template_controller":
            if 'controller_' not in name:
                raise AttributeError(f'this controller named "{name}" does not contain "controller_". Controllers should start or contain "controller_" and the key in the config_controllers.yml file should follow the underscore')
            return name.replace("controller_", "").replace("_", "-").lower()
        else:
            raise AttributeError()
    
    @property
    def controller_data_for_csv(self):
        return {}

    @property
    def computation_library(self) -> ComputationLibrary:
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
