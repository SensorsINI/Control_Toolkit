import os
from typing import Optional

from Control_Toolkit_ASF.Cost_Functions.CartPole.cartpole_trajectory_generator import cartpole_trajectory_generator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import yaml
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper

from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.others.globals_and_utils import import_optimizer_by_name

from torch import inference_mode

from others.globals_and_utils import load_or_reload_config_if_modified, update_attributes

from Control_Toolkit.others.get_logger import get_logger
log = get_logger(__name__)


config_optimizers = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")), Loader=yaml.FullLoader)
config_cost_function = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_functions.yml")), Loader=yaml.FullLoader)


class controller_mpc(template_controller):

    _has_optimizer = True

    def __init__(
            self,
            dt: float,
            environment_name: str,
            num_states: int,
            num_control_inputs: int,
            control_limits: "Tuple[np.ndarray, np.ndarray]",
            initial_environment_attributes: "dict[str, TensorType]",
    ):
        super().__init__(dt, environment_name, num_states, num_control_inputs, control_limits,
                         initial_environment_attributes)
        self.cartpole_trajectory_generator:cartpole_trajectory_generator = cartpole_trajectory_generator() # TODO awkward to add this specific cost function generator here to generic controller_mpc

    def configure(self, optimizer_name: Optional[str]=None, predictor_specification: Optional[str]=None):
        if optimizer_name in {None, ""}:
            optimizer_name = str(self.config_controller["optimizer"])
            log.info(f'Using optimizer \'{optimizer_name}\' specified in controller config file')
        if predictor_specification in {None, ""}:
            predictor_specification: Optional[str] = self.config_controller.get("predictor_specification", None)
            log.info(f"Using predictor_specification='{predictor_specification}' specified in controller config file")

        config_optimizer = config_optimizers[optimizer_name]

        # Create cost function
        cost_function_specification = self.config_controller.get("cost_function_specification", None)
        log.info(f'using cost_function_specification=\'{cost_function_specification}\' in config {self.config_controller}')
        self.cost_function_wrapper = CostFunctionWrapper()
        self.cost_function_wrapper.configure(self, cost_function_specification=cost_function_specification)

        # Create predictor
        self.predictor_wrapper = PredictorWrapper()
        
        # MPC Controller always has an optimizer
        Optimizer = import_optimizer_by_name(optimizer_name)
        self.optimizer: template_optimizer = Optimizer(
            predictor=self.predictor_wrapper,
            cost_function=self.cost_function_wrapper,
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
        
        self.predictor_wrapper.configure(
            batch_size=self.optimizer.num_rollouts,
            horizon=self.optimizer.mpc_horizon,
            dt=self.config_controller["dt"],
            computation_library=self.computation_library,
            predictor_specification=predictor_specification
        )

        if self.lib.lib == 'Pytorch':
            self.step = inference_mode()(self.step)
        else:
            self.step = self.step

        
    def step(self, state: np.ndarray, time:float=None, updated_attributes: "dict[str, TensorType]" = {})->float:
        """ Compute one step of control.

        :param state: the current state as 1d state vector
        :param time: the current time in seconds
        :param updated_attributes: a dict of values to update tensorflow assignment values in the cost function

        :returns: the control vector (for cartpole, a scalar cart acceleration)

        """
        # log.debug(f'step time={time:.3f}s')


        # following gets target_position, target_equilibrium, and target_trajectory passed to tensorflow. The trajectory is passed in
        # as updated_attributes, and is transferred to tensorflow by the update_attributes call
        cost_function=self.cost_function_wrapper.cost_function
        update_attributes(updated_attributes,cost_function) # update target_position and target_equilibrium in cost function to use
        updated_attributes.clear()
        self.cartpole_trajectory_generator.\
                generate_cartpole_trajectory(time=time, state=state, controller=self, cost_function=self.cost_function_wrapper.cost_function)

        # now we fill this dict with config file changes if there are any and update attributes in the controller, the cost function, and the optimizer
        # detect any changes in config scalar values and pass to this controller or the cost function or optimizer

        # note that the cost function that has its attributes updated is the enclosed cost function of the wrapper!
        # note the following mapping for config files
        #   config_controller.yml -> self (i.e. controller_mpc)
        #   config_cost_functions.yml -> self.cost_function_wrapper.cost_function
        #   config_optimizers.yml -> self.optimizer AND self.predictor_wrapper.predictor
        for (objs,config) in (((self,),'config_controllers.yml'), ((self.cost_function_wrapper.cost_function,), 'config_cost_functions.yml'), ((self.optimizer,self.predictor_wrapper.predictor), 'config_optimizers.yml')):
            (config,changes)=load_or_reload_config_if_modified(os.path.join('Control_Toolkit_ASF',config), search_path=['CartPoleSimulation']) # all the config files are currently assumed to be in Control_Toolkit_ASF folder
            # process changes to configs using new returned change list
            if not changes is None:
                for k,v in changes.items():
                    if isinstance(v, (int, float, str)):
                        updated_attributes[k]=v
                        for o in objs: # for each object in objs, update its attributes
                            update_attributes(updated_attributes,o)
                log.debug(f'updated {objs} with scalar updated_attributes {updated_attributes}')

        # obtain the next control action from the optimizer
        u = self.optimizer.step(state, time)

        # for analysis of model mismatch, obtain the prediction of next state with this control action


        self.update_logs(self.optimizer.logging_values)
        return u

    def controller_reset(self):
        self.optimizer.optimizer_reset()
        self.cartpole_trajectory_generator.reset()

    def keyboard_input(self, c):
      try:
          self.cartpole_trajectory_generator.keyboard_input(c)
      except AttributeError:
          pass

    def print_keyboard_help(self):
        try:
            self.cartpole_trajectory_generator.print_keyboard_help()
        except AttributeError:
            pass
