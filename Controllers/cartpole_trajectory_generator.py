# stub for genrating desired future trajectory of cartpole
import numpy
import numpy as np
from torch import TensorType

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import ComputationLibrary
from CartPole import state_utilities

period=1

log=get_logger(__name__)

class cartpole_trajectory_generator:
    """ Generates target state trajectory for the cartpole """
    def __init__(self, lib:ComputationLibrary, controller:template_controller=None):
        """ Construct the trajectory generator.

        :param lib: the computation library, e.g. tensorflow
        :param horizon: the MPC horizon in timesteps
        """
        self.lib = lib
        self.controller:template_controller=controller

    def step(self, time: float, horizon: int, dt:float) -> TensorType:
        """ Computes the desired future state trajectory at this time.

        :param time: the scalar time in seconds
        :param horizon: the number of horizon steps
        :param dt: the timestep in seconds

        :returns: the target state trajectory of cartpole.
        It should be a Tensor with NaN as at least first entries for don't care states, and otherwise the desired state values.

        """

        traj=np.zeros((state_utilities.NUM_STATES, horizon)) # must be numpy here because tensor is immutable
        traj[:]=self.lib.nan # set all states undetermined

        cost_function=self.controller.cost_function_wrapper.cost_function # use cost_function to access attributes (fields) set in config_cost_functions.yml
        controller=self.controller # use controller to access attributes set in config_optimizers

        policy=cost_function.policy
        if policy is None:
            raise RuntimeError(f'set policy in config_cost_functions.yml')

        if policy == 'spin': # spin pole CW or CCW depending on target_equilibrium up or down
            traj[state_utilities.POSITION_IDX] = controller.target_position
            # traj[state_utilities.ANGLE_COS_IDX, :] = controller.target_equilibrium
            # traj[state_utilities.ANGLE_SIN_IDX, :] = 0
            # traj[state_utilities.ANGLE_IDX, :] = self.lib.pi * controller.target_equilibrium
            traj[state_utilities.ANGLED_IDX, :] = 1000*controller.target_equilibrium # 1000 rad/s is arbitrary, not sure if this is best target
            # traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy == 'balance': # balance upright or down at desired cart position
            traj[state_utilities.POSITION_IDX] = controller.target_position
            target_angle=self.lib.pi * (1-controller.target_equilibrium)/2 # either 0 for up and pi for down
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            traj[state_utilities.ANGLE_IDX, :] = target_angle
            traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy == 'shimmy': # cart follows a desired cart position shimmy while keeping pole up or down
            shimmy_period=cost_function.shimmy_period # seconds
            shimmy_amp=cost_function.shimmy_amp # meters
            endtime=time+horizon*dt
            times=np.linspace(time,endtime,num=horizon)
            shimmy=shimmy_amp*np.sin((2*np.pi/shimmy_period)*times)
            shimmyd=np.gradient(shimmy,dt)
            traj[state_utilities.POSITION_IDX] = controller.target_position+shimmy
            target_angle=self.lib.pi * (1-controller.target_equilibrium)/2 # either 0 for up and pi for down
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            traj[state_utilities.ANGLE_IDX, :] = target_angle
            # traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = shimmyd
        else:
            log.error(f'cost policy "{policy}" is unknown')

        # traj=self.lib.to_variable(traj, self.lib.float32)

        return traj
