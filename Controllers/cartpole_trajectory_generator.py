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

    def step(self, time: float, horizon: int, dt:float, state:np.ndarray) -> TensorType:
        """ Computes the desired future state trajectory at this time.

        :param time: the scalar time in seconds
        :param horizon: the number of horizon steps
        :param dt: the timestep in seconds
        :param state: the current state of the cartpole

        :returns: the target state trajectory of cartpole.
        It should be a Tensor with NaN as at least first entries for don't care states, and otherwise the desired future state values.

        """

        traj=np.zeros((state_utilities.NUM_STATES, horizon)) # must be numpy here because tensor is immutable
        traj[:]=self.lib.nan # set all states undetermined

        cost_function=self.controller.cost_function_wrapper.cost_function # use cost_function to access attributes (fields) set in config_cost_functions.yml
        controller=self.controller # use controller to access attributes set in config_optimizers

        policy=self.controller.cost_function_wrapper.cost_function.policy
        if policy is None:
            raise RuntimeError(f'set policy in config_self.controller.cost_function_wrapper.cost_functions.yml')

        gui_target_position=self.controller.target_position # GUI slider position
        gui_target_equilibrium=self.controller.target_equilibrium # GUI switch +1 or -1 to make pole target up or down position

        if policy == 'spin': # spin pole CW or CCW depending on target_equilibrium up or down
            traj[state_utilities.POSITION_IDX] = gui_target_position
            # traj[state_utilities.ANGLE_COS_IDX, :] = gui_target_equilibrium
            # traj[state_utilities.ANGLE_SIN_IDX, :] = 0
            endtime=horizon*dt
            times=np.linspace(0,endtime,num=horizon)
            s_per_rev_target=cost_function.spin_rev_period_sec
            rad_per_s_target=2*np.pi/s_per_rev_target
            rad_per_dt=rad_per_s_target*dt
            current_angle=state[state_utilities.ANGLE_IDX]
            traj[state_utilities.ANGLE_IDX, :] = current_angle+gui_target_equilibrium*times*rad_per_dt
            # traj[state_utilities.ANGLED_IDX, :] = rad_per_s_target*gui_target_equilibrium # 1000 rad/s is arbitrary, not sure if this is best target
            # traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy == 'balance': # balance upright or down at desired cart position
            traj[state_utilities.POSITION_IDX] = gui_target_position
            target_angle=self.lib.pi * (1-gui_target_equilibrium)/2 # either 0 for up and pi for down
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            # traj[state_utilities.ANGLE_IDX, :] = target_angle
            traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = 0
        elif policy == 'shimmy': # cart follows a desired cart position shimmy while keeping pole up or down
            per=self.controller.cost_function_wrapper.cost_function.shimmy_per # seconds
            amp=self.controller.cost_function_wrapper.cost_function.shimmy_amp # meters
            endtime=time+horizon*dt
            times=np.linspace(time,endtime,num=horizon)
            cartpos=amp*np.sin((2*np.pi/per)*times)
            cartvel=np.gradient(cartpos,dt)
            traj[state_utilities.POSITION_IDX] = gui_target_position+cartpos
            target_angle=self.lib.pi * (1-gui_target_equilibrium)/2 # either 0 for up and pi for down
            traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            traj[state_utilities.ANGLE_IDX, :] = target_angle
            # traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = cartvel
        elif policy == 'cartonly': # cart follows the trajectory, pole ignored
            per=self.controller.cost_function_wrapper.cost_function.cartonly_per # seconds
            amp=self.controller.cost_function_wrapper.cost_function.cartonly_amp # meters
            endtime=time+horizon*dt
            times=np.linspace(time,endtime,num=horizon)
            from scipy.signal import sawtooth
            cartpos=amp*sawtooth((2*np.pi/per)*times, width=cost_function.cartonly_duty_cycle) # width=.5 makes triangle https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
            cartvel=np.gradient(cartpos,dt)
            traj[state_utilities.POSITION_IDX] = gui_target_position+cartpos
            # target_angle=self.lib.pi * (1-gui_target_equilibrium)/2 # either 0 for up and pi for down
            # traj[state_utilities.ANGLE_COS_IDX, :] = np.cos(target_angle)
            # traj[state_utilities.ANGLE_SIN_IDX, :] = np.sin(target_angle)
            # traj[state_utilities.ANGLE_IDX, :] = target_angle
            # traj[state_utilities.ANGLED_IDX, :] = 0
            traj[state_utilities.POSITIOND_IDX, :] = cartvel
        else:
            log.error(f'cost policy "{policy}" is unknown')

        # traj=self.lib.to_variable(traj, self.lib.float32)

        return traj
