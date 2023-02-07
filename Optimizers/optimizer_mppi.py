from typing import Tuple

from Control_Toolkit.others.globals_and_utils import get_logger
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorFlowLibrary, PyTorchLibrary, \
    TensorType

import numpy as np

from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive

log = get_logger(__name__)

class optimizer_mppi(template_optimizer):
    """ Model Predictive Path Integral optimizer, based on

    The equations and parameters are defined in the following:

    Williams, G., P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou. 2016. “Aggressive Driving with Model Predictive Path Integral Control.” In 2016 IEEE International Conference on Robotics and Automation (ICRA), 1433–40. https://doi.org/10.1109/ICRA.2016.7487277.

    A longer paper with all the math is

    Williams, Grady, Andrew Aldrich, and Evangelos A. Theodorou. 2017. “Model Predictive Path Integral Control: From Theory to Parallel Computation.” Journal of Guidance, Control, and Dynamics: A Publication of the American Institute of Aeronautics and Astronautics Devoted to the Technology of Dynamics and Control 40 (2): 344–57. https://doi.org/10.2514/1.G001921.

    The following paper uses a LWPR model to fly crazyfly drones using another type of MPPI with minibatches of rollouts

    Williams, Grady, Eric Rombokas, and Tom Daniel. 2015. “GPU Based Path Integral Control with Learned Dynamics.” arXiv [cs.RO]. arXiv. http://arxiv.org/abs/1503.00330.

    """
    supported_computation_libraries = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}
    
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: CostFunctionWrapper,
        num_states: int,
        num_control_inputs: int,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        computation_library: "type[ComputationLibrary]",
        seed: int,
        cc_weight: float,
        R: float,
        LBD: float,
        mpc_horizon: int,
        num_rollouts: int,
        NU: float,
        SQRTRHOINV: float,
        period_interpolation_inducing_points: int,
        optimizer_logging: bool,
    ):
        """Instantiate MPPI optimizer, see
        Williams et al. 2017, 'Model Predictive Path Integral Control: From Theory to Parallel Computation'

        :param predictor: Predictor to compute trajectory rollouts with. Instance of a wrapper class, which at this time can have unspecified parameters.
        :type predictor: PredictorWrapper
        :param cost_function: Instance containing the objective functions to minimize.
        :type cost_function: CostFunctionWrapper
        :param num_states: Length of the system's state vector.
        :type num_states: int
        :param num_control_inputs: Length of the system's input vector.
        :type num_control_inputs: int
        :param control_limits: Bounds on the input, one array each for lower/upper.
        :type control_limits: Tuple[np.ndarray, np.ndarray]
        :param computation_library: The numerical package to use for optimization. One of our custom 'computation library' classes.
        :type computation_library: type[ComputationLibrary]
        :param seed: Random seed for reproducibility. Manage the seed lifecycle and (re-)use outside of this class.
        :type seed: int
        :param cc_weight: Positive scalar weight placed on the MPPI-specific quadratic control cost term.
        :type cc_weight: float
        :param R: Weight of quadratic cost term.
        :type R: float
        :param LBD: Positive parameter defining greediness of the optimizer in the weighted sum of rollouts.
        If lambda is strongly positive, all trajectories get roughly the same weight even if one has much lower cost than the other.
        As lambda approaches 0, the relative importance of the most low-cost trajectories when determining the weighted average increases.
        :type LBD: float
        :param mpc_horizon: Length of the MPC horizon in steps. dt is implicitly given by the predictor provided.
        :type mpc_horizon: int
        :param num_rollouts: Number of parallel rollouts. Generally, MPPI's control performance improves when this value increases. Typical range is 5e2-5e3.
        :type num_rollouts: int
        :param NU: Adjustment of exploration variance. Increase to increase input exploration.
        :type NU: float
        :param SQRTRHOINV: Positive scalar. Has an effect on the sampling variance. An increase results in wider sampling of inputs around the nominal plan.
        :type SQRTRHOINV: float
        :param period_interpolation_inducing_points: Positive distance in number of steps between two inducing points along the MPC horizon.
        If set to 1, this means independent sampling of each input.
        If larger than 1, inputs between inducing points are obtained by linear interpolation.
        :type period_interpolation_inducing_points: int
        :param optimizer_logging: Whether to store rollouts, evaluated trajectory costs, etc. in a 'logging_values' variable. Consumes extra memory if True.
        :type optimizer_logging: bool
        """
        super().__init__(
            predictor=predictor,
            cost_function=cost_function,
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            control_limits=control_limits,
            optimizer_logging=optimizer_logging,
            seed=seed,
            num_rollouts=num_rollouts,
            mpc_horizon=mpc_horizon,
            computation_library=computation_library,
        )
        
        # Create second predictor for computing optimal trajectories
        self.predictor_single_trajectory = self.predictor.copy()
        
        
        # MPPI parameters
        self.cc_weight = self.lib.to_variable(cc_weight, self.lib.float32)
        self.R = self.lib.to_variable(R, self.lib.float32)
        self.LBD = self.lib.to_variable(LBD, self.lib.float32)
        self.NU = self.lib.to_variable(NU, self.lib.float32)
        self._SQRTRHOINV = SQRTRHOINV

        self.update_internal_state = self.update_internal_state_of_RNN  # FIXME: There is one unnecessary operation in this function in case it is not an RNN.

        if True:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted

        self.Interpolator = Interpolator(self.mpc_horizon, period_interpolation_inducing_points, self.num_control_inputs, self.lib)

        # here the predictor, cost computer, and optimizer are compiled to native instrutions by tensorflow graphs and XLA JIT

        self.predict_and_cost = CompileAdaptive(self._predict_and_cost)
        self.predict_optimal_trajectory = CompileAdaptive(self._predict_optimal_trajectory)

        self.optimizer_reset()
    
    def configure(self, dt: float, predictor_specification: str, **kwargs):
        self.SQRTRHODTINV = self.lib.to_tensor(np.array(self._SQRTRHOINV) * (1 / np.sqrt(dt)), self.lib.float32)
        del self._SQRTRHOINV
        
        self.predictor_single_trajectory.configure(
            batch_size=1, horizon=self.mpc_horizon, dt=dt,  # TF requires constant batch size
            predictor_specification=predictor_specification,
        )
        
    def return_all(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, rollout_trajectory, traj_cost, u_run

    def return_restricted(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, None, None, None

    def check_dimensions_s(self, s):
        # Make sure the input is at least 2d
        if self.lib.ndim(s) == 1:
            s = s[self.lib.newaxis, :]
        return s

    #mppi correction
    def mppi_correction_cost(self, u, delta_u, time=None):
        return self.lib.sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), (1, 2))

    #total cost of the trajectory
    def get_mppi_trajectory_cost(self, state_horizon ,u, u_prev, delta_u, time:float=None):
        """ Compute the total trajectory costs for all the rollouts

        :param state_horizon: the states as [rollouts,timesteps,states]
        :param u: the control as ??? TODO
        :param u_prev: the previous control input
        :param delta_u: change in control input, TODO passed in for efficiency?
        :param time: the time in seconds

         :returns: the total mppi cost for each rollout, i.e. 1d-vector of costs per rollout
         """
        total_cost = self.cost_function.get_trajectory_cost(state_horizon,u, u_prev,time=time)
        mppi_correction_cost =self.mppi_correction_cost(u, delta_u, time=time)
        total_mppi_cost = total_cost +mppi_correction_cost
        return total_mppi_cost

    def reward_weighted_average(self, S, delta_u):
        rho = self.lib.reduce_min(S, 0)
        exp_s = self.lib.exp(-1.0/self.LBD * (S-rho))
        a = self.lib.sum(exp_s, 0)
        b = self.lib.sum(exp_s[:, self.lib.newaxis, self.lib.newaxis]*delta_u, 0)/a
        return b

    def inizialize_pertubation(self, random_gen):
        stdev = self.SQRTRHODTINV

        delta_u = random_gen.normal(
            [self.num_rollouts, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
            dtype=self.lib.float32) * stdev

        delta_u = self.Interpolator.interpolate(delta_u)

        return delta_u

    def _predict_and_cost(self, state:TensorType, u_nom:TensorType, random_gen, u_old:TensorType, time:float=None):
        """ Predict dynamics and compute costs of trajectories

        :param state: the current state of system, dimensions are [rollouts, timesteps, states]
        :param u_nom: the nominal control input
        :param random_gen: the random generator
        :param u_old: previous control input
        :param time: time in seconds

        :returns: u, u_nom: the new control input TODO what are u and u_nom?
        """
        state = self.lib.tile(state, (self.num_rollouts, 1))
        # generate random input sequence and clip to control limits
        u_nom = self.lib.concat([u_nom[:, 1:, :], u_nom[:, -1:, :]], 1)
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = self.lib.tile(u_nom, (self.num_rollouts, 1, 1))+delta_u
        u_run = self.lib.clip(u_run, self.action_low, self.action_high)
        rollout_trajectory = self.predictor.predict_tf(state, u_run, time=time)
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u, time=time)
        u_nom = self.lib.clip(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        u = u_nom[0, 0, :]
        self.update_internal_state(state, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run)

    def update_internal_state_of_RNN(self, s, u_nom):
        u_tiled = self.lib.tile(u_nom[:, :1, :], (self.num_rollouts, 1, 1))
        self.predictor.update(s=s, Q0=u_tiled)

    def _predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, state: np.ndarray, time=None):
        """ Does one timestep of control

        :param state: the current state
        :param time: the current time in seconds

        :returns: u, the new control input to system
        """
        if self.optimizer_logging:
            self.logging_values = {"s_logged": state.copy()}
        state = self.lib.to_tensor(state, self.lib.float32)
        state = self.check_dimensions_s(state)

        tf_time=self.lib.to_tensor(time,self.lib.float32) # must pass scalar tensor for time to prevent recompiling tensorflow functions over and over

        self.u, self.u_nom, rollout_trajectory, traj_cost, u_run = self.predict_and_cost(state, self.u_nom, self.rng, self.u, time=tf_time)
        self.u = self.lib.to_numpy(self.lib.squeeze(self.u))

        # print(f'mean traj cost={np.mean(traj_cost.numpy()):.2f}') # todo debug


        if self.optimizer_logging:
            self.logging_values["Q_logged"] = self.lib.to_numpy(u_run)
            self.logging_values["J_logged"] = self.lib.to_numpy(traj_cost)
            self.logging_values["rollout_trajectories_logged"] = self.lib.to_numpy(self.rollout_trajectories)
            self.logging_values["u_logged"] = self.u

        if False:
            self.optimal_trajectory = self.lib.to_numpy(self.predict_optimal_trajectory(state, self.u_nom))

        return self.u

    def optimizer_reset(self):
        self.u_nom = (
            0.5 * self.lib.to_tensor(self.action_low + self.action_high, self.lib.float32)
            * self.lib.ones([1, self.mpc_horizon, self.num_control_inputs])
        )
