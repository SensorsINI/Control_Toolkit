from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorFlowLibrary, PyTorchLibrary

import numpy as np

from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileAdaptive
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


class optimizer_mppi(template_optimizer):
    supported_computation_libraries = {NumpyLibrary, TensorFlowLibrary, PyTorchLibrary}
    
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: CostFunctionWrapper,
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
        calculate_optimal_trajectory: bool,
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
        self.cc_weight = cc_weight
        self.R = self.lib.to_tensor(R, self.lib.float32)
        self.LBD = LBD
        self.NU = self.lib.to_tensor(NU, self.lib.float32)
        self._SQRTRHOINV = SQRTRHOINV

        self.update_internal_state = self.update_internal_state_of_RNN  # FIXME: There is one unnecessary operation in this function in case it is not an RNN.

        if True:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted

        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.Interpolator = None

        self.predict_and_cost = CompileAdaptive(self._predict_and_cost)

        self.calculate_optimal_trajectory = calculate_optimal_trajectory
        self.optimal_trajectory = None
        self.optimal_control_sequence = None
        self.predict_optimal_trajectory = CompileAdaptive(self._predict_optimal_trajectory)

    
    def configure(self,
                  num_states: int,
                  num_control_inputs: int,
                  dt: float,
                  predictor_specification: str,
                  **kwargs):

        super().configure(
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            default_configure=False,
        )

        self.Interpolator = Interpolator(self.mpc_horizon, self.period_interpolation_inducing_points, self.num_control_inputs, self.lib)

        self.SQRTRHODTINV = self.lib.to_tensor(np.array(self._SQRTRHOINV) * (1 / np.sqrt(dt)), self.lib.float32)
        del self._SQRTRHOINV
        
        self.predictor_single_trajectory.configure(
            batch_size=1, horizon=self.mpc_horizon, dt=dt,  # TF requires constant batch size
            predictor_specification=predictor_specification,
        )

        self.optimizer_reset()
        
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
    def mppi_correction_cost(self, u, delta_u):
        return self.lib.sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), (1, 2))

    #total cost of the trajectory
    def get_mppi_trajectory_cost(self, state_horizon ,u, u_prev, delta_u):
        total_cost = self.cost_function.get_trajectory_cost(state_horizon,u, u_prev)
        total_mppi_cost = total_cost + self.mppi_correction_cost(u, delta_u)
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

    def _predict_and_cost(self, s, u_nom, random_gen, u_old):
        s = self.lib.tile(s, (self.num_rollouts, 1))
        # generate random input sequence and clip to control limits
        u_nom = self.lib.concat([u_nom[:, 1:, :], u_nom[:, -1:, :]], 1)
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = self.lib.tile(u_nom, (self.num_rollouts, 1, 1))+delta_u
        u_run = self.lib.clip(u_run, self.action_low, self.action_high)
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u)
        u_nom = self.lib.clip(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        u = u_nom[0, 0, :]
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run)

    def update_internal_state_of_RNN(self, s, u_nom):
        u_tiled = self.lib.tile(u_nom[:, :1, :], (self.num_rollouts, 1, 1))
        self.predictor.update(s=s, Q0=u_tiled)

    def _predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
        s = self.lib.to_tensor(s, self.lib.float32)
        s = self.check_dimensions_s(s)

        self.u, self.u_nom, self.rollout_trajectories, traj_cost, u_run = self.predict_and_cost(s, self.u_nom, self.rng, self.u)
        self.u = self.lib.to_numpy(self.lib.squeeze(self.u))

        if self.optimizer_logging:
            self.logging_values["Q_logged"] = self.lib.to_numpy(u_run)
            self.logging_values["J_logged"] = self.lib.to_numpy(traj_cost)
            self.logging_values["rollout_trajectories_logged"] = self.lib.to_numpy(self.rollout_trajectories)
            self.logging_values["u_logged"] = self.u

        self.optimal_control_sequence = self.lib.to_numpy(self.u_nom)

        if self.calculate_optimal_trajectory:
            self.optimal_trajectory = self.lib.to_numpy(self.predict_optimal_trajectory(s, self.u_nom))

        return self.u

    def optimizer_reset(self):
        self.u_nom = (
            0.5 * self.lib.to_tensor(self.action_low + self.action_high, self.lib.float32)
            * self.lib.ones([1, self.mpc_horizon, self.num_control_inputs])
        )
