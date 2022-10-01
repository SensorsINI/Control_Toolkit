from importlib import import_module

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import create_rng, Compile

from Control_Toolkit.Controllers import template_controller


class controller_mppi_tf(template_controller):
    def __init__(self, environment: EnvironmentBatched, seed: int, num_control_inputs: int, cc_weight: float, R: float, LBD: float, mpc_horizon: int, num_rollouts: int, dt: float, predictor_intermediate_steps: int, NU: float, SQRTRHOINV: float, GAMMA: float, SAMPLING_TYPE: str, NET_NAME: str, predictor_name: str, **kwargs):
        super().__init__(environment)
        
        #First configure random sampler
        self.rng_mppi = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        self.num_rollouts = num_rollouts

        self.cc_weight = cc_weight

        self.predictor_name = predictor_name
        self.mppi_samples = mpc_horizon  # Number of steps in MPC horizon

        self.R = tf.convert_to_tensor(R)
        self.LBD = LBD
        self.NU = tf.convert_to_tensor(NU)
        self.SQRTRHODTINV = tf.convert_to_tensor(np.array(SQRTRHOINV) * (1 / np.math.sqrt(dt)), dtype=tf.float32)
        self.GAMMA = GAMMA
        self.SAMPLING_TYPE = SAMPLING_TYPE
        
        self.clip_control_input_low = self.env_mock.action_space.low
        self.clip_control_input_high = self.env_mock.action_space.high

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.env_mock.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.mppi_samples,
            dt=dt,
            intermediate_steps=predictor_intermediate_steps,
            disable_individual_compilation=True,
            batch_size=num_rollouts,
            net_name=NET_NAME,
            planning_environment=self.env_mock,
        )
        if predictor_name == "predictor_autoregressive_tf":
            self.predictor_single_trajectory = getattr(predictor_module, predictor_name)(
            horizon=self.mppi_samples,
            dt=dt,
            intermediate_steps=predictor_intermediate_steps,
            disable_individual_compilation=True,
            batch_size=1,
            net_name=NET_NAME,
        )
        else:
            self.predictor_single_trajectory = self.env_mock.predictor

        self.get_rollouts_from_mppi = True
        self.get_optimal_trajectory = False

        self.controller_reset()
        self.rollout_trajectory = None
        self.traj_cost = None
        self.optimal_trajectory = None

        # Defining function - the compiled part must not have if-else statements with changing output dimensions
        if predictor_name == 'predictor_autoregressive_tf':
            self.update_internal_state = self.update_internal_state_of_RNN
        else:
            self.update_internal_state = lambda s, u_nom: ...

        if self.get_rollouts_from_mppi:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted
        
    def return_all(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, rollout_trajectory, traj_cost, u_run

    def return_restricted(self, u, u_nom, rollout_trajectory, traj_cost, u_run):
        return u, u_nom, None, None, None

    def check_dimensions_s(self, s):
        # Make sure the input is at least 2d
        if tf.rank(s) == 1:
            s = s[tf.newaxis, :]
        return s

    #mppi correction
    def mppi_correction_cost(self, u, delta_u):
        return tf.math.reduce_sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), axis=[1, 2])

    #total cost of the trajectory
    def get_mppi_trajectory_cost(self, s_hor ,u, u_prev, delta_u):
        stage_cost = self.env_mock.cost_functions.get_trajectory_cost(s_hor,u, u_prev)
        total_cost = stage_cost + self.mppi_correction_cost(u, delta_u)
        return total_cost

    def reward_weighted_average(self, S, delta_u):
        rho = tf.math.reduce_min(S)
        exp_s = tf.exp(-1.0/self.LBD * (S-rho))
        a = tf.math.reduce_sum(exp_s)
        b = tf.math.reduce_sum(exp_s[:, tf.newaxis, tf.newaxis]*delta_u, axis=0)/a
        return b

    def inizialize_pertubation(self, random_gen):
        stdev = self.SQRTRHODTINV
        sampling_type = self.SAMPLING_TYPE
        if sampling_type == "interpolated":
            step = 10
            range_stop = int(tf.math.ceil(self.mppi_samples / step)*step) + 1
            t = tf.range(range_stop, delta = step)
            t_interp = tf.cast(tf.range(range_stop), tf.float32)
            delta_u = random_gen.normal([self.num_rollouts, t.shape[0], self.num_control_inputs], dtype=tf.float32) * stdev
            interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
            delta_u = interp[:,:self.mppi_samples, :]
        else:
            delta_u = random_gen.normal([self.num_rollouts, self.mppi_samples, self.num_control_inputs], dtype=tf.float32) * stdev
        return delta_u

    @Compile
    def predict_and_cost(self, s, u_nom, random_gen, u_old):
        s = tf.tile(s, tf.constant([self.num_rollouts, 1]))
        # generate random input sequence and clip to control limits
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1:, :]], axis=1)
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1])+delta_u
        u_run = tf.clip_by_value(u_run, self.clip_control_input_low, self.clip_control_input_high)
        rollout_trajectory = self.env_mock.predictor.predict_tf(s, u_run)
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.clip_control_input_low, self.clip_control_input_high)
        u = u_nom[0, 0, :]
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run)

    def update_internal_state_of_RNN(self, s, u_nom):
        u_tiled = tf.tile(u_nom[:, :1, :], tf.constant([self.num_rollouts, 1, 1]))
        self.env_mock.predictor.update_internal_state_tf(s=s, Q0=u_tiled)

    @Compile
    def predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        if self.predictor_name ==  'predictor_autoregressive_tf':
            self.predictor_single_trajectory.update_internal_state_tf(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s = self.check_dimensions_s(s)

        self.u, self.u_nom, rollout_trajectory, traj_cost, u_run = self.predict_and_cost(s, self.u_nom, self.rng_mppi, self.u)
        
        self.u_logged = self.u
        self.Q_logged, self.J_logged = u_run.numpy(), traj_cost.numpy()
        self.rollout_trajectories_logged = rollout_trajectory.numpy()

        if self.get_rollouts_from_mppi:
            self.rollout_trajectory = rollout_trajectory.numpy()
            self.traj_cost = traj_cost.numpy()

        if self.get_optimal_trajectory:
            self.optimal_trajectory = self.predict_optimal_trajectory(s, self.u_nom).numpy()

        return tf.squeeze(self.u).numpy()
    
    def controller_report(self):
        pass

    def controller_reset(self):
        self.u_nom = (
            0.5 * (self.env_mock.action_space.low + self.env_mock.action_space.high)
            * tf.ones([1, self.mppi_samples, self.num_control_inputs], dtype=tf.float32)
        )
        self.u = 0.0
