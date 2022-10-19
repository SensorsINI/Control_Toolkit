import copy
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box


class optimizer_mppi_tf(template_optimizer):
    def __init__(
        self,
        controller: template_controller,
        predictor: PredictorWrapper,
        cost_function: cost_function_base,
        action_space: Box,
        observation_space: Box,
        seed: int,
        cc_weight: float,
        R: float,
        LBD: float,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_specification: str,
        NU: float,
        SQRTRHOINV: float,
        GAMMA: float,
        SAMPLING_TYPE: str,
        optimizer_logging: bool,
    ):
        super().__init__(
            controller=controller,
            predictor=predictor,
            cost_function=cost_function,
            predictor_specification=predictor_specification,
            action_space=action_space,
            observation_space=observation_space,
            seed=seed,
            num_rollouts=num_rollouts,
            mpc_horizon=mpc_horizon,
            optimizer_logging=optimizer_logging,
        )
        
        # Create second predictor for computing optimal trajectories
        self.predictor_single_trajectory = self.predictor.copy()
        self.predictor_single_trajectory.configure(
            batch_size=1, horizon=self.mpc_horizon,  # TF requires constant batch size
            predictor_specification=predictor_specification,
        )
        
        # MPPI parameters
        self.cc_weight = cc_weight
        dt = self.predictor.predictor_config['dt']
        self.R = tf.convert_to_tensor(R)
        self.LBD = LBD
        self.NU = tf.convert_to_tensor(NU)
        self.SQRTRHODTINV = tf.convert_to_tensor(np.array(SQRTRHOINV) * (1 / np.sqrt(dt)), dtype=tf.float32)
        self.GAMMA = GAMMA
        self.SAMPLING_TYPE = SAMPLING_TYPE

        self.update_internal_state = self.update_internal_state_of_RNN  # FIXME: There is one unnecessary operation in this function in case it is not an RNN.

        if True:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted
        
        self.optimizer_reset()
        
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
        stage_cost = self.cost_function.get_trajectory_cost(s_hor,u, u_prev)
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
            range_stop = int(tf.math.ceil(self.mpc_horizon / step)*step) + 1
            t = tf.range(range_stop, delta = step)
            t_interp = tf.cast(tf.range(range_stop), tf.float32)
            delta_u = random_gen.normal([self.num_rollouts, t.shape[0], self.num_control_inputs], dtype=tf.float32) * stdev
            interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
            delta_u = interp[:,:self.mpc_horizon, :]
        else:
            delta_u = random_gen.normal([self.num_rollouts, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32) * stdev
        return delta_u

    @CompileTF
    def predict_and_cost(self, s, u_nom, random_gen, u_old):
        s = tf.tile(s, tf.constant([self.num_rollouts, 1]))
        # generate random input sequence and clip to control limits
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1:, :]], axis=1)
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1])+delta_u
        u_run = tf.clip_by_value(u_run, self.action_low, self.action_high)
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u)
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        u = u_nom[0, 0, :]
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run)

    def update_internal_state_of_RNN(self, s, u_nom):
        u_tiled = tf.tile(u_nom[:, :1, :], tf.constant([self.num_rollouts, 1, 1]))
        self.predictor.update(s=s, Q0=u_tiled)

    @CompileTF
    def predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            logging_values = {"s_logged": s.copy()}
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s = self.check_dimensions_s(s)

        self.u, self.u_nom, rollout_trajectory, traj_cost, u_run = self.predict_and_cost(s, self.u_nom, self.rng, self.u)
        self.u = tf.squeeze(self.u).numpy()
        
        if self.optimizer_logging:
            logging_values["Q_logged"] = u_run.numpy()
            logging_values["J_logged"] = traj_cost.numpy()
            logging_values["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            logging_values["u_logged"] = self.u
            self.send_logs_to_controller(logging_values)

        if False:
            self.optimal_trajectory = self.predict_optimal_trajectory(s, self.u_nom).numpy()

        return self.u
    
    def controller_report(self):
        pass

    def optimizer_reset(self):
        self.u_nom = (
            0.5 * (self.action_low + self.action_high)
            * tf.ones([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        )
