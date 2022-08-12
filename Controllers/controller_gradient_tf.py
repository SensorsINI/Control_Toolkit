from importlib import import_module

import numpy as np
import tensorflow as tf
from others.globals_and_utils import create_rng
from SI_Toolkit.TF.TF_Functions.Compile import Compile

from Control_Toolkit.Controllers import template_controller


class controller_gradient_tf(template_controller):
    def __init__(
        self,
        environment,
        seed: int,
        num_control_inputs: int,
        dt: float,
        mpc_horizon: int,
        gradient_steps: int,
        mpc_rollouts: int,
        initial_action_stdev: float,
        predictor_name: str,
        predictor_intermediate_steps: int,
        CEM_NET_NAME: str,
        learning_rate: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        gradmax_clip: float,
        rtol: float,
        **kwargs,
    ):
        # First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        # MPC params
        self.num_rollouts = mpc_rollouts
        self.gradient_steps = gradient_steps
        self.cem_samples = mpc_horizon  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps
        self.initial_action_stdev = initial_action_stdev

        self.NET_NAME = CEM_NET_NAME

        self.optim = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        self.gradmax_clip = gradmax_clip
        self.rtol = rtol

        # instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=self.NET_NAME,
        )

        super().__init__(environment)
        self.action_low = self.env_mock.action_space.low
        self.action_high = self.env_mock.action_space.high

        # Initialization
        self.controller_reset()
        self.u = 0

    @Compile
    def gradient_optimization(self, s: tf.Tensor, Q_tf: tf.Variable, optim):
        # rollout the trajectories and get cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q_tf)
            traj_cost, rollout_trajectory = self.predict_and_cost(s, Q_tf)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q_tf)
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # use optimizer to applay gradients and retrieve next set of input sequences
        optim.apply_gradients(zip([dc_dQ_prc], [Q_tf]))
        # clip
        Q = tf.clip_by_value(Q_tf, self.action_low, self.action_high) 
    
        # traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
        return traj_cost, rollout_trajectory, Q
    
    @Compile
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    # step function to find control
    def step(self, s: np.ndarray, time=None):
        # Start all trajectories in current state
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        # generate random input sequence and clip to control limits
        Q = tf.tile(self.dist_mue, (self.num_rollouts, 1, 1)) + tf.multiply(
            self.rng_cem.normal(
                shape=[self.num_rollouts, self.cem_samples, self.num_control_inputs],
                dtype=tf.float32,
            ),
            self.stdev,
        )
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)

        # Perform gradient descent steps
        prev_cost = 0.0
        for _ in range(self.gradient_steps):
            self.Q_tf.assign(Q)
            traj_cost, rollout_trajectory, Q = self.gradient_optimization(s, self.Q_tf, self.optim)

            traj_cost = traj_cost.numpy()
            if np.mean(np.abs((traj_cost - prev_cost) / (prev_cost + self.rtol))) < self.rtol:
                # Assume that we have converged sufficiently
                break
            prev_cost = traj_cost.copy()
        
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0]

        self.u: np.ndarray = tf.squeeze(Q[best_idx, 0, :]).numpy()

        self.Q_logged, self.J_logged = Q.numpy(), traj_cost.numpy()
        self.rollout_trajectories_logged = rollout_trajectory.numpy()
        self.u_logged = self.u.copy()

        return self.u

    def controller_reset(self):
        self.dist_mue = (self.action_high + self.action_low) * 0.5 * tf.ones([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = self.initial_action_stdev * tf.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = tf.sqrt(self.dist_var)
        self.Q_tf = tf.Variable(
            tf.zeros([self.num_rollouts, self.cem_samples, self.num_control_inputs]),
            trainable=True,
            dtype=tf.float32,
        )
