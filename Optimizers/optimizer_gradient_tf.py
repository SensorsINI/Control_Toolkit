from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


class optimizer_gradient_tf(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}
    
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: CostFunctionWrapper,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        computation_library: "type[ComputationLibrary]",
        seed: int,
        mpc_horizon: int,
        gradient_steps: int,
        num_rollouts: int,
        initial_action_stdev: float,
        learning_rate: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        gradmax_clip: float,
        rtol: float,
        warmup: bool,
        warmup_iterations: int,
        optimizer_logging: bool,
        calculate_optimal_trajectory: bool,
    ):
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
        
        # MPC parameters
        self.gradient_steps = gradient_steps
        self.initial_action_stdev = initial_action_stdev

        # Initialize optimizer
        self.optim = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        self.gradmax_clip = gradmax_clip
        self.rtol = rtol
        self.warmup = warmup
        self.warmup_iterations = warmup_iterations

        # Setup warmup
        self.first_iter_count = self.gradient_steps
        if self.warmup:
            self.first_iter_count = self.warmup_iterations
    
    def _predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory
    
    @CompileTF
    def predict_and_cost(self, s, Q):
        return self._predict_and_cost(s, Q)

    @CompileTF
    def gradient_optimization(self, s: tf.Tensor, Q_tf: tf.Variable, optim):
        # rollout the trajectories and get cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q_tf)
            traj_cost, _ = self._predict_and_cost(s, Q_tf)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q_tf)
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # use optimizer to applay gradients and retrieve next set of input sequences
        optim.apply_gradients(zip([dc_dQ_prc], [Q_tf]))
        # clip
        Q = tf.clip_by_value(Q_tf, self.action_low, self.action_high) 
    
        # traj_cost, rollout_trajectory = self._predict_and_cost(s, Q)
        return Q
    
    # step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
        # Start all trajectories in current state
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        
        # warm start setup
        if self.count == 0:
            iters = self.first_iter_count
        else:
            iters = self.gradient_steps

        # Perform gradient descent steps
        # prev_cost = 0.0
        for _ in range(iters):
            Q = self.gradient_optimization(s, self.Q_tf, self.optim)
            self.Q_tf.assign(Q)

            # traj_cost = traj_cost.numpy()
            # if np.mean(np.abs((traj_cost - prev_cost) / (prev_cost + self.rtol))) < self.rtol:
            #     # Assume that we have converged sufficiently
            #     break
            # prev_cost = traj_cost.copy()
        
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0]

        self.u: np.ndarray = tf.squeeze(Q[best_idx, 0, :]).numpy()
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = Q.numpy()
            self.logging_values["J_logged"] = traj_cost.numpy()
            self.logging_values["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.logging_values["u_logged"] = self.u

        # Shift Q, Adam weights by one time step
        self.count += 1
        Q_s = self.rng.uniform(
            shape=[self.num_rollouts, 1, self.num_control_inputs],
            minval=self.action_low,
            maxval=self.action_high,
            dtype=tf.float32,
        )
        Q_shifted = tf.concat([self.Q_tf[:, 1:, :], Q_s], axis=1)
        self.Q_tf.assign(Q_shifted)

        adam_weights = self.optim.get_weights()
        if len(adam_weights) > 0:
            # if it is not time to reset, all optimizer weights are shifted for a warmstart
            w1 = tf.concat(
                [
                    adam_weights[1][:, 1:, :],
                    tf.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                ],
                axis=1,
            )
            w2 = tf.concat(
                [
                    adam_weights[2][:, 1:, :],
                    tf.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                ],
                axis=1,
            )
            self.optim.set_weights([adam_weights[0], w1, w2])

        return self.u

    def optimizer_reset(self):
        # generate random input sequence and clip to control limits
        Q = self.rng.uniform(
            [self.num_rollouts, self.mpc_horizon, self.num_control_inputs],
            self.action_low,
            self.action_high,
            dtype=tf.float32,
        )
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
        self.Q_tf = tf.Variable(Q, dtype=tf.float32)

        self.count = 0

        adam_weights = self.optim.get_weights()
        self.optim.set_weights([tf.zeros_like(el) for el in adam_weights])
