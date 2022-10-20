from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
import tensorflow as tf
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box
from SI_Toolkit.Functions.TF.Compile import CompileTF


class optimizer_random_action_tf(template_optimizer):
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: cost_function_base,
        action_space: Box,
        observation_space: Box,
        seed: int,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_specification: str,
        optimizer_logging: bool,
    ):
        super().__init__(
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
        
        self.optimizer_reset()
    
    @CompileTF
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    # step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
        # Start all trajectories in current state
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        
        Q = self.rng.uniform(
            shape=[self.num_rollouts, self.mpc_horizon, self.num_control_inputs],
            minval=self.action_low,
            maxval=self.action_high,
            dtype=tf.float32,
        )
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

        return self.u

    def optimizer_reset(self):
        # generate random input sequence and clip to control limits
        Q = self.rng.uniform(
                shape=[self.num_rollouts, self.mpc_horizon, self.num_control_inputs],
                minval=self.action_low,
                maxval=self.action_high,
                dtype=tf.float32,
            )
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
