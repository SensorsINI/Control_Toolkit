import numpy as np
import tensorflow as tf
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box
from SI_Toolkit.Functions.TF.Compile import CompileTF
from SI_Toolkit.Predictors import template_predictor


class controller_random_action(template_controller):
    def __init__(
        self,
        predictor: template_predictor,
        cost_function: cost_function_base,
        seed: int,
        action_space: Box,
        observation_space: Box,
        mpc_horizon: int,
        num_rollouts: int,
        controller_logging: bool,
        **kwargs,
    ):
        super().__init__(predictor=predictor, cost_function=cost_function, seed=seed, action_space=action_space, observation_space=observation_space, mpc_horizon=mpc_horizon, num_rollouts=num_rollouts, controller_logging=controller_logging)
        self.controller_reset()
    
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
        if self.controller_logging:
            self.current_log["s_logged"] = s.copy()
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
        
        if self.controller_logging:
            self.current_log["Q_logged"] = Q.numpy()
            self.current_log["J_logged"] = traj_cost.numpy()
            self.current_log["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.current_log["u_logged"] = self.u

        return self.u

    def controller_reset(self):
        # generate random input sequence and clip to control limits
        Q = self.rng.uniform(
                shape=[self.num_rollouts, self.mpc_horizon, self.num_control_inputs],
                minval=self.action_low,
                maxval=self.action_high,
                dtype=tf.float32,
            )
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
