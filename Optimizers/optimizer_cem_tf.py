from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


#cem class
class optimizer_cem_tf(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}
    
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: CostFunctionWrapper,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        computation_library: "type[ComputationLibrary]",
        seed: int,
        mpc_horizon: int,
        cem_outer_it: int,
        cem_initial_action_stdev: float,
        num_rollouts: int,
        cem_stdev_min: float,
        cem_best_k: int,
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
        
        # CEM parameters
        self.cem_outer_it = cem_outer_it
        self.cem_initial_action_stdev = cem_initial_action_stdev
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.warmup = warmup
        self.warmup_iterations = warmup_iterations

    @CompileTF
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(rollout_trajectory, Q, self.u)
        return traj_cost, rollout_trajectory

    @CompileTF
    def update_distribution(self, s: tf.Tensor, Q: tf.Tensor, traj_cost: tf.Tensor, rollout_trajectory: tf.Tensor, dist_mue: tf.Tensor, stdev: tf.Tensor, rng: tf.random.Generator):
        #generate random input sequence and clip to control limits
        Q = tf.tile(dist_mue,(self.num_rollouts,1,1)) + tf.multiply(rng.normal(
            shape=(self.num_rollouts, self.mpc_horizon, self.num_control_inputs), dtype=tf.float32), stdev)
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)

        #rollout the trajectories and get cost
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
        rollout_trajectory = tf.ensure_shape(rollout_trajectory, [self.num_rollouts, self.mpc_horizon+1, self.num_states])

        #sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[:self.cem_best_k]
        elite_Q = tf.gather(Q, best_idx, axis=0)
        #update the distribution for next inner loop
        dist_mue = tf.reduce_mean(elite_Q, axis=0, keepdims=True)
        stdev = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        
        return dist_mue, stdev, Q, elite_Q, traj_cost, rollout_trajectory

    # step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        Q = tf.zeros((self.num_rollouts, self.mpc_horizon, self.num_control_inputs), dtype=tf.float32)
        rollout_trajectory = tf.zeros((self.num_rollouts, self.mpc_horizon+1, self.num_states), dtype=tf.float32)
        traj_cost = tf.zeros((self.num_rollouts), dtype=tf.float32)

        iterations = self.warmup_iterations if self.warmup and self.count == 0 else self.cem_outer_it
        for _ in range(0, iterations):
            self.dist_mue, self.stdev, Q, elite_Q, traj_cost, rollout_trajectory = self.update_distribution(s, Q, traj_cost, rollout_trajectory, self.dist_mue, self.stdev, self.rng)
        
        Q, traj_cost, rollout_trajectory = Q.numpy(), traj_cost.numpy(), rollout_trajectory.numpy()

        #after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.stdev = tf.clip_by_value(self.stdev, self.cem_stdev_min, 1.0e8)
        self.stdev = tf.concat([self.stdev[:,1:,:], self.cem_initial_action_stdev*tf.ones((1,1,self.num_control_inputs))], axis=1)
        self.u = tf.squeeze(elite_Q[0,0,:]).numpy()
        self.dist_mue = tf.concat([self.dist_mue[:,1:,:], (self.action_low + self.action_high) * 0.5 * tf.ones((1,1,self.num_control_inputs))], axis=1)
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = Q
            self.logging_values["J_logged"] = traj_cost
            self.logging_values["rollout_trajectories_logged"] = rollout_trajectory
            self.logging_values["u_logged"] = self.u

        self.count += 1
        return self.u

    def optimizer_reset(self):
        self.dist_mue = (self.action_low + self.action_high) * 0.5 * tf.ones([1, self.mpc_horizon, self.num_control_inputs])
        self.stdev = self.cem_initial_action_stdev * tf.ones([1, self.mpc_horizon, self.num_control_inputs])
        self.count = 0
        self.u = 0.0
