#Controller equivalent to the cem+grad controller from Bharadhwaj et al. 2020
#
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
import tensorflow as tf
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import CompileTF
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box


#controller class
class controller_cem_naive_grad_tf(template_controller):
    def __init__(
        self,
        cost_function: cost_function_base,
        seed: int,
        action_space: Box,
        observation_space: Box,
        mpc_horizon: int,
        cem_outer_it: int,
        num_rollouts: int,
        predictor_specification: str,
        cem_initial_action_stdev: float,
        cem_stdev_min: float,
        cem_best_k: int,
        learning_rate: float,
        gradmax_clip: float,
        controller_logging: bool,
        **kwargs,
    ):
        super().__init__(cost_function=cost_function, seed=seed, action_space=action_space, observation_space=observation_space, mpc_horizon=mpc_horizon, num_rollouts=num_rollouts, controller_logging=controller_logging)
        
        # Predictor
        self.predictor = PredictorWrapper()
        self.predictor.configure(
            batch_size=self.num_rollouts, horizon=self.mpc_horizon, predictor_specification=predictor_specification
        )
        
        # CEM parameters
        self.cem_outer_it = cem_outer_it
        self.cem_initial_action_stdev = cem_initial_action_stdev
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k

        # Optimization parameters
        self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        
        self.controller_reset()

    @CompileTF
    def predict_and_cost(self, s, rng, dist_mue, stdev):
        # generate random input sequence and clip to control limits
        Q = tf.tile(dist_mue, [self.num_rollouts, 1, 1]) + rng.normal(
            [self.num_rollouts, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32) * stdev
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
        # rollout the trajectories and record gradient
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.cost_function.get_trajectory_cost(rollout_trajectory, Q, self.u)
        # retrieve gradient
        dc_dQ = tape.gradient(traj_cost, Q)
        # modify gradients: makes sure norm of each gradient is at most "gradmax_clip".
        Q_update = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # update Q with gradient descent step
        Qn = Q-self.learning_rate*Q_update
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)
        #rollout all trajectories a last time
        rollout_trajectory = self.predictor.predict_tf(s, Qn)
        traj_cost = self.cost_function.get_trajectory_cost(rollout_trajectory, Qn, self.u)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0:self.cem_best_k]
        elite_Q = tf.gather(Qn, best_idx, axis=0)
        # update the distribution for next inner loop
        self.dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        self.stdev = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)
        return self.dist_mue, self.stdev, Qn, traj_cost, rollout_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.controller_logging:
            self.current_log["s_logged"] = s.copy()
        # tile s and convert inputs to tensor
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        #cem steps updating distribution
        for _ in range(0,self.cem_outer_it):
            self.dist_mue, self.stdev, Q, J, rollout_trajectory = self.predict_and_cost(s, self.rng, self.dist_mue, self.stdev)
        
        #after all inner loops, clip std min, so enough is explored
        #and shove all the values down by one for next control input
        self.stdev = tf.clip_by_value(self.stdev, self.cem_stdev_min, 10.0)
        self.stdev = tf.concat([self.stdev[:, 1:, :], self.cem_initial_action_stdev*tf.ones(shape=(1,1,self.num_control_inputs))], axis=1)
        self.u = tf.squeeze(self.dist_mue[0,0,:]).numpy()
        self.dist_mue = tf.concat([self.dist_mue[:, 1:, :], tf.constant((self.action_low + self.action_high) * 0.5, shape=(1,1,self.num_control_inputs))], axis=1)
        
        if self.controller_logging:
            self.current_log["Q_logged"] = Q.numpy()
            self.current_log["J_logged"] = J.numpy()
            self.current_log["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.current_log["u_logged"] = self.u
        
        return self.u

    def controller_reset(self):
        #reset controller initial distribution
        self.dist_mue = (self.action_low + self.action_high) * 0.5 * tf.ones([1, self.mpc_horizon, self.num_control_inputs])
        self.stdev = self.cem_initial_action_stdev * tf.ones([1, self.mpc_horizon, self.num_control_inputs])
