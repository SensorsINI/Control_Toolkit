from importlib import import_module

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfpd
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import create_rng, Compile

from Control_Toolkit.Controllers import template_controller


# CEM with Gaussian Mixture Model Sampling Distribution
class controller_cem_gmm_tf(template_controller):
    def __init__(self, environment: EnvironmentBatched, seed: int, num_control_inputs: int, dt: float, mpc_horizon: int, cem_outer_it: int, num_rollouts: int, predictor_name: str, predictor_intermediate_steps: int, CEM_NET_NAME: str, cem_stdev_min: float, cem_best_k: int, **kwargs):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #cem params
        self.num_rollouts = num_rollouts
        self.cem_outer_it = cem_outer_it
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.cem_samples = mpc_horizon  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.NET_NAME = CEM_NET_NAME

        #instantiate predictor
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

        self.controller_reset()
        self.u = 0.0

    @Compile
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        return traj_cost, rollout_trajectory

    @Compile
    def update_distribution(self, s: tf.Tensor, Q: tf.Tensor, traj_cost: tf.Tensor, rollout_trajectory: tf.Tensor, sampling_dist: tfpd.Distribution, rng: tf.random.Generator):
        #generate random input sequence and clip to control limits
        Q = sampling_dist.sample(sample_shape=[self.num_rollouts])
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)

        #rollout the trajectories and get cost
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
        rollout_trajectory = tf.ensure_shape(rollout_trajectory, [self.num_rollouts, self.cem_samples+1, self.env_mock.num_states])

        #sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[:self.cem_best_k]
        elite_Q = tf.gather(Q, best_idx, axis=0)

        #update the distribution for next inner loop
        distances_to_top_Q = tf.norm((elite_Q[..., tf.newaxis]-tf.transpose(elite_Q, perm=(1,2,0))[tf.newaxis, ...]), axis=[1, 2])
        selection_tensor = distances_to_top_Q[2:, :2]
        selection_idx = tf.argmin(selection_tensor, axis=1)
        closest_Q_1 = tf.concat([elite_Q[0:1,:,:], elite_Q[2:,:,:][selection_idx==0]], axis=0)
        closest_Q_2 = tf.concat([elite_Q[1:2,:,:], elite_Q[2:,:,:][selection_idx==1]], axis=0)
        num_Q_1 = tf.cast(tf.shape(closest_Q_1)[0], dtype=tf.float32)
        prob_Q_1 = num_Q_1 / self.cem_best_k
        sampling_dist = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=[prob_Q_1, 1.0 - prob_Q_1]),
            components_distribution=tfpd.Normal(
                loc=tf.stack([
                    tf.reduce_mean(closest_Q_1, axis=0),
                    tf.reduce_mean(closest_Q_2, axis=0),
                ], axis=-1),
                scale=tf.stack([
                    tf.clip_by_value(tf.math.reduce_std(closest_Q_1, axis=0), self.cem_stdev_min, 1.0e4),
                    tf.clip_by_value(tf.math.reduce_std(closest_Q_2, axis=0), self.cem_stdev_min, 1.0e4)
                ], axis=-1)
            )
        )
        # sampling_dist_updated = sampling_dist.experimental_fit(elite_Q, validate_args=True)
        
        return sampling_dist, Q, elite_Q, traj_cost, rollout_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        Q = tf.zeros((self.num_rollouts, self.cem_samples, self.num_control_inputs), dtype=tf.float32)
        rollout_trajectory = tf.zeros((self.num_rollouts, self.cem_samples+1, self.env_mock.num_states), dtype=tf.float32)
        traj_cost = tf.zeros((self.num_rollouts), dtype=tf.float32)

        for _ in range(0, self.cem_outer_it):
            self.sampling_dist, Q, elite_Q, traj_cost, rollout_trajectory = self.update_distribution(s, Q, traj_cost, rollout_trajectory, self.sampling_dist, self.rng_cem)
        
        Q, traj_cost, rollout_trajectory = Q.numpy(), traj_cost.numpy(), rollout_trajectory.numpy()
        self.u = tf.squeeze(elite_Q[0, 0, :]).numpy()
        
        # Shift distribution parameters
        prev_mue = self.sampling_dist.components_distribution.mean()
        prev_stdev = self.sampling_dist.components_distribution.stddev()
        self.sampling_dist = tfpd.MixtureSameFamily(
            mixture_distribution=self.sampling_dist.mixture_distribution,
            components_distribution=tfpd.Normal(
                loc=tf.concat([prev_mue[1:,:,:], prev_mue[-1:,:,:]], axis=0),
                scale=tf.concat([prev_stdev[1:,:,:], prev_stdev[-1:,:,:]], axis=0),
            )
        )

        self.Q_logged, self.J_logged = Q, traj_cost
        self.rollout_trajectories_logged = rollout_trajectory
        self.u_logged = self.u

        return self.u

    def controller_reset(self):
        dist_mue = (self.action_low + self.action_high) * 0.5 * tf.ones([self.cem_samples, self.num_control_inputs])
        dist_var = 0.5 * tf.ones([self.cem_samples, self.num_control_inputs])
        dist_stdev = tf.math.sqrt(dist_var)
        self.sampling_dist = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfpd.Normal(loc=tf.stack(2*[dist_mue], axis=-1), scale=tf.stack(2*[dist_stdev], axis=-1)),
        )