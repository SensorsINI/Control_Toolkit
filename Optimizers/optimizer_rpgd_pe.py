from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF, get_logger
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

# FOR VISUALIZING TRAJECTORIES--------------------------------------------
from Control_Toolkit.others.trajectory_visualize import TrajectoryVisualizer
from Control_Toolkit.others.standalone_visualizers import *
# ------------------------------------------------------------------------

# FOR KPF-----------------------------------------------------------------
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import tensorflow_probability as tfp
import random
import math

# ------------------------------------------------------------------------

logger = get_logger(__name__)


class optimizer_rpgd_pe(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}

    def __init__(
            self,
            predictor: PredictorWrapper,
            cost_function: CostFunctionWrapper,
            control_limits: "Tuple[np.ndarray, np.ndarray]",
            computation_library: "type[ComputationLibrary]",
            seed: int,
            mpc_horizon: int,
            num_rollouts: int,
            outer_its: int,
            sample_stdev: float,
            sample_mean: float,
            sample_whole_control_space: bool,
            uniform_dist_min: float,
            uniform_dist_max: float,
            resamp_per: int,
            period_interpolation_inducing_points: int,
            SAMPLING_DISTRIBUTION: str,
            shift_previous: int,
            warmup: bool,
            warmup_iterations: int,
            learning_rate: float,
            opt_keep_k_ratio: float,
            gradmax_clip: float,
            rtol: float,
            adam_beta_1: float,
            adam_beta_2: float,
            adam_epsilon: float,
            optimizer_logging: bool,
            calculate_optimal_trajectory: bool,
            visualize: bool,
            view_unoptimized: bool,
            kpf_sample_stdev: float,
            kpf_sample_mean: float,
            kpf_keep_ratio: float,
            # kpf_keep_best: int,
            kpf_perturb_best_ratio: float,
            kpf_perturb_sigma_ratio: float,
            visualize_color_coded: bool,
            visualize_color_coded_advanced: bool,
            visualize_control_distributions: bool,
            visualize_control_2d: bool,
            visualize_per: int,
            kpf_g_sigma: float,
            kpf_cdf_interp_num: int,
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

        # Create second predictor for computing optimal trajectories
        self.predictor_single_trajectory = self.predictor.copy()

        # RPGD parameters
        self.outer_its = outer_its

        self.sample_stdev = tf.convert_to_tensor(sample_stdev, dtype=tf.float32)
        self.sample_mean = tf.convert_to_tensor(sample_mean, dtype=tf.float32)

        self.sample_whole_control_space = sample_whole_control_space
        if self.sample_whole_control_space:
            self.sample_min = tf.convert_to_tensor(self.action_low, dtype=tf.float32)
            self.sample_max = tf.convert_to_tensor(self.action_high, dtype=tf.float32)
        else:
            self.sample_min = tf.convert_to_tensor(uniform_dist_min, dtype=tf.float32)
            self.sample_max = tf.convert_to_tensor(uniform_dist_max, dtype=tf.float32)

        self.resamp_per = resamp_per
        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.shift_previous = shift_previous
        self.do_warmup = warmup
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = int(max(int(num_rollouts * opt_keep_k_ratio), 1))
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol
        self.SAMPLING_DISTRIBUTION = SAMPLING_DISTRIBUTION

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.Interpolator = None

        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )

        self.calculate_optimal_trajectory = calculate_optimal_trajectory
        self.optimal_trajectory = None
        self.optimal_control_sequence = None
        self.predict_optimal_trajectory = CompileTF(self._predict_optimal_trajectory)

        # VISUALS:
        # TV class
        self.visualize = visualize
        self.view_unoptimized = view_unoptimized
        if self.visualize:
            self.TV = TrajectoryVisualizer()
        # Standalone
        self.visualize_color_coded = visualize_color_coded
        self.visualize_color_coded_advanced = visualize_color_coded_advanced
        self.visualize_control_distributions = visualize_control_distributions
        self.visualize_control_2d = visualize_control_2d
        self.visualize_per = visualize_per

        # Kinda Particle Filter:
        # Now done below due to num_control_inputs not being initialized
        # self.kpf_dimensions = (self.num_rollouts, self.mpc_horizon, self.num_control_inputs)
        # Initialize weights of size num_rollouts
        self.kpf_weights = tf.fill(self.num_rollouts, 1. / self.num_rollouts)
        self.kpf_dimensions = None
        self.kpf_keep_ratio = kpf_keep_ratio
        self.kpf_keep_number = int(max(int(num_rollouts * self.kpf_keep_ratio), 1))
        # self.kpf_keep_best = kpf_keep_best
        self.kpf_sample_mean = kpf_sample_mean
        self.kpf_sample_stdev = kpf_sample_stdev
        self.kpf_g_sigma = kpf_g_sigma
        """self.kpf_cdf_interp_num = kpf_cdf_interp_num"""
        self.kpf_num_resample = 0
        """self.kpf_limits_low = - (self.action_high - self.action_low) / 2
        self.kpf_limits_high = - self.kpf_limits_low"""
        self.kpf_perturb_best = int(max(int(num_rollouts * kpf_perturb_best_ratio), 1))
        self.kpf_perturb_sigma = (self.action_high - self.action_low) * kpf_perturb_sigma_ratio

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

        self.Interpolator = Interpolator(self.mpc_horizon, self.period_interpolation_inducing_points,
                                         self.num_control_inputs, self.lib)

        self.predictor_single_trajectory.configure(
            batch_size=1, horizon=self.mpc_horizon, dt=dt,  # TF requires constant batch size
            predictor_specification=predictor_specification,
        )

        # KPF Initialize dimensions --------------------------
        self.kpf_dimensions = (self.num_rollouts, self.mpc_horizon, self.num_control_inputs)
        # ----------------------------------------------------

        self.optimizer_reset()

    # @CompileTF
    def sample_actions(self, rng_gen: tf.random.Generator, batch_size: int):
        if self.SAMPLING_DISTRIBUTION == "normal":
            Qn = rng_gen.normal(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                mean=self.kpf_sample_mean,  # Was sample mean before!
                stddev=self.kpf_sample_stdev,  # Was sample stdev before!
                dtype=tf.float32,
            )
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            Qn = rng_gen.uniform(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                minval=self.sample_min,
                maxval=self.sample_max,
                dtype=tf.float32,
            )
        else:
            raise ValueError(f"RPGD cannot interpret sampling type {self.SAMPLING_DISTRIBUTION}")
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)

        Qn = self.Interpolator.interpolate(Qn)

        return Qn

    @CompileTF
    def get_kpf_weights(self, best_idx, rt):
        rt_dim1, rt_dim2, rt_dim3 = rt.shape

        sliced_rt = rt[:, :, 5:7]  # makes sense to select 5 and 6 (x,y) in last dim

        # squeezed_rt = tf.reshape(sliced_rt, shape=(rt_dim1, rt_dim2 * sliced_rt.shape[2]))
        squeezed_rt = sliced_rt

        delta = squeezed_rt[:, None] - squeezed_rt  # 32,32,16,2

        distances_separate_timesteps = tf.norm(delta, axis=-1)  # 32,32,16
        distances0 = tf.reduce_mean(distances_separate_timesteps, axis=-1)  # 32,32 mean distance over timesteps
        distances = distances0 / tf.reduce_max(distances0)

        g_similarity = tf.exp(-(distances ** 2) / (2 * self.kpf_g_sigma ** 2))
        g_similarity_min = tf.linalg.set_diag(g_similarity, - tf.ones(rt_dim1) * np.inf)  # np.inf if not reduce_min below!

        # find the closest similarity to any neighbor, use that as a divergence metric
        divergence_metric_min = tf.reduce_min(1 - g_similarity_min, axis=1)
        divergence_metric_mean = tf.reduce_mean(1 - g_similarity, axis=1)

        if False:
            divergence_metric_min_no_g = tf.reduce_mean(distances, axis=1)
            divergence_metric_mean_no_g = tf.reduce_mean(distances, axis=1)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_min,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_min_no_g,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean_no_g,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)

        if False:
            g_similarity0 = tf.exp(-(distances ** 2) / (2 * 0.01 ** 2))
            g_similarity2 = tf.exp(-(distances ** 2) / (2 * 0.05 ** 2))
            g_similarity3 = tf.exp(-(distances ** 2) / (2 * 1 ** 2))
            divergence_metric_mean0 = tf.reduce_mean(1 - g_similarity0, axis=1)
            divergence_metric_mean2 = tf.reduce_mean(1 - g_similarity2, axis=1)
            divergence_metric_mean3 = tf.reduce_mean(1 - g_similarity3, axis=1)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean0,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean2,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)
            """visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean2,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)"""
            visualize_color_coded_trajectories(self.rollout_trajectories,
                                               divergence_metric_mean3,
                                               None,
                                               None,
                                               self.cost_function.cost_function.variable_parameters.lidar_points,
                                               self.cost_function.cost_function.variable_parameters.next_waypoints)

        divergence_metric = divergence_metric_mean

        # -------------------------------------------------------------------------------------------------------

        # find indices of furthest (best) points according to predefined kpf_keep_number
        # divergence_metric decreases with more similarity
        sorted_indices = tf.argsort(divergence_metric)
        num_resample = self.num_rollouts - len(best_idx)

        """furthest_indices = sorted_indices[-self.kpf_keep_number:]
        # furthest_indices = tf.math.top_k(divergence_metric, k=self.kpf_keep_number).indices
        total_keep_idx = tf.concat([furthest_indices, best_idx[:self.kpf_keep_best]], 0)
        total_keep_idx, _ = tf.unique(total_keep_idx)
        total_keep_idx = tf.cast(total_keep_idx, tf.int32)

        num_resample = self.num_rollouts - len(total_keep_idx)"""

        kpf_perturb_idx = sorted_indices[-self.kpf_perturb_best:]
        kpf_weights = divergence_metric

        return kpf_weights, num_resample, kpf_perturb_idx, sorted_indices

    @CompileTF
    def get_kpf_samples(self, Q, noise, pick_perturb, perturb_idx):
        # return self.sample_actions(self.rng, self.kpf_num_resample)

        # return self.sample_actions(self.rng, self.kpf_num_resample)
        """perturb_indices = sorted_indices[-self.kpf_perturb_best:]
        ratio = tf.cast(tf.math.ceil(self.kpf_num_resample / self.kpf_perturb_best), dtype=tf.int32)
        tiled_pi = tf.tile(perturb_indices, [ratio])[:self.kpf_num_resample]"""

        picked_perturbed = tf.gather(perturb_idx, pick_perturb)
        Q_picked = tf.gather(Q, picked_perturbed)
        Q_final = tf.add(Q_picked, noise)

        return Q_final

    def predict_and_cost(self, s: tf.Tensor, Q: tf.Variable):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    @CompileTF
    def grad_step(
            self, s: tf.Tensor, Q: tf.Variable, opt: tf.keras.optimizers.Optimizer
    ):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            traj_cost, _ = self.predict_and_cost(s, Q)
        # retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # use optimizer to apply gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Qn = tf.clip_by_value(Q, self.action_low, self.action_high)
        return Qn, traj_cost

    @CompileTF
    def get_action(self, s: tf.Tensor, Q: tf.Variable):
        # Rollout trajectories and retrieve cost
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)

        # CHANGED FOR KPF---------------------------------------------------
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[: self.opt_keep_k]
        worst_idx = sorted_cost[-self.opt_keep_k:]
        # ------------------------------------------------------------------

        # Retrieve optimal input and warmstart for next iteration
        Qn = tf.concat(
            [Q[:, self.shift_previous:, :], self.lib.tile(Q[:, -1:, :], (1, self.shift_previous, 1))]
            , axis=1)
        return Qn, best_idx, worst_idx, traj_cost, rollout_trajectory

    def _predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}

        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        # warm start setup
        if self.count == 0:
            iters = self.first_iter_count
        else:
            iters = self.outer_its

        # VISUALIZE UNOPTIMIZED TRAJECTORIES --------------------
        # Calculate unoptimized trajectories:
        unoptimized_Q = None
        unoptimized_rollout_trajectories = None

        if (self.visualize and self.view_unoptimized) or self.visualize_color_coded_advanced or self.visualize_control_2d:
            (
                _,
                _,
                _,
                _,
                unoptimized_rollout_trajectories,
            ) = self.get_action(s, self.Q_tf)
            unoptimized_Q = self.Q_tf
        # --------------------------------------------

        # optimize control sequences with gradient based optimization
        # prev_cost = tf.convert_to_tensor(np.inf, dtype=tf.float32)
        for _ in range(0, iters):
            Qn, traj_cost = self.grad_step(s, self.Q_tf, self.opt)
            self.Q_tf.assign(Qn)

            # check for convergence of optimization
            # if bool(
            #     tf.reduce_mean(
            #         tf.math.abs((traj_cost - prev_cost) / (prev_cost + self.rtol))
            #     )
            #     < self.rtol
            # ):
            #     # assume that we have converged sufficiently
            #     break
            # prev_cost = tf.identity(traj_cost)

        # Prepare warmstart
        (
            Qn,
            best_idx,
            worst_idx,
            J,
            self.rollout_trajectories,
        ) = self.get_action(s, self.Q_tf)
        self.u_nom = self.Q_tf[tf.newaxis, best_idx[0], :, :]
        self.u = self.u_nom[0, 0, :].numpy()

        # VISUALIZE TRAJECTORIES --------------------
        if self.visualize:
            self.TV.plot_update(self.rollout_trajectories, self.Q_tf, unoptimized_rollout_trajectories, unoptimized_Q)
        # --------------------------------------------

        if self.optimizer_logging:
            self.logging_values["Q_logged"] = self.Q_tf.numpy()
            self.logging_values["J_logged"] = J.numpy()
            self.logging_values["rollout_trajectories_logged"] = self.rollout_trajectories.numpy()
            self.logging_values["trajectory_ages_logged"] = self.trajectory_ages.numpy()
            self.logging_values["u_logged"] = self.u

        self.optimal_control_sequence = self.lib.to_numpy(self.u_nom)

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstarts the initial guess, but also the initial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            """# if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(
                self.rng, self.num_rollouts - self.opt_keep_k
            )
            Q_keep = tf.gather(Qn, best_idx)  # resorting according to costs"""

            (
                self.kpf_weights,
                self.kpf_num_resample,
                kpf_perturb_idx,
                sorted_indices
            ) = self.get_kpf_weights(best_idx, self.rollout_trajectories)

            total_keep_idx = best_idx

            """noise = tf.random.uniform(shape=(self.kpf_num_resample,
                                             self.mpc_horizon,
                                             self.num_control_inputs),
                                      minval=self.kpf_limits_low, maxval=self.kpf_limits_high)"""

            noise = tf.random.normal(shape=(self.kpf_num_resample, self.mpc_horizon, self.num_control_inputs),
                                     mean=0,
                                     stddev=self.kpf_perturb_sigma)

            num_perturb = len(kpf_perturb_idx)
            random_floats = tf.random.uniform(shape=(self.kpf_num_resample,),
                                              minval=0.0,
                                              maxval=float(num_perturb),
                                              dtype=tf.float32)
            random_pick_perturb = tf.cast(random_floats, dtype=tf.int32)

            Qres = self.get_kpf_samples(self.Q_tf, noise, random_pick_perturb, kpf_perturb_idx)
            # Qres = self.sample_actions(self.rng, self.kpf_num_resample)

            if False and self.count % 20 == 0:
                (_, _, _, _, kpf_trajectories,) = self.get_action(s,
                                                                  tf.concat([Qres, tf.zeros(
                                                                      shape=(self.num_rollouts - self.kpf_num_resample,
                                                                             self.mpc_horizon,
                                                                             self.num_control_inputs
                                                                             )
                                                                  )
                                                                             ], axis=0)
                                                                  )
                unop_trajectories = unoptimized_rollout_trajectories
                visualize_obstacles(self.rollout_trajectories,
                                   self.kpf_weights,
                                   unop_trajectories,
                                   kpf_trajectories,
                                   self.cost_function.cost_function.variable_parameters.lidar_points,
                                   self.cost_function.cost_function.variable_parameters.next_waypoints)
                visualize_obstacles_rt(self.rollout_trajectories,
                                    self.kpf_weights,
                                    unop_trajectories,
                                    kpf_trajectories,
                                    self.cost_function.cost_function.variable_parameters.lidar_points,
                                    self.cost_function.cost_function.variable_parameters.next_waypoints)
                visualize_obstacles_kpf(self.rollout_trajectories,
                                    self.kpf_weights,
                                    unop_trajectories,
                                    kpf_trajectories,
                                    self.cost_function.cost_function.variable_parameters.lidar_points,
                                    self.cost_function.cost_function.variable_parameters.next_waypoints)

            # VISUALIZE COLOR CODED TRAJECTORIES-----------------------------------------
            if (self.visualize_color_coded or self.visualize_color_coded_advanced) and self.count % self.visualize_per == 0:
                unop_trajectories, kpf_trajectories = None, None
                if self.visualize_color_coded_advanced:
                    (_, _, _, _, kpf_trajectories,) = self.get_action(s,
                                                                      tf.concat([Qres, tf.zeros(
                                                                        shape=(self.num_rollouts - self.kpf_num_resample,
                                                                               self.mpc_horizon,
                                                                               self.num_control_inputs
                                                                               )
                                                                                )
                                                                      ], axis=0)
                                                                      )
                    unop_trajectories = unoptimized_rollout_trajectories
                visualize_color_coded_trajectories(self.rollout_trajectories,
                                                   self.kpf_weights,
                                                   unop_trajectories,
                                                   kpf_trajectories,
                                                   self.cost_function.cost_function.variable_parameters.lidar_points,
                                                   self.cost_function.cost_function.variable_parameters.next_waypoints)
            # ---------------------------------------------------------------------------

            # VISUALIZE DISTRIBUTION WITH WEIGHTS ---------------------------------------
            if self.visualize_control_distributions and self.count % self.visualize_per == 0:
                visualize_control_input_distributions(self.action_low,
                                                      self.action_high,
                                                      self.kpf_weights,
                                                      self.mpc_horizon,
                                                      self.kpf_cdf_interp_num,
                                                      self.Q_tf,
                                                      Qres)
            # ---------------------------------------------------------------------------

            # VISUALIZE CONTROL INPUTS IN 2D ---------------------------------------
            if self.visualize_control_2d and self.count % self.visualize_per == 0:
                visualize_control_input_2d(self.action_low,
                                           self.action_high,
                                           self.kpf_weights,
                                           self.mpc_horizon,
                                           self.Q_tf,
                                           Qres,
                                           unoptimized_Q)
            # ---------------------------------------------------------------------------

            Q_keep = tf.gather(Qn, total_keep_idx)

            Qn = tf.concat([Qres, Q_keep], axis=0)

            self.trajectory_ages = tf.concat([
                tf.zeros(self.kpf_num_resample, dtype=tf.int32),
                tf.gather(self.trajectory_ages, total_keep_idx),
            ], axis=0)  # total_keep_idx WAS BEST_IDX; num_resample WAS OPT_KEEP_K BEFORE!!!!!!
            # --------------------------------------------------------------------------------------------------------------------

            # Updating the weights of adam:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            if len(adam_weights) > 0:
                wk1 = tf.concat(
                    [
                        tf.gather(adam_weights[1], best_idx)[:, 1:, :],   # CHANGE to total_keep_idx
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                wk2 = tf.concat(
                    [
                        tf.gather(adam_weights[2], best_idx)[:, 1:, :],    # CHANGE to total_keep_idx
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                # For the new trajectories they are reset to 0
                w1 = tf.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.mpc_horizon,
                        self.num_control_inputs,
                    ]
                )
                w2 = tf.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.mpc_horizon,
                        self.num_control_inputs,
                    ]
                )
                w1 = tf.concat([w1, wk1], axis=0)
                w2 = tf.concat([w2, wk2], axis=0)
                # Set weights
                self.opt.set_weights([adam_weights[0], w1, w2])
        else:
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
                self.opt.set_weights([adam_weights[0], w1, w2])
        self.trajectory_ages += 1
        self.Q_tf.assign(Qn)
        self.count += 1

        if self.calculate_optimal_trajectory:
            self.optimal_trajectory = self.lib.to_numpy(self.predict_optimal_trajectory(s[:1, :], self.u_nom))

        return self.u

    def optimizer_reset(self):
        # # unnecessary part: Adaptive sampling distribution
        # self.dist_mue = (
        #     (self.action_low + self.action_high)
        #     * 0.5
        #     * tf.ones([1, self.mpc_horizon, self.num_control_inputs])
        # )
        # self.stdev = self.sample_stdev * tf.ones(
        #     [1, self.mpc_horizon, self.num_control_inputs]
        # )
        # # end of unnecessary part

        # sample new initial guesses for trajectories
        Qn = self.sample_actions(self.rng, self.num_rollouts)
        if hasattr(self, "Q_tf"):
            self.Q_tf.assign(Qn)
        else:
            self.Q_tf = tf.Variable(Qn)
        self.count = 0

        # reset optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        self.trajectory_ages: tf.Tensor = tf.zeros((self.num_rollouts), dtype=tf.int32)




# GRAVEYARD

        '''x = self.rollout_trajectories[:, :, 5]
        y = self.rollout_trajectories[:, :, 6]

        new_x = x[:, None]
        new_y = y[:, None]

        d_x = x[:, None] - x
        d_y = y[:, None] - y

        distances = tf.sqrt(tf.square(d_x) + tf.square(d_y))

        distances_sum = tf.reduce_mean(distances, axis=2)

        # g_distances = 1 - tf.exp(-distances_sum ** 2 / (2 * self.kpf_g_sigma ** 2))
        g_distances = distances_sum

        g_distances = tf.linalg.set_diag(g_distances, tf.ones(rt_dim1) * np.inf)
        divergence_metric = tf.reduce_min(g_distances, axis=1)

        """g_distances = tf.linalg.set_diag(g_distances, tf.zeros(rt_dim1))
        divergence_metric = tf.reduce_mean(g_distances, axis=1)"""'''

        # METHOD 1 - trajectory similarity using kernels-------------------------------------------------------
        # becomes (n_rollouts x n_chosen_output_states)
        """squeezed_rt = tf.reshape(self.rollout_trajectories[:, :, 5:7], (rt_dim1, rt_dim2 * 2))
        distances = tf.norm(squeezed_rt[:, None] - squeezed_rt, axis=-1)

        # width of Gaussian kernel and distances
        g_distances = 1 - tf.exp(-distances ** 2 / (2 * self.kpf_g_sigma ** 2))
        g_distances = tf.linalg.set_diag(g_distances, tf.ones(rt_dim1) * np.inf)  # np.inf if not reduce_min below!

        # find the closest similarity to any neighbor, use that as a divergence metric
        divergence_metric = tf.reduce_min(g_distances, axis=1)
        # divergence_metric = tf.reduce_mean(g_distances, axis=1)"""
        # -------------------------------------------------------------------------------------------------------

        # METHOD 2 - calculate the distances between endpoints--------------------------------------------------
        """reshaped_rt = tf.reshape(self.rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))
        end_rollout_trajectories = tf.squeeze(reshaped_rt, axis=1)
        distances = tf.norm(end_rollout_trajectories[:, None] - end_rollout_trajectories, axis=-1)

        distances = tf.linalg.set_diag(distances, tf.ones(rt_dim1) * np.inf)
        divergence_metric = tf.reduce_min(distances, axis=1)"""

        # get threshold distance for resampling
        # threshold_distance = tf.cast(tf.reduce_max(worst_values), dtype=tf.float32)
