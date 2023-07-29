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

# ------------------------------------------------------------------------

logger = get_logger(__name__)


"""@CompileTF
def tf_interp(x, xs, ys):
    # Normalize data types
    ys = tf.cast(ys, tf.float32)
    xs = tf.cast(xs, tf.float32)
    x = tf.cast(x, tf.float32)

    # Pad control points for extrapolation
    xs = tf.concat([[xs[0]], xs, [xs[-1]]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # Compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # Solve for intercepts
    bs = ys - ms * xs

    # Find the line parameters at each input data point using searchsorted
    i = tf.searchsorted(xs, x)
    m = tf.gather(ms, i)
    b = tf.gather(bs, i)

    # Apply the linear mapping at each input data point
    y = m * x + b
    return y


@CompileTF
def get_interp_pdf_cdf(x_val, x_min, x_max, y_val, num_interp_pts):
    # Sort the x and y using TensorFlow operations
    x_sorted_indices = tf.argsort(x_val)
    x_sorted = tf.gather(x_val, x_sorted_indices)
    y_sorted = tf.gather(y_val, x_sorted_indices)

    # Linear interpolation using tf_interp
    x_interp = tf.linspace(x_min, x_max, num_interp_pts)
    y_interp = tf_interp(x_interp, x_sorted, y_sorted)

    # pdf using TensorFlow operations
    y_pdf = y_interp / tf.reduce_sum(y_interp)

    # cdf using TensorFlow operations
    y_cdf = tf.math.cumsum(y_pdf)

    return x_interp, y_pdf, y_cdf"""


class optimizer_rpgd_pfi(template_optimizer):
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
            kpf_keep_best: int,
            visualize_color_coded: bool,
            visualize_control_distributions: bool,
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
        self.visualize = visualize
        self.view_unoptimized = view_unoptimized
        if self.visualize:
            self.TV = TrajectoryVisualizer()
        self.visualize_color_coded = visualize_color_coded
        self.visualize_control_distributions = visualize_control_distributions
        self.visualize_per = visualize_per

        # Kinda Particle Filter:
        # Now done below due to num_control_inputs not being initialized
        # self.kpf_dimensions = (self.num_rollouts, self.mpc_horizon, self.num_control_inputs)
        # Initialize weights of size num_rollouts
        self.kpf_weights = tf.fill(self.num_rollouts, 1. / self.num_rollouts)
        self.kpf_dimensions = None
        self.kpf_keep_ratio = kpf_keep_ratio
        self.kpf_keep_number = int(max(int(num_rollouts * self.kpf_keep_ratio), 1))
        self.kpf_keep_best = kpf_keep_best
        self.kpf_sample_mean = kpf_sample_mean
        self.kpf_sample_stdev = kpf_sample_stdev
        self.kpf_g_sigma = kpf_g_sigma
        self.kpf_cdf_interp_num = kpf_cdf_interp_num
        self.kpf_num_resample = 0

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
    def get_kpf_weights(self, best_idx):
        rt_dim1, rt_dim2, rt_dim3 = self.rollout_trajectories.shape

        # METHOD 1 - trajectory similarity using kernels-------------------------------------------------------
        # becomes (n_rollouts x n_chosen_output_states)
        squeezed_rt = tf.reshape(self.rollout_trajectories[:, :, 5:7], (rt_dim1, rt_dim2 * 2))
        distances = tf.norm(squeezed_rt[:, None] - squeezed_rt, axis=-1)

        # width of Gaussian kernel and distances
        g_distances = 1 - tf.exp(-distances ** 2 / (2 * self.kpf_g_sigma ** 2))
        g_distances = tf.linalg.set_diag(g_distances, tf.ones(rt_dim1) * np.inf)

        # find the closest similarity to any neighbor, use that as a divergence metric
        divergence_metric = tf.reduce_min(g_distances, axis=1)
        # divergence_metric = tf.reduce_mean(g_distances, axis=1)
        # -------------------------------------------------------------------------------------------------------

        # METHOD 2 - calculate the distances between endpoints--------------------------------------------------
        """reshaped_rt = tf.reshape(self.rollout_trajectories[:, rt_dim2 - 1, 5:7], (rt_dim1, 1, 2))
        end_rollout_trajectories = tf.squeeze(reshaped_rt, axis=1)
        distances = tf.norm(end_rollout_trajectories[:, None] - end_rollout_trajectories, axis=-1)

        distances = tf.linalg.set_diag(distances, tf.ones(rt_dim1) * np.inf)
        divergence_metric = tf.reduce_min(distances, axis=1)"""

        # get threshold distance for resampling
        # threshold_distance = tf.cast(tf.reduce_max(worst_values), dtype=tf.float32)
        # -------------------------------------------------------------------------------------------------------

        # find indices of furthest (best) points according to predefined kpf_keep_number
        # divergence_metric decreases with more similarity
        furthest_indices = tf.math.top_k(-divergence_metric, k=self.kpf_keep_number).indices
        total_keep_idx = tf.concat([furthest_indices, best_idx[:self.kpf_keep_best]], 0)
        total_keep_idx, _ = tf.unique(total_keep_idx)
        total_keep_idx = tf.cast(total_keep_idx, tf.int32)

        num_resample = self.num_rollouts - len(total_keep_idx)

        # ALTERNATIVE 1: simple resampling from given distribution--------------------------------------
        # Qres = self.sample_actions(self.rng, self.num_rollouts - len(total_keep_idx))
        # -----------------------------------------------------------------------------------------------

        # ALTERNATIVE 2: smart KPF resampling using weights--------------------------------------------------
        # update weights and normalize
        kpf_weights = divergence_metric / tf.reduce_sum(divergence_metric)

        return kpf_weights, num_resample, total_keep_idx

    # @CompileTF
    def kpf_step(self, Qn):
        # uniform_samples = tf.random.uniform(shape=(num_resample, self.mpc_horizon, self.num_control_inputs))

        # Qn_reduced = tf.reshape(Qn, (self.num_rollouts, self.mpc_horizon * self.num_control_inputs))

        """for i in range(self.num_control_inputs):
            for j in range(self.mpc_horizon):
                # Sort the x and y using TensorFlow operations
                sliced_inputs = Qn[:, j, i]

                sorted_indices = tf.argsort(sliced_inputs)
                Qn_sorted = tf.gather(sliced_inputs, sorted_indices)
                weights_sorted = tf.gather(self.kpf_weights, sorted_indices)
                weights_cdf = tf.cumsum(weights_sorted)

                uniform_samples = tf.linspace(0.0, 1.0, num_resample)  # MOVE OUT!

                # Find indices for the sampled points using binary search
                indices = tf.math.minimum(tf.searchsorted(weights_cdf, uniform_samples), self.num_rollouts - 1)

                # n1 = uniform_samples
                # n2 = tf.gather(weights_cdf, tf.math.maximum(indices - 1, 0))
                # n3 = tf.gather(weights_cdf, indices)

                prev_indices = tf.math.maximum(indices - 1, 0)

                alphas = tf.math.maximum(tf.math.divide(uniform_samples - tf.gather(weights_cdf, prev_indices),
                                                        tf.gather(weights_cdf, indices) - tf.gather(weights_cdf, prev_indices))
                                         , 0)

                samples_per_ci_per_timestep = tf.gather(Qn_sorted, prev_indices) + alphas * (tf.gather(Qn_sorted, indices) - tf.gather(Qn_sorted, prev_indices))"""

        Q_kpf_res = self.sample_actions(self.rng, self.kpf_num_resample)

        """trajectories = []
        for i in range(num_resample):
            trajectory = []
            for j in range(self.mpc_horizon):
                inputs_per_timestep = []
                for k in range(self.num_control_inputs):
                    (
                        sample_x,
                        _,
                        cdf_y
                    ) = get_interp_pdf_cdf(Qn[:, j, k],
                                           self.action_low[k],
                                           self.action_high[k],
                                           self.kpf_weights,
                                           self.kpf_cdf_interp_num)
                    rev_sampled_input = tf_interp(uniform_samples[i, j, k], cdf_y, sample_x)
                    inputs_per_timestep.append(rev_sampled_input)
                trajectory.append(inputs_per_timestep)
            trajectories.append(trajectory)

        Q_kpf_res = tf.convert_to_tensor(trajectories, dtype=tf.float32)
        Q_kpf_res = tf.reshape(Q_kpf_res, (num_resample, self.mpc_horizon, self.num_control_inputs))"""
        # ----------------------------------------------------------------------------------------------------

        return Q_kpf_res

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
        # ------------------------------------------------------------------

        # # Unnecessary Part
        # # get distribution of kept trajectories. This is actually unnecessary for this optimizer, might be incorparated into another one tho
        # elite_Q = tf.gather(Q, best_idx, axis=0)
        # dist_mue = tf.math.reduce_mean(elite_Q, axis=0, keepdims=True)
        # dist_std = tf.math.reduce_std(elite_Q, axis=0, keepdims=True)

        # dist_mue = tf.concat(
        #     [
        #         dist_mue[:, 1:, :],
        #         (self.action_low + self.action_high)
        #         * 0.5
        #         * tf.ones([1, 1, self.num_control_inputs]),
        #     ],
        #     axis=1,
        # )

        # # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        # dist_std = tf.clip_by_value(dist_std, self.sample_stdev, 10.0)
        # dist_std = tf.concat(
        #     [
        #         dist_std[:, 1:, :],
        #         self.sample_stdev
        #         * tf.ones(shape=[1, 1, self.num_control_inputs]),
        #     ],
        #     axis=1,
        # )
        # # End of unnecessary part

        # Retrieve optimal input and warmstart for next iteration
        Qn = tf.concat(
            [Q[:, self.shift_previous:, :], self.lib.tile(Q[:, -1:, :], (1, self.shift_previous, 1))]
            , axis=1)
        return Qn, best_idx, traj_cost, rollout_trajectory

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

        if self.visualize and self.view_unoptimized:
            (
                unoptimized_Q,
                _,
                _,
                unoptimized_rollout_trajectories,
            ) = self.get_action(s, self.Q_tf)
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
            J,
            self.rollout_trajectories,
        ) = self.get_action(s, self.Q_tf)
        self.u_nom = self.Q_tf[tf.newaxis, best_idx[0], :, :]
        self.u = self.u_nom[0, 0, :].numpy()

        # VISUALIZE TRAJECTORIES --------------------
        if self.visualize:
            self.TV.plot_update(self.rollout_trajectories, Qn, unoptimized_rollout_trajectories, unoptimized_Q)
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
                total_keep_idx
            ) = self.get_kpf_weights(best_idx)

            # KPF STEP ------------------------------------------------------------------------------------------------
            Qres = self.kpf_step(Qn)

            # Qres = self.sample_actions(self.rng, self.kpf_num_resample)
            # ------------------------------------------------------------------------------------------

            # VISUALIZE COLOR CODED TRAJECTORIES-----------------------------------------
            if self.visualize_color_coded and self.count % self.visualize_per == 0:
                visualize_color_coded_trajectories(self.rollout_trajectories,
                                                   self.kpf_weights,
                                                   self.cost_function.cost_function.variable_parameters.lidar_points,
                                                   self.cost_function.cost_function.variable_parameters.next_waypoints)
                visualize_color_coded_trajectories(unoptimized_rollout_trajectories,
                                                   self.kpf_weights,
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
                                                      Qn)
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
                        tf.gather(adam_weights[1], best_idx)[:, 1:, :],
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                wk2 = tf.concat(
                    [
                        tf.gather(adam_weights[2], best_idx)[:, 1:, :],
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
