from typing import Tuple, Union
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF, get_logger
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

logger = get_logger(__name__)


class optimizer_rpgd_particle_tf(template_optimizer):
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
        resamp_per: int,
        period_interpolation_inducing_points: int,
        SAMPLING_DISTRIBUTION: str,
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
        
        # RPGD parameters
        self.outer_its = outer_its
        self.sample_stdev = sample_stdev
        self.resamp_per = resamp_per
        self.do_warmup = warmup
        self.SAMPLING_DISTRIBUTION = SAMPLING_DISTRIBUTION
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = int(max(int(num_rollouts * opt_keep_k_ratio), 1))
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.Interpolator = None
        self.inducing_points_indices = None

        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        
        if self.SAMPLING_DISTRIBUTION == "normal":
            self.theta_min = tf.stack([
                self.action_low, 0.01 * tf.ones_like(self.action_low)
            ], axis=1)
            self.theta_max = tf.stack([
                self.action_high, 1.e2 * tf.ones_like(self.action_high)
            ], axis=1)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            self.theta_min = tf.repeat(tf.expand_dims(self.action_low, 1), 2, 1)
            self.theta_max = tf.repeat(tf.expand_dims(self.action_high, 1), 2, 1)
        else:
            raise ValueError(f"Unsupported sampling distribution {self.SAMPLING_DISTRIBUTION}")

    def configure(self,
                  num_states: int,
                  num_control_inputs: int,
                  **kwargs):

        super().configure(
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            default_configure=False,
        )

        self.Interpolator = Interpolator(self.mpc_horizon, self.period_interpolation_inducing_points,
                                         self.num_control_inputs, self.lib)
        self.inducing_points_indices = tf.cast(tf.linspace(0, self.mpc_horizon - 1, self.Interpolator.number_of_interpolation_inducing_points), tf.int32)


        self.optimizer_reset()
    
    def predict_and_cost(self, s: tf.Tensor, Q: tf.Variable):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    @CompileTF
    def sample_actions(self, rng_gen: tf.random.Generator, batch_size: int):
        if self.SAMPLING_DISTRIBUTION == "normal":
            Qn = rng_gen.normal(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                mean=0.0,
                stddev=self.sample_stdev,
                dtype=tf.float32,
            )
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            Qn = rng_gen.uniform(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                minval=self.action_low,
                maxval=self.action_high,
                dtype=tf.float32,
            )
        else:
            raise ValueError(f"RPGD cannot interpret sampling type {self.SAMPLING_DISTRIBUTION}")
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)
        Qn = self.Interpolator.interpolate(Qn)
        return Qn

    @CompileTF
    def resample_actions(self, rng_gen: tf.random.Generator, input_plans: tf.Tensor):
        input_plans_at_inducing_points = tf.gather(input_plans, self.inducing_points_indices, axis=1)
        Q_resampled = input_plans_at_inducing_points + self.sample_stdev * rng_gen._standard_normal(tf.shape(input_plans_at_inducing_points), tf.float32)
        Q_resampled = tf.clip_by_value(Q_resampled, self.action_low, self.action_high)
        Q_resampled = self.Interpolator.interpolate(Q_resampled)
        return Q_resampled

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
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[: self.opt_keep_k]

        # Warmstart for next iteration        
        Qn = tf.concat([Q[:, 1:, :], Q[:, -1:, :]], axis=1)
        return Qn, best_idx, traj_cost, rollout_trajectory

    def index_to_2d(n: int, c: Union[tf.Tensor, int]) -> Union[tf.Tensor, int]:
        """Given an index c that continues through a 2D tensor of shape n x n in row-major order,
        return the index pair (i, j). Input c can be a batch of indices
        """
        return tf.stack(
            [tf.math.mod(c, n), tf.math.floormod(c, n)],
            axis=1,
        )
        
        
    
    @CompileTF
    def get_plans_to_resample(self, Qn: tf.Tensor, terminal_states: tf.Tensor, number_of_plans: int) -> tf.Tensor:
        """Find out which of the terminal states are in the least dense region.
        The input plans that produced them should be the mean for resampling

        :param Qn: Has shape(batch_size x MPC_horizon x num_action_dims)
        :type Qn: tf.Tensor
        :param terminal_states: Has shape(batch_size x num_state_dims)
        :type terminal_states: tf.Tensor
        :param number_of_plans: How many plans to resample about
        :type number_of_plans: int
        :return: Tensor of actions which are to be resampled about
        :rtype: tf.Tensor
        """
        # batch_size = Qn.shape[0]
        # Collect terminal states' distances as a (batch_size x batch_size) matrix
        distances = tf.reduce_sum((terminal_states[:, tf.newaxis, :] - terminal_states[tf.newaxis, :, :]) ** 2, axis=2)
        distances += tf.cast(tf.abs(distances) < 1e-8, dtype=tf.float32) * 1.0e8
        # Find which state has the largest minimum distance to any other
        distances_min = tf.reduce_min(distances, axis=1)
        # indices_of_min = tf.where(distances == tf.repeat(distances_min[:, tf.newaxis], batch_size, axis=1))
        # Sanity check: Is np.all(tf.gather_nd(distances, indices_of_min) == distances_min) == True?
        
        # Determine which of the plans to gather
        gather_indices = tf.argsort(distances_min, direction="DESCENDING")[:number_of_plans]
        
        return tf.gather(Qn, gather_indices, axis=0)        

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

        # retrieve optimal input and prepare warmstart
        (
            Qn,
            best_idx,
            J,
            rollout_trajectory,
        ) = self.get_action(s, self.Q_tf)
        self.u = tf.squeeze(self.Q_tf[best_idx[0], 0, :])
        self.u = self.u.numpy()
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = self.Q_tf.numpy()
            self.logging_values["J_logged"] = J.numpy()
            self.logging_values["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.logging_values["trajectory_ages_logged"] = self.trajectory_ages.numpy()
            self.logging_values["u_logged"] = self.u

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            plans_to_resample = self.get_plans_to_resample(Qn, rollout_trajectory[:, -1, :], self.num_rollouts - self.opt_keep_k)
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.resample_actions(
                self.rng, plans_to_resample
            )
            Q_keep = tf.gather(Qn, best_idx)  # resorting according to costs
            Qn = tf.concat([Qres, Q_keep], axis=0)
            self.trajectory_ages = tf.concat([
                tf.zeros(self.num_rollouts - self.opt_keep_k, dtype=tf.int32),
                tf.gather(self.trajectory_ages, best_idx),
            ], axis=0)
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
        return self.u

    def optimizer_reset(self):
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
