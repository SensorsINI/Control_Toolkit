from typing import Tuple
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
        num_states: int,
        num_control_inputs: int,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        computation_library: "type[ComputationLibrary]",
        seed: int,
        mpc_horizon: int,
        num_rollouts: int,
        outer_its: int,
        sample_stdev: float,
        resamp_per: int,
        period_interpolation_inducing_points: int,
        warmup: bool,
        warmup_iterations: int,
        learning_rate: float,
        opt_keep_k: int,
        gradmax_clip: float,
        rtol: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        optimizer_logging: bool,
    ):
        super().__init__(
            predictor=predictor,
            cost_function=cost_function,
            num_states=num_states,
            num_control_inputs=num_control_inputs,
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
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = opt_keep_k
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        self.Interpolator = Interpolator(self.mpc_horizon, period_interpolation_inducing_points,
                                         self.num_control_inputs, self.lib)

        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        
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
        Qn = rng_gen.uniform(
            [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
            minval=self.action_low,
            maxval=self.action_high,
            dtype=tf.float32,
        )
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)
        Qn = self.Interpolator.interpolate(Qn)
        return Qn

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

        # Warmstart for next iteration
        
        Qn = tf.concat([Q[:, 1:, :], Q[:, -1:, :]], axis=1)
        return Qn, best_idx, traj_cost, rollout_trajectory

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
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(
                self.rng, self.num_rollouts - self.opt_keep_k
            )
            Q_keep = tf.gather(Qn, best_idx)  # resorting according to costs
            Qn = tf.concat([Qres, Q_keep], axis=0)
            self.trajectory_ages = tf.concat([
                tf.gather(self.trajectory_ages, best_idx),
                tf.zeros(self.num_rollouts - self.opt_keep_k, dtype=tf.int32)
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
