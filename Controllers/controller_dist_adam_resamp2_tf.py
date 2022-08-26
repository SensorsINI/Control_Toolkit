from importlib import import_module

import numpy as np
import tensorflow as tf
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import create_rng, Compile

from Control_Toolkit.Controllers import template_controller


class controller_dist_adam_resamp2_tf(template_controller):
    def __init__(
        self,
        environment: EnvironmentBatched,
        seed: int,
        num_control_inputs: int,
        dt: float,
        mpc_horizon: int,
        num_rollouts: int,
        outer_its: int,
        sample_stdev: float,
        resamp_per: int,
        predictor_name: str,
        predictor_intermediate_steps: int,
        NET_NAME: str,
        SAMPLING_TYPE: str,
        interpolation_step: int,
        warmup: bool,
        cem_LR: float,
        opt_keep_k: int,
        gradmax_clip: float,
        rtol: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        **kwargs,
    ):
        # configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed, use_tf=True)

        # set Environment References
        super().__init__(environment)
        self.action_low: tf.Tensor = tf.convert_to_tensor(
            self.env_mock.action_space.low, dtype=tf.float32
        )
        self.action_high: tf.Tensor = tf.convert_to_tensor(
            self.env_mock.action_space.high, dtype=tf.float32
        )

        # basic params
        self.num_control_inputs = num_control_inputs

        self.num_rollouts = num_rollouts
        self.outer_its = outer_its
        self.sample_stdev = sample_stdev
        self.cem_samples = mpc_horizon  # number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.resamp_per = resamp_per

        self.NET_NAME = NET_NAME
        self.predictor_name = predictor_name

        self.SAMPLING_TYPE = SAMPLING_TYPE
        self.interpolation_step = interpolation_step
        self.do_warmup = warmup

        # optimization params
        self.opt_keep_k = opt_keep_k
        self.cem_LR = tf.constant(cem_LR, dtype=tf.float32)

        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol

        # instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=NET_NAME,
        )

        # warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.cem_samples * self.outer_its

        # if sampling type is "interpolated" setup linear interpolation as a matrix multiplication
        if SAMPLING_TYPE == "interpolated":
            step = interpolation_step
            self.num_valid_vals = int(np.ceil(self.cem_samples / step) + 1)
            self.interp_mat = np.zeros(
                (
                    (self.num_valid_vals - 1) * step,
                    self.num_valid_vals,
                    self.num_control_inputs,
                ),
                dtype=np.float32,
            )
            step_block = np.zeros((step, 2, self.num_control_inputs), dtype=np.float32)
            for j in range(step):
                step_block[j, 0, :] = (step - j) * np.ones(
                    (self.num_control_inputs), dtype=np.float32
                )
                step_block[j, 1, :] = j * np.ones(
                    (self.num_control_inputs), dtype=np.float32
                )
            for i in range(self.num_valid_vals - 1):
                self.interp_mat[i * step : (i + 1) * step, i : i + 2, :] = step_block
            self.interp_mat = self.interp_mat[: self.cem_samples, :, :] / step
            self.interp_mat = tf.constant(
                tf.transpose(self.interp_mat, perm=(1, 0, 2)), dtype=tf.float32
            )
        else:
            self.interp_mat = None
            self.num_valid_vals = self.cem_samples

        self.opt = tf.keras.optimizers.Adam(
            learning_rate=cem_LR,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )

        # setup sampling distribution
        self.controller_reset()
        self.u = 0.0

        self.bestQ = None

    @Compile
    def sample_actions(self, rng_gen: tf.random.Generator, batch_size: int):
        Qn = self.sample_stdev * rng_gen.normal(
            [batch_size, self.num_valid_vals, self.num_control_inputs], dtype=tf.float32
        )
        Qn = tf.clip_by_value(Qn, self.action_low, self.action_high)
        if self.SAMPLING_TYPE == "interpolated":
            Qn = tf.transpose(
                tf.matmul(
                    tf.transpose(Qn, perm=(2, 0, 1)),
                    tf.transpose(self.interp_mat, perm=(2, 0, 1)),
                ),
                perm=(1, 2, 0),
            )
        return Qn

    @Compile
    def grad_step(
        self, s: tf.Tensor, Q: tf.Variable, opt: tf.keras.optimizers.Optimizer
    ):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.env_mock.cost_functions.get_trajectory_cost(
                rollout_trajectory, Q, self.u
            )
        # retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=[1, 2])
        # use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Qn = tf.clip_by_value(Q, self.action_low, self.action_high)
        return Qn, traj_cost

    @Compile
    def get_action(self, s: tf.Tensor, Q: tf.Variable):
        # Rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[: self.opt_keep_k]
        elite_Q = tf.gather(Q, best_idx, axis=0)

        # # Unnecessary Part
        # # get distribution of kept trajectories. This is actually unnecessary for this controller, might be incorparated into another one tho
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
        u = tf.squeeze(elite_Q[0, 0, :])
        Qn = tf.concat([Q[:, 1:, :], Q[:, -1:, :]], axis=1)
        return u, Qn, best_idx, traj_cost, rollout_trajectory

    def step(self, s: np.ndarray, time=None):
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        # warm start setup
        if self.count == 0:
            iters = self.first_iter_count
        else:
            iters = self.outer_its

        # optimize control sequences with gradient based optimization
        prev_cost = 0.0
        for _ in range(0, iters):
            Qn, traj_cost = self.grad_step(s, self.Q_tf, self.opt)
            self.Q_tf.assign(Qn)

            # check for convergence of optimization
            if bool(
                tf.reduce_mean(
                    tf.math.abs((traj_cost - prev_cost) / (prev_cost + self.rtol))
                )
                < self.rtol
            ):
                # assume that we have converged sufficiently
                break
            prev_cost = tf.identity(traj_cost)

        # retrieve optimal input and prepare warmstart
        (
            self.u,
            Qn,
            self.bestQ,
            J,
            rollout_trajectory,
        ) = self.get_action(s, self.Q_tf)

        self.u_logged = self.u
        self.Q_logged, self.J_logged = self.Q_tf.numpy(), J.numpy()
        self.rollout_trajectories_logged = rollout_trajectory.numpy()

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(
                self.rng_cem, self.num_rollouts - self.opt_keep_k
            )
            Q_keep = tf.gather(Qn, self.bestQ)  # resorting according to costs
            Qn = tf.concat([Qres, Q_keep], axis=0)
            # Updating the weights of adam:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            if len(adam_weights) > 0:
                wk1 = tf.concat(
                    [
                        tf.gather(adam_weights[1], self.bestQ)[:, 1:, :],
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                wk2 = tf.concat(
                    [
                        tf.gather(adam_weights[2], self.bestQ)[:, 1:, :],
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                # For the new trajectories they are reset to 0
                w1 = tf.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.cem_samples,
                        self.num_control_inputs,
                    ]
                )
                w2 = tf.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.cem_samples,
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
        self.Q_tf.assign(Qn)
        self.count += 1
        return self.u.numpy()

    def controller_reset(self):
        # # unnecessary part: Adaptive sampling distribution
        # self.dist_mue = (
        #     (self.action_low + self.action_high)
        #     * 0.5
        #     * tf.ones([1, self.cem_samples, self.num_control_inputs])
        # )
        # self.stdev = self.sample_stdev * tf.ones(
        #     [1, self.cem_samples, self.num_control_inputs]
        # )
        # # end of unnecessary part

        # sample new initial guesses for trajectories
        Qn = self.sample_actions(self.rng_cem, self.num_rollouts)
        self.Q_tf = tf.Variable(Qn)
        self.count = 0

        # reset optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
