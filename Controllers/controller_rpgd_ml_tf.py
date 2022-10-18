from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import CompileTF, get_logger
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box

logger = get_logger(__name__)


class controller_rpgd_ml_tf(template_controller):
    """Maximum Entropy RPGD controller.

    :param template_controller: A base controller with the required interface
    :type template_controller: abc.ABCMeta
    """
    def __init__(
        self,
        cost_function: cost_function_base,
        seed: int,
        action_space: Box,
        observation_space: Box,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_specification: str,
        outer_its: int,
        sample_stdev: float,
        resamp_per: int,
        SAMPLING_TYPE: str,
        SAMPLING_DISTRIBUTION: str,
        interpolation_step: int,
        warmup: bool,
        warmup_iterations: int,
        learning_rate: float,
        opt_keep_k: int,
        gradmax_clip: float,
        rtol: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        maximum_entropy_alpha: float,
        controller_logging: bool,
        **kwargs,
    ):
        super().__init__(cost_function=cost_function, seed=seed, action_space=action_space, observation_space=observation_space, mpc_horizon=mpc_horizon, num_rollouts=num_rollouts, controller_logging=controller_logging)
        
        # Predictor
        self.predictor = PredictorWrapper()
        self.predictor.configure(
            batch_size=self.num_rollouts, horizon=self.mpc_horizon, predictor_specification=predictor_specification
        )
        
        # RPGD parameters
        self.outer_its = outer_its
        self.sample_stdev = sample_stdev
        self.resamp_per = resamp_per
        self.SAMPLING_TYPE = SAMPLING_TYPE
        self.SAMPLING_DISTRIBUTION = SAMPLING_DISTRIBUTION
        self.interpolation_step = interpolation_step
        self.do_warmup = warmup
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = opt_keep_k
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol
        self.maximum_entropy_alpha = maximum_entropy_alpha
        self.gaussian = tfp.distributions.Normal(loc=0., scale=1.)

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        # if sampling type is "interpolated" setup linear interpolation as a matrix multiplication
        if SAMPLING_TYPE == "interpolated":
            step = interpolation_step
            self.num_valid_vals = int(np.ceil(self.mpc_horizon / step) + 1)
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
            self.interp_mat = self.interp_mat[: self.mpc_horizon, :, :] / step
            self.interp_mat = tf.constant(
                tf.transpose(self.interp_mat, perm=(1, 0, 2)), dtype=tf.float32
            )
        else:
            self.interp_mat = None
            self.num_valid_vals = self.mpc_horizon

        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        
        # Theta bound has dimension (num_actions, num_params_per_action)
        # Example: Environment with 3 control inputs and U[a,b] sampling => (3, 2)
        if self.SAMPLING_DISTRIBUTION == "normal":
            self.theta_min = tf.stack([
                self.action_low, tf.zeros_like(self.action_low)
            ], axis=0)
            self.theta_max = tf.stack([
                self.action_high, 1.e2 * tf.ones_like(self.action_high)
            ], axis=0)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            self.theta_min = tf.repeat(tf.expand_dims(self.action_low, 1), 2, 1)
            self.theta_max = tf.repeat(tf.expand_dims(self.action_high, 1), 2, 1)
        else:
            raise ValueError(f"Unsupported sampling distribution {self.SAMPLING_DISTRIBUTION}")
        
        self.controller_reset()
    
    def zeta(self, theta: tf.Variable, epsilon: tf.Tensor):
        """Corresponds to N(mu, stdev) with each sample independent."""
        if self.SAMPLING_DISTRIBUTION == "normal":
            mu, stdev = tf.unstack(theta, 2, -1)
            Q = mu + stdev * epsilon
            Q_clipped = tf.clip_by_value(Q, self.action_low, self.action_high)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            l, r = tf.unstack(theta, 2, -1)
            Q = (r - l) * self.gaussian.cdf(epsilon) + l  # Transform Gaussian sample into uniform
            Q_clipped = tf.clip_by_value(Q, self.action_low, self.action_high)
        return Q_clipped

    def ML_estimation(self, epsilon: tf.Tensor, Q: tf.Variable):
        ep = tf.transpose(epsilon, [1, 2, 0])
        Y = tf.convert_to_tensor(Q)
        Y = tf.transpose(Y, [1, 2, 0])[...,tf.newaxis]
        if self.SAMPLING_DISTRIBUTION == "normal":
            X = tf.stack([tf.ones_like(ep), ep], axis=-1)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            X = tf.stack([1.0 - self.gaussian.cdf(ep), self.gaussian.cdf(ep)], axis=-1)
        else:
            raise ValueError()
        theta_ML: tf.Tensor = tf.linalg.lstsq(X, Y)
        theta_ML = tf.transpose(theta_ML, [3, 0, 1, 2])  # Output should have shape of theta
        return theta_ML
        

    def entropy(self, theta):
        """Computes the Shannon entropy of a univariate Gaussian N(mu, sigma). theta = [mu, sigma].
        See https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/"""
        if self.SAMPLING_DISTRIBUTION == "normal":
            stdev = theta[..., 1:]
            h = 0.5 * tf.math.log(2 * np.pi * stdev**2) + 0.5
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            l, r = tf.unstack(theta, 2, -1)
            h = tf.math.log(tf.maximum(r - l, 1e-8))
        else:
            raise ValueError(f"Unsupported sampling distribution {self.SAMPLING_DISTRIBUTION}")
        return h
        
    @CompileTF
    def sample_actions(self, rng_gen: tf.random.Generator, batch_size: int):
        # Reparametrization trick
        epsilon = rng_gen.normal([batch_size, self.num_valid_vals, self.num_control_inputs], dtype=tf.float32)
        
        # Compute plan interpolation if required
        if self.SAMPLING_TYPE == "interpolated":
            epsilon = tf.transpose(
                tf.matmul(
                    tf.transpose(epsilon, perm=(2, 0, 1)),
                    tf.transpose(self.interp_mat, perm=(2, 0, 1)),
                ),
                perm=(1, 2, 0),
            )
        return epsilon

    @CompileTF
    def grad_step(
        self, s: tf.Tensor, Q: tf.Variable, opt: tf.keras.optimizers.Optimizer
    ):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.cost_function.get_trajectory_cost(
                rollout_trajectory, Q, self.u
            )
            # entropy_cost = tf.reduce_sum(self.entropy(theta))  # Taking sum here is optional for gradient. If no reduce_sum, then tf gradient is the same
            traj_cost_mc_estimate = tf.reduce_mean(traj_cost)  # - self.alpha * entropy_cost
        # retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost_mc_estimate, Q)
        dc_dQ_prc = tf.clip_by_norm(dc_dQ, self.gradmax_clip, axes=-1)
        # use optimizer to apply gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
        return Q, traj_cost

    @CompileTF
    def get_action(self, s: tf.Tensor, epsilon: tf.Tensor, theta: tf.Variable):
        Q = self.zeta(theta, epsilon)
        # Rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[: self.opt_keep_k]

        # # Unnecessary Part
        # # get distribution of kept trajectories. This is actually unnecessary for this controller, might be incorparated into another one tho
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
        u = tf.squeeze(Q[sorted_cost[0], 0, :])
        epsilon_shifted = tf.concat([epsilon[:, 1:, :], epsilon[:, -1:, :]], axis=1)
        theta_shifted = tf.concat([theta[:, 1:, :, :], theta[:, -1:, :, :]], axis=1)
        return u, epsilon_shifted, theta_shifted, best_idx, traj_cost, rollout_trajectory

    def step(self, s: np.ndarray, time=None):
        if self.controller_logging:
            self.current_log["s_logged"] = s.copy()
            
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
            self.Q.assign(self.zeta(self.theta, self.epsilon))
            _Q, traj_cost = self.grad_step(s, self.Q, self.opt)
            self.Q.assign(_Q)

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

        # Maximum likelihood estimation of theta
        _theta = self.ML_estimation(self.epsilon, self.Q)
        self.theta.assign(_theta)
        
        # retrieve optimal input and prepare warmstart
        (
            self.u,
            epsilon_shifted,
            theta_shifted,
            best_idx,
            J,
            rollout_trajectory,
        ) = self.get_action(s, self.epsilon, self.theta)
        self.epsilon = epsilon_shifted
        self.theta.assign(theta_shifted)
        
        self.u = self.u.numpy()
        
        if self.controller_logging:
            self.current_log["Q_logged"] = self.zeta(self.theta, self.epsilon).numpy()
            self.current_log["J_logged"] = J.numpy()
            self.current_log["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.current_log["trajectory_ages_logged"] = self.trajectory_ages.numpy()
            self.current_log["u_logged"] = self.u

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstarts the initial guess, but also the intial optimizer weights
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            epsilon_resampled = self.sample_actions(self.rng, self.num_rollouts - self.opt_keep_k)
            epsilon_retained = tf.gather(self.epsilon, best_idx, axis=0)  # resorting according to costs
            self.epsilon = tf.concat([epsilon_resampled, epsilon_retained], axis=0)
            self.trajectory_ages = tf.concat([
                tf.gather(self.trajectory_ages, best_idx, axis=0),
                tf.zeros(self.num_rollouts - self.opt_keep_k, dtype=tf.int32)
            ], axis=0)
        
        # Updating the weights of Adam:
        if len(adam_weights) > 0:
            # For the trajectories which are kept, the weights are shifted for a warmstart
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
        self.count += 1
        return self.u

    def controller_reset(self):
        # Adaptive sampling distribution (1, mpc_horizon, dim_theta)        
        if self.SAMPLING_DISTRIBUTION == "normal":
            # A) Gaussian distribution
            initial_theta = tf.stack([
                0.5 * (self.action_low + self.action_high), self.sample_stdev * tf.ones_like(self.action_low)
            ], 1)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            # B) Uniform distribution
            initial_theta = tf.stack([self.action_low, self.action_high], 1)
        
        # Theta has shape (1, mpc_horizon, num_actions, distribution parameters per action)
        initial_theta = np.tile(initial_theta, (self.mpc_horizon, 1, 1))[np.newaxis]
        if hasattr(self, "theta"):
            self.theta.assign(initial_theta)
        else:
            self.theta = tf.Variable(initial_theta, dtype=tf.float32)
        
        initial_Q = tf.zeros([self.num_rollouts, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        if hasattr(self, "Q"):
            self.Q.assign(initial_Q)
        else:
            self.Q = tf.Variable(initial_Q, dtype=tf.float32)
        
        self.alpha = tf.constant(self.maximum_entropy_alpha, dtype=tf.float32)
            
        # Sample new initial guesses for trajectories
        self.epsilon = self.sample_actions(self.rng, self.num_rollouts)
        self.count = 0

        # Reset optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        self.trajectory_ages: tf.Tensor = tf.zeros((self.num_rollouts), dtype=tf.int32)
