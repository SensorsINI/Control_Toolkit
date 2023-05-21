from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF, get_logger
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

logger = get_logger(__name__)


class optimizer_rpgd_me_tf(template_optimizer):
    """Maximum Entropy RPGD optimizer.

    :param template_optimizer: A base optimizer with the required interface
    :type template_optimizer: abc.ABCMeta
    """
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
        maximum_entropy_alpha: float,
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
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = int(max(int(num_rollouts * opt_keep_k_ratio), 1))
        self.gradmax_clip = tf.constant(gradmax_clip, dtype=tf.float32)
        self.rtol = rtol
        self.maximum_entropy_alpha = maximum_entropy_alpha
        self.gaussian = tfp.distributions.Normal(loc=0., scale=1.)
        self.SAMPLING_DISTRIBUTION = SAMPLING_DISTRIBUTION

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        self.period_interpolation_inducing_points = None
        self.Interpolator = None

        self.opt_theta = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        self.opt_Q = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
        )
        
        # Theta bound has dimension (num_actions, num_params_per_action)
        # Example: Environment with 3 control inputs and U[a,b] sampling => (3, 2)
        if self.SAMPLING_DISTRIBUTION == "normal":
            self.theta_min = tf.stack([
                self.action_low, 0.01 * tf.ones_like(self.action_low)
            ], axis=1)
            self.theta_max = tf.stack([
                self.action_high, 1.0e2 * tf.ones_like(self.action_high)
            ], axis=1)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            self.theta_min = tf.repeat(tf.expand_dims(self.action_low, 1), 2, 1)
            self.theta_max = tf.repeat(tf.expand_dims(self.action_high, 1), 2, 1)
        else:
            raise ValueError(f"Unsupported sampling distribution {self.SAMPLING_DISTRIBUTION}")


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

        self.optimizer_reset()

    def zeta(self, theta: tf.Variable, epsilon: tf.Variable):
        """Corresponds to N(mu, stdev) with each sample independent."""
        if self.SAMPLING_DISTRIBUTION == "normal":
            mu, stdev = tf.unstack(theta, 2, -1)
            Q = mu + stdev * epsilon
            Q_clipped = tf.clip_by_value(Q, self.action_low, self.action_high)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            l, r = tf.unstack(theta, 2, -1)
            Q = (r - l) * self.gaussian.cdf(epsilon) + l
            Q_clipped = tf.clip_by_value(Q, self.action_low, self.action_high)
        
        # If sampling interpolation active, compute interpolated inputs here
        Q_clipped = self.Interpolator.interpolate(Q_clipped)
        return Q_clipped

    def zeta_inv(self, theta: tf.Variable, Q: tf.Variable):
        """Given a control Q and distribution parameters theta, return what was the standard Gaussian sample epsilon.
        Q = zeta(theta, epsilon)
        epsilon = zeta_inv(theta, Q)
        """
        inducing_point_indices = tf.cast(tf.linspace(0, self.mpc_horizon-1, self.Interpolator.number_of_interpolation_inducing_points), tf.int32)
        Q_at_inducing_points = tf.gather(Q, inducing_point_indices, axis=1)
        if self.SAMPLING_DISTRIBUTION == "normal":
            mu, stdev = tf.unstack(theta, 2, -1)
            epsilon = (Q_at_inducing_points - mu) / tf.clip_by_value(stdev, 1.0e-6, 1.0e8)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            l, r = tf.unstack(theta, 2, -1)
            epsilon = self.gaussian.quantile(tf.clip_by_value((Q_at_inducing_points - l) / tf.clip_by_value((r - l), 1.0e-6, 1.0e8), 1.0e-6, 1.0-1.0e-6))
        return epsilon

    def entropy(self, theta):
        """
        Computes the Shannon entropy of either one of:
        - a univariate Gaussian N(mu, sigma). theta = [mu, sigma]
            - See https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
        - a uniform Distribution U[l, r]. theta = [l, r]
        """
        if self.SAMPLING_DISTRIBUTION == "normal":
            _, stdev = tf.unstack(theta, 2, -1)
            return 0.5 * tf.math.log(2 * np.pi * stdev**2) + 0.5
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            l, r = tf.unstack(theta, 2, -1)
            return tf.math.log(tf.maximum(r - l, 1e-8))
        else:
            raise ValueError(f"Unsupported sampling distribution {self.SAMPLING_DISTRIBUTION}")

    def predict_and_cost(self, s: tf.Tensor, Q: tf.Variable):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory
        
    @CompileTF
    def sample_actions(self, rng_gen: tf.random.Generator, batch_size: int):
        # Reparametrization trick
        # Generate epsilon only at inducing points
        # Sampling interpolation is done after converting epsilons to actual samples u through this function
        return rng_gen.normal([batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs], dtype=tf.float32)
    
    @CompileTF
    def grad_step(
        self, s: tf.Tensor, epsilon: tf.Variable, theta: tf.Variable, Q_tf: tf.Variable, opt_Q: tf.keras.optimizers.Optimizer, opt_theta: tf.keras.optimizers.Optimizer
    ):
        # rollout trajectories and retrieve cost
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(theta)
            Q = self.zeta(theta, epsilon)
            traj_cost, _ = self.predict_and_cost(s, Q)
            entropy_cost = - self.alpha * self.entropy(theta)[..., tf.newaxis]
            traj_cost_mc_estimate = tf.reduce_mean(traj_cost[:, tf.newaxis, tf.newaxis, tf.newaxis], keepdims=True)
            traj_cost_total = traj_cost_mc_estimate + entropy_cost
        # retrieve gradient of cost w.r.t. input sequence
        dc_du = tape.gradient(traj_cost, Q)  # Update single samples to minimze their cost
        dc_du = tf.clip_by_norm(dc_du, self.gradmax_clip, axes=-1)
        dc_dt = tape.gradient(traj_cost_total, theta)  # Update sampling distribution to minimize maximum entropy cost
        dc_dt = tf.clip_by_norm(dc_dt, self.gradmax_clip, axes=-1)
        # use optimizer to apply gradients and retrieve next set of input sequences
        Q_tf.assign(Q)
        opt_Q.apply_gradients(zip([dc_du], [Q_tf]))
        opt_theta.apply_gradients(zip([dc_dt], [theta]))
        # clip
        theta = tf.clip_by_value(theta, self.theta_min, self.theta_max)
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
        return theta, Q

    @CompileTF
    def get_action(self, s: tf.Tensor, Q: tf.Variable):
        # Rollout trajectories and retrieve cost
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[: self.opt_keep_k]

        return best_idx, traj_cost, rollout_trajectory

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
            _theta, _Q = self.grad_step(s, self.epsilon, self.theta, self.Q_tf, self.opt_Q, self.opt_theta)
            _epsilon = self.zeta_inv(_theta, _Q)
            self.epsilon.assign(_epsilon)
            self.theta.assign(_theta)

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
            best_idx,
            J,
            rollout_trajectory,
        ) = self.get_action(s, _Q)
        
        # Warmstart for next iteration
        epsilon_new = tf.concat([self.epsilon[:, 1:, :], self.epsilon[:, -1:, :]], axis=1)
        self.epsilon.assign(epsilon_new)
        theta_new = tf.concat([self.theta[:, 1:, :, :], self.theta[:, -1:, :, :]], axis=1)
        self.theta.assign(theta_new)
        
        self.u = tf.squeeze(_Q[best_idx[0], 0, :])
        self.u = self.u.numpy()
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = _Q.numpy()
            self.logging_values["J_logged"] = J.numpy()
            self.logging_values["rollout_trajectories_logged"] = rollout_trajectory.numpy()
            self.logging_values["trajectory_ages_logged"] = self.trajectory_ages.numpy()
            self.logging_values["u_logged"] = self.u

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        adam_weights_Q = self.opt_Q.get_weights()
        adam_weights_theta = self.opt_theta.get_weights()
        
        if self.count % self.resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            epsilon_res = self.sample_actions(
                self.rng, self.num_rollouts - self.opt_keep_k
            )
            epsilon_keep = tf.gather(self.epsilon, best_idx, axis=0)  # resorting according to costs
            self.epsilon.assign(tf.concat([epsilon_res, epsilon_keep], axis=0))
            self.trajectory_ages = tf.concat([
                tf.zeros(self.num_rollouts - self.opt_keep_k, dtype=tf.int32),
                tf.gather(self.trajectory_ages, best_idx, axis=0),
            ], axis=0)
            # Updating the weights of adam:
            if len(adam_weights_Q) > 0:
                # For the trajectories which are kept, the weights are shifted for a warmstart
                wk1 = tf.concat(
                    [
                        tf.gather(adam_weights_Q[1], best_idx, axis=0)[:, 1:, :],
                        tf.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                wk2 = tf.concat(
                    [
                        tf.gather(adam_weights_Q[2], best_idx, axis=0)[:, 1:, :],
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
                self.opt_Q.set_weights([adam_weights_Q[0], w1, w2])
        else:
            if len(adam_weights_Q) > 0:
                # For the trajectories which are kept, the weights are shifted for a warmstart
                w1 = tf.concat(
                    [
                        adam_weights_Q[1][:, 1:, :],
                        tf.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                w2 = tf.concat(
                    [
                        adam_weights_Q[2][:, 1:, :],
                        tf.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                    ],
                    axis=1,
                )
                # Set weights
                self.opt_Q.set_weights([adam_weights_Q[0], w1, w2])
        
        if len(adam_weights_theta) > 0:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            w1 = tf.concat(
                [
                    adam_weights_theta[1][:, 1:, :, :],
                    tf.zeros([1, 1, *self.theta.shape[-2:]]),
                ],
                axis=1,
            )
            w2 = tf.concat(
                [
                    adam_weights_theta[2][:, 1:, :, :],
                    tf.zeros([1, 1, *self.theta.shape[-2:]]),
                ],
                axis=1,
            )
            # Set weights
            self.opt_theta.set_weights([adam_weights_theta[0], w1, w2])
        self.trajectory_ages += 1
        self.count += 1
        return self.u

    def optimizer_reset(self):
        # Adaptive sampling distribution (1, mpc_horizon, dim_theta)        
        if self.SAMPLING_DISTRIBUTION == "normal":
            # A) Gaussian distribution
            initial_theta = tf.stack([
                0.5 * (self.action_low + self.action_high), self.sample_stdev * tf.ones_like(self.action_low)
            ], 1)
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            # B) Uniform distribution
            initial_theta = tf.stack([self.action_low, self.action_high], 1)
            
        # Theta has shape (1, number_of_inducing_points, num_actions, num_params_per_action)
        initial_theta = np.tile(initial_theta, (self.Interpolator.number_of_interpolation_inducing_points, 1, 1))[np.newaxis]
        if hasattr(self, "theta"):
            self.theta.assign(initial_theta)
        else:
            self.theta = tf.Variable(initial_theta, dtype=tf.float32)
        self.alpha = tf.constant(self.maximum_entropy_alpha, dtype=tf.float32)
            
        # Sample new initial guesses for trajectories
        initial_epsilon = self.sample_actions(self.rng, self.num_rollouts)
        if hasattr(self, "epsilon"):
            self.epsilon.assign(initial_epsilon)
        else:
            self.epsilon = tf.Variable(initial_epsilon, dtype=tf.float32)
            self.Q_tf = tf.Variable(tf.zeros([self.num_rollouts, self.mpc_horizon, self.num_control_inputs]), dtype=tf.float32)
        self.count = 0

        # Reset optimizer
        adam_weights_Q = self.opt_Q.get_weights()
        self.opt_Q.set_weights([tf.zeros_like(el) for el in adam_weights_Q])
        adam_weights_theta = self.opt_theta.get_weights()
        self.opt_theta.set_weights([tf.zeros_like(el) for el in adam_weights_theta])
        self.trajectory_ages: tf.Tensor = tf.zeros((self.num_rollouts), dtype=tf.int32)
