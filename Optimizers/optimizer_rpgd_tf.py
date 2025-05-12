from typing import Tuple, Any
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary, PyTorchLibrary, TensorType, VariableType

import numpy as np
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileAdaptive, get_logger
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

logger = get_logger(__name__)


class ADAM:
    """
    A class to represent the ADAM optimizer.
    """

    def __init__(self, lib, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

        self.lib = lib

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.adam = None

    def build_optimizer(self, num_rollouts: int, mpc_horizon: int, control_limits: Tuple[np.ndarray, np.ndarray]):

        # Build optimizer based on library
        if isinstance(self.lib, TensorFlowLibrary):
            import tensorflow as tf

            self.adam = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.epsilon,
            )
        else:
            # PyTorch Adam
            import torch
            self.adam = torch.optim.Adam(
                params=[self.lib.to_tensor(np.zeros((num_rollouts, mpc_horizon, control_limits[0].shape[-1])),
                                           self.lib.float32)],
                lr=self.learning_rate,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
            )

    def apply_gradients(self, grads_and_vars):
        self.adam.apply_gradients(grads_and_vars)

    def get_weights(self):
        # Gather every tf.Variable storing Adam’s state:
        #   iteration count, beta-power accumulators, then m/v for each var.
        state_vars = self.adam.variables()
        # Convert each to a NumPy array so they can be serialized/passed around.
        return [var.numpy() for var in state_vars]

    def set_weights(self, weights):
        # Retrieve the same list of state variables
        state_vars = self.adam.variables()
        # Guard against wrong number of arrays
        if len(weights) != len(state_vars):
            raise ValueError(
                f"Expected {len(state_vars)} arrays but got {len(weights)}"
            )
        # Assign each provided array back into the optimizer’s variables.
        for var, w in zip(state_vars, weights):
            # .assign will convert Python/numpy to the correct tf.dtype automatically.
            var.assign(w)

    def reset(self):
        adam_weights = self.get_weights()
        self.set_weights([self.lib.zeros_like(el) for el in adam_weights])



class optimizer_rpgd_tf(template_optimizer):
    supported_computation_libraries = (TensorFlowLibrary, PyTorchLibrary)
    
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
        **kwargs,
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

        self.sample_stdev = self.lib.to_tensor(sample_stdev, self.lib.float32)
        self.sample_mean = self.lib.to_tensor(sample_mean, self.lib.float32)

        self.sample_whole_control_space = sample_whole_control_space
        if self.sample_whole_control_space:
            self.sample_min = self.lib.to_tensor(self.action_low, self.lib.float32)
            self.sample_max = self.lib.to_tensor(self.action_high, self.lib.float32)
        else:
            self.sample_min = self.lib.to_tensor(uniform_dist_min, self.lib.float32)
            self.sample_max = self.lib.to_tensor(uniform_dist_max, self.lib.float32)

        self.resamp_per = resamp_per
        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.shift_previous = shift_previous
        self.do_warmup = warmup
        self.warmup_iterations = warmup_iterations
        self.opt_keep_k = int(max(int(num_rollouts * opt_keep_k_ratio), 1))
        self.gradmax_clip = self.lib.constant(gradmax_clip, self.lib.float32)
        self.rtol = rtol
        self.SAMPLING_DISTRIBUTION = SAMPLING_DISTRIBUTION

        # Warmup setup
        self.first_iter_count = self.outer_its
        if self.do_warmup:
            self.first_iter_count = self.warmup_iterations

        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.Interpolator = None

        self.opt = ADAM(self.lib, learning_rate=learning_rate, beta_1=adam_beta_1,
                        beta_2=adam_beta_2, epsilon=adam_epsilon)
        self.opt.build_optimizer(num_rollouts, mpc_horizon, control_limits)

        self.calculate_optimal_trajectory = calculate_optimal_trajectory
        self.optimal_trajectory = None
        self.summed_stage_cost = None
        self.optimal_control_sequence = None

        self.get_action = CompileAdaptive(self._get_action)
        self.grad_step = CompileAdaptive(self._grad_step)
        self.predict_optimal_trajectory = CompileAdaptive(self._predict_optimal_trajectory)

    def configure(self,
                  num_states: int,
                  num_control_inputs: int,
                  **kwargs):

        dt = kwargs.get("dt", None)
        predictor_specification = kwargs.get("predictor_specification", None)

        super().configure(
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            default_configure=False,
        )

        self.Interpolator = Interpolator(self.mpc_horizon, self.period_interpolation_inducing_points,
                                         self.num_control_inputs, self.lib)

        if dt is not None and predictor_specification is not None:
            self.predictor_single_trajectory.configure(
                batch_size=1, horizon=self.mpc_horizon, dt=dt,  # TF requires constant batch size
                predictor_specification=predictor_specification,
            )
        else:
            raise ValueError("RPGD requires dt and predictor_specification to be passed.")

        self.optimizer_reset()

    def sample_actions(self, rng_gen: Any, batch_size: int):
        if self.SAMPLING_DISTRIBUTION == "normal":
            Qn = rng_gen.normal(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                mean=self.sample_mean,
                stddev=self.sample_stdev,
                dtype=self.lib.float32,
            )
        elif self.SAMPLING_DISTRIBUTION == "uniform":
            Qn = rng_gen.uniform(
                [batch_size, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs],
                minval=self.sample_min,
                maxval=self.sample_max,
                dtype=self.lib.float32,
            )
        else:
            raise ValueError(f"RPGD cannot interpret sampling type {self.SAMPLING_DISTRIBUTION}")
        Qn = self.lib.clip(Qn, self.action_low, self.action_high)

        Qn = self.Interpolator.interpolate(Qn)

        return Qn

    def predict_and_cost(self, s: TensorType, Q: VariableType):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_core(s, Q)
        traj_cost = self.cost_function.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    def _grad_step(self, s: TensorType, Q: VariableType, opt: Any):
        # dispatch to the TF or Torch implementation *and* return its outputs
        if isinstance(self.lib, TensorFlowLibrary):
            return self._grad_step_tf(s, Q, opt)
        else:
            return self._grad_step_torch(s, Q, opt)

    def _grad_step_tf(
        self, s: TensorType, Q: VariableType, opt: Any
    ):
        # rollout trajectories and retrieve cost
        with self.lib.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            traj_cost, _ = self.predict_and_cost(s, Q)
        # retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        dc_dQ_prc = self.lib.clip_by_norm(dc_dQ, self.gradmax_clip, [1, 2])
        # use optimizer to apply gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        # clip
        Qn = self.lib.clip(Q, self.action_low, self.action_high)
        return Qn, traj_cost

    def _grad_step_torch(
            self, s: TensorType, Q: VariableType, opt: Any
    ):
        # --- PyTorch branch ---------------------------
        # ensure Q collects gradients
        Q.requires_grad_(True)
        traj_cost, _ = self.predict_and_cost(s, Q)
        # sum over rollouts to get a scalar
        loss = traj_cost.sum()
        opt.zero_grad()
        loss.backward()
        # raw gradient
        dc_dQ = Q.grad
        # clip by norm
        dc_dQ_prc = self.lib.clip_by_norm(dc_dQ, self.gradmax_clip, [1, 2])
        # manually inject the clipped grad before stepping
        Q.grad = dc_dQ_prc
        opt.step()
        # clip outputs, detach for next iteration
        Qn = self.lib.clip(Q.detach(), self.action_low, self.action_high)
        # set up new leaf with grad enabled
        return Qn.clone().detach().requires_grad_(True), traj_cost.detach()

    def _get_action(self, s: TensorType, Q: VariableType):
        # Rollout trajectories and retrieve cost
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
        
        # sort the costs and find best k costs
        sorted_cost = self.lib.argsort(traj_cost, 0)
        best_idx = sorted_cost[: self.opt_keep_k]

        # # Unnecessary Part
        # # get distribution of kept trajectories. This is actually unnecessary for this optimizer, might be incorparated into another one tho
        # elite_Q = self.lib.gather(Q, best_idx, axis=0)
        # dist_mue = self.lib.reduce_mean(elite_Q, axis=0, keepdims=True)
        # dist_std = self.lib.reduce_std(elite_Q, axis=0, keepdims=True)

        # dist_mue = self.lib.concat(
        #     [
        #         dist_mue[:, 1:, :],
        #         (self.action_low + self.action_high)
        #         * 0.5
        #         * self.lib.ones([1, 1, self.num_control_inputs]),
        #     ],
        #     1,
        # )

        # # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        # dist_std = self.lib.clip(dist_std, self.sample_stdev, 10.0)
        # dist_std = self.lib.concat(
        #     [
        #         dist_std[:, 1:, :],
        #         self.sample_stdev
        #         * self.lib.ones(shape=[1, 1, self.num_control_inputs]),
        #     ],
        #     1,
        # )
        # # End of unnecessary part

        # Retrieve optimal input and warmstart for next iteration
        Qn = self.lib.concat(
            [Q[:, self.shift_previous:, :], self.lib.tile(Q[:, -1:, :], (1, self.shift_previous, 1))]
            , axis=1)
        return Qn, best_idx, traj_cost, rollout_trajectory

    def _predict_optimal_trajectory(self, s, u_nom, u):
        optimal_trajectory = self.predictor_single_trajectory.predict_core(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        summed_stage_cost = self.cost_function.get_summed_stage_cost(optimal_trajectory, u_nom[:1, :, :], u)
        return optimal_trajectory, summed_stage_cost

    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
            
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, self.lib.constant([self.num_rollouts, 1], self.lib.int64))
        s = self.lib.to_tensor(s, self.lib.float32)

        # warm start setup
        if self.count == 0:
            iters = self.first_iter_count
        else:
            iters = self.outer_its

        # optimize control sequences with gradient based optimization
        # prev_cost = self.lib.to_tensor(np.inf, self.lib.float32)
        for _ in range(0, iters):
            Qn, traj_cost = self.grad_step(s, self.Q_tf, self.opt)
            self.Q_tf.assign(Qn)

            # check for convergence of optimization
            # if bool(
            #     self.lib.reduce_mean(
            #         self.lib.abs((traj_cost - prev_cost) / (prev_cost + self.rtol))
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
        self.u_nom = self.Q_tf[self.lib.newaxis, best_idx[0], :, :]
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = self.Q_tf.numpy()
            self.logging_values["J_logged"] = J.numpy()
            self.logging_values["rollout_trajectories_logged"] = self.rollout_trajectories.numpy()
            self.logging_values["trajectory_ages_logged"] = self.trajectory_ages.numpy()
            self.logging_values["u_logged"] = self.u

        self.optimal_control_sequence = self.lib.to_numpy(self.u_nom)

        # modify adam optimizers. The optimizer optimizes all rolled out trajectories at once
        # and keeps weights for all these, which need to get modified.
        # The algorithm not only warmstrats the initial guess, but also the intial optimizer weights
        # Solution for new TF>2.10 adam optimizer, now using legacy instead
        # adam_weights_variables = self.opt.variables()
        # adam_weights = [v.numpy() for v in adam_weights_variables]
        adam_weights = self.opt.get_weights()
        if self.count % self.resamp_per == 0:
            # if it is time to resample, new random input sequences are drawn for the worst bunch of trajectories
            Qres = self.sample_actions(
                self.rng, self.num_rollouts - self.opt_keep_k
            )
            Q_keep = self.lib.gather(Qn, best_idx, 0)  # resorting according to costs
            Qn = self.lib.concat([Qres, Q_keep], 0)
            self.trajectory_ages = self.lib.concat([
                self.lib.zeros((self.num_rollouts - self.opt_keep_k,)),
                self.lib.gather(self.trajectory_ages, best_idx, 0),
            ], 0)
            # Updating the weights of adam:
            # For the trajectories which are kept, the weights are shifted for a warmstart
            if len(adam_weights) > 0:
                wk1 = self.lib.concat(
                    [
                        self.lib.gather(adam_weights[1], best_idx, 0)[:, 1:, :],
                        self.lib.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    1,
                )
                wk2 = self.lib.concat(
                    [
                        self.lib.gather(adam_weights[2], best_idx, 0)[:, 1:, :],
                        self.lib.zeros([self.opt_keep_k, 1, self.num_control_inputs]),
                    ],
                    1,
                )
                # For the new trajectories they are reset to 0
                w1 = self.lib.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.mpc_horizon,
                        self.num_control_inputs,
                    ]
                )
                w2 = self.lib.zeros(
                    [
                        self.num_rollouts - self.opt_keep_k,
                        self.mpc_horizon,
                        self.num_control_inputs,
                    ]
                )
                w1 = self.lib.concat([w1, wk1], 0)
                w2 = self.lib.concat([w2, wk2], 0)
                # Set weights
                self.opt.set_weights([adam_weights[0], w1, w2])
        else:
            if len(adam_weights) > 0:
                # if it is not time to reset, all optimizer weights are shifted for a warmstart
                w1 = self.lib.concat(
                    [
                        adam_weights[1][:, 1:, :],
                        self.lib.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                    ],
                    1,
                )
                w2 = self.lib.concat(
                    [
                        adam_weights[2][:, 1:, :],
                        self.lib.zeros([self.num_rollouts, 1, self.num_control_inputs]),
                    ],
                    1,
                )
                self.opt.set_weights([adam_weights[0], w1, w2])
        self.trajectory_ages += 1
        self.Q_tf.assign(Qn)
        self.count += 1

        if self.calculate_optimal_trajectory:
            optimal_trajectory, summed_stage_cost = self.predict_optimal_trajectory(s[:1, :], self.u_nom, self.u)
            self.optimal_trajectory = self.lib.to_numpy(optimal_trajectory)
            self.summed_stage_cost = self.lib.to_numpy(summed_stage_cost)

        self.u = self.u_nom[0, 0, :].numpy()
        return self.u


    def optimizer_reset(self):
        # # unnecessary part: Adaptive sampling distribution
        # self.dist_mue = (
        #     (self.action_low + self.action_high)
        #     * 0.5
        #     * self.lib.ones([1, self.mpc_horizon, self.num_control_inputs])
        # )
        # self.stdev = self.sample_stdev * self.lib.ones(
        #     [1, self.mpc_horizon, self.num_control_inputs]
        # )
        # # end of unnecessary part

        # sample new initial guesses for trajectories
        Qn = self.sample_actions(self.rng, self.num_rollouts)
        if hasattr(self, "Q_tf"):
            self.Q_tf.assign(Qn)
        else:
            self.Q_tf = self.lib.to_variable(Qn, self.lib.float32)
        self.count = 0

        self.opt.reset()
        self.trajectory_ages = self.lib.zeros((self.num_rollouts,))
