from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from Control_Toolkit.others.Interpolator import Interpolator
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper


# optimizer class
class optimizer_mppi_var_tf(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}
    
    def __init__(
        self,
        predictor: PredictorWrapper,
        cost_function: CostFunctionWrapper,
        control_limits: "Tuple[np.ndarray, np.ndarray]",
        computation_library: "type[ComputationLibrary]",
        seed: int,
        cc_weight: float,
        R: float,
        LBD_mc: float,
        mpc_horizon: int,
        num_rollouts: int,
        NU_mc: float,
        SQRTRHOINV_mc: float,
        LR: float,
        max_grad_norm: float,
        STDEV_min: float,
        STDEV_max: float,
        period_interpolation_inducing_points: int,
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
        
        # MPPI parameters
        self.cc_weight = cc_weight
        self.R = R
        self.LBD = LBD_mc
        self.NU = NU_mc
        self.mppi_lr = LR
        self.stdev_min = STDEV_min
        self.stdev_max = STDEV_max
        self.max_grad_norm = max_grad_norm
        self._SQRTRHOINV_mc = SQRTRHOINV_mc

        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.Interpolator = None

        self.u_nom = None  # nominal u
        self.nuvec = None  # vector of variances to be optimized


    def configure(self,
                  num_states: int,
                  num_control_inputs: int,
                  dt: float,
                  **kwargs):

        super().configure(
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            default_configure=False,
        )

        self.Interpolator = Interpolator(self.mpc_horizon, self.period_interpolation_inducing_points,
                                         self.num_control_inputs, self.lib)

        # Set up nominal u
        self.u_nom = tf.zeros([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        # Set up vector of variances to be optimized
        self.nuvec = np.math.sqrt(self.NU)*tf.ones([1, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs])
        self.nuvec = tf.Variable(self.nuvec)

        self.SQRTRHODTINV = self._SQRTRHOINV_mc * (1 / np.sqrt(dt))
        del self._SQRTRHOINV_mc

        self.optimizer_reset()
    
    #mppi correction
    def mppi_correction_cost(self, u, delta_u, nuvec):
        nudiv = self.Interpolator.interpolate(nuvec)
        return tf.reduce_sum(self.cc_weight * (0.5 * (1 - 1.0 / nudiv**2) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), axis=2)

    #mppi averaging of trajectories
    def reward_weighted_average(self, S, delta_u):
        rho = tf.math.reduce_min(S)
        exp_s = tf.exp(-1.0/self.LBD * (S-rho))
        a = tf.math.reduce_sum(exp_s)
        b = tf.math.reduce_sum(exp_s[:,tf.newaxis,tf.newaxis]*delta_u, axis=0, keepdims=True)/a
        return b

    #initialize the pertubations
    def inizialize_pertubation(self, random_gen, nuvec):
        delta_u = random_gen.normal([self.num_rollouts, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs], dtype=tf.float32) * nuvec * self.SQRTRHODTINV
        delta_u = self.Interpolator.interpolate(delta_u)
        return delta_u

    @CompileTF
    def do_step(self, s, u_nom, random_gen, u_old, nuvec):
        #start gradient tape
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(nuvec) #watch variances on tape
            delta_u = self.inizialize_pertubation(random_gen, nuvec) #initialize pertubations
            #build real input and clip, preserving gradient
            u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1]) + delta_u
            u_run = tfp.math.clip_by_value_preserve_gradient(u_run, self.action_low, self.action_high)
            #rollout and cost
            rollout_trajectory = self.predictor.predict_tf(s, u_run)
            unc_cost = self.cost_function.get_trajectory_cost(rollout_trajectory, u_run, u_old)
            mean_uncost = tf.math.reduce_mean(unc_cost)
            #retrieve gradient
            dc_ds = tape.gradient(mean_uncost, nuvec)
            dc_ds = tf.clip_by_norm(dc_ds, self.max_grad_norm,axes = [1])
        #correct cost of mppi
        cor_cost = self.mppi_correction_cost(u_run, delta_u, nuvec)
        cor_cost = tf.math.reduce_sum(cor_cost, axis=1)
        traj_cost = unc_cost + cor_cost
        #build optimal input
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        u = u_nom[0, 0, :]
        u_nom = tf.concat([u_nom[:, 1:, :], tf.constant(0.0, shape=[1, 1, self.num_control_inputs])], axis=1)
        #adapt variance
        new_nuvec = nuvec-self.mppi_lr*dc_ds
        new_nuvec = tf.clip_by_value(new_nuvec, self.stdev_min, self.stdev_max)
        return u, u_nom, new_nuvec, u_run, traj_cost

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.optimizer_logging:
            self.logging_values = {"s_logged": s.copy()}
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        self.u, self.u_nom, new_nuvec, u_run, traj_cost = self.do_step(s, self.u_nom, self.rng, self.u, self.nuvec)
        
        if self.optimizer_logging:
            self.logging_values["Q_logged"] = u_run.numpy()
            self.logging_values["J_logged"] = traj_cost.numpy()
            self.logging_values["u_logged"] = self.u
        
        self.nuvec.assign(new_nuvec)
        return tf.squeeze(self.u).numpy()

    #reset to initial values
    def optimizer_reset(self):
        self.u_nom = tf.zeros([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        self.nuvec.assign(np.math.sqrt(self.NU)*tf.ones([1, self.Interpolator.number_of_interpolation_inducing_points, self.num_control_inputs]))
