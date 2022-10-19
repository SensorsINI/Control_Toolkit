from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from gym.spaces.box import Box


#controller class
class optimizer_mppi_var_tf(template_optimizer):
    def __init__(
        self,
        controller: template_controller,
        predictor: PredictorWrapper,
        cost_function: cost_function_base,
        action_space: Box,
        observation_space: Box,
        seed: int,
        cc_weight: float,
        R: float,
        LBD_mc: float,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_specification: str,
        dt: float,
        NU_mc: float,
        SQRTRHOINV_mc: float,
        GAMMA: float,
        SAMPLING_TYPE: str,
        LR: float,
        max_grad_norm: float,
        STDEV_min: float,
        STDEV_max: float,
        interpolation_step: int,
        optimizer_logging: bool,
    ):
        super().__init__(
            controller=controller,
            predictor=predictor,
            cost_function=cost_function,
            predictor_specification=predictor_specification,
            action_space=action_space,
            observation_space=observation_space,
            seed=seed,
            num_rollouts=num_rollouts,
            mpc_horizon=mpc_horizon,
            optimizer_logging=optimizer_logging,
        )
        
        # MPPI parameters
        self.SAMPLING_TYPE = SAMPLING_TYPE
        self.cc_weight = cc_weight
        self.R = R
        self.LBD = LBD_mc
        self.NU = NU_mc
        self.SQRTRHODTINV = SQRTRHOINV_mc * (1 / np.math.sqrt(dt))
        self.GAMMA = GAMMA
        self.mppi_lr = LR
        self.stdev_min = STDEV_min
        self.stdev_max = STDEV_max
        self.max_grad_norm = max_grad_norm

        # Setup interpolation matrix
        if SAMPLING_TYPE == "interpolated":
            step = interpolation_step
            num_valid_vals = int(np.ceil(self.mpc_horizon / step) + 1)
            interp_mat = np.zeros(((num_valid_vals - 1) * step, num_valid_vals, self.num_control_inputs), dtype=np.float32)
            step_block = np.zeros((step, 2, self.num_control_inputs), dtype=np.float32)
            for j in range(step):
                step_block[j, 0, :] = (step - j) * np.ones((self.num_control_inputs), dtype=np.float32)
                step_block[j, 1, :] = j * np.ones((self.num_control_inputs), dtype=np.float32)
            for i in range(num_valid_vals - 1):
                interp_mat[i * step:(i + 1) * step, i:i + 2, :] = step_block
            interp_mat = interp_mat[:self.mpc_horizon, :, :] / step
            interp_mat = tf.constant(tf.transpose(interp_mat, perm=(1,0,2)), dtype=tf.float32)
        else:
            interp_mat = None
            num_valid_vals = self.mpc_horizon
        self.num_valid_vals, self.interp_mat = num_valid_vals, interp_mat
        
        # Set up nominal u
        self.u_nom = tf.zeros([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        # Set up vector of variances to be optimized
        self.nuvec = np.math.sqrt(self.NU)*tf.ones([1, num_valid_vals, self.num_control_inputs])
        self.nuvec = tf.Variable(self.nuvec)

        self.optimizer_reset()
    
    #mppi correction
    def mppi_correction_cost(self, u, delta_u, nuvec):
        if self.SAMPLING_TYPE == "interpolated":
            nudiv = tf.transpose(tf.matmul(tf.transpose(nuvec, perm=(2,0,1)), tf.transpose(self.interp_mat, perm=(2,0,1))), perm=(1,2,0))
        else:
            nudiv = nuvec
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
        delta_u = random_gen.normal([self.num_rollouts, self.num_valid_vals, self.num_control_inputs], dtype=tf.float32) * nuvec * self.SQRTRHODTINV
        if self.SAMPLING_TYPE == "interpolated":
            delta_u = tf.transpose(tf.matmul(tf.transpose(delta_u, perm=(2,0,1)), tf.transpose(self.interp_mat, perm=(2,0,1))), perm=(1,2,0)) #here interpolation is simply a multiplication with a matrix
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
            logging_values = {"s_logged": s.copy()}
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        self.u, self.u_nom, new_nuvec, u_run, traj_cost = self.do_step(s, self.u_nom, self.rng, self.u, self.nuvec)
        
        if self.optimizer_logging:
            logging_values["Q_logged"] = u_run.numpy()
            logging_values["J_logged"] = traj_cost.numpy()
            logging_values["u_logged"] = self.u
            self.send_logs_to_controller(logging_values)
        
        self.nuvec.assign(new_nuvec)
        return tf.squeeze(self.u).numpy()

    #reset to initial values
    def optimizer_reset(self):
        self.u_nom = tf.zeros([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32)
        self.nuvec.assign(np.math.sqrt(self.NU)*tf.ones([1, self.num_valid_vals, self.num_control_inputs]))
