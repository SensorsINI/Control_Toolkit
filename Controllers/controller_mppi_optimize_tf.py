import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit_ASF.Cost_Functions import cost_function_base
from Control_Toolkit.others.globals_and_utils import CompileTF
from gym.spaces.box import Box
from SI_Toolkit.Predictors import template_predictor


#controller class
class controller_mppi_optimize_tf(template_controller):
    def __init__(
        self,
        cost_function: cost_function_base,
        seed: int,
        action_space: Box,
        observation_space: Box,
        cc_weight: float,
        R: float,
        LBD: float,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_specification: str,
        dt: float,
        NU: float,
        SQRTRHOINV: float,
        GAMMA: float,
        SAMPLING_TYPE: str,
        gradmax_clip: float,
        optim_steps: int,
        mppi_LR: float,
        adam_beta_1: float,
        adam_beta_2: float,
        adam_epsilon: float,
        controller_logging: bool,
        **kwargs,
    ):
        super().__init__(cost_function=cost_function, seed=seed, action_space=action_space, observation_space=observation_space, mpc_horizon=mpc_horizon, num_rollouts=num_rollouts, predictor_specification=predictor_specification, controller_logging=controller_logging)
        
        # Cost function parameters
        self.cc_weight = cc_weight
        self.R = R
        self.LBD = LBD

        # MPPI parameters
        self.NU = NU
        self.SQRTRHODTINV = SQRTRHOINV * (1 / np.math.sqrt(dt))
        self.GAMMA = GAMMA
        self.SAMPLING_TYPE = SAMPLING_TYPE

        # Optimization params
        self.gradmax_clip = tf.constant(gradmax_clip, dtype = tf.float32)
        self.optim_steps = optim_steps

        # Setup prototype control sequence
        self.Q_opt = tf.Variable(tf.zeros([1,self.mpc_horizon,self.num_control_inputs], dtype=tf.float32))
        
        # Setup Adam optimizer
        mppi_LR = tf.constant(mppi_LR, dtype=tf.float32)
        self.opt = tf.keras.optimizers.Adam(learning_rate=mppi_LR, beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)
        
        self.controller_reset()

    #mppi correction for importance sampling
    def mppi_correction_cost(self, u, delta_u):
        return tf.reduce_sum(self.cc_weight * (0.5 * (1 - 1.0 / self.NU) * self.R * (delta_u ** 2) + self.R * u * delta_u + 0.5 * self.R * (u ** 2)), axis=2)

    #total cost of the trajectory
    def get_mppi_trajectory_cost(self, s_hor ,u, u_prev, delta_u):
        #stage costs
        stage_cost = self.cost_function.get_stage_cost(s_hor[:,1:,:],u, u_prev)
        stage_cost = stage_cost + self.mppi_correction_cost(u, delta_u)
        #reduce alonge rollouts and add final cost
        total_cost = tf.math.reduce_sum(stage_cost,axis=1)
        total_cost = total_cost + self.cost_function.get_terminal_cost(s_hor)
        return total_cost

    #path integral approximation: sum deltaU's weighted with exponential funciton of trajectory costs
    #according to mppi theory
    def reward_weighted_average(self, S, delta_u):
        rho = tf.math.reduce_min(S)
        exp_s = tf.exp(-1.0/self.LBD * (S-rho))
        a = tf.math.reduce_sum(exp_s)
        b = tf.math.reduce_sum(exp_s[:,tf.newaxis,tf.newaxis]*delta_u, axis=0, keepdims=True)/a
        return b

    #initialize pertubation
    def inizialize_pertubation(self, random_gen):
        #if interpolation on, interpolate with method from tensor flow probability
        stdev = self.SQRTRHODTINV
        sampling_type = self.SAMPLING_TYPE
        if sampling_type == "interpolated":
            step = 10
            range_stop = int(tf.math.ceil(self.mpc_horizon / step) * step) + 1
            t = tf.range(range_stop, delta = step)
            t_interp = tf.cast(tf.range(range_stop), tf.float32)
            delta_u = random_gen.normal([self.num_rollouts, t.shape[0], self.num_control_inputs], dtype=tf.float32) * stdev
            interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
            delta_u = interp[:,:self.mpc_horizon,:]
        else:
            #otherwise i.i.d. generation
            delta_u = random_gen.normal([self.num_rollouts, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32) * stdev
        return delta_u

    @CompileTF
    def mppi_prior(self, s, u_nom, random_gen, u_old):
        # generate random input sequence and clip to control limits
        delta_u = self.inizialize_pertubation(random_gen)
        u_run = tf.tile(u_nom, [self.num_rollouts, 1, 1]) + delta_u
        u_run = tf.clip_by_value(u_run, self.action_low, self.action_high)
        #predict trajectories
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        #rollout cost
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u)
        #retrive control sequence via path integral
        u_nom = tf.clip_by_value(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        return u_nom

    @CompileTF
    def grad_step(self, s, Q, opt):
        #do a gradient descent step
        #setup gradient tape
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            #rollout trajectory and retrive cost
            rollout_trajectory = self.predictor.predict_tf(s, Q)
            traj_cost = self.cost_function.get_trajectory_cost(rollout_trajectory, Q, self.u)
        #retrieve gradient of cost w.r.t. input sequence
        dc_dQ = tape.gradient(traj_cost, Q)
        #modify gradients: makes sure biggest entry of each gradient is at most "gradmax_clip". (For this controller only one sequence
        dc_dQ_max = tf.math.reduce_max(tf.abs(dc_dQ), axis=1, keepdims=True) #find max gradient for every sequence
        mask = (dc_dQ_max > self.gradmax_clip) #generate binary mask
        invmask = tf.logical_not(mask)
        dc_dQ_prc = ((dc_dQ / dc_dQ_max) * tf.cast(mask, tf.float32) * self.gradmax_clip + dc_dQ * tf.cast(
            invmask, tf.float32)) #modify gradients
        #use optimizer to applay gradients and retrieve next set of input sequences
        opt.apply_gradients(zip([dc_dQ_prc], [Q]))
        #clip
        Qn = tf.clip_by_value(Q, self.action_low, self.action_high)
        return Qn, traj_cost

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        if self.controller_logging:
            self.current_log["s_logged"] = s.copy()
        # tile inital state and convert inputs to tensorflow tensors
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)

        #first retrieve suboptimal control sequence with mppi
        Q_mppi = self.mppi_prior(s, self.Q_opt, self.rng, self.u)
        self.Q_opt.assign(Q_mppi)

        #optimize control sequence with gradient based optimization
        for _ in range(self.optim_steps):
            Q_opt, traj_cost = self.grad_step(s, self.Q_opt, self.opt)
            self.Q_opt.assign(Q_opt)

        self.u = np.squeeze(self.Q_opt[0, 0, :].numpy())
        
        if self.controller_logging:
            self.current_log["Q_logged"] = self.Q_opt.numpy()
            self.current_log["J_logged"] = traj_cost.numpy()
            self.current_log["u_logged"] = self.u
        
        self.Q_opt.assign(tf.concat([self.Q_opt[:, 1:, :], tf.zeros([1,1,self.num_control_inputs])], axis=1)) #shift and initialize new input with 0
        #reset adam optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
        return self.u

    def controller_reset(self):
        #reset prototype control sequence
        self.Q_opt.assign(tf.zeros([1, self.mpc_horizon, self.num_control_inputs], dtype=tf.float32))
        self.u = 0.0
        #reset adam optimizer
        adam_weights = self.opt.get_weights()
        self.opt.set_weights([tf.zeros_like(el) for el in adam_weights])
