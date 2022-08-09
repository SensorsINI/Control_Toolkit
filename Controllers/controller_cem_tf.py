from importlib import import_module

import numpy as np
import tensorflow as tf
from Control_Toolkit.others.environment import EnvironmentBatched
from Control_Toolkit.others.globals_and_utils import create_rng, Compile

from Control_Toolkit.Controllers import template_controller


#cem class
class controller_cem_tf(template_controller):
    def __init__(self, environment: EnvironmentBatched, seed: int, num_control_inputs: int, dt: float, mpc_horizon: float, cem_outer_it: int, cem_rollouts: int, predictor_name: str, predictor_intermediate_steps: int, CEM_NET_NAME: str, cem_stdev_min: float, cem_best_k: int, **kwargs):
        #First configure random sampler
        self.rng_cem = create_rng(self.__class__.__name__, seed)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        #cem params
        self.num_rollouts = cem_rollouts
        self.mpc_horizon = mpc_horizon
        self.cem_outer_it = cem_outer_it
        self.cem_stdev_min = cem_stdev_min
        self.cem_best_k = cem_best_k
        self.cem_samples = int(mpc_horizon / dt)  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps

        self.NET_NAME = CEM_NET_NAME

        #instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.cem_samples,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=self.NET_NAME,
        )

        super().__init__(environment)
        self.action_low = self.env_mock.action_space.low
        self.action_high = self.env_mock.action_space.high

        self.controller_reset()
        self.u = 0.0

    @Compile
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(rollout_trajectory, Q, self.u)
        return traj_cost, rollout_trajectory

    #step function to find control
    def step(self, s: np.ndarray, time=None):
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        for _ in range(0,self.cem_outer_it):
            #generate random input sequence and clip to control limits
            Q = np.tile(self.dist_mue,(self.num_rollouts,1,1)) + np.multiply(self.rng_cem.standard_normal(
                size=(self.num_rollouts, self.cem_samples, self.num_control_inputs), dtype=np.float32), self.stdev)
            Q = np.clip(Q, self.action_low, self.action_high, dtype=np.float32)

            Q = tf.convert_to_tensor(Q, dtype=tf.float32)

            #rollout the trajectories and get cost
            traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)
            Q, traj_cost, rollout_trajectory = Q.numpy(), traj_cost.numpy(), rollout_trajectory.numpy()
            #sort the costs and find best k costs
            sorted_cost = np.argsort(traj_cost)
            best_idx = sorted_cost[:self.cem_best_k]
            elite_Q = Q[best_idx,:,:]
            #update the distribution for next inner loop
            self.dist_mue = np.mean(elite_Q, axis=0, keepdims=True)
            self.stdev = np.std(elite_Q, axis=0, keepdims=True)

        #after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.stdev = np.clip(self.stdev, self.cem_stdev_min, None)
        self.stdev = np.append(self.stdev[:,1:,:], np.sqrt(0.5)*np.ones((1,1,self.num_control_inputs)), axis=1).astype(np.float32)
        self.u = np.squeeze(self.dist_mue[0,0,:])
        self.dist_mue = np.append(self.dist_mue[:,1:,:], (self.action_low + self.action_high) * 0.5 * np.ones((1,1,self.num_control_inputs)), axis=1).astype(np.float32)
        
        self.Q_logged, self.J_logged = Q, traj_cost
        self.rollout_trajectories_logged = rollout_trajectory
        self.u_logged = self.u

        return self.u

    def controller_reset(self):
        self.dist_mue = (self.action_low + self.action_high) * 0.5 * np.ones([1, self.cem_samples, self.num_control_inputs])
        self.dist_var = 0.5 * np.ones([1, self.cem_samples, self.num_control_inputs])
        self.stdev = np.sqrt(self.dist_var)
