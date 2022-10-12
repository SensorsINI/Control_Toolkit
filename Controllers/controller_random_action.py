from importlib import import_module

import numpy as np
import tensorflow as tf
from Control_Toolkit.others.environment import EnvironmentBatched
from others.globals_and_utils import create_rng
from SI_Toolkit.Functions.TF.Compile import CompileTF

from Control_Toolkit.Controllers import template_controller


class controller_random_action(template_controller):
    def __init__(
        self,
        environment_model: EnvironmentBatched,
        seed: int,
        num_control_inputs: int,
        dt: float,
        mpc_horizon: int,
        num_rollouts: int,
        predictor_name: str,
        predictor_intermediate_steps: int,
        CEM_NET_NAME: str,
        **kwargs,
    ):
        # First configure random sampler
        self.rng = create_rng(self.__class__.__name__, seed, use_tf=True)

        # Parametrization
        self.num_control_inputs = num_control_inputs

        # MPC params
        self.num_rollouts = num_rollouts
        self.num_horizon = mpc_horizon  # Number of steps in MPC horizon
        self.intermediate_steps = predictor_intermediate_steps
        self.NET_NAME = CEM_NET_NAME

        # instantiate predictor
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        self.predictor = getattr(predictor_module, predictor_name)(
            horizon=self.num_horizon,
            dt=dt,
            intermediate_steps=self.intermediate_steps,
            disable_individual_compilation=True,
            batch_size=self.num_rollouts,
            net_name=self.NET_NAME,
            planning_environment=environment_model,
        )

        super().__init__(environment_model)
        self.action_low = self.env_mock.action_space.low
        self.action_high = self.env_mock.action_space.high

        # Initialization
        self.controller_reset()
        self.u = 0
    
    @CompileTF
    def predict_and_cost(self, s, Q):
        # rollout trajectories and retrieve cost
        rollout_trajectory = self.predictor.predict_tf(s, Q)
        traj_cost = self.env_mock.cost_functions.get_trajectory_cost(
            rollout_trajectory, Q, self.u
        )
        return traj_cost, rollout_trajectory

    # step function to find control
    def step(self, s: np.ndarray, time=None):
        # Start all trajectories in current state
        s = np.tile(s, tf.constant([self.num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        
        Q = self.rng.uniform(
            shape=[self.num_rollouts, self.num_horizon, self.num_control_inputs],
            minval=self.action_low,
            maxval=self.action_high,
            dtype=tf.float32,
        )
        traj_cost, rollout_trajectory = self.predict_and_cost(s, Q)

        # sort the costs and find best k costs
        sorted_cost = tf.argsort(traj_cost)
        best_idx = sorted_cost[0]

        self.u: np.ndarray = tf.squeeze(Q[best_idx, 0, :]).numpy()

        self.Q_logged, self.J_logged = Q.numpy(), traj_cost.numpy()
        self.rollout_trajectories_logged = rollout_trajectory.numpy()
        self.u_logged = self.u.copy()

        return self.u

    def controller_reset(self):
        # generate random input sequence and clip to control limits
        Q = self.rng.uniform(
                shape=[self.num_rollouts, self.num_horizon, self.num_control_inputs],
                minval=self.action_low,
                maxval=self.action_high,
                dtype=tf.float32,
            )
        Q = tf.clip_by_value(Q, self.action_low, self.action_high)
