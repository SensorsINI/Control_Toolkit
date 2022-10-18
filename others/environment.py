from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.others.globals_and_utils import get_logger
from gym.spaces import Box

log = get_logger(__name__)


class EnvironmentBatched:
    """Has no __init__ method."""
    class cost_functions_wrapper:
        def __init__(self, env) -> None:
            self.env: EnvironmentBatched = env

        def get_terminal_cost(self, s_hor):
            return 0.0

        def get_stage_cost(self, s, u, u_prev):
            return -self.env.get_reward(s, u)

        def get_trajectory_cost(self, s_hor, u, u_prev=None):
            return (
                self.env.lib.sum(self.get_stage_cost(s_hor[:, :-1, :], u, None), 1)
                + self.get_terminal_cost(s_hor)
            )
    
    action_space: Box
    observation_space: Box
    cost_functions: cost_functions_wrapper
    dt: float

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        return NotImplementedError()
    
    def step_tf(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        return NotImplementedError()

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return NotImplementedError()

    def _set_up_rng(self, seed: int = None) -> None:
        if seed is None:
            seed = 0
            log.warn(f"Environment set up with no seed specified. Setting to {seed}.")

        self.rng = self.lib.create_rng(seed)

    def is_done(self, state):
        return NotImplementedError()

    def get_reward(self, state, action):
        return NotImplementedError()

    def _apply_actuator_noise(self, action: TensorType):
        disturbance = (
            self._actuator_noise
            * (self.action_space.high - self.action_space.low)
            * self.lib.standard_normal(
                self.rng, (self._batch_size, len(self._actuator_noise))
            )
        )
        return self.lib.clip(action + disturbance, self.action_space.low, self.action_space.high)

    def _expand_arrays(
        self,
        state: Union[np.ndarray, tf.Tensor, torch.Tensor],
        action: Union[np.ndarray, tf.Tensor, torch.Tensor],
    ):
        if self.lib.ndim(action) < 2:
            action = self.lib.reshape(
                action, (self._batch_size, sum(self.action_space.shape))
            )
        if self.lib.ndim(state) < 2:
            state = self.lib.reshape(
                state, (self._batch_size, sum(self.observation_space.shape))
            )
        return state, action

    def _get_reset_return_val(self, return_info: bool = False):
        if self._batch_size == 1:
            self.state = self.lib.to_numpy(self.lib.squeeze(self.state))

        if return_info:
            return tuple((self.state, {}))
        return self.state

    def set_computation_library(self, computation_lib: "type[ComputationLibrary]"):
        try:
            self.lib = computation_lib
        except KeyError as error:
            log.exception(error)

    # Overloading properties/methods for Bharadhwaj implementation
    @property
    def a_size(self):
        return self.action_space.shape[0]

    def reset_state(self, batch_size):
        self.reset()

    @property
    def B(self):
        return self._batch_size

    def rollout(self, actions, return_traj=False):
        # Uncoditional action sequence rollout
        # actions: shape: TxBxA (time, batch, action)
        assert actions.dim() == 3
        assert actions.size(1) == self.B, "{}, {}".format(actions.size(1), self.B)
        assert actions.size(2) == self.a_size
        T = actions.size(0)
        rs = []
        ss = []

        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        for i in range(T):
            # Reshape for step function: BxTxA
            s, r, _, _ = self.step(actions[i])
            rs.append(r)
            ss.append(s)
            total_r = total_r + r
            # if(done):
            #     break
        if return_traj:
            return rs, ss
        else:
            return total_r
