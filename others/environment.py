from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.others.globals_and_utils import get_logger
from gym.spaces import Box

log = get_logger(__name__)


class EnvironmentBatched:
    """Has no __init__ method."""
    action_space: Box
    observation_space: Box
    dt: float
    _predictor = None
    
    @property
    def predictor(self):
        if self._predictor is None:
            raise ValueError("Predictor not set for this environment yet")
        return self._predictor

    @predictor.setter
    def predictor(self, x):
        self._predictor = x

    def step(
        self, action: TensorType
    ) -> Tuple[
        TensorType,
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        Union[np.ndarray, bool],
        dict,
    ]:
        """Step function with new OpenAI Gym API (gym>=0.26)

        :param action: control input to system
        :type action: TensorType
        :return: observation, reward, terminated, truncated, info
        :rtype: Tuple[ TensorType, Union[np.ndarray, float], Union[np.ndarray, bool], Union[np.ndarray, bool], dict, ]
        """
        return NotImplementedError()
    
    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        return NotImplementedError()

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        """Reset function with new OpenAI Gym API (gym>=0.26)

        :param state: State to set environment to, set random if default (None) is specified
        :type state: np.ndarray, optional
        :param seed: Seed for random number generator, defaults to None
        :type seed: Optional[int], optional
        :param options: Additional information to specify how the environment is reset, defaults to None. This can include a "state" key
        :type options: Optional[dict], optional
        :return: Observation of the initial state and auxiliary information
        :rtype: Tuple[np.ndarray, Optional[dict]]
        """
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

    def _get_reset_return_val(self):
        if self._batch_size == 1:
            self.state = self.lib.to_numpy(self.lib.squeeze(self.state))
        return self.state, {}

    def set_computation_library(self, ComputationLib: "type[ComputationLibrary]"):
        try:
            self.lib = ComputationLib
        except KeyError as error:
            log.exception(error)
    
    def set_logs(self, logs: dict[str, Any]):
        self._logs = logs

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
