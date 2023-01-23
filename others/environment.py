from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch

from SI_Toolkit.computation_library import ComputationLibrary, TensorType
from Control_Toolkit.others.globals_and_utils import get_logger
from gymnasium.spaces import Box

log = get_logger(__name__)


class EnvironmentBatched:
    """Has no __init__ method."""
    action_space: Box
    observation_space: Box
    dt: float
    num_states: int
    num_actions: int
    _predictor = None
    _actuator_noise: Union[np.ndarray, float]
    _batch_size: int
    environment_attributes = {}
    
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
        """Step function with new OpenAI Gym API (gymnasium>=0.26)

        :param action: control input to system
        :type action: TensorType
        :return: observation, reward, terminated, truncated, info
        :rtype: Tuple[ TensorType, Union[np.ndarray, float], Union[np.ndarray, bool], Union[np.ndarray, bool], dict, ]
        """
        raise NotImplementedError()
    
    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        raise NotImplementedError()

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        """Reset function with new OpenAI Gym API (gymnasium>=0.26)

        :param state: State to set environment to, set random if default (None) is specified
        :type state: np.ndarray, optional
        :param seed: Seed for random number generator, defaults to None
        :type seed: Optional[int], optional
        :param options: Additional information to specify how the environment is reset, defaults to None. This can include a "state" key
        :type options: Optional[dict], optional
        :return: Observation of the initial state and auxiliary information
        :rtype: Tuple[np.ndarray, Optional[dict]]
        """
        raise NotImplementedError()

    def _set_up_rng(self, seed: int = None) -> None:
        if seed is None:
            seed = 0
            log.warn(f"Environment set up with no seed specified. Setting to {seed}.")

        self.rng = self.lib.create_rng(seed)

    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, *args, **kwargs):
        # Static method that describes when the environment is done
        # Each subclassed environment can specify any list of args and kwargs that may be needed
        raise NotImplementedError()

    def get_reward(self, state, action):
        raise NotImplementedError()

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
        state: TensorType,
        action: TensorType,
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
    
    @property
    def logs(self):
        return getattr(self, "_logs", dict())
    
    def set_logs(self, logs: "dict[str, Any]"):
        self._logs = logs
