from typing import Callable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from numpy.random import SFC64, Generator
from Control_Toolkit.others.globals_and_utils import get_logger
from gym.spaces import Box

log = get_logger(__name__)


TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
RandomGeneratorType = Union[Generator, tf.random.Generator, torch.Generator]
NumericType = Union[float, int]


class LibraryHelperFunctions:
    @staticmethod
    def set_to_value(v: TensorType, x: TensorType):
        v[...] = x

    @staticmethod
    def set_to_variable(v: tf.Variable, x: tf.Tensor):
        v.assign(x)


class ComputationLibrary:
    lib = None
    reshape: Callable[[TensorType, "tuple[int]"], TensorType] = None
    permute: Callable[[TensorType, "tuple[int]"], TensorType] = None
    newaxis = None
    shape: Callable[[TensorType], "list[int]"] = None
    to_numpy: Callable[[TensorType], np.ndarray] = None
    to_variable: Callable[[TensorType], np.ndarray] = None
    to_tensor: Callable[[TensorType, type], TensorType] = None
    constant: Callable[[TensorType, type], TensorType] = None
    unstack: Callable[[TensorType, int, int], "list[TensorType]"] = None
    ndim: Callable[[TensorType], int] = None
    clip: Callable[[TensorType, float, float], TensorType] = None
    sin: Callable[[TensorType], TensorType] = None
    asin: Callable[[TensorType], TensorType] = None
    cos: Callable[[TensorType], TensorType] = None
    tan: Callable[[TensorType], TensorType] = None
    squeeze: Callable[[TensorType], TensorType] = None
    unsqueeze: Callable[[TensorType, int], TensorType] = None
    stack: Callable[["list[TensorType]", int], TensorType] = None
    cast: Callable[[TensorType, type], TensorType] = None
    floormod: Callable[[TensorType], TensorType] = None
    float32 = None
    int32 = None
    bool = None
    tile: Callable[[TensorType, "tuple[int]"], TensorType] = None
    gather: Callable[[TensorType, TensorType, int], TensorType] = None
    arange: Callable[[Optional[NumericType], NumericType, Optional[NumericType]], TensorType] = None
    zeros: Callable[["tuple[int]"], TensorType] = None
    zeros_like: Callable[[TensorType], TensorType] = None
    ones: Callable[["tuple[int]"], TensorType] = None
    sign: Callable[[TensorType], TensorType] = None
    create_rng: Callable[[int], RandomGeneratorType] = None
    standard_normal: Callable[[RandomGeneratorType, "tuple[int]"], TensorType] = None
    uniform: Callable[
        [RandomGeneratorType, "tuple[int]", TensorType, TensorType, type], TensorType
    ] = None
    sum: Callable[[TensorType, int], TensorType] = None
    set_shape: Callable[[TensorType, "list[int]"], None] = None
    concat: Callable[["list[TensorType]", int], TensorType]
    pi: TensorType = None
    any: Callable[[TensorType], bool] = None
    all: Callable[[TensorType], bool] = None
    reduce_any: Callable[[TensorType, int], bool] = None
    reduce_all: Callable[[TensorType, int], bool] = None
    reduce_max: Callable[[TensorType, int], bool] = None
    less: Callable[[TensorType, TensorType], TensorType] = None
    greater: Callable[[TensorType, TensorType], TensorType] = None
    logical_not: Callable[[TensorType], TensorType] = None
    min: Callable[[TensorType, TensorType], TensorType] = None
    max: Callable[[TensorType, TensorType], TensorType] = None
    atan2: Callable[[TensorType], TensorType] = None
    abs: Callable[[TensorType], TensorType] = None
    sqrt: Callable[[TensorType], TensorType] = None
    argpartition: Callable[[TensorType, int], TensorType] = None
    norm: Callable[[TensorType, int], bool] = None
    cross: Callable[[TensorType, TensorType], TensorType] = None
    dot: Callable[[TensorType, TensorType], TensorType] = None
    stop_gradient: Callable[[TensorType], TensorType] = None
    assign: Callable[[Union[TensorType, tf.Variable], TensorType], Union[TensorType, tf.Variable]] = None


class NumpyLibrary(ComputationLibrary):
    lib = 'Numpy'
    reshape = lambda x, shape: np.reshape(x, shape)
    permute = np.transpose
    newaxis = np.newaxis
    shape = np.shape
    to_numpy = lambda x: np.array(x)
    to_variable = np.array
    to_tensor = lambda x, dtype: np.array(x, dtype=dtype)
    constant = lambda x, t: np.array(x, dtype=t)
    unstack = lambda x, num, axis: list(np.moveaxis(x, axis, 0))
    ndim = np.ndim
    clip = np.clip
    sin = np.sin
    asin = np.arcsin
    cos = np.cos
    tan = np.tan
    squeeze = np.squeeze
    unsqueeze = np.expand_dims
    stack = np.stack
    cast = lambda x, t: x.astype(t)
    floormod = np.mod
    float32 = np.float32
    int32 = np.int32
    bool = np.bool_
    tile = np.tile
    gather = lambda x, i, a: np.take(x, i, axis=a)
    arange = np.arange
    zeros = np.zeros
    zeros_like = np.zeros_like
    ones = np.ones
    sign = np.sign
    create_rng = lambda seed: Generator(SFC64(seed))
    standard_normal = lambda generator, shape: generator.standard_normal(size=shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        low=low, high=high, size=shape
    ).astype(dtype)
    sum = lambda x, a: np.sum(x, axis=a, keepdims=False)
    set_shape = lambda x, shape: x
    concat = lambda x, a: np.concatenate(x, axis=a)
    pi = np.array(np.pi).astype(np.float32)
    any = np.any
    all = np.all
    reduce_any = lambda a, axis: np.any(a, axis=axis)
    reduce_all = lambda a, axis: np.all(a, axis=axis)
    reduce_max = lambda a, axis: np.max(a, axis=axis)
    less = lambda x, y: np.less(x, y)
    greater = lambda x, y: np.greater(x, y)
    logical_not = lambda x: np.logical_not(x)
    min = np.minimum
    max = np.maximum
    atan2 = np.arctan2
    abs = np.abs
    sqrt = np.sqrt
    argpartition = lambda x, k: np.argpartition(x, k)[..., :k]
    norm = lambda x, axis: np.linalg.norm(x, axis=axis)
    cross = np.cross
    dot = np.dot
    stop_gradient = lambda x: x
    assign = LibraryHelperFunctions.set_to_value


class TensorFlowLibrary(ComputationLibrary):
    lib = 'TF'
    reshape = tf.reshape
    permute = tf.transpose
    newaxis = tf.newaxis
    shape = lambda x: x.get_shape()  # .as_list()
    to_numpy = lambda x: x.numpy()
    to_variable = tf.Variable
    to_tensor = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)
    constant = lambda x, t: tf.constant(x, dtype=t)
    unstack = lambda x, num, axis: tf.unstack(x, num=num, axis=axis)
    ndim = tf.rank
    clip = tf.clip_by_value
    sin = tf.sin
    asin = tf.asin
    cos = tf.cos
    tan = tf.tan
    squeeze = tf.squeeze
    unsqueeze = tf.expand_dims
    stack = tf.stack
    cast = lambda x, t: tf.cast(x, dtype=t)
    floormod = tf.math.floormod
    float32 = tf.float32
    int32 = tf.int32
    bool = tf.bool
    tile = tf.tile
    gather = lambda x, i, a: tf.gather(x, i, axis=a)
    arange = tf.range
    zeros = tf.zeros
    zeros_like = tf.zeros_like
    ones = tf.ones
    sign = tf.sign
    create_rng = lambda seed: tf.random.Generator.from_seed(seed)
    standard_normal = lambda generator, shape: generator.normal(shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        shape, minval=low, maxval=high, dtype=dtype
    )
    sum = lambda x, a: tf.reduce_sum(x, axis=a, keepdims=False)
    set_shape = lambda x, shape: x.set_shape(shape)
    concat = lambda x, a: tf.concat(x, a)
    pi = tf.convert_to_tensor(np.array(np.pi), dtype=tf.float32)
    any = tf.reduce_any
    all = tf.reduce_all
    reduce_any = lambda a, axis: tf.reduce_any(a, axis=axis)
    reduce_all = lambda a, axis: tf.reduce_all(a, axis=axis)
    reduce_max = lambda a, axis: tf.reduce_max(a, axis=axis)
    less = lambda x, y: tf.math.less(x, y)
    greater = lambda x, y: tf.math.greater(x, y)
    logical_not = lambda x: tf.math.logical_not(x)
    min = tf.minimum
    max = tf.maximum
    atan2 = tf.atan2
    abs = tf.abs
    sqrt = tf.sqrt
    argpartition = lambda x, k: tf.math.top_k(-x, k, sorted=False)[1]
    norm = lambda x, axis: tf.norm(x, axis=axis)
    cross = tf.linalg.cross
    dot = lambda a, b: tf.tensordot(a, b, 1)
    stop_gradient = tf.stop_gradient
    assign = LibraryHelperFunctions.set_to_variable


class PyTorchLibrary(ComputationLibrary):
    lib = 'Pytorch'
    reshape = torch.reshape
    permute = torch.permute
    newaxis = None
    shape = lambda x: list(x.size())
    to_numpy = lambda x: x.cpu().detach().numpy()
    to_variable = torch.as_tensor
    to_tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype)
    constant = lambda x, t: torch.as_tensor(x, dtype=t)
    unstack = lambda x, num, dim: torch.unbind(x, dim=dim)
    ndim = lambda x: x.ndim
    clip = torch.clamp
    sin = torch.sin
    asin = torch.asin
    cos = torch.cos
    tan = torch.tan
    squeeze = torch.squeeze
    unsqueeze = torch.unsqueeze
    stack = torch.stack
    cast = lambda x, t: x.type(t)
    floormod = torch.remainder
    float32 = torch.float32
    int32 = torch.int32
    bool = torch.bool
    tile = torch.tile
    gather = lambda x, i, a: torch.gather(x, dim=a, index=i)
    arange = torch.arange
    zeros = torch.zeros
    zeros_like = torch.zeros_like
    ones = torch.ones
    sign = torch.sign
    create_rng = lambda seed: torch.Generator().manual_seed(seed)
    standard_normal = lambda generator, shape: torch.normal(
        torch.zeros(shape), 1.0, generator=generator
    )
    uniform = (
        lambda generator, shape, low, high, dtype: (high - low)
        * torch.rand(*shape, generator=generator, dtype=dtype)
        + low
    )
    sum = lambda x, a: torch.sum(x, a, keepdim=False)
    set_shape = lambda x, shape: x
    concat = lambda x, a: torch.concat(x, dim=a)
    pi = torch.from_numpy(np.array(np.pi)).float()
    any = torch.any
    all = torch.all
    reduce_any = lambda a, axis: torch.any(a, dim=axis)
    reduce_all = lambda a, axis: torch.all(a, dim=axis)
    reduce_max = lambda a, axis: torch.max(a, dim=axis)
    less = lambda x, y: torch.less(x, y)
    greater = lambda x, y: torch.greater(x, y)
    logical_not = lambda x: torch.logical_not(x)
    min = torch.minimum
    max = torch.maximum
    atan2 = torch.atan2
    abs = torch.abs
    sqrt = torch.sqrt
    argpartition = torch.topk
    norm = lambda x, axis: torch.linalg.norm(x, dim=axis)
    cross = torch.linalg.cross
    dot = torch.dot
    stop_gradient = tf.stop_gradient # FIXME: How to imlement this in torch?
    assign = LibraryHelperFunctions.set_to_value


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
