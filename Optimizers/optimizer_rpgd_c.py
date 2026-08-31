import ctypes
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Tuple

# libgomp reads this while native dependencies are loading.
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")

import numpy as np

from CartPole.cartpole_parameters import CartPoleParameters
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary
from SI_Toolkit.load_and_normalize import load_yaml


class RpgdConfig(ctypes.Structure):
    _fields_ = [
        ("mpc_horizon", ctypes.c_int),
        ("num_rollouts", ctypes.c_int),
        ("outer_its", ctypes.c_int),
        ("resamp_per", ctypes.c_int),
        ("period_interpolation_inducing_points", ctypes.c_int),
        ("intermediate_steps", ctypes.c_int),
        ("shift_previous", ctypes.c_int),
        ("sampling_distribution", ctypes.c_int),
        ("sample_whole_control_space", ctypes.c_int),
        ("warmup", ctypes.c_int),
        ("warmup_iterations", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("reserve_threads", ctypes.c_int),
        ("seed", ctypes.c_uint),
        ("mpc_timestep", ctypes.c_float),
        ("learning_rate", ctypes.c_float),
        ("adam_beta_1", ctypes.c_float),
        ("adam_beta_2", ctypes.c_float),
        ("adam_epsilon", ctypes.c_float),
        ("gradmax_clip", ctypes.c_float),
        ("opt_keep_k_ratio", ctypes.c_float),
        ("sample_stdev", ctypes.c_float),
        ("sample_mean", ctypes.c_float),
        ("uniform_dist_min", ctypes.c_float),
        ("uniform_dist_max", ctypes.c_float),
        ("action_low", ctypes.c_float),
        ("action_high", ctypes.c_float),
        ("k", ctypes.c_float),
        ("m_cart", ctypes.c_float),
        ("m_pole", ctypes.c_float),
        ("g", ctypes.c_float),
        ("J_fric", ctypes.c_float),
        ("M_fric", ctypes.c_float),
        ("L", ctypes.c_float),
        ("u_max", ctypes.c_float),
        ("track_half_length", ctypes.c_float),
        ("dd_quadratic_weight_up", ctypes.c_float),
        ("db_weight_up", ctypes.c_float),
        ("ep_weight_up", ctypes.c_float),
        ("ekp_weight_up", ctypes.c_float),
        ("cc_weight_up", ctypes.c_float),
        ("vel_penalty_reg", ctypes.c_float),
        ("R", ctypes.c_float),
        ("permissible_track_fraction", ctypes.c_float),
    ]


class RpgdRuntime(ctypes.Structure):
    _fields_ = [
        ("target_position", ctypes.c_float),
        ("target_equilibrium", ctypes.c_float),
        ("L", ctypes.c_float),
        ("m_pole", ctypes.c_float),
    ]


class optimizer_rpgd_c(template_optimizer):
    supported_computation_libraries = (NumpyLibrary, TensorFlowLibrary, PyTorchLibrary)

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
        intermediate_steps: int = 10,
        num_threads: int = 0,
        reserve_threads: int = 1,
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
        self.seed = 1 if seed is None else int(seed)
        self.outer_its = outer_its
        self.sample_stdev = sample_stdev
        self.sample_mean = sample_mean
        self.sample_whole_control_space = bool(sample_whole_control_space)
        self.uniform_dist_min = uniform_dist_min
        self.uniform_dist_max = uniform_dist_max
        self.resamp_per = resamp_per
        self.period_interpolation_inducing_points = period_interpolation_inducing_points
        self.intermediate_steps = int(intermediate_steps)
        if SAMPLING_DISTRIBUTION not in ("normal", "uniform"):
            raise ValueError(
                "RPGD-C SAMPLING_DISTRIBUTION must be 'normal' or 'uniform', "
                f"got {SAMPLING_DISTRIBUTION!r}"
            )
        self.sampling_distribution = 1 if SAMPLING_DISTRIBUTION == "uniform" else 0
        self.shift_previous = shift_previous
        self.warmup = bool(warmup)
        self.warmup_iterations = warmup_iterations
        self.learning_rate = learning_rate
        self.opt_keep_k_ratio = opt_keep_k_ratio
        self.gradmax_clip = gradmax_clip
        self.rtol = rtol
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.adam_epsilon = adam_epsilon
        self.calculate_optimal_trajectory = calculate_optimal_trajectory
        self.num_threads = int(num_threads)
        self.reserve_threads = int(reserve_threads)

        self._c_lib = None
        self._solver = None
        self._cfg = None
        self._lock = threading.RLock()
        self._runtime = RpgdRuntime()
        self._state_arr = (ctypes.c_float * 6)()
        self._state_np = np.ctypeslib.as_array(self._state_arr)
        self.u = np.zeros((1,), dtype=np.float32)
        self.optimal_control_sequence = None
        self.rollout_trajectories = None

    def configure(self, num_states: int, num_control_inputs: int, **kwargs):
        super().configure(num_states=num_states, num_control_inputs=num_control_inputs, default_configure=False)
        with self._lock:
            if self._c_lib is None:
                self._setup_c_backend()
            cfg = self._make_config(kwargs.get("dt", None))
            validation_status = self._c_lib.rpgd_validate_config(ctypes.byref(cfg))
            if validation_status != 0:
                raise ValueError(f"Invalid RPGD-C configuration (status {validation_status})")
            solver = self._c_lib.rpgd_create(ctypes.byref(cfg))
            if not solver:
                raise RuntimeError("Failed to create RPGD C solver")
            old_solver = self._solver
            self._cfg = cfg
            self._solver = solver
            self.thread_count = self._c_lib.rpgd_get_num_threads(solver)
            if old_solver is not None:
                self._c_lib.rpgd_destroy(old_solver)

    def step(self, s: np.ndarray, time=None):
        with self._lock:
            if self._solver is None:
                raise RuntimeError("RPGD-C optimizer is not configured")
            if self.optimizer_logging:
                self.logging_values = {"s_logged": s.copy()}
            np.copyto(self._state_np, np.asarray(s, dtype=np.float32)[:6])
            self._runtime.target_position = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "target_position", 0.0))
            self._runtime.target_equilibrium = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "target_equilibrium", 1.0))
            self._runtime.L = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "L", self._cfg.L))
            self._runtime.m_pole = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "m_pole", self._cfg.m_pole))
            u = self._c_lib.rpgd_step(self._solver, self._state_arr, ctypes.byref(self._runtime))
            status = self._c_lib.rpgd_get_last_status(self._solver)
            if status in (-1, -4, -6):
                raise RuntimeError(f"RPGD-C step failed safely (status {status})")
            self.u = np.asarray([u], dtype=np.float32)
            self.optimal_control_sequence = self.u.reshape(1, 1, 1)
            return self.u

    def optimizer_reset(self):
        with self._lock:
            if self._solver is not None:
                self._c_lib.rpgd_reset(self._solver, ctypes.c_uint(self.seed))

    def debug_get_q(self):
        with self._lock:
            out = np.empty((self.num_rollouts, self.mpc_horizon), dtype=np.float32)
            self._c_lib.rpgd_debug_get_q(self._solver, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            return out

    def debug_set_q(self, q):
        with self._lock:
            q_arr = np.asarray(q, dtype=np.float32).reshape(self.num_rollouts, self.mpc_horizon)
            self._c_lib.rpgd_debug_set_q(self._solver, q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def debug_get_costs(self):
        with self._lock:
            out = np.empty((self.num_rollouts,), dtype=np.float32)
            self._c_lib.rpgd_debug_get_costs(self._solver, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            return out

    def debug_get_indices(self):
        with self._lock:
            out = np.empty((self.num_rollouts,), dtype=np.int32)
            self._c_lib.rpgd_debug_get_indices(self._solver, out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
            return out

    def debug_gradient_adjoint(self, state, q, runtime=None):
        with self._lock:
            state_arr = np.asarray(state, dtype=np.float32).reshape(6)
            q_arr = np.asarray(q, dtype=np.float32).reshape(self.mpc_horizon)
            grad = np.empty((self.mpc_horizon,), dtype=np.float32)
            rt = runtime if runtime is not None else self._runtime_from_cost_parameters()
            self._c_lib.rpgd_debug_gradient_adjoint(
                ctypes.byref(self._cfg),
                ctypes.byref(rt),
                state_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return grad

    def _runtime_from_cost_parameters(self):
        rt = RpgdRuntime()
        rt.target_position = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "target_position", 0.0))
        rt.target_equilibrium = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "target_equilibrium", 1.0))
        rt.L = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "L", self._cfg.L))
        rt.m_pole = self._to_float(getattr(self.cost_function.cost_function.variable_parameters, "m_pole", self._cfg.m_pole))
        return rt

    def close(self):
        with self._lock:
            if self._solver is not None:
                self._c_lib.rpgd_destroy(self._solver)
                self._solver = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _to_float(value):
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return float(value)

    def _make_config(self, dt):
        cp = CartPoleParameters()
        cost_cfg = load_yaml(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))
        minimal = cost_cfg["CartPole"]["quadratic_boundary_grad_minimal"]
        action_low = self._to_float(self.action_low)
        action_high = self._to_float(self.action_high)
        return RpgdConfig(
            int(self.mpc_horizon),
            int(self.num_rollouts),
            int(self.outer_its),
            int(self.resamp_per),
            int(self.period_interpolation_inducing_points),
            int(self.intermediate_steps),
            int(self.shift_previous),
            int(self.sampling_distribution),
            int(self.sample_whole_control_space),
            int(self.warmup),
            int(self.warmup_iterations),
            int(self.num_threads),
            int(self.reserve_threads),
            int(self.seed),
            float(dt if dt is not None else 0.02),
            float(self.learning_rate),
            float(self.adam_beta_1),
            float(self.adam_beta_2),
            float(self.adam_epsilon),
            float(self.gradmax_clip),
            float(self.opt_keep_k_ratio),
            float(self.sample_stdev),
            float(self.sample_mean),
            float(self.uniform_dist_min),
            float(self.uniform_dist_max),
            action_low,
            action_high,
            self._to_float(cp.k),
            self._to_float(cp.m_cart),
            self._to_float(cp.m_pole),
            self._to_float(cp.g),
            self._to_float(cp.J_fric),
            self._to_float(cp.M_fric),
            self._to_float(cp.L),
            self._to_float(cp.u_max),
            self._to_float(cp.TrackHalfLength),
            float(minimal["dd_quadratic_weight_up"]),
            float(minimal["db_weight_up"]),
            float(minimal["ep_weight_up"]),
            float(minimal["ekp_weight_up"]),
            float(minimal["cc_weight_up"]),
            float(minimal["vel_penalty_reg"]),
            float(minimal["R"]),
            float(minimal["permissible_track_fraction"]),
        )

    def _setup_c_backend(self):
        c_dir = Path(__file__).resolve().parent / "rpgd_c"
        ext = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}[sys.platform]
        lib_path = c_dir / f"librpgd_cartpole{ext}"
        sources = [
            c_dir / "rpgd_cartpole.c",
            c_dir / "cartpole_model.c",
            c_dir / "cartpole_cost.c",
        ]
        headers = [
            c_dir / "rpgd_cartpole.h",
            c_dir / "cartpole_model.h",
            c_dir / "cartpole_cost.h",
            c_dir / "rpgd_platform.h",
            c_dir / "rpgd_worker.h",
        ]
        newest_source = max(path.stat().st_mtime for path in sources + headers)
        if (not lib_path.exists()) or lib_path.stat().st_mtime < newest_source:
            self._build_c_library(c_dir, lib_path.name)
        self._c_lib = ctypes.CDLL(str(lib_path))
        self._c_lib.rpgd_get_abi_version.argtypes = []
        self._c_lib.rpgd_get_abi_version.restype = ctypes.c_uint
        self._c_lib.rpgd_get_config_size.argtypes = []
        self._c_lib.rpgd_get_config_size.restype = ctypes.c_size_t
        if self._c_lib.rpgd_get_abi_version() != 1:
            raise RuntimeError("Unsupported RPGD-C library ABI")
        if self._c_lib.rpgd_get_config_size() != ctypes.sizeof(RpgdConfig):
            raise RuntimeError("RPGD-C/Python RpgdConfig layout mismatch")
        self._c_lib.rpgd_validate_config.argtypes = [ctypes.POINTER(RpgdConfig)]
        self._c_lib.rpgd_validate_config.restype = ctypes.c_int
        self._c_lib.rpgd_create.argtypes = [ctypes.POINTER(RpgdConfig)]
        self._c_lib.rpgd_create.restype = ctypes.c_void_p
        self._c_lib.rpgd_destroy.argtypes = [ctypes.c_void_p]
        self._c_lib.rpgd_destroy.restype = None
        self._c_lib.rpgd_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._c_lib.rpgd_reset.restype = None
        self._c_lib.rpgd_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(RpgdRuntime),
        ]
        self._c_lib.rpgd_step.restype = ctypes.c_float
        self._c_lib.rpgd_get_num_threads.argtypes = [ctypes.c_void_p]
        self._c_lib.rpgd_get_num_threads.restype = ctypes.c_int
        self._c_lib.rpgd_debug_rollout_cost.argtypes = [
            ctypes.POINTER(RpgdConfig),
            ctypes.POINTER(RpgdRuntime),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._c_lib.rpgd_debug_rollout_cost.restype = ctypes.c_float
        self._c_lib.rpgd_debug_rollout_final_state.argtypes = [
            ctypes.POINTER(RpgdConfig),
            ctypes.POINTER(RpgdRuntime),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._c_lib.rpgd_debug_rollout_final_state.restype = None
        self._c_lib.rpgd_debug_gradient_adjoint.argtypes = [
            ctypes.POINTER(RpgdConfig),
            ctypes.POINTER(RpgdRuntime),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._c_lib.rpgd_debug_gradient_adjoint.restype = None
        self._c_lib.rpgd_debug_gradient_fd.argtypes = self._c_lib.rpgd_debug_gradient_adjoint.argtypes
        self._c_lib.rpgd_debug_gradient_fd.restype = None
        self._c_lib.rpgd_debug_set_q.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._c_lib.rpgd_debug_set_q.restype = None
        self._c_lib.rpgd_debug_get_q.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._c_lib.rpgd_debug_get_q.restype = None
        self._c_lib.rpgd_debug_set_adam.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._c_lib.rpgd_debug_set_adam.restype = None
        self._c_lib.rpgd_debug_get_adam.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._c_lib.rpgd_debug_get_adam.restype = None
        self._c_lib.rpgd_debug_get_costs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self._c_lib.rpgd_debug_get_costs.restype = None
        self._c_lib.rpgd_debug_get_indices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        self._c_lib.rpgd_debug_get_indices.restype = None
        self._c_lib.rpgd_get_workspace_bytes.argtypes = [ctypes.c_void_p]
        self._c_lib.rpgd_get_workspace_bytes.restype = ctypes.c_size_t
        self._c_lib.rpgd_get_static_workspace_bytes.argtypes = []
        self._c_lib.rpgd_get_static_workspace_bytes.restype = ctypes.c_size_t
        self._c_lib.rpgd_get_last_status.argtypes = [ctypes.c_void_p]
        self._c_lib.rpgd_get_last_status.restype = ctypes.c_int
        self._c_lib.rpgd_is_baremetal.argtypes = []
        self._c_lib.rpgd_is_baremetal.restype = ctypes.c_int

    @staticmethod
    def _build_c_library(c_dir: Path, lib_name: str, extra_cflags=None, allow_openmp=True):
        compiler = os.environ.get("CC", "gcc")
        sources = [
            "rpgd_cartpole.c",
            "cartpole_model.c",
            "cartpole_cost.c",
        ]
        extra = list(extra_cflags or [])
        output_name = f".{lib_name}.{os.getpid()}.{threading.get_ident()}.tmp"

        def build_command(*flags, pthread=False):
            return [
                compiler,
                "-std=c11",
                "-O3",
                "-fPIC",
                "-shared",
                *flags,
                *extra,
                *sources,
                "-lm",
                *(["-pthread"] if pthread else []),
                "-o",
                output_name,
            ]

        base_cmd = build_command(pthread=True)
        lto_cmd = build_command("-flto", pthread=True)
        openmp_lto_cmd = build_command("-flto", "-fopenmp")
        openmp_cmd = build_command("-fopenmp")
        baremetal_cmd = build_command()
        commands = [openmp_lto_cmd, openmp_cmd, lto_cmd, base_cmd] if allow_openmp else [baremetal_cmd]
        last_error = None
        for cmd in commands:
            try:
                result = subprocess.run(cmd, cwd=c_dir, capture_output=True, text=True)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd,
                        output=result.stdout,
                        stderr=result.stderr,
                    )
                if allow_openmp and "-fopenmp" not in cmd:
                    print("Built RPGD C backend without OpenMP; using pthread rollout workers.")
                os.replace(c_dir / output_name, c_dir / lib_name)
                return
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                last_error = exc
                try:
                    (c_dir / output_name).unlink()
                except FileNotFoundError:
                    pass
        raise RuntimeError(f"Could not build RPGD C backend on {platform.platform()}: {last_error}")
