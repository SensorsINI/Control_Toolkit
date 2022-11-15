"""
This is a linear-quadratic optimizer
The Jacobian of the model needs to be provided
"""

from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
import scipy

from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

from CartPoleSimulation.CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                                         POSITIOND_IDX)
from Control_Toolkit.others.globals_and_utils import create_rng

# Forces
from forces import forcespro
import forcespro.nlp
import numpy as np
from forces import get_userid
import casadi
import os
import pickle


class optimizer_lqr_forces(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}

    def __init__(
            self,
            predictor: PredictorWrapper,
            cost_function: CostFunctionWrapper,
            num_states: int,
            num_control_inputs: int,
            control_limits: "Tuple[np.ndarray, np.ndarray]",
            computation_library: "type[ComputationLibrary]",
            seed: int,
            mpc_horizon: int,
            num_rollouts: int,
            optimizer_logging: bool,
            jacobian_path: str,
            action_max: float,
            state_max: list[float],
            q: float,
            r: float
    ):
        super().__init__(
            predictor=predictor,
            cost_function=cost_function,
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            control_limits=control_limits,
            optimizer_logging=optimizer_logging,
            seed=seed,
            num_rollouts=1,
            mpc_horizon=mpc_horizon,
            computation_library=computation_library,
        )

        # dynamically import jacobian module
        module_path, jacobian_method = jacobian_path.rsplit('.', 1)
        self.module = __import__(module_path, fromlist=["cartpole_jacobian"])
        self.jacobian = getattr(self.module, jacobian_method)

        self.action_low = -action_max
        self.action_high = +action_max
        self.q = q
        self.r = r

        self.optimizer_reset()

        # lower and upper bounds
        # constrained_idx = [i for i, x in enumerate(state_max) if x != 'inf']
        #
        # xmax = np.array([state_max[i] for i in constrained_idx])
        xmax = np.array([s if s != 'inf' else np.inf for s in state_max])
        xmin = -xmax

        umin = np.array([self.action_low])
        umax = np.array([self.action_high])

        # ubidx = [1] + [i + 2 for i in constrained_idx]
        # lbidx = ubidx

        self.nxc = len(state_max)
        self.nx = len(state_max)
        self.nu = 1

        # # Cost matrices for LQR controller
        # self.Q = np.diag([self.P] * self.nx).astype(np.float32)  # How much to punish x
        # self.R = np.diag([self.R] * self.nu).astype(np.float32)  # How much to punish u

        # for readability
        N = self.mpc_horizon
        nxc = self.nxc
        nx = self.nx
        nu = self.nu
        # terminal weight obtained from discrete-time Riccati equation

        # Model Definition
        # ----------------

        # Problem dimensions
        self.model = forcespro.nlp.SymbolicModel(N)  # horizon length
        model = self.model
        model.nvar = nu + nx  # number of variables
        model.neq = nx  # number of equality constraints
        model.nh = 0  # number of inequality constraint functions
        model.npar = 1  # number of runtime parameters

        Tf = 2  # final time

        Q = np.diag([q] * self.nx)
        R = np.diag([r] * self.nu)
        sqrt_weights = [np.sqrt(p) for p in [r] * nu + [q] * nx]

        # model.objective = lambda z, p: (sqrt_weights*z).T @ z

        model.LSobjective = lambda z, p: np.array(sqrt_weights) * z
        model.continuous_dynamics = self.linear_dynamics

        # We use an explicit RK4 integrator here to discretize continuous dynamics
        integrator_stepsize = Tf / (model.N - 1)

        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        model.E = np.concatenate([np.zeros((nx, nu)), np.identity(nx)], axis=1)

        # Inequality constraints
        # upper/lower variable bounds lb <= x <= ub
        model.lb = np.concatenate((umin, xmin), 0)
        model.ub = np.concatenate((umax, xmax), 0)

        model.xinitidx = range(nu, nu + nx)  # indexes affected by initial condition

        # Generate solver
        # ---------------

        # Define solver options
        codeoptions = forcespro.CodeOptions()
        codeoptions.maxit = 200  # Maximum number of iterations
        codeoptions.printlevel = 2  # Use printlevel = 2 to print progress (but not for timings)
        codeoptions.optlevel = 3  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        codeoptions.nlp.integrator.Ts = integrator_stepsize
        codeoptions.nlp.integrator.nodes = 5
        codeoptions.nlp.integrator.type = 'ERK4'
        codeoptions.solvemethod = 'SQP_NLP'
        # codeoptions.solvemethod = 'ADMM'
        codeoptions.sqp_nlp.rti = 1
        codeoptions.sqp_nlp.maxSQPit = 1
        codeoptions.sqp_nlp.reg_hessian = 5e-9
        codeoptions.nlp.hessian_approximation = 'gauss-newton'
        # codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.forcenonconvex = 1

        # try:
        #     with open('model.pickle', 'rb') as handle:
        #         saved_codeoptions = pickle.load(handle)
        # except Exception:
        #     saved_codeoptions = None

        generate_new_code = True
        if generate_new_code:
            # Generate FORCESPRO solver
            self.solver = model.generate_solver(codeoptions)
            # with open('model.pickle', 'wb') as handle:
            #     pickle.dump(codeoptions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Read already generated solver
            gympath = '/'.join(os.path.abspath(__file__).split('/')[:-3])
            self.solver = forcespro.nlp.Solver.from_directory(os.path.join(gympath, 'FORCES_NLP_solver'))
            pass

    def cartpole_order2jacobian_order(self, s: np.ndarray):
        # Jacobian does not match the state order, permutation is needed
        new_s = np.ndarray(4)
        new_s[0] = s[POSITION_IDX]
        new_s[1] = s[POSITIOND_IDX]
        new_s[2] = s[ANGLE_IDX]
        new_s[3] = s[ANGLED_IDX]
        return new_s

    def jacobian_order2cartpole_order(self, s: np.ndarray):
        # Jacobian does not match the state order, permutation is needed
        new_s = np.ndarray(6)
        new_s[POSITION_IDX] = s[0]
        new_s[POSITIOND_IDX] = s[1]
        new_s[2] = np.cos(new_s[0])
        new_s[3] = np.sin(new_s[0])
        new_s[ANGLE_IDX] = s[2]
        new_s[ANGLED_IDX] = s[3]
        return new_s

    def LSobjective(self, z, p):
        sqrt_weights = [np.sqrt(p) for p in [self.r] * self.nu + [self.q] * self.nx]
        return casadi.vertcat([sqrt_weights[i] * z[i] for i in range(len(sqrt_weights))])

    def linear_dynamics(self, s, u):
        # calculate dx/dt evaluating f(x,u) = A(x,u)*x + B(x,u)*u
        jacobian = self.jacobian(s, 0.0)  # linearize around u=0.0
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * self.action_high
        return A @ s + B @ u

    def step(self, s: np.ndarray, time=None):
        s = self.cartpole_order2jacobian_order(s).astype(np.float32)
        s = np.hstack((s, np.zeros((1,))))
        x0 = np.transpose(np.tile(s, (1, self.mpc_horizon)))
        problem = {"x0": x0}
        problem["all_parameters"] = np.ones((self.model.N, 1))
        output, exitflag, info = self.solver.solve(problem)
        u = output["x01"][0:self.nu]

        return u.astype(np.float32)

    def optimizer_reset(self):
        pass

