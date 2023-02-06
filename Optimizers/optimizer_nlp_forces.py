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


# Forces
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(".", "forces")))
from forces import forcespro
import forcespro.nlp
import numpy as np
from forces import get_userid
import casadi
import os
import pickle
import Control_Toolkit.others.dynamics_forces_interface

class optimizer_nlp_forces(template_optimizer):
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
            dynamics: str,
            action_max: list[float],
            state_max: list[float],
            optimize_over: list[int],
            q: list[float],
            r: list[float]
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

        # dynamically import dynamics of the model
        self.dynamics = getattr(Control_Toolkit.others.dynamics_forces_interface, dynamics)

        self.optimize_over = optimize_over

        self.action_low = -np.array(action_max)
        self.action_high = -self.action_low
        self.q = q
        self.r = r
        self.optimizer_reset()

        self.rsnorms = []
        self.res_eqs = []

        # lower and upper bounds
        # constrained_idx = [i for i, x in enumerate(state_max) if x != 'inf']
        #
        # xmax = np.array([state_max[i] for i in constrained_idx])
        xmax = np.array([s if s != 'inf' else np.inf for s in state_max])
        xmin = -xmax

        umin = self.action_low
        umax = self.action_high

        # ubidx = [1] + [i + 2 for i in constrained_idx]
        # lbidx = ubidx

        self.nx = len(optimize_over)
        self.nu = len(action_max)

        # # Cost matrices for LQR controller
        # self.Q = np.diag([self.P] * self.nx).astype(np.float32)  # How much to punish x
        # self.R = np.diag([self.R] * self.nu).astype(np.float32)  # How much to punish u

        # for readability
        N = self.mpc_horizon

        nx = self.nx
        nu = self.nu

        self.j = 0              #Global counter for debug
        self.open_loop_solution = dict() #Global variable for debug

        # Model Definition
        # ----------------

        # Problem dimensions
        self.model = forcespro.nlp.SymbolicModel(N)  # horizon length
        model = self.model
        model.nvar = nu + nx  # number of variables
        model.neq = nx  # number of equality constraints
        model.nh = 0  # number of inequality constraint functions
        model.npar = nu + nx  # number of runtime parameters


        sqrt_weights = [np.sqrt(p) for p in r + q]

        # model.objective = lambda z, p: (sqrt_weights*z).T @ (sqrt_weights*z)

        # model.LSobjective = lambda z, p: np.array(sqrt_weights) * casadi.fmin(2*np.pi -z,z)

        self.target = np.zeros((nu + nx,))
        model.LSobjective = lambda z, p: np.array(sqrt_weights) * (z - p)
        model.continuous_dynamics = self.dynamics       #continuous_dynamics : (s, u) --> ds/dx

        # We use an explicit RK4 integrator here to discretize continuous dynamics

        self.integrator_stepsize = 0.02

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
        codeoptions.maxit = 40                                  # Maximum number of iterations
        codeoptions.printlevel = 1                              # Use printlevel = 2 to print progress (but not for timings)
        codeoptions.optlevel = 2                                # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        codeoptions.nlp.integrator.Ts = self.integrator_stepsize
        codeoptions.nlp.integrator.nodes = 1
        # codeoptions.nlp.integrator.type = 'ERK'
        codeoptions.nlp.integrator.type = 'ForwardEuler'
        # codeoptions.solvemethod = 'SQP_NLP'
        codeoptions.solvemethod = 'PDIP_NLP'
        # codeoptions.solvemethod = 'ADMM'

        codeoptions.ADMMrho = 6
        codeoptions.ADMMfactorize = 1
        codeoptions.sqp_nlp.rti = 10
        codeoptions.sqp_nlp.maxSQPit = 100
        codeoptions.sqp_nlp.reg_hessian = 5e-2
        codeoptions.sqp_nlp.qpinit = 0                             # 0 for cold start, 1 for centered start

        codeoptions.nlp.hessian_approximation = 'gauss-newton'
        # codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.forcenonconvex = 1
        # codeoptions.floattype = 'float'
        # codeoptions.threadSafeStorage = True;
        codeoptions.overwrite = 1
        codeoptions.nlp.TolStat = 1E-1                          # inf norm tol.on stationarity
        codeoptions.nlp.TolEq = 1E-1                            # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1E-3                          # tol.on inequality constraints
        codeoptions.nlp.TolComp = 1E-3                          # tol.on complementarity
        codeoptions.mu0 = 10                                    #complementary slackness
        codeoptions.accuracy.eq = 1e-2  # infinity norm of residual for equalities

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

    def rungekutta4(self, x, u, dt):
        k1 = self.model.continuous_dynamics(x, u, 0)
        k2 = self.model.continuous_dynamics(x + dt / 2 * k1, u, 0)
        k3 = self.model.continuous_dynamics(x + dt / 2 * k2, u, 0)
        k4 = self.model.continuous_dynamics(x + dt * k3, u, 0)
        new_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        new_x = np.array([float(new_x[i]) for i in range(x.shape[0])])
        return new_x

    def step(self, s: np.ndarray, time=None):
        s = s[self.optimize_over].astype(np.float32)
        nx = len(s)
        nu = self.nu
        u0 = 0.0
        x0 = np.hstack((np.ones((1,))*u0, s))                              # add initial guess for input 0

        dt = self.integrator_stepsize
        for i in range(self.model.N-1):
            new_x = self.rungekutta4(x0[-nx:], u0, dt)
            x0 = np.hstack((x0, u0, new_x))



        # x0 = np.transpose(np.tile(s, (1, self.mpc_horizon)))
        problem = {"x0": x0}
        self.target[3] = self.cost_function.cost_function.controller.target_position.numpy()
        # problem["all_parameters"] = np.ones((self.model.N, self.model.npar))
        problem["all_parameters"] = np.tile(self.target, (self.model.N, 1))
        problem["xinit"] = s

        initial_obj = self.test_initial_condition(problem)
        output, exitflag, info = self.solver.solve(problem)
        solution_obj = self.test_open_loop_solution(problem, output)

        open_loop = True
        if open_loop:
            if self.j == 0:
                self.open_loop_solution = output.copy()
            # u = x0[self.j * (nx + nu):self.j * (nx + nu) + nu]
            u = self.open_loop_solution[self.int_to_dict_key(self.j)][0:self.nu]
            self.j += 1
            if self.j == self.model.N - 1:
                self.j = 0

        else:
            u = output["x01"][0:self.nu]
        # sD = self.model.continuous_dynamics(s, u)
        self.rsnorms.append(info.rsnorm)
        self.res_eqs.append(info.res_eq)
        self.previous_exitflag = exitflag
        return u.astype(np.float32)

    def int_to_dict_key(self, n):
        return 'x' + str(n+1).zfill(2)

    def test_initial_condition(self, problem):
        x0 = problem['x0']
        pars = problem['all_parameters']
        total_obj = 0
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = x0[ss * self.model.nvar:(ss + 1) * self.model.nvar]
            p = pars[ss,:]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            # self.cost_function.get_stage_cost(z[self.nu:self.nu+self.nx].astype(np.float32), z[0:self.nu].astype(np.float32),
            #                                   z[0:self.nu].astype(np.float32))
            assert not (np.any(np.isnan(c)) or np.any(np.isinf(c))), 'Encountered NaN in c at stage ' + str(ss)
            assert not (np.any(np.isnan(np.sum(jacc))) or np.any(np.isinf(np.sum(jacc)))), 'Encountered NaN in jacc at stage ' + str(ss)
            assert not (np.any(np.isnan(ineq)) or np.any(np.isinf(ineq))), 'Encountered NaN in ineq at stage ' + str(ss)
            assert not (np.any(np.isnan(jacineq)) or np.any(np.isinf(jacineq))), 'Encountered NaN in jacineq at stage ' + str(ss)
            assert not (np.any(np.isnan(obj)) or np.any(np.isinf(obj))), 'Encountered NaN in obj at stage ' + str(ss)
            assert not (np.any(np.isnan(gradobj)) or np.any(np.isinf(gradobj))), 'Encountered NaN in gradobj at stage ' + str(ss)
            total_obj += obj
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj
        print('Did not encounter NaNs')

    def test_open_loop_solution(self, problem, output):
        pars = problem['all_parameters']
        total_obj = 0
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = output[("x0" if ss + 1 < 10 else "x") + str(ss + 1)]
            p = pars[ss,:]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            total_obj += obj
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj
        print('Did not encounter NaNs')

    def optimizer_reset(self):
        pass

