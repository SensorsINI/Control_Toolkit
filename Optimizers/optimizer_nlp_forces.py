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
        model.continuous_dynamics = self.dynamics       #continuous_dynamics : (s, u) --> ds/dx

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
        codeoptions.maxit = 200                                 # Maximum number of iterations
        codeoptions.printlevel = 2                              # Use printlevel = 2 to print progress (but not for timings)
        codeoptions.optlevel = 3                                # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        codeoptions.nlp.integrator.Ts = integrator_stepsize
        codeoptions.nlp.integrator.nodes = 5
        codeoptions.nlp.integrator.type = 'ERK4'
        codeoptions.solvemethod = 'SQP_NLP'
        # codeoptions.solvemethod = 'PDIP'
        # codeoptions.solvemethod = 'ADMM'

        codeoptions.sqp_nlp.rti = 1
        codeoptions.sqp_nlp.maxSQPit = 1
        codeoptions.sqp_nlp.reg_hessian = 5e-9
        codeoptions.nlp.hessian_approximation = 'gauss-newton'
        # codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.forcenonconvex = 1
        # codeoptions.floattype = 'float'
        # codeoptions.threadSafeStorage = True;
        codeoptions.overwrite = 1
        # codeoptions.nlp.TolStat = 1E-3                          # inf norm tol.on stationarity
        # codeoptions.nlp.TolEq = 1E-3                            # tol. on equality constraints
        # codeoptions.nlp.TolIneq = 1E-3                          # tol.on inequality constraints
        # codeoptions.nlp.TolComp = 1E-3                          # tol.on complementarity


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



    def step(self, s: np.ndarray, time=None):
        s = s[self.optimize_over].astype(np.float32)
        u0 = 0.1
        s = np.hstack((np.ones((1,))*u0, s))                              # add initial guess for input 0
        x0 = np.transpose(np.tile(s, (1, self.mpc_horizon)))
        problem = {"x0": x0}
        problem["all_parameters"] = np.ones((self.model.N, 1))*0.7
        # problem["xinit"] = s
        self.test_solver(problem)
        output, exitflag, info = self.solver.solve(problem)
        u = output["x01"][0:self.nu]
        # sD = self.model.continuous_dynamics(s, u)
        self.rsnorms.append(info.rsnorm)
        self.res_eqs.append(info.res_eq)
        return u.astype(np.float32)

    def test_solver(self, problem):
        x0 = problem['x0']
        pars = problem['all_parameters']
        for ss in range(self.model.N-1):
            z = x0[ss*self.model.nvar:(ss+1)*self.model.nvar]
            p = pars[ss]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            assert not (np.any(np.isnan(c)) or np.any(np.isinf(c))), 'Encountered NaN in c at stage ' + str(ss)
            assert not (np.any(np.isnan(np.sum(jacc))) or np.any(np.isinf(np.sum(jacc)))), 'Encountered NaN in jacc at stage ' + str(ss)
            assert not (np.any(np.isnan(ineq)) or np.any(np.isinf(ineq))), 'Encountered NaN in ineq at stage ' + str(ss)
            assert not (np.any(np.isnan(jacineq)) or np.any(np.isinf(jacineq))), 'Encountered NaN in jacineq at stage ' + str(ss)
            assert not (np.any(np.isnan(obj)) or np.any(np.isinf(obj))), 'Encountered NaN in obj at stage ' + str(ss)
            assert not (np.any(np.isnan(gradobj)) or np.any(np.isinf(gradobj))), 'Encountered NaN in gradobj at stage ' + str(ss)
        print('Did not encounter NaNs')

    def optimizer_reset(self):
        pass

