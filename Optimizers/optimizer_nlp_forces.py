"""
This is a non-linear optimizer
Requires ForcesPro software in folder 'forces' in the working directory
Requires environment specific parameters in config_optimizers and environment specific dynamics in
others.dynamics_forces_interface.py
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
import Control_Toolkit.others.cost_forces_interface
import Control_Toolkit.others.initial_guess_forces_interface

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
            optimizer_logging: bool,
            seed: int,
            mpc_horizon: int,
            initial_guess: str,
            num_rollouts: int,
            environment_specific_parameters: dict
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
        # save cost function for debug purpose
        self.cost_function = cost_function

        # retrieve environment specific parameters of the optimizer
        environment_name = self.cost_function.cost_function.controller.environment_name
        environment_name = '_'.join(environment_name.split('_')[:-1])
        env_pars = environment_specific_parameters[environment_name]
        self.environment_name = environment_name

        # set attributes
        self.optimize_over = env_pars['optimize_over']
        self.is_angle = env_pars['is_angle']
        self.dt = env_pars['dt']
        self.q = env_pars['q']
        self.r = env_pars['r']
        self.initial_strategy = getattr(Control_Toolkit.others.initial_guess_forces_interface, initial_guess)
        self.dynamics = getattr(Control_Toolkit.others.dynamics_forces_interface, env_pars['dynamics'])
        self.cost = getattr(Control_Toolkit.others.cost_forces_interface, env_pars['cost']) if env_pars[
                                                                                             'cost'] != None else None
        self.state_max = env_pars['state_max']
        self.action_max = env_pars['action_max']
        self.action_high = np.array(self.action_max)
        self.nx = len(self.optimize_over)
        self.nu = len(self.action_max)

        # for readability
        N = self.mpc_horizon
        nx = self.nx
        nu = self.nu
        xmax = np.array([s if s != 'inf' else np.inf for s in self.state_max])
        xmin = -xmax
        umin = self.action_low
        umax = self.action_high

        # global debug variables
        self.j = 0
        self.open_loop_solution = dict()
        self.rsnorms = []
        self.res_eqs = []
        self.action_low = -self.action_high
        self.open_loop_errors = np.zeros((N, 1))

        # reset optimizer
        self.optimizer_reset()

        # Model Definition
        # ----------------

        # Problem dimensions
        model = forcespro.nlp.SymbolicModel(N)
        model.nvar = nu + nx  # number of variables
        model.neq = nx  # number of equality constraints
        model.nh = 0  # number of inequality constraint functions
        model.npar = nu + nx  # number of runtime parameters
        model.xinitidx = range(nu, nu + nx)  # indexes affected by initial condition
        self.model = model

        # Cost function
        # More info at https://forces.embotech.com/Documentation/solver_options/index.html?highlight=lsobjective#gauss-newton-options
        self.target = np.zeros((nu + nx,))
        if self.cost != None:
            model.objective = self.cost
            model.objectiveN = lambda z, p: model.objective(z, p)
        else:
            sqrt_weights = [np.sqrt(p) for p in self.r + self.q]
            model.LSobjective = lambda z, p: np.array(sqrt_weights) * (z - p)
            model.LSobjectiveN = lambda z, p: model.LSobjective(z, p)

        # Dynamics for equality costraints
        model.continuous_dynamics = self.dynamics  # continuous_dynamics : (s, u) --> ds/dx
        # Inequality constraints
        # upper/lower variable bounds lb <= z <= ub
        model.lb = np.concatenate((umin, xmin), 0)
        model.ub = np.concatenate((umax, xmax), 0)

        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        model.E = np.concatenate([np.zeros((nx, nu)), np.identity(nx)], axis=1)

        # Generate solver
        # ---------------

        # Solver options
        # Documented at: https://forces.embotech.com/Documentation/solver_options/index.html?highlight=erk2#high-level-interface-options
        codeoptions = forcespro.CodeOptions()
        codeoptions.maxit = 1000  # Maximum number of iterations
        codeoptions.printlevel = 2  # Use printlevel = 2 to print progress (but not for timings)
        codeoptions.optlevel = 2  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
        codeoptions.solvemethod = 'PDIP_NLP'
        # codeoptions.solvemethod = 'SQP_NLP'
        codeoptions.nlp.hessian_approximation = 'gauss-newton' if str(type(model.LSobjective)) == "<class 'function'>" else 'bfgs'
        codeoptions.nlp.hessian_approximation = 'bfgs' # Works with both LSobjective and objective
        codeoptions.forcenonconvex = 1
        # codeoptions.floattype = 'float'
        # codeoptions.threadSafeStorage = True;
        codeoptions.overwrite = 1

        # Integration
        codeoptions.nlp.integrator.Ts = self.dt
        codeoptions.nlp.integrator.nodes = 1
        # codeoptions.nlp.integrator.type = 'ERK2'
        codeoptions.nlp.integrator.type = 'ForwardEuler'

        # Tolerances
        codeoptions.nlp.TolStat = 1E-1  # inf norm tol.on stationarity
        codeoptions.nlp.TolEq = 5E-2  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1E-3  # tol.on inequality constraints
        codeoptions.nlp.TolComp = 1E-3  # tol.on complementarity
        codeoptions.mu0 = 10  # complementary slackness
        codeoptions.accuracy.eq = 1e-2  # infinity norm of residual for equalities

        # Method specific parameters, override generic ones
        # codeoptions.ADMMrho = 6
        # codeoptions.ADMMfactorize = 1
        codeoptions.sqp_nlp.rti = 10
        codeoptions.sqp_nlp.maxSQPit = 100
        codeoptions.sqp_nlp.reg_hessian = 5e-2
        codeoptions.sqp_nlp.qpinit = 1  # 0 for cold start, 1 for centered start

        generate_new_code = False
        if generate_new_code:
            # Generate ForcesPRO solver
            self.solver = model.generate_solver(codeoptions)
        else:
            # Read already generated solver
            gympath = '/'.join(os.path.abspath(__file__).split('/')[:-3])
            self.solver = forcespro.nlp.Solver.from_directory(os.path.join(gympath, 'FORCES_NLP_solver'))
            pass

        # Open loop is useful for debug purposes
        self.open_loop = False

    def rungekutta4(self, x, u, dt):
        k1 = self.model.continuous_dynamics(x, u, 0)
        k2 = self.model.continuous_dynamics(x + dt / 2 * k1, u, 0)
        k3 = self.model.continuous_dynamics(x + dt / 2 * k2, u, 0)
        k4 = self.model.continuous_dynamics(x + dt * k3, u, 0)
        new_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        new_x = np.array([float(new_x[i]) for i in range(x.shape[0])])
        return new_x

    def int_to_dict_key(self, n):
        return 'x' + str(n + 1).zfill(2)

    def offset_angles(self, s, is_angle):
        f = lambda x: x + 2 * np.pi if x < 0 else x
        for i in is_angle:
            s[i] = f(s[i])
        return

    def initial_trajectory_guess(self, s0, target, initial_control_plan):
        x0 = np.ndarray((0,))
        s = s0
        u = initial_control_plan(s, target)
        x0 = np.hstack((x0, u, s))

        for i in range(self.model.N-1):
            # new_x = self.rungekutta4(x0[-self.nx:], u, self.dt)
            new_x = self.solver.dynamics(x0[-(self.nx+self.nu):], p=np.zeros((self.model.npar,)), stage=0)[0].squeeze()
            u = initial_control_plan(new_x, target)
            x0 = np.hstack((x0, u, new_x))

        return x0

    def step(self, s: np.ndarray, time=None):

        # Offset angles
        self.offset_angles(s, self.is_angle)

        # Select only the indipendent variables
        s = s[self.optimize_over].astype(np.float32)


        # Define the problem
        try:
            self.target[3] = self.cost_function.cost_function.controller.target_position.numpy()    #Cartpole
        except AttributeError:
            pass

        # Build initial guess x0
        x0 = self.initial_trajectory_guess(s, self.target, self.initial_strategy)
        problem = {"x0": x0}

        # problem["all_parameters"] = np.ones((self.model.N, self.model.npar))
        problem["all_parameters"] = np.tile(self.target, (self.model.N, 1))
        problem["xinit"] = s

        if not self.open_loop or self.j == 0:
            # Solve
            self.initial_obj = self.test_initial_condition(problem)  # DEBUG
            output, exitflag, info = self.solver.solve(problem)
            self.solution_obj = self.test_open_loop_solution(problem, output)  # DEBUG

            # If solver failed use previous output
            if exitflag >= 0 or self.open_loop_solution == {}:
                self.j = 0
                self.open_loop_solution = output.copy()
            else:
                self.j += 1

            # Get input
            u = self.open_loop_solution[self.int_to_dict_key(self.j)][0:self.nu]

            # Debug infos
            self.rsnorms.append(info.rsnorm)
            self.res_eqs.append(info.res_eq)
            self.previous_exitflag = exitflag

            if self.open_loop:
                self.open_loop_solution = output.copy()
                self.j += 1
        else:
            # Retrieve jth element from the open loop solution
            u = self.open_loop_solution[self.int_to_dict_key(self.j)][0:self.nu]

            # DEBUG
            self.open_loop_errors[self.j] = np.linalg.norm(
                self.open_loop_solution[self.int_to_dict_key(self.j)][1:] - s)
            print('\n\n' + 'Open loop prediction: ' + str(self.open_loop_solution[self.int_to_dict_key(self.j)][1:]))
            print('Open loop error: ' + str(self.open_loop_errors[self.j,0]))

            self.j += 1
            if self.j == self.model.N:
                self.j = 0

        return u.astype(np.float32)

    def test_initial_condition(self, problem):
        x0 = problem['x0']
        pars = problem['all_parameters']
        total_obj = 0
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = x0[ss * self.model.nvar:(ss + 1) * self.model.nvar]
            p = pars[ss, :]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            # self.cost_function.get_stage_cost(z[self.nu:self.nu+self.nx].astype(np.float32), z[0:self.nu].astype(np.float32),
            #                                   z[0:self.nu].astype(np.float32))
            assert not (np.any(np.isnan(c)) or np.any(np.isinf(c))), 'Encountered NaN in c at stage ' + str(ss)
            assert not (np.any(np.isnan(np.sum(jacc))) or np.any(
                np.isinf(np.sum(jacc)))), 'Encountered NaN in jacc at stage ' + str(ss)
            assert not (np.any(np.isnan(ineq)) or np.any(np.isinf(ineq))), 'Encountered NaN in ineq at stage ' + str(ss)
            assert not (np.any(np.isnan(jacineq)) or np.any(
                np.isinf(jacineq))), 'Encountered NaN in jacineq at stage ' + str(ss)
            assert not (np.any(np.isnan(obj)) or np.any(np.isinf(obj))), 'Encountered NaN in obj at stage ' + str(ss)
            assert not (np.any(np.isnan(gradobj)) or np.any(
                np.isinf(gradobj))), 'Encountered NaN in gradobj at stage ' + str(ss)
            total_obj += obj
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj
        print('Did not encounter NaNs')

    def test_open_loop_solution(self, problem, output):
        pars = problem['all_parameters']
        total_obj = 0
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = output[self.int_to_dict_key(ss)]
            p = pars[ss, :]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            total_obj += obj
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj
        print('Did not encounter NaNs')

    def optimizer_reset(self):
        pass
