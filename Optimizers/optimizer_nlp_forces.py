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
import matplotlib.pyplot as plt
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
            generate_new_solver: bool,
            terminal_constraint_at_target: bool,
            terminal_set_width: float,
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
                                                                                             'cost'] is not None else None
        self.generate_new_solver = generate_new_solver
        self.terminal_constraint_at_target = terminal_constraint_at_target
        self.terminal_set_width = terminal_set_width
        self.idx_terminal_set = env_pars['idx_terminal_set']
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
        # self.action_low = -self.action_high
        self.open_loop_errors = np.zeros((N, 1))
        self.suboptimizer = None
        self.environment = None     #Bad practice, only for debug
        self.step_counter = 0
        self.logs = np.zeros((4,0))

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
        if self.cost_function.cost_function.controller.environment_name == 'obstacle_avoidance_batched':
            N_obst = self.cost_function.cost_function.controller.obstacle_positions.shape[0]
            model.npar = 3 + 4*N_obst
        model.xinitidx = range(nu, nu + nx)  # indexes affected by initial condition
        # model.xfinalidx = range(nu, nu + nx) if terminal_constraint_at_target else None
        self.model = model

        # Cost function
        # More info at https://forces.embotech.com/Documentation/solver_options/index.html?highlight=lsobjective#gauss-newton-options
        self.target = np.zeros((nu + nx,))
        if self.cost is not None:
            model.objective = self.cost
            model.objectiveN = lambda z, p: model.objective(z, p)
        else:
            sqrt_weights = [np.sqrt(p) for p in self.r + self.q]
            model.LSobjective = lambda z, p: np.array(sqrt_weights) * (z - p)
            model.LSobjectiveN = lambda z, p: model.LSobjective(z, p)

        # Dynamics for equality costraints
        model.continuous_dynamics = self.dynamics  # continuous_dynamics : (s, u) --> ds/dx
        # Inequality constraints
        # # upper/lower variable bounds lb <= z <= ub
        lb = np.concatenate((umin, xmin), 0)
        ub = np.concatenate((umax, xmax), 0)
        if terminal_set_width <= 0:
            model.lb, model.ub = lb, ub
        else:
            model.lbidx = list(range(model.nvar))
            model.ubidx = list(range(model.nvar))
            # self.lb_tiled = np.tile(lb, (self.model.N,))
            # self.ub_tiled = np.tile(ub, (self.model.N,))

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
        codeoptions.parallel = 1
        codeoptions.solvemethod = 'PDIP_NLP'
        # codeoptions.solvemethod = 'SQP_NLP'
        codeoptions.nlp.hessian_approximation = 'gauss-newton' if str(type(model.LSobjective)) == "<class 'function'>" else 'bfgs'
        # codeoptions.nlp.hessian_approximation = 'bfgs' # Works with both LSobjective and objective
        codeoptions.forcenonconvex = 1
        # codeoptions.floattype = 'float'
        # codeoptions.threadSafeStorage = True;
        codeoptions.overwrite = 1
        # if terminal_set_width > 0:
        #     codeoptions.nlp.stack_parambounds = True

        # Integration
        codeoptions.nlp.integrator.Ts = self.dt
        codeoptions.nlp.integrator.nodes = 1
        # codeoptions.nlp.integrator.type = 'ERK2'
        codeoptions.nlp.integrator.type = 'ForwardEuler'

        # Tolerances
        codeoptions.nlp.TolStat = 1E-2  # inf norm tol.on stationarity
        codeoptions.nlp.TolEq = 1E-3  # tol. on equality constraints
        codeoptions.nlp.TolIneq = 1E-3  # tol.on inequality constraints
        codeoptions.nlp.TolComp = 1E-3  # tol.on complementarity
        # codeoptions.mu0 = 10  # complementary slackness
        codeoptions.accuracy.eq = 1e-2  # infinity norm of residual for equalities

        # Method specific parameters, override generic ones
        # codeoptions.ADMMrho = 6
        # codeoptions.ADMMfactorize = 1
        # codeoptions.sqp_nlp.rti = 10
        codeoptions.sqp_nlp.maxqps = 200
        # codeoptions.sqp_nlp.maxSQPit = 100
        # codeoptions.sqp_nlp.reg_hessian = 5e-2
        # codeoptions.sqp_nlp.qpinit = 1  # 0 for cold start, 1 for centered start
        codeoptions.sqp_nlp.qp_timeout = 0
        # Save codeoptions
        self.codeoptions = codeoptions

        if self.generate_new_solver:
            # Generate ForcesPRO solver
            self.solver = model.generate_solver(codeoptions)
        else:
            # Read already generated solver
            gympath = '/'.join(os.path.abspath(__file__).split('/')[:-3])
            self.solver = forcespro.nlp.Solver.from_directory(os.path.join(gympath, 'FORCES_NLP_solver'))
            pass

        # Open loop is useful for debug purposes
        self.open_loop = False
        # self.open_loop = True

        # Specify fixed parameters of the problem
        self.problem = {}

        if terminal_set_width > 0:
            # Lower and upper bounds
            self.problem['lb01'] = lb[:nu]
            self.problem['ub01'] = ub[:nu]
            for i in range(1, N):
                key = "{:02d}".format(i + 1)
                self.problem['lb' + key] = lb
                self.problem['ub' + key] = ub

        # Override SQP initial guess at every step
        if self.codeoptions.solvemethod == 'SQP_NLP':
            self.problem['reinitialize'] = True

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
        # f = lambda x: x + 2 * np.pi if x < 0 else x
        f = lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi
        for i in is_angle:
            s[i] = f(s[i])
        return

    def initial_trajectory_guess(self, s0, target, control_strategy):
        x0 = np.ndarray((0,))
        s = s0
        u = control_strategy(s, target)
        u = np.ones(self.nu) * u
        x0 = np.hstack((x0, u, s))

        for i in range(self.model.N-1):
            # new_x = self.rungekutta4(x0[-self.nx:], u, self.dt)
            new_x = self.solver.dynamics(x0[-(self.nx+self.nu):], p=np.zeros((self.model.npar,)), stage=0)[0].squeeze()
            u = control_strategy(new_x, target)
            u = np.ones(self.nu)*u
            x0 = np.hstack((x0, u, new_x))

        return x0

    def trajectory_from_suboptimizer(self, s0):
        x0 = np.ndarray((0,))
        s = s0
        u = self.suboptimizer.best_input_trajectory[0,:].numpy()
        x0 = np.hstack((x0, u, s))

        for i in range(self.model.N - 1):
            # new_x = self.rungekutta4(x0[-self.nx:], u, self.dt)
            new_x = self.solver.dynamics(x0[-(self.nx + self.nu):], p=np.zeros((self.model.npar,)), stage=0)[
                0].squeeze()
            # new_x = self.environment.step_dynamics(s[np.newaxis, :], u[np.newaxis,:], dt=self.dt)
            u = self.suboptimizer.best_input_trajectory[0, :].numpy()
            x0 = np.hstack((x0, u, new_x))

        return x0

    def compare_dynamics(self):
        for _ in range(100):
            s = np.array([np.random.uniform(-self.state_max[i], self.state_max[i]) for i in range(self.nx)])
            u = np.array([np.random.uniform(self.action_low[i], self.action_high[i]) for i in range(self.nu)])
            next_s_env = self.environment.step_dynamics(s[np.newaxis, :], u[np.newaxis,:], dt=self.dt)
            next_s_solver = self.solver.dynamics(np.hstack((u, s)), p=np.zeros((self.model.npar,)), stage=0)[
                0]
            err = np.linalg.norm(next_s_env[0,:].numpy()-next_s_solver[:,0])
            pass

    def solution_to_rollout(self, output):
        rollout = np.vstack((output[k][:] for k in output))
        return rollout

    def step(self, s: np.ndarray, time=None):

        # self.compare_dynamics()
        u = self.suboptimizer.step(s)
        rpgd_solution = self.trajectory_from_suboptimizer(s)

        # Offset angles
        self.offset_angles(s, self.is_angle)

        # Select only the indipendent variables
        s = s[self.optimize_over].astype(np.float32)


        # Define the problem
        #@TODO Refactor
        try:
            self.target[3] = self.cost_function.cost_function.controller.target_position.numpy()    #Cartpole

        except AttributeError:
            pass

        p = self.target
        if self.cost_function.cost_function.controller.environment_name == 'obstacle_avoidance_batched':
            self.target = self.cost_function.cost_function.controller.target_point.numpy()
            obst = self.cost_function.cost_function.controller.obstacle_positions
            obst = np.ndarray.flatten(obst.numpy())
            p = np.hstack((self.target, obst))

        # if self.cost_function.cost_function.controller.environment_name == 'dubinscar_batched':
        #     self.cost_function.cost_function.controller.target_point = np.array([0.9, 0.0, 0.0])

        # Build initial guess x0
        x0 = self.initial_trajectory_guess(s, self.target, self.initial_strategy)
        # x0 = rpgd_solution
        self.problem["x0"] = x0


        # Terminal set around target
        if self.terminal_set_width > 0:
            key = "{:02d}".format(self.model.N)
            self.problem['lb' + key] = self.problem['lb' + key].copy()
            self.problem['ub' + key] = self.problem['ub' + key].copy()
            self.problem['lb' + key][self.nu:][self.idx_terminal_set] = \
                self.target[self.nu:][self.idx_terminal_set] - self.terminal_set_width
            self.problem['ub' + key][self.nu:][self.idx_terminal_set] = \
                self.target[self.nu:][self.idx_terminal_set] + self.terminal_set_width

        # problem["all_parameters"] = np.ones((self.model.N, self.model.npar))
        self.problem['all_parameters'] = np.tile(p, (self.model.N, 1))
        self.problem['xinit'] = s
        # problem['xfinal'] = self.target[self.nu:] if self.terminal_constraint_at_target else None

        if not self.open_loop or self.j == 0:
            # Solve
            self.initial_obj, self.initial_obj_env = self.test_initial_condition(self.problem, self.problem['x0'])
            output, exitflag, info = self.solver.solve(self.problem)
            self.solution_obj, self.solution_obj_env = self.test_open_loop_solution(self.problem, output)  # DEBUG

            # Solve 2
            self.problem["x0"] = rpgd_solution
            self.rpgd_obj, self.rpgd_obj_env =  self.test_initial_condition(self.problem, rpgd_solution)# DEBUG
            output2, exitflag2, info2 = self.solver.solve(self.problem)
            self.solution2_obj, self.solution2_obj_env = self.test_open_loop_solution(self.problem, output2)  # DEBUG



            self.logs = np.hstack((self.logs, np.array([self.initial_obj,
                                                                    self.solution_obj,
                                                                    self.solution2_obj,
                                                                    self.rpgd_obj])[:,np.newaxis]))

            # Plot
            L = 20
            if self.step_counter % L == (L - 1):
                # pass
                # fig = plt.figure(2)
                # times = np.arange(0, self.step_counter + 1) * self.dt
                # # for i in range(self.logs.shape[0]):
                # #     plt.plot(times, self.logs[i, :])
                # plt.plot(times, self.logs.transpose())
                # plt.xlabel('t')
                # plt.ylabel('objectives')
                # plt.yscale('log')
                # plt.title('Plot of a Numpy Vector')
                # plt.show()
                # fig.savefig('plot.pdf', dpi=fig.dpi)
                with open('data.npy', 'wb') as f:
                    np.save(f, self.logs)
                pass
            self.step_counter += 1

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
                self.open_loop_solution[self.int_to_dict_key(self.j)][self.nu:] - s)
            print('\n\n' + 'Open loop prediction: ' + str(self.open_loop_solution[self.int_to_dict_key(self.j)][self.nu:]))
            print('Open loop error: ' + str(self.open_loop_errors[self.j,0]))

            self.j += 1
            if self.j == self.model.N:
                self.j = 0

        return u.astype(np.float32)


    def test_initial_condition(self, problem, x0):
        # x0 = problem['x0']
        pars = problem['all_parameters']
        total_obj = 0.
        total_obj_env = 0.0
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = x0[ss * self.model.nvar:(ss + 1) * self.model.nvar]
            p = pars[ss, :]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            obj_env = self.cost_function.get_stage_cost(np.float32(z[self.nu:])[np.newaxis, np.newaxis],
                                              np.float32(z[:self.nu])[np.newaxis, np.newaxis], None).numpy()[0,0]
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
            total_obj_env += obj_env
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj, total_obj_env
        print('Did not encounter NaNs')

    def test_open_loop_solution(self, problem, output):
        pars = problem['all_parameters']
        total_obj = 0.
        total_obj_env = 0.
        initial_trajectory = np.zeros((self.model.N, self.model.neq))
        for ss in range(self.model.N - 1):
            z = output[self.int_to_dict_key(ss)]
            p = pars[ss, :]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            ineq, jacineq = self.solver.ineq(z, p, stage=ss)
            obj, gradobj = self.solver.objective(z, p, stage=ss)
            obj_env = self.cost_function.get_stage_cost(np.float32(z[self.nu:])[np.newaxis, np.newaxis],
                                                        np.float32(z[:self.nu])[np.newaxis, np.newaxis], None).numpy()[
                0, 0]
            total_obj += obj
            total_obj_env += obj_env
            initial_trajectory[ss, :, np.newaxis] = c
        return total_obj, total_obj_env
        print('Did not encounter NaNs')

    def optimizer_reset(self):
        pass
