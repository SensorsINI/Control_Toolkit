"""
This is a non-linear optimizer
Requires ForcesPro software in folder 'forces' in the working directory
Requires environment specific parameters in config_optimizers and environment specific dynamics in
others.dynamics_forces_interface.py
"""

from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary
from Control_Toolkit_ASF.Forces_interfaces.dynamics_forces_interface import casadi_to_numpy

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
from itertools import chain

sys.path.insert(0, os.path.abspath(os.path.join(".", "forces")))
from forces import forcespro
import forcespro.nlp
from forces import get_userid
import casadi
import pickle
from line_profiler_pycharm import profile
import Control_Toolkit_ASF.Forces_interfaces.dynamics_forces_interface
import Control_Toolkit_ASF.Forces_interfaces.cost_forces_interface
import Control_Toolkit_ASF.Forces_interfaces.initial_guess_forces_interface
import Control_Toolkit_ASF.Forces_interfaces.target_forces_interface


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
            mppi_reinitialization: bool,
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
        if environment_name == '': environment_name = self.cost_function.cost_function.controller.environment_name
        env_pars = environment_specific_parameters[environment_name]
        self.environment_name = environment_name

        # set attributes
        self.optimize_over = env_pars['optimize_over']
        self.is_angle = env_pars['is_angle']
        self.dt = env_pars['dt']
        self.q = env_pars['q']
        self.r = env_pars['r']
        self.initial_strategy = getattr(Control_Toolkit_ASF.Forces_interfaces.initial_guess_forces_interface,
                                        initial_guess)
        self.dynamics = getattr(Control_Toolkit_ASF.Forces_interfaces.dynamics_forces_interface, env_pars['dynamics'])
        # self.env_dynamics = getattr(Control_Toolkit_ASF.Forces_interfaces.dynamics_forces_interface,
        #                             environment_name + '_env')
        self.cost = getattr(Control_Toolkit_ASF.Forces_interfaces.cost_forces_interface,
                            env_pars['cost']) if 'cost' in env_pars.keys() else None
        self.target_function = getattr(Control_Toolkit_ASF.Forces_interfaces.target_forces_interface,
                                       env_pars['target'] if 'target' in env_pars.keys() else 'standard_target')
        self.generate_new_solver = generate_new_solver
        self.terminal_constraint_at_target = terminal_constraint_at_target
        self.terminal_set_width = terminal_set_width
        self.idx_terminal_set = env_pars['idx_terminal_set']
        self.state_max = env_pars['state_max']
        self.action_max = env_pars['action_max']
        self.action_high = np.array(self.action_max)
        self.nx = len(self.optimize_over)
        self.nu = len(self.action_max)
        self.nz = self.nx + self.nu
        self.previous_input = np.zeros((self.nu,))
        self.exitflag = -1

        self.control_limits = control_limits
        self.optimizer_logging = optimizer_logging,
        self.computation_library = computation_library,
        self.mppi_reinitialization = mppi_reinitialization
        self.mppi_optimizer = None  # only for reinitialization if the option is active
        self.mppi_x = None
        self.rollout_trajectories = np.zeros((1, mpc_horizon+1, 9))
        # check env and interface dynamics match
        # self.compare_dynamics(200)

        # for readability
        N = self.mpc_horizon
        nx = self.nx
        nu = self.nu
        xmax = np.array([s if s != 'inf' else np.inf for s in self.state_max])
        xmin = -xmax
        self.action_low = -self.action_high
        umin = self.action_low
        umax = self.action_high

        # global debug variables
        self.j = N
        self.open_loop_solution = dict()
        self.rsnorms = []
        self.res_eqs = []
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
        model.npar = env_pars['npar'] if 'npar' in env_pars.keys() else nu + nx  # number of runtime parameters
        model.xinitidx = range(nu, nu + nx)  # indexes affected by initial condition
        # model.xfinalidx = range(nu, nu + nx) if terminal_constraint_at_target else None
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
        # codeoptions.parallel = 1              #Makes it slower for some reason
        codeoptions.solvemethod = 'PDIP_NLP'
        # codeoptions.solvemethod = 'SQP_NLP'
        codeoptions.nlp.hessian_approximation = 'gauss-newton' if str(
            type(model.LSobjective)) == "<class 'function'>" else 'bfgs'
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
        TolStat = 1E-2  # inf norm tol.on stationarity
        TolEq = 1E-2  # tol. on equality constraints

        # Method specific parameters
        # PDIP
        codeoptions.nlp.TolStat = TolStat  # inf norm tol.on stationarity
        codeoptions.nlp.TolEq = TolEq  # tol. on equality constraints
        # codeoptions.nlp.TolIneq = 1E-3  # tol.on inequality constraints
        codeoptions.nlp.TolComp = 1E-3  # tol.on complementarity
        # codeoptions.mu0 = 10  # complementary slackness
        codeoptions.accuracy.eq = 1e-2  # infinity norm of residual for equalities
        codeoptions.warmstart = True
        # codeoptions.debug = True
        codeoptions.nlp.BarrStrat = 'monotone'
        codeoptions.nlp.linear_solver = 'symm_indefinite_legacy'

        # SQP
        codeoptions.sqp_nlp.TolStat = TolStat
        codeoptions.sqp_nlp.TolEq = TolEq
        # codeoptions.sqp_nlp.rti = 10
        codeoptions.sqp_nlp.maxqps = 100
        # codeoptions.sqp_nlp.maxSQPit = 100
        # codeoptions.sqp_nlp.reg_hessian = 5e-2
        codeoptions.sqp_nlp.qpinit = 1  # 0 for cold start, 1 for centered start
        codeoptions.sqp_nlp.qp_timeout = 0
        self.max_it = codeoptions.sqp_nlp.maxqps if codeoptions.solvemethod == 'SQP_NLP' else codeoptions.maxit

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
        # self.open_loop = True
        self.open_loop = False

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
            self.problem['reinitialize'] = False

    def rungekutta4(self, x, u, dt):
        k1 = self.model.continuous_dynamics(x, u, 0)
        k2 = self.model.continuous_dynamics(x + dt / 2 * k1, u, 0)
        k3 = self.model.continuous_dynamics(x + dt / 2 * k2, u, 0)
        k4 = self.model.continuous_dynamics(x + dt * k3, u, 0)
        new_x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        new_x = np.array([float(new_x[i]) for i in range(x.shape[0])])
        return new_x

    def int_to_dict_key(self, n):
        return 'x' + str(n + 1).zfill(1 if self.model.N<=9 else 2)

    def offset_angles(self, s, is_angle):
        f = lambda x: x + 2 * np.pi if x < 0 else x
        for i in is_angle:
            s[i] = f(s[i])
        return

    def add_state_to_guess(self, x0, target, control_strategy):
        # new_x = self.rungekutta4(x0[-self.nx:], u, self.dt)
        new_x = self.solver.dynamics(x0[-(self.nx + self.nu):], p=np.zeros((self.model.npar,)), stage=0)[0].squeeze()
        # u = control_strategy(new_x, target)
        # u = np.ones(self.nu, ) * u
        u = x0[-(self.nu + self.nx):-self.nx]
        x0 = np.hstack((x0, u, new_x))
        return x0

    def state_to_obs(self, s):
        obs = np.zeros((9,))
        obs[[5, 6, 8, 1, 2, 0, 7]] = s
        obs[3] = np.cos(s[4])
        obs[4] = np.sin(s[4])
        return obs

    def mppi_solution(self, s0):
        if self.mppi_optimizer is None:
            self.initiate_mppi_optimizer()
        obs = self.state_to_obs(s0)
        self.mppi_optimizer.step(obs, time=None)
        trajs = self.mppi_optimizer.rollout_trajectories
        U = self.mppi_optimizer.u_nom[0, :, :]
        x = np.ndarray((0,))
        # x0 = np.hstack((x0, U[0], s0))
        s = s0
        for i in range(0, self.model.N):
            x = np.hstack((x, U[i], s))
            u = U[i].numpy()
            s = s + self.dt * self.env_dynamics(s, u, None)
        pass
        return x

    def initial_trajectory_guess(self, s0, target, control_strategy):
        x0 = np.ndarray((0,))
        s = s0
        u = control_strategy(s, target)
        # u = np.ones((self.nu,))*u
        x0 = np.hstack((x0, u, s))

        for i in range(self.model.N - 1):
            x0 = self.add_state_to_guess(x0, target, control_strategy)

        return x0

    def initiate_mppi_optimizer(self):
        # If mppi_reinitialization==True use MPPI for initialization
        from Control_Toolkit.Optimizers.optimizer_mppi import optimizer_mppi
        import yaml
        config_optimizers = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml")),
                                      Loader=yaml.FullLoader)
        config_mppi = config_optimizers['mppi']
        self.mppi_optimizer: template_optimizer = optimizer_mppi(
            predictor=self.predictor,
            cost_function=self.cost_function,
            num_states=self.num_states,
            num_control_inputs=self.num_control_inputs,
            control_limits=self.control_limits,
            optimizer_logging=self.optimizer_logging[0],
            computation_library=self.computation_library[0],
            **config_mppi,
        )
        # Some optimizers require additional controller parameters (e.g. predictor_specification or dt) to be fully configured.
        # Do this here. If the optimizer does not require any additional parameters, it will ignore them.
        self.mppi_optimizer.configure(dt=self.dt, predictor_specification='ODE_TF')
        return

    @profile
    def step(self, s: np.ndarray, time=None):

        # Offset angles
        self.offset_angles(s, self.is_angle)

        # Select only the indipendent variables
        s = s[self.optimize_over].astype(np.float32)

        # Set parameters
        parameter_map = {'previous_input': self.previous_input, 'nz': self.nz}
        self.target = self.target_function(self.cost_function.cost_function.controller, parameter_map)
        self.problem['all_parameters'] = np.tile(self.target, (self.model.N, 1))

        if self.mppi_reinitialization:
            self.mppi_x = self.mppi_solution(s)

        # Build initial guess x0
        if True or 'x0' not in self.problem.keys() or self.j >= self.model.N - 1 or self.exitflag<0:
            x0 = self.initial_trajectory_guess(s, self.target, self.initial_strategy) if not self.mppi_reinitialization else self.mppi_x
        else:   #warm start
            x0 = np.hstack(tuple(self.open_loop_solution[key] for key in self.open_loop_solution))[
                 (self.j + 1) * self.nz:]
            for i in range(self.model.N - self.j - 1, self.model.N):
                x0 = self.add_state_to_guess(x0, self.target, self.initial_strategy)
        self.problem['x0'] = x0

        # Terminal set around target
        if self.terminal_set_width > 0:
            key = "{:02d}".format(self.model.N)
            self.problem['lb' + key] = self.problem['lb' + key].copy()
            self.problem['ub' + key] = self.problem['ub' + key].copy()
            self.problem['lb' + key][self.nu:][self.idx_terminal_set] = \
                self.target[self.nu:][self.idx_terminal_set] - self.terminal_set_width
            self.problem['ub' + key][self.nu:][self.idx_terminal_set] = \
                self.target[self.nu:][self.idx_terminal_set] + self.terminal_set_width

        self.problem['xinit'] = s
        # problem['xfinal'] = self.target[self.nu:] if self.terminal_constraint_at_target else None

        if not self.open_loop:
            # Solve
            self.initial_obj = self.test_initial_condition(self.problem)  # DEBUG
            output, exitflag, info = self.solver.solve(self.problem)
            self.rollout_trajectories = self.solution_to_envtrajectory_format(output)
            self.solution_obj = self.test_open_loop_solution(self.problem, output)  # DEBUG
            #
            # If reached max it set as failed
            if info.it == self.max_it:
                exitflag = -1

            self.exitflag = exitflag
            # solution_trajectory, env_trajectory, solver_trajectory, interface_trajectory = self.compare_open_loop_behaviour(
            #     self.problem, output)



            # If solver succeded use copy output
            if True or exitflag<0 : #and self.j == self.model.N - 1:
                # self.j = self.model.N - 1  # if the solver fails and we run out of input solutions use the initial guess
                self.j = 0
                self.open_loop_solution = self.trajectory_to_solution_format(x0)
            elif exitflag >= 0:
                self.j = 0
                self.open_loop_solution = output.copy()
            # else use previous output

            # Get input
            u = self.open_loop_solution[self.int_to_dict_key(self.j)][0:self.nu]

            # # Debug infos
            # self.rsnorms.append(info.rsnorm)
            # self.res_eqs.append(info.res_eq)
            # self.previous_exitflag = exitflag

        else:
            if self.j == self.model.N:
                # Solve
                # self.initial_obj = self.test_initial_condition(self.problem)  # DEBUG
                output, exitflag, info = self.solver.solve(self.problem)
                # self.solution_obj = self.test_open_loop_solution(self.problem, output)  # DEBUG
                # solution_trajectory, env_trajectory, solver_trajectory, interface_trajectory = self.compare_open_loop_behaviour(
                #     self.problem, output)
                if exitflag >= 0:
                    # if False:
                    self.open_loop_solution = output.copy()
                else:
                    # self.j = self.model.N - 1  # if the solver fails and we run out of input solutions use the initial guess
                    self.open_loop_solution = self.trajectory_to_solution_format(x0)

                # Retrieve jth element from the open loop solution
                self.j = 0
            u = self.open_loop_solution[self.int_to_dict_key(self.j)][0:self.nu]

            # DEBUG
            open_loop_prediction = self.open_loop_solution[self.int_to_dict_key(self.j)][self.nu:]
            self.open_loop_errors[self.j] = np.linalg.norm(open_loop_prediction - s, ord=np.inf)
            np.set_printoptions(suppress=True)
            print('\n\n' + 'Input: ' +'\t'*6 + np.array2string(u, precision=3, floatmode='fixed'))
            print('Open loop prediction: \t\t' + np.array2string(open_loop_prediction, precision=3,
                                                                 floatmode='fixed'))
            print('Actual state: ' +'\t'*4 + np.array2string(s, precision=3, floatmode='fixed'))
            print('Open loop max error: \t\t' + "{0:0.3f}".format(self.open_loop_errors[self.j, 0]))
            np.set_printoptions(suppress=False)

            self.j += 1

        # self.rollout_trajectories = self.solution_to_envtrajectory_format(self.open_loop_solution)

        # if self.mppi_reinitialization:
        #     self.rollout_trajectories = np.stack(
        #         (self.solution_to_envtrajectory_format(self.trajectory_to_solution_format(self.mppi_x)), self.rollout_trajectories),
        #         1).squeeze()
        self.optimal_trajectory = self.rollout_trajectories
        self.previous_input = u.astype(np.float32)
        return u.astype(np.float32)

    def trajectory_to_solution_format(self, arr):
        solution = {self.int_to_dict_key(i): arr[self.nz * i:self.nz * (i + 1)] for i in range(self.mpc_horizon)}
        return solution

    def solution_to_envtrajectory_format(self, d):
        envtrajectory = np.vstack(tuple(self.state_to_obs(d[self.int_to_dict_key(i)][self.nu:]) for i in chain(range(self.model.N), [self.model.N-1])))[np.newaxis,:]
        return envtrajectory

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
        # print('Did not encounter NaNs')
        return total_obj

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
        # print('Did not encounter NaNs')
        return total_obj

    def compare_open_loop_behaviour(self, problem, output):
        pars = problem['all_parameters']
        solution_trajectory = np.zeros((self.model.N, self.nx))
        env_trajectory = np.zeros((self.model.N, self.nx))
        solver_trajectory = np.zeros((self.model.N, self.nx))
        interface_trajectory = np.zeros((self.model.N, self.nx))

        for ss in range(self.model.N - 1):
            z = output[self.int_to_dict_key(ss)]
            s = z[self.nu:]
            u = z[:self.nu]
            p = pars[ss, :]
            c, jacc = self.solver.dynamics(z, p, stage=ss)
            solution_trajectory[ss, :] = output[self.int_to_dict_key(ss + 1)][self.nu:] if ss != self.model.N - 1 else s
            solver_trajectory[ss, :, np.newaxis] = c
            env_trajectory[ss, :] = s + self.dt * self.env_dynamics(s, u, p)
            interface_trajectory[ss, :] = s + self.dt * casadi_to_numpy(self.dynamics(s, u, p))
        return solution_trajectory, env_trajectory, solver_trajectory, interface_trajectory

    def compare_dynamics(self, M: int):
        for i in range(M):
            s = (np.random.random_sample((7,)) - 0.5) * 30
            u = (np.random.random_sample((2,)) - 0.5) * 10
            sD_env = self.env_dynamics(s, u, None)
            sD_forces = casadi_to_numpy(self.dynamics(s, u, None))
            error = np.linalg.norm(sD_env - sD_forces)
            # if error > 0.001: raise Exception('env and interface dynamics don''t match')
            pass
        return

    def optimizer_reset(self):
        pass
