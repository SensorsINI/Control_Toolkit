from CartPole.cartpole_model import _cartpole_ode
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX,
                                      cartpole_state_vector_to_jacobian_order)
from CartPoleSimulation.CartPole.cartpole_jacobian import cartpole_jacobian
import numpy as np
import casadi
"""
Forces requires a function of the dynamics in the form
dx/dt = f(x,u,p)
to derive equality constraints
"""

def cartpole_linear_dynamics(s, u, p):
    # calculate dx/dt evaluating f(x,u) = A(x,u)*x + B(x,u)*u
    action_high = 2.62
    jacobian = cartpole_jacobian(s, 0.0)  # linearize around u=0.0
    A = jacobian[:, :-1]
    B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * action_high
    return A @ s + B @ u

def cartpole_non_linear_dynamics(s, u, p: 0):
    ca, sa, angleD, positionD = np.cos(s[0]), np.sin(s[0]), s[1], s[3]
    angleDD, positionDD = _cartpole_ode(ca, sa, angleD, positionD, u)
    sD = casadi.SX.sym('sD', 4, 1)
    sD[0] = s[3]
    sD[1] = angleDD
    sD[2] = s[1]
    sD[3] = positionDD
    return sD

def pendulum_dynamics(s, u, p):
    # th, thD, sth, cth = s[0], s[1], s[2], s[3]
    g = 10.0
    l = 1.0
    m = 1.0
    sD = casadi.SX.sym('sD', 2, 1)
    sD[0] = s[1]
    sD[1] = 3 * g / (2 * l) * np.sin(s[0]) + 3.0 / (m * l ** 2) * u
    return sD
