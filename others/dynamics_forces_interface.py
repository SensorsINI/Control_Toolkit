from CartPole.cartpole_model import _cartpole_ode
from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                      ANGLED_IDX, POSITION_IDX, POSITIOND_IDX,
                                      cartpole_state_vector_to_jacobian_order)
from CartPoleSimulation.CartPole.cartpole_jacobian import cartpole_jacobian
import numpy as np

"""
Forces requires a function of the dynamics in the form
dx/dt = f(x,u)
to derive equality constraints
"""

def cartpole_linear_dynamics(s, u):
    # calculate dx/dt evaluating f(x,u) = A(x,u)*x + B(x,u)*u
    action_high = 2.62
    jacobian = cartpole_jacobian(s, 0.0)  # linearize around u=0.0
    A = jacobian[:, :-1]
    B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * action_high
    return A @ s + B @ u

def cartpole_non_linear_dynamics(s, u):
    ca, sa, angleD, positionD = np.cos(s[0]), np.sin(s[0]), s[1], s[3]
    angleDD, positionDD = _cartpole_ode(ca, sa, angleD, positionD, u)
    sD = np.ndarray(s[3], angleDD, s[1], positionDD)
    return sD
