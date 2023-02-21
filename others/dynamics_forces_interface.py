import numpy as np
import casadi
"""
Forces requires a function of the dynamics in the form
dx/dt = f(x,u,p)
to derive equality constraints
"""

def import_cartpole():
    from CartPoleSimulation.CartPole.cartpole_model import _cartpole_ode
    from CartPoleSimulation.CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                                             ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)
    from CartPoleSimulation.CartPole.cartpole_jacobian import cartpole_jacobian
    return

def cartpole_linear_dynamics(s, u, p):
    # calculate dx/dt evaluating f(x,u) = A(x,u)*x + B(x,u)*u
    import_cartpole()
    action_high = 2.62
    jacobian = cartpole_jacobian(s, 0.0)  # linearize around u=0.0
    A = jacobian[:, :-1]
    B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * action_high
    return A @ s + B @ u

def cartpole_non_linear_dynamics(s, u, p: 0):
    import_cartpole()
    u_max = 2.62
    ca, sa, angleD, positionD = np.cos(s[0]), np.sin(s[0]), s[1], s[3]
    angleDD, positionDD = _cartpole_ode(ca, sa, angleD, positionD, u*u_max)
    sD = casadi.SX.sym('sD', 4, 1)
    sD[0] = angleD
    sD[1] = angleDD
    sD[2] = positionD
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

def acrobot_dynamics(s, u, p):
    from Environments.acrobot_batched import acrobot_batched
    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_POS_1
    lc2 = LINK_COM_POS_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    g = 9.8

    # theta1, theta2, dtheta1, dtheta2 = tuple(list(s))
    # theta1, theta2, dtheta1, dtheta2 = np.unstack(s, 4, 1)
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]

    a = u
    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * casadi.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * casadi.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * casadi.cos(theta1 + theta2 - casadi.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * casadi.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * casadi.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * casadi.cos(theta1 - casadi.pi / 2)
        + phi2
    )

    # the following line is consistent with the java implementation and the
    # book
    ddtheta2 = (
        a
        + d2 / d1 * phi1
        - m2 * l1 * lc2 * dtheta1**2 * casadi.sin(theta2)
        - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    sD = casadi.SX.sym('sD', 4, 1)
    sD[0] = dtheta1
    sD[1] = dtheta2
    sD[2] = ddtheta1
    sD[3] = ddtheta2

    return sD

def continuous_mountaincar(s,u,p):
    power = 0.0015
    force = u
    min_position = -1.2

    position = s[0]
    velocity = s[1]
    sD = casadi.SX.sym('sD', 2, 1)

    sD[0] = s[1] * casadi.logic_not(casadi.logic_and((position <= min_position), (velocity < 0)))
    sD[1] = force * power - 0.0025 * casadi.cos(3 * position)

    return sD

def vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # # constraints
    # u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

def vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # # constraints
    # u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

def f1tenth_dynamics(s, u, p):
    params = {
        'mu': 1.0489,  # friction coefficient  [-]
        'C_Sf': 4.718,  # cornering stiffness front [1/rad]
        'C_Sr': 5.4562,  # cornering stiffness rear [1/rad]
        'lf': 0.15875,  # distance from venter of gracity to front axle [m]
        'lr': 0.17145,  # distance from venter of gracity to rear axle [m]
        'h': 0.074,  # center of gravity height of toal mass [m]
        'm': 3.74,  # Total Mass of car [kg]
        'I': 0.04712,  # Moment of inertia for entire mass about z axis  [kgm^2]
        's_min': -0.4189,  # Min steering angle [rad]
        's_max': 0.4189,  # Max steering angle [rad]
        'sv_min': -3.2,  # Min steering velocity [rad/s]
        'sv_max': 3.2,  # Max steering velocity [rad/s]
        'v_switch': 7.319,  # switching velocity [m/s]
        'a_max': 9.51,  # Max acceleration [m/s^2]
        'v_min': -5.0,  # Min velocity [m/s]
        'v_max': 20.0,  # Max velocity [m/s]
        'width': 0.31,  # Width of car [m]
        'length': 0.58  # Length of car [m]
    }
    sD = vehicle_dynamics_st(
        s,
        np.array([u[0], u[1]]),
        params['mu'],
        params['C_Sf'],
        params['C_Sr'],
        params['lf'],
        params['lr'],
        params['h'],
        params['m'],
        params['I'],
        params['s_min'],
        params['s_max'],
        params['sv_min'],
        params['sv_max'],
        params['v_switch'],
        params['a_max'],
        params['v_min'],
        params['v_max'])
    return sD