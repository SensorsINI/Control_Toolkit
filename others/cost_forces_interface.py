import casadi
import numpy as np
import yaml
import os
"""
Forces requires a function of the cost in the form
objective = f(z,p)
to derive equality constraints
"""

config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)


def pendulum(z, p):
    return -casadi.cos(z[1]) + 0.05*z[0]**2

def pendulum_square_norm(z, p):
    return z[1]**2

def continuous_mountaincar(z, p):
    # z[1] = casadi.fmax(z[1], -0.5)
    z[1] = casadi.if_else(casadi.le(z[1],-0.5), -0.5, z[1])
    return -casadi.sin(3 * z[1]) #/ ((z[1] - 1.25) ** 2)

def continuous_mountaincar_approximated(z, p):
    return -1.27*(z[1] + 0.4)**3 - 1.56 * (z[1] + 0.4)**2 + 0.00326758 * (z[1] + 0.4) + 0.322505

def cartpole_weights():
    cartpole_angle_weight = float(config["cartpole_simulator_batched"]["default"]["angle_weight"])
    cartpole_position_weight = float(config["cartpole_simulator_batched"]["default"]["position_weight"])
    return cartpole_angle_weight, cartpole_position_weight

def cartpole_simulator1(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return -cartpole_angle_weight*casadi.cos(z[1]) + cartpole_position_weight*(z[3] - p[3])**2

def cartpole_simulator2(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return -cartpole_angle_weight*(z[1]**2) + cartpole_position_weight*(z[3] - p[3])**2

def obstacles():
    # seed = 49607
    # target = (-1, 1, 0)
    obst = np.array([[-0.63589, -0.12645, 0.72638, 0.09140],
        [0.09512, 0.86974, -0.14193, 0.14091],
        [0.32933, 0.01162, -0.11440, 0.08833],
        [-0.67929, 0.19865, 0.14162, 0.28421],
        [-0.28223, 0.82019, 0.68124, 0.10455],
        [0.62852, -0.54213, -0.13166, 0.18236],
        [-0.60734, 0.31083, -0.26043, 0.12806],
        [-0.12491, -0.43055, 0.50640, 0.27649],
        [-0.44553, 0.33577, 0.54104, 0.12525],
        [0.31719, 0.08227, 0.09898, 0.23555],
        [0.48858, -0.58945, -0.89945, 0.20899],
        [-0.80885, 0.59042, -0.60242, 0.19431],
        [0.23766, 0.70330, -0.81050, 0.09609],
        [-0.57289, 0.66760, -0.11860, 0.27896],
        [0.68795, 0.31120, 0.77682, 0.29616],
        [-0.63207, 0.82611, 0.13359, 0.16158]])
    return obst


def obstacle_avoidance(z, p):
    # p = [target] + [o_x, o_y, o_z, o_r] * n_obstacles
    target = p[0:3]
    # n_obstacles = int((p.shape[0]-3)/4)
    cost = 0.0
    obstacle_cost = 0.0
    obst = np.ndarray.flatten(obstacles())
    for i in range(0, obst.shape[0], 4):
        radius = obst[i+3]
        # d = casadi.norm_2(z[3:6] - obst[i:i+3])
        d = casadi.sumsqr(z[3:6] - obst[i:i + 3])
        if radius> 0.21:
            obstacle_cost += 1.0 - (casadi.fmin(1.0, 1.0*d / radius))
        # obstacle_cost += casadi.le(casadi.sumsqr(obst[i:i+3] - z[3:6]), radius)
    # target_cost = casadi.norm_2(z[3:6] - target)
    target_cost = casadi.sumsqr(target - z[3:6])
    # target_cost = casadi.sumsqr(z[3:6] - -0.9)
    # target_cost = (target[0] - z[3])**2 + (target[1] - z[4])**2 + (target[2] - z[5])**2
    target_cost = (-0.9 - z[3]) ** 2 + (0.9 - z[4]) ** 2 + (0.0 - z[5]) ** 2

    goal_reward = -casadi.le(casadi.sumsqr(target - z[3:6]), 0.1)
    cost = target_cost + obstacle_cost
    # cost = goal_reward + 0.1*target_cost
    # cost = goal_reward #+ obstacle_cost
    return cost

