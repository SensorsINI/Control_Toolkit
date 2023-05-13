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
    return -casadi.cos(z[1]) + 0.05 * z[0] ** 2


def pendulum_square_norm(z, p):
    return z[1] ** 2


def continuous_mountaincar(z, p):
    # z[1] = casadi.fmax(z[1], -0.5)
    z[1] = casadi.if_else(casadi.le(z[1], -0.5), -0.5, z[1])
    return -casadi.sin(3 * z[1])  # / ((z[1] - 1.25) ** 2)


def continuous_mountaincar_approximated(z, p):
    return -1.27 * (z[1] + 0.4) ** 3 - 1.56 * (z[1] + 0.4) ** 2 + 0.00326758 * (z[1] + 0.4) + 0.322505


def cartpole_weights():
    cartpole_angle_weight = float(config["cartpole_simulator_batched"]["default"]["angle_weight"])
    cartpole_position_weight = float(config["cartpole_simulator_batched"]["default"]["position_weight"])
    return cartpole_angle_weight, cartpole_position_weight


def cartpole_simulator1(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return -casadi.cos(z[1]) + -0.1 * casadi.cos(
        z[2])  # + -0.01*casadi.cos(z[4])#+ cartpole_position_weight*(z[3] - p[3])**2


def cartpole_simulator2(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return casadi.fabs(z[1])  # + cartpole_position_weight*(z[3] - p[3])**2


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
        radius = obst[i + 3]
        # d = casadi.norm_2(z[3:6] - obst[i:i+3])
        d = casadi.sumsqr(z[3:6] - obst[i:i + 3])
        if radius > 0.21:
            obstacle_cost += 1.0 - (casadi.fmin(1.0, 1.0 * d / radius))
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


def dubins_obstacles():
    return [[0.31868297, 0.7749577, 0.28699964], [-0.49458456, -0.11240232, 0.27588597],
            [0.19568837, -0.5972406, 0.082922354], [0.06757867, -0.42966557, 0.2968905],
            [0.6409783, 0.36124045, 0.29200518], [-0.52833635, 0.17657721, 0.19466914],
            [0.36637014, -0.600214, 0.20858128], [-0.040501416, -0.16123867, 0.10538994],
            [0.39486748, -0.2967993, 0.23954614], [-0.47237647, 0.27628905, 0.13882904],
            [-0.49906766, 0.5697846, 0.1728635], [-0.61574525, 0.73863643, 0.14459442],
            [0.41435724, 0.4362622, 0.05848365]]


def dubins_car(z, p):
    x, y, yaw_car, steering_rate = (z[i] for i in range(2, 6))
    # target = np.array([])
    x_target, y_target, yaw_target = 0.9, 0.0, 0.0
    target = casadi.SX((x_target, y_target, yaw_target))

    # head_to_target = dubins_car_batched.get_heading(self.lib, states, self.lib.unsqueeze(target, 0))
    # alpha = head_to_target - yaw_car
    # ld = dubins_car_batched.get_distance(self.lib, states, self.lib.unsqueeze(target, 0))
    # crossTrackError = self.lib.sin(alpha) * ld

    # car_in_bounds = dubins_car_batched._car_in_bounds(self.lib, x, y)
    # car_at_target = dubins_car_batched._car_at_target(self.lib, x, y, x_target, y_target)

    obstacles = np.array(dubins_obstacles())
    obstacles_cost = 0.0

    for i in range(0, obstacles.shape[0]):
        radius = obstacles[i, 2]
        # d = casadi.norm_2(z[3:6] - obst[i:i+3])
        d = casadi.norm_2(z[2:4] - obstacles[i, :-1])
        if radius > 0.20:
            obstacles_cost = casadi.fmax(obstacles_cost, 1.0 - (casadi.fmin(1.0, 1.0 * d / radius))**2)

    cost = (
        # self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * (-10.0)
        # + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32) * (
            0.125 * (
        # 3 * crossTrackError**2
            0.0
            # + casadi.fabs(x_target - x)
            # + casadi.fabs(y_target - y)
            + 0.1 * (x_target - x) ** 2
            + 0.1 * (y_target - y) ** 2
            # - casadi.cos(casadi.atan((y_target-y)/(x_target-x)))
            # + 3 * (head_to_target - yaw_car)**2 / MAX_STEER
            + 5.0 * obstacles_cost
    )
        # )
        # + self.lib.cast(~car_in_bounds, self.lib.float32)
    )
    # cost = casadi.if_else(casadi.le(casadi.norm_2(z[2:4]-target[0:2]), 0.01), 0.0 , cost)

    return cost - 1.0
