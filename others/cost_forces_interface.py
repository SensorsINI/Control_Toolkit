import casadi
"""
Forces requires a function of the cost in the form
objective = f(z,p)
to derive equality constraints
"""

def continuous_mountaincar(z, p):
    return -casadi.sin(3 * z[1]) / ((z[1] - 1.25) ** 2)

def continuous_mountaincar_approximated(z, p):
    return -1.27*(z[1] + 0.4)**3 - 1.56 * (z[1] + 0.4)**2 + 0.00326758 * (z[1] + 0.4) + 0.322505

def cartpole_simulator(z, p):
    return -casadi.cos(z[1])