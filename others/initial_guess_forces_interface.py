def PD(x, x_f):
    P_gain = 1.0
    D_gain = 0.5
    u = P_gain*(x_f[0] - x[0]) + D_gain*(x_f[1] - x[1])
    return 0.2*u

def PD_3d(x, x_f):
    P_gain = 10.0
    D_gain = 0.5
    u = P_gain*(x_f - x[0:3]) #+ D_gain*(x_f[1] - x[1])
    return u

def no_action(x, x_f):
    return 0.0