"""pointmass equations of motion"""


def simulate(x, u, dt=0.1):
    # print "x, u, dt", x, u, dt
    v = x[1] + (u[0]) * dt
    p = x[0] + x[1] * dt

    x[0] = p
    x[1] = v

    return x
