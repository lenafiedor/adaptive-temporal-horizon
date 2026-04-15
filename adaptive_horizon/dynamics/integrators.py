import numpy as np


def rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_coupled(x, Q, dt, f, jacobian):
    """
    One RK4 step for coupled system:
        dx/dt = f(x)
        dQ/dt = J(x) Q
    """
    k1_x = f(x)
    J1 = np.array(jacobian(*x))
    k1_Q = J1 @ Q

    x2 = x + 0.5 * dt * k1_x
    Q2 = Q + 0.5 * dt * k1_Q
    k2_x = f(x2)
    J2 = np.array(jacobian(*x2))
    k2_Q = J2 @ Q2

    x3 = x + 0.5 * dt * k2_x
    Q3 = Q + 0.5 * dt * k2_Q
    k3_x = f(x3)
    J3 = np.array(jacobian(*x3))
    k3_Q = J3 @ Q3

    x4 = x + dt * k3_x
    Q4 = Q + dt * k3_Q
    k4_x = f(x4)
    J4 = np.array(jacobian(*x4))
    k4_Q = J4 @ Q4

    x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    Q_next = Q + (dt / 6.0) * (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q)

    return x_next, Q_next
