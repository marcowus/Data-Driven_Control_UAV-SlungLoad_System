import numpy as np


def backstepping_inner_loop(q_r, q_r_dot, q_r_ddot, q, q_dot, q_ddot):
    """Backstepping inner-loop controller for UAV attitude.

    Parameters are 8x1 vectors for states and derivatives. Returns the
    incremental torque command for roll, pitch and yaw."""
    k1 = np.diag([258, 258, 258])
    k2 = np.diag([222, 222, 222])
    g_bar = np.diag([950, 950, 950])

    q_r = np.asarray(q_r)
    q_r_dot = np.asarray(q_r_dot)
    q_r_ddot = np.asarray(q_r_ddot)
    q = np.asarray(q)
    q_dot = np.asarray(q_dot)
    q_ddot = np.asarray(q_ddot)

    e1 = q_r[3:6] - q[3:6]
    e2 = q_r_dot[3:6] + k1.dot(e1) - q_dot[3:6]

    delta_u = np.linalg.solve(g_bar, e1 + k2.dot(e2) - q_ddot[3:6] + q_r_ddot[3:6] + k1.dot(q_r_dot[3:6] - q_dot[3:6]))
    return delta_u
