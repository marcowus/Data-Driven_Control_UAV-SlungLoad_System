import numpy as np
from .backstepping_inner_loop import backstepping_inner_loop
from .hexacopter_model import hexacopter_model


def run_adaptive_control(Ts: float = 0.001, time: float = 50.0):
    ga = 9.81
    M = 0.4
    m = 0.03

    steps = int(time / Ts)
    q = np.zeros((8, steps))
    qd = np.zeros((8, steps))
    qdd = np.zeros((8, steps))
    swing = np.zeros((2, steps))

    q[:, 0] = np.array([0.2, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0001, 0.0])
    q[:, 1] = q[:, 0]

    tau = np.zeros((6, steps))
    delta_tau = np.zeros((3, steps))
    q_r = np.zeros((8, steps))
    q_r[:, 0] = np.array([0.2, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_r[:, 1] = q_r[:, 0]
    qd_r = np.zeros((8, steps))
    qdd_r = np.zeros((8, steps))

    gama_a = np.diag([42, 42, 42])
    gama_u = np.array([[0.1, 0], [0, 0]])
    ro_a = np.diag([48, 48, 48])
    ro_u = np.array([[0.4, 0.2], [0.4, 0.2], [0.4, 0.2]])
    sigma = 2 * np.eye(3)

    s = np.zeros((3, steps))
    sd = np.zeros((3, steps))
    K_t = np.zeros((3, steps))

    s[:, 0] = ro_a @ ((-qd_r[0:3, 0] + qd[0:3, 0]) + gama_a @ (-q_r[0:3, 0] + q[0:3, 0])) + \
              ro_u @ ((-qd_r[6:8, 0] + qd[6:8, 0]) + gama_u @ (-q_r[6:8, 0] + q[6:8, 0]))
    sd[:, 0] = ro_a @ ((-qdd_r[0:3, 0] + qdd[0:3, 0]) + gama_a @ (-qd_r[0:3, 0] + qd[0:3, 0])) + \
               ro_u @ ((-qdd_r[6:8, 0] + qdd[6:8, 0]) + gama_u @ (-qd_r[6:8, 0] + qd[6:8, 0]))

    delta_u = np.zeros(3)
    delta_u2 = np.zeros(3)
    K_bar = np.array([0.01, 0.01, 0.01])
    K = np.array([0.01, 0.01, 0.01])
    U_epsilon = np.ones(3)
    mu = np.array([0.001, 0.001, 0.001])
    K_dot = np.zeros(3)
    mu_ep = np.array([0.2, 0.2, 0.2])

    for k in range(1, steps - 1):
        b_hat = np.diag([250, 250, 250])

        q_r[0:3, k + 1] = np.array([0.2, k * 0.001, 0.02])
        qd_r[0:3, k + 1] = np.array([0.0, 0.001, 0.0])

        s[:, k] = ro_a @ ((-qd_r[0:3, k] + qd[0:3, k]) + gama_a @ (-q_r[0:3, k] + q[0:3, k])) + \
                  ro_u @ ((-qd_r[6:8, k] + qd[6:8, k]) + gama_u @ (-q_r[6:8, k] + q[6:8, k]))
        sd[:, k] = ro_a @ ((-qdd_r[0:3, k] + qdd[0:3, k]) + gama_a @ (-qd_r[0:3, k] + qd[0:3, k])) + \
                   ro_u @ ((-qdd_r[6:8, k] + qdd[6:8, k]) + gama_u @ (-qd_r[6:8, k] + qd[6:8, k]))

        U_epsilon = -K * np.sign(s[:, k])

        for i in range(3):
            if K[i] < mu[i]:
                K_dot[i] = mu[i]
            else:
                K_dot[i] = K_bar[i] * np.linalg.norm(s[i, k]) * np.sign(np.linalg.norm(s[i, k]) - mu_ep[i])
        K = K + K_dot * Ts
        K_t[:, k] = K

        delta_u = np.linalg.solve(b_hat, -sigma @ s[:, k] - sd[:, k - 1] + U_epsilon)
        delta_tau[:, k] = delta_u

        tau[0:3, k] = tau[0:3, k - 1] + delta_u

        phi_r = np.arcsin((tau[0, k] * np.sin(q[3, k]) - tau[1, k] * np.cos(q[3, k])) /
                            np.sqrt(tau[0, k] ** 2 + tau[1, k] ** 2 + (tau[2, k] + ga) ** 2))
        theta_r = np.arctan((tau[0, k] * np.cos(q[3, k]) + tau[1, k] * np.sin(q[3, k])) /
                              (tau[2, k] + ga))
        q_r[3:6, k] = np.array([0.0, theta_r, phi_r])

        delta_u2 = backstepping_inner_loop(q_r[:, k], qd_r[:, k], qdd_r[:, k], q[:, k], qd[:, k], qdd[:, k])
        tau[3:6, k] = tau[3:6, k - 1] + delta_u2

        qdd[:, k + 1], u1 = hexacopter_model(q[:, k], qd[:, k], tau[:, k])

        qd[:, k + 1] = qd[:, k] + qdd[:, k + 1] * Ts
        q[:, k + 1] = q[:, k] + qd[:, k + 1] * Ts

        q[3:6, k + 1] = (q[3:6, k + 1] + np.pi) % (2 * np.pi) - np.pi

        if q[6, k + 1] < 0:
            q[6, k + 1] = abs(q[6, k + 1])
            qd[6, k + 1] = -qd[6, k + 1]
            q[7, k + 1] = q[7, k + 1] + np.pi
        q[7, k + 1] = (q[7, k + 1] + np.pi) % (2 * np.pi) - np.pi

        swing[0, k + 1] = np.rad2deg(q[6, k + 1])
        swing[1, k + 1] = np.rad2deg(q[7, k + 1])

        a1 = (np.cos(q[5, k + 1]) * np.sin(q[4, k + 1]) * np.cos(q[3, k + 1]) +
              np.sin(q[5, k + 1]) * np.sin(q[3, k + 1])) / (m + M)
        a2 = (np.cos(q[5, k + 1]) * np.sin(q[4, k + 1]) * np.sin(q[3, k + 1]) -
              np.sin(q[5, k + 1]) * np.cos(q[3, k + 1])) / (m + M)
        a3 = np.cos(q[5, k + 1]) * np.cos(q[4, k + 1]) / (m + M)
        tau[0:3, k] = np.array([a1 * u1, a2 * u1, a3 * u1 - ga])

    return {
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "tau": tau,
        "swing": swing,
        "K_t": K_t,
    }


if __name__ == "__main__":
    run_adaptive_control(time=1.0)
    print("Adaptive controller simulation completed")
