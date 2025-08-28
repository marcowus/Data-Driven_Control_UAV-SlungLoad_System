import numpy as np
from casadi import SX, Function, nlpsol, vertcat, reshape
from .backstepping_inner_loop import backstepping_inner_loop
from .hexacopter_model import hexacopter_model


def run_smpc(Ts: float = 0.001, time: float = 50.0, N: int = 20):
    ga = 9.81
    x_r = 0.2
    z_r = 0.02

    steps = int(time / Ts)
    q = np.zeros((8, steps))
    q[0, 0] = x_r
    q[3, 0] = 0.0
    q[6, 0] = 0.0001
    q[2, 0] = z_r

    qd = np.zeros((8, steps))
    qdd = np.zeros((8, steps))
    u = np.zeros((6, steps))
    q_r = np.zeros((8, steps))
    q_r[0, 0] = x_r
    q_r[2, 0] = z_r
    qd_r = np.zeros((8, steps))
    qdd_r = np.zeros((8, steps))

    e_a = np.zeros((3, steps))
    ed_a = np.zeros((3, steps))
    e_u = np.zeros((2, steps))
    ed_u = np.zeros((2, steps))

    e_a[:, 0] = q[0:3, 0] - q_r[0:3, 0]
    ed_a[:, 0] = qd[0:3, 0] - qd_r[0:3, 0]
    e_u[:, 0] = q[6:8, 0] - q_r[6:8, 0]
    ed_u[:, 0] = qd[6:8, 0] - qd_r[6:8, 0]

    lamda_a = np.diag([60, 60, 60])
    c_a = np.diag([9, 9, 9])
    lamda_u = np.array([[1, 0.1], [1, 0.1], [1, 0.1]])
    c_u = np.array([[15, 0], [0, 0]])

    s_a = np.zeros((3, steps))
    s_u = np.zeros((3, steps))
    s = np.zeros((3, steps))

    s_a[:, 0] = lamda_a @ (ed_a[:, 0] + c_a @ e_a[:, 0])
    s_u[:, 0] = lamda_u @ (ed_u[:, 0] + c_u @ e_u[:, 0])
    s[:, 0] = s_a[:, 0] + s_u[:, 0]

    gbar = 0.010
    X = np.zeros((6, steps))

    A = np.block([[np.diag([2, 2, 2]), np.diag([-1, -1, -1])],
                  [np.diag([1, 1, 1]), np.zeros((3, 3))]])
    B = np.vstack((np.diag([gbar * Ts, gbar * Ts, gbar * Ts]), np.zeros((3, 3))))

    Q = np.diag([25, 25, 25, 15, 15, 15])
    R = np.diag([1, 1, 1])

    s21, s22, s23 = SX.sym('s21'), SX.sym('s22'), SX.sym('s23')
    s11, s12, s13 = SX.sym('s11'), SX.sym('s12'), SX.sym('s13')
    states_sym = vertcat(s21, s22, s23, s11, s12, s13)
    n_states = states_sym.size1()
    delta_ux, delta_uy, delta_uz = SX.sym('delta_ux'), SX.sym('delta_uy'), SX.sym('delta_uz')
    delta_u_sym = vertcat(delta_ux, delta_uy, delta_uz)
    rhs = A @ states_sym + B @ delta_u_sym
    f = Function('f', [states_sym, delta_u_sym], [rhs])

    U_sym = SX.sym('U', 3, N)
    P_sym = SX.sym('P', 2 * n_states)
    Xk = P_sym[0:n_states]
    obj = 0
    for k in range(N):
        err = Xk - P_sym[n_states:2 * n_states]
        obj = obj + err.T @ Q @ err + U_sym[:, k].T @ R @ U_sym[:, k]
        Xk = f(Xk, U_sym[:, k])

    OPT_variables = reshape(U_sym, 3 * N, 1)
    nlp_prob = {'f': obj, 'x': OPT_variables, 'p': P_sym}
    opts = {'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

    args_lbx = -np.inf * np.ones(3 * N)
    args_ubx = np.inf * np.ones(3 * N)

    u0 = np.zeros((3, N))
    X[:, 0] = np.concatenate((s[:, 0], np.zeros_like(s[:, 0])))
    swing = np.zeros((2, steps))

    for i in range(steps - 1):
        x0 = X[:, i]
        xs = np.zeros(n_states)
        p = np.concatenate((x0, xs))
        sol = solver(x0=u0.reshape(-1, 1), lbx=args_lbx, ubx=args_ubx, p=p)
        delta_u_star = np.array(sol['x']).reshape(3, N)
        u0 = np.hstack((delta_u_star[:, 1:], delta_u_star[:, -1:]))
        delta_u = delta_u_star[:, 0]

        if i == 0:
            u[0:3, i] = delta_u
        else:
            u[0:3, i] = u[0:3, i - 1] + delta_u

        denom = np.sqrt(u[0, i] ** 2 + u[1, i] ** 2 + (u[2, i] + ga) ** 2)
        denom = np.maximum(denom, 1e-6)
        arg = (u[0, i] * np.sin(q[3, i]) - u[1, i] * np.cos(q[3, i])) / denom
        arg = np.clip(arg, -1.0, 1.0)
        phi_r = np.arcsin(arg)
        theta_den = u[2, i] + ga
        theta_den = theta_den if abs(theta_den) > 1e-6 else np.sign(theta_den) * 1e-6
        theta_r = np.arctan((u[0, i] * np.cos(q[3, i]) + u[1, i] * np.sin(q[3, i])) / theta_den)
        q_r[3:6, i] = np.array([0.0, theta_r, phi_r])

        delta_u2 = backstepping_inner_loop(q_r[:, i], qd_r[:, i], qdd_r[:, i], q[:, i], qd[:, i], qdd[:, i])
        if i == 0:
            u[3:6, i] = delta_u2
        else:
            u[3:6, i] = u[3:6, i - 1] + delta_u2

        qdd[:, i + 1], _ = hexacopter_model(q[:, i], qd[:, i], u[:, i])
        qd[:, i + 1] = qd[:, i] + qdd[:, i + 1] * Ts
        q[:, i + 1] = q[:, i] + qd[:, i + 1] * Ts

        q[3:6, i + 1] = (q[3:6, i + 1] + np.pi) % (2 * np.pi) - np.pi
        if q[6, i + 1] < 0:
            q[6, i + 1] = abs(q[6, i + 1])
            qd[6, i + 1] = -qd[6, i + 1]
            q[7, i + 1] = q[7, i + 1] + np.pi
        q[7, i + 1] = (q[7, i + 1] + np.pi) % (2 * np.pi) - np.pi
        swing[:, i + 1] = np.rad2deg(q[6:8, i + 1])

        q_r[0:3, i + 1] = np.array([x_r, i * 0.001, z_r])
        qd_r[0:3, i + 1] = np.array([0.0, 0.001, 0.0])
        qdd_r[0:3, i + 1] = np.array([0.0, 0.0, 0.0])

        e_a[:, i + 1] = q[0:3, i + 1] - q_r[0:3, i + 1]
        e_u[:, i + 1] = q[6:8, i + 1] - q_r[6:8, i + 1]
        ed_a[:, i + 1] = qd[0:3, i + 1] - qd_r[0:3, i + 1]
        ed_u[:, i + 1] = qd[6:8, i + 1] - qd_r[6:8, i + 1]

        s_a[:, i + 1] = lamda_a @ (ed_a[:, i + 1] + c_a @ e_a[:, i + 1])
        s_u[:, i + 1] = lamda_u @ (ed_u[:, i + 1] + c_u @ e_u[:, i + 1])
        s[:, i + 1] = s_a[:, i + 1] + s_u[:, i + 1]
        X[:, i + 1] = np.concatenate((s[:, i + 1], s[:, i]))

    return {'q': q, 'qd': qd, 'qdd': qdd, 'u': u, 'swing': swing}


if __name__ == '__main__':
    run_smpc(time=1.0)
    print("SMPC simulation completed")
