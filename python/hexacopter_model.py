import numpy as np


def hexacopter_model(q: np.ndarray, q_dot: np.ndarray, tau: np.ndarray):
    """Compute accelerations of a UAV carrying a suspended load.

    Parameters
    ----------
    q : array-like, shape (8,)
        [x, y, z, psi, theta, phi, alpha, beta]
    q_dot : array-like, shape (8,)
        First derivative of q.
    tau : array-like, shape (6,)
        Control vector [tau_x, tau_y, tau_z, tau_psi, tau_theta, tau_phi].

    Returns
    -------
    qdd : ndarray, shape (8,)
        Second derivative of q.
    u1 : float
        Collective thrust.
    """
    M = 0.4
    m = 0.03
    g = 9.81
    d = 0.1
    l = 0.35
    I_p = 1.00e-6
    I_x = 1.77e-3
    I_y = 1.77e-3
    I_z = 3.54e-3

    q = np.asarray(q).reshape(8)
    q_dot = np.asarray(q_dot).reshape(8)
    tau = np.asarray(tau).reshape(6)

    u1 = (M + m) * np.sqrt(tau[0] ** 2 + tau[1] ** 2 + (tau[2] + g) ** 2)
    u = np.array([u1, tau[3], tau[4], tau[5]])

    x, y, z, psi, theta, phi, alpha, beta = q
    x_d, y_d, z_d, psi_d, theta_d, phi_d, alpha_d, beta_d = q_dot

    m11 = M + m
    m22 = m11
    m33 = m11
    m17 = m * l * np.cos(alpha) * np.cos(beta)
    m18 = -m * l * np.sin(alpha) * np.sin(beta)
    m27 = m * l * np.cos(alpha) * np.sin(beta)
    m28 = m * l * np.sin(alpha) * np.cos(beta)
    m37 = m * l * np.sin(alpha)
    m44 = I_x * np.sin(theta) ** 2 + np.cos(theta) ** 2 * (I_y * np.sin(phi) ** 2 + I_z * np.cos(phi) ** 2)
    m45 = (I_y - I_z) * (np.cos(theta) * np.sin(phi) * np.cos(phi))
    m55 = I_y * np.cos(phi) ** 2 + I_z * np.sin(phi) ** 2
    m77 = m * l ** 2 + I_p
    m88 = m * l ** 2 * np.sin(alpha) ** 2 + I_p

    M_q = np.array([
        [m11, 0, 0, 0, 0, 0, m17, m18],
        [0, m22, 0, 0, 0, 0, m27, m28],
        [0, 0, m33, 0, 0, 0, m37, 0],
        [0, 0, 0, m44, m45, -I_x * np.sin(theta), 0, 0],
        [0, 0, 0, m45, m55, 0, 0, 0],
        [0, 0, 0, -I_x * np.sin(theta), 0, I_x, 0, 0],
        [m17, m27, m37, 0, 0, 0, m77, 0],
        [m18, m28, 0, 0, 0, 0, 0, m88],
    ])

    c17 = -m * l * (np.cos(alpha) * np.sin(beta) * beta_d + np.sin(alpha) * np.cos(beta) * alpha_d)
    c18 = -m * l * (np.cos(alpha) * np.sin(beta) * alpha_d + np.sin(alpha) * np.cos(beta) * beta_d)
    c27 = m * l * (np.cos(alpha) * np.cos(beta) * beta_d - np.sin(alpha) * np.sin(beta) * alpha_d)
    c28 = m * l * (np.cos(alpha) * np.cos(beta) * alpha_d - np.sin(alpha) * np.sin(beta) * beta_d)
    c44 = (I_x * theta_d * np.sin(theta) * np.cos(theta)
           - (I_y + I_z) * (theta_d * np.sin(theta) * np.cos(theta) * np.sin(phi) ** 2)
           + (I_y - I_z) * phi_d * np.cos(theta) ** 2 * np.sin(phi) * np.cos(phi))
    c45 = (I_x * psi_d * np.sin(theta) * np.cos(theta)
           - (I_y - I_z) * (theta_d * np.sin(theta) * np.cos(phi) * np.sin(phi) + phi_d * np.cos(theta) * np.sin(phi) ** 2)
           - (I_y + I_z) * (psi_d * np.sin(theta) * np.cos(theta) * np.cos(phi) ** 2 - phi_d * np.cos(theta) * np.cos(phi) ** 2))
    c46 = -(I_x * theta_d * np.cos(theta) - (I_y - I_z) * (psi_d * np.cos(theta) ** 2 * np.sin(phi) * np.cos(phi)))
    c54 = psi_d * np.sin(theta) * np.cos(theta) * (-I_x + I_y * np.sin(phi) ** 2 + I_z * np.cos(phi) ** 2)
    c55 = -(I_y - I_z) * (phi_d * np.sin(phi) * np.cos(phi))
    c56 = (I_x * psi_d * np.cos(theta)
           + (I_y - I_z) * (-theta_d * np.sin(theta) * np.cos(phi) + psi_d * np.cos(theta) * (np.cos(phi) ** 2 - np.sin(phi) ** 2)))
    c64 = -(I_y - I_z) * (psi_d * np.cos(theta) ** 2 * np.sin(phi) * np.cos(phi))
    c65 = (-I_x * psi_d * np.cos(theta)
           + (I_y - I_z) * (theta_d * np.sin(phi) * np.cos(phi) + psi_d * np.cos(theta) * (np.sin(phi) ** 2 - np.cos(phi) ** 2)))

    C_q = np.array([
        [0, 0, 0, 0, 0, 0, c17, c18],
        [0, 0, 0, 0, 0, 0, c27, c28],
        [0, 0, 0, 0, 0, 0, m * l * np.cos(alpha) * alpha_d, 0],
        [0, 0, 0, c44, c45, c46, 0, 0],
        [0, 0, 0, c54, c55, c56, 0, 0],
        [0, 0, 0, c64, c65, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -m * l ** 2 * np.sin(alpha) * np.cos(alpha) * beta_d],
        [0, 0, 0, 0, 0, 0, m * l ** 2 * np.sin(alpha) * np.cos(alpha) * beta_d,
         m * l ** 2 * np.sin(alpha) * np.cos(alpha) * alpha_d],
    ])

    G_q = np.array([0, 0, (M + m) * g, 0, 0, 0, m * l * g * np.sin(alpha), 0])

    b = np.array([
        [np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta), 0, 0, 0],
        [np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi), 0, 0, 0],
        [np.cos(theta) * np.cos(phi), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    qdd = np.linalg.solve(M_q, -C_q.dot(q_dot) - G_q + b.dot(u))
    return qdd, float(u1)
