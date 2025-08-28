import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from .backstepping_inner_loop import backstepping_inner_loop
from .hexacopter_model import hexacopter_model

# ============================================================
# Van der Pol plant dynamics for demonstration
# ============================================================

def plant_dynamics_vdp(t, x, u, theta_true, disturbance_amp=0.1):
    """Second-order Van der Pol oscillator with control and disturbance."""
    x1, x2 = x
    phi = (1 - x1**2) * x2
    disturbance = disturbance_amp * np.sin(np.pi * t)
    dx = np.zeros_like(x)
    dx[0] = x2
    dx[1] = -x1 + theta_true * phi + u + disturbance
    return dx


# ============================================================
# Data generator tailored for the Van der Pol example
# ============================================================
class DataGenerator:
    def __init__(self, system_params):
        self.params = system_params
        self.dt = system_params['dt']

    def generate_excitation(self, t, phase=0):
        return (
            5 * np.sin(0.5 * t + phase)
            + 3 * np.sin(2.5 * t)
            + 2 * (t % 4 < 2) - 1.0
            + 0.5 * np.random.randn()
        )

    def generate_data(self, num_trajectories=20, steps_per_traj=200):
        X, Y, U = [], [], []
        for i in range(num_trajectories):
            x = np.random.uniform(-2, 2, 2)
            phase = np.random.uniform(0, 2 * np.pi)
            for k in range(steps_per_traj):
                t = k * self.dt
                u = np.clip(self.generate_excitation(t, phase), -20, 20)
                dx = plant_dynamics_vdp(t, x, u, self.params['theta_true'])
                x_next = x + dx * self.dt
                if np.any(np.abs(x_next) > 50):
                    break
                X.append(x.copy())
                Y.append(x_next.copy())
                U.append(u)
                x = x_next
        return np.array(X), np.array(Y), np.array(U)


# ============================================================
# Simple Koopman regressor using lifted observables
# ============================================================
class EnhancedKoopman(BaseEstimator):
    def __init__(self, n_obs=10, lambda_reg=0.1):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A_aug = None

    def lift(self, x):
        x1, x2 = x
        return np.array([
            x1, x2,
            x1**2, x2**2,
            x1 * x2,
            np.sin(x1), np.cos(x1),
            np.sin(x2), np.cos(x2),
            x1**3, x2**3,
        ])[: self.n_obs]

    def fit(self, X, Y, U):
        Psi_X = np.array([self.lift(x) for x in X])
        Psi_Y = np.array([self.lift(y) for y in Y])
        Phi = np.hstack([Psi_X, U.reshape(-1, 1)])
        reg = self.lambda_reg * np.eye(Phi.shape[1])
        self.A_aug = np.linalg.lstsq(Phi.T @ Phi + reg, Phi.T @ Psi_Y, rcond=None)[0].T
        return self

    def predict_dot(self, x, u, dt):
        if self.A_aug is None:
            return np.zeros_like(x)
        psi = self.lift(x)
        phi_aug = np.hstack([psi, [u]])
        psi_next = self.A_aug @ phi_aug
        x_next = psi_next[:2]
        return (x_next - x) / dt


# ============================================================
# Prescribed-time backstepping controller with Koopman compensation
# ============================================================
class ImprovedBacksteppingController:
    def __init__(self, Tp=5.0, sigma=(8.0, 8.0), gamma=0.8):
        self.Tp = Tp
        self.sigma = sigma
        self.gamma = gamma
        self.eps = 1e-3
        self.theta_hat = 0.0
        self.koopman = None
        self.alpha = 0.95
        self.K_linear = np.array([10.0, 5.0])

        # Gains for translational outer-loop used with the hexacopter model
        self.Kp_pos = np.diag([4.0, 4.0, 4.0])
        self.Kd_pos = np.diag([2.0, 2.0, 2.0])

    def bind_koopman(self, koopman_model):
        if not hasattr(koopman_model, "lift"):
            raise ValueError("Invalid Koopman model provided")
        self.koopman = koopman_model

    def compute_vdp_control(self, x, t, dt):
        if t >= self.alpha * self.Tp:
            return self.linear_control(x)
        return self.time_varying_control(x, t, dt)

    def compute_control(self, q, q_dot, q_ddot, q_r, q_r_dot, q_r_ddot):
        """Return full 6-D control vector for ``hexacopter_model``.

        Parameters
        ----------
        q, q_dot, q_ddot : array_like, shape (8,)
            Current state, its first and second derivative.
        q_r, q_r_dot, q_r_ddot : array_like, shape (8,)
            Reference state and derivatives.

        Returns
        -------
        tau : ndarray, shape (6,)
            [tau_x, tau_y, tau_z, tau_psi, tau_theta, tau_phi]
        """

        # Outer-loop for translational dynamics (tau_x, tau_y, tau_z)
        e_pos = q_r[0:3] - q[0:3]
        e_vel = q_r_dot[0:3] - q_dot[0:3]
        thrust_cmd = (
            q_r_ddot[0:3]
            + self.Kd_pos.dot(e_vel)
            + self.Kp_pos.dot(e_pos)
        )

        # Inner-loop for attitude using backstepping_inner_loop (tau_psi, tau_theta, tau_phi)
        torque_cmd = backstepping_inner_loop(q_r, q_r_dot, q_r_ddot, q, q_dot, q_ddot)

        return np.concatenate([thrust_cmd, torque_cmd])

    def time_varying_control(self, x, t, dt):
        x1, x2 = x
        Tp_t = max(self.Tp - t, self.eps)
        z1 = x1
        alpha1 = -self.sigma[0] / Tp_t * z1
        d_alpha1_dt = (self.sigma[0] / Tp_t**2) * z1 - (self.sigma[0] / Tp_t) * x2
        z2 = x2 - alpha1
        phi_val = (1 - x1**2) * x2
        u = -z1 - (self.sigma[1] / Tp_t) * z2 + x1 - self.theta_hat * phi_val + d_alpha1_dt
        theta_dot = self.gamma * z2 * phi_val
        self.theta_hat += theta_dot * dt
        self.theta_hat = np.clip(self.theta_hat, -5.0, 5.0)
        return np.clip(u, -50, 50)

    def linear_control(self, x):
        u = -self.K_linear @ x
        return np.clip(u, -10, 10)


def hexacopter_step(controller, q, q_dot, q_ddot, q_r, q_r_dot, q_r_ddot, dt):
    """Propagate one step of the hexacopter dynamics.

    This utility demonstrates how the controller output is combined with
    the thrust commands and fed to :func:`hexacopter_model`.
    """

    tau = controller.compute_control(q, q_dot, q_ddot, q_r, q_r_dot, q_r_ddot)
    qdd, u1 = hexacopter_model(q, q_dot, tau)
    q_dot_next = q_dot + qdd * dt
    q_next = q + q_dot_next * dt
    return q_next, q_dot_next, tau, u1


# ============================================================
# Simulation harness
# ============================================================

def run_simulation(system_params, initial_state, koopman_model, simulation_time=10.0):
    controller = ImprovedBacksteppingController(Tp=5.0)
    controller.bind_koopman(koopman_model)
    t = np.arange(0, simulation_time, system_params['dt'])
    x = initial_state.copy()
    log = {
        'time': t,
        'states': np.zeros((len(t), 2)),
        'control': np.zeros(len(t)),
        'theta_hat': np.zeros(len(t)),
    }
    for i, current_time in enumerate(t):
        u = controller.compute_vdp_control(x, current_time, system_params['dt'])
        dx = plant_dynamics_vdp(current_time, x, u, system_params['theta_true'])
        x += dx * system_params['dt']
        log['states'][i] = x
        log['control'][i] = u
        log['theta_hat'][i] = controller.theta_hat
    return log


# ============================================================
# Entry point
# ============================================================

def main():
    system_params = {'theta_true': 1.5, 'dt': 0.01}
    dg = DataGenerator(system_params)
    X_train, Y_train, U_train = dg.generate_data(num_trajectories=20, steps_per_traj=200)
    koopman_model = EnhancedKoopman(n_obs=10, lambda_reg=0.1)
    koopman_model.fit(X_train, Y_train, U_train)
    np.random.seed(42)
    initial_states = [np.random.uniform(-2, 2, 2) for _ in range(3)]
    logs = [run_simulation(system_params, s, koopman_model) for s in initial_states]
    colors = plt.cm.viridis(np.linspace(0, 1, len(logs)))
    plt.figure(figsize=(10, 6))
    for i, log in enumerate(logs):
        plt.plot(log['time'], log['states'][:, 0], color=colors[i], alpha=0.8)
    plt.axvline(5.0, color='r', linestyle='--', label='Tp=5s')
    plt.ylabel('State $x_1$')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('koopman_state_x1.svg', format='svg')

    plt.figure(figsize=(10, 6))
    for i, log in enumerate(logs):
        plt.plot(log['time'], log['control'], color=colors[i], alpha=0.8)
    plt.axvline(5.0, color='r', linestyle='--', label='Tp=5s')
    plt.ylabel('Control')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('koopman_control.svg', format='svg')

    plt.figure(figsize=(10, 6))
    for i, log in enumerate(logs):
        plt.plot(log['time'], log['theta_hat'], color=colors[i], alpha=0.8)
    plt.axhline(system_params['theta_true'], color='k', linestyle='--', label='true $\theta$')
    plt.axvline(5.0, color='r', linestyle='--', label='Tp=5s')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\hat{\theta}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('koopman_theta_hat.svg', format='svg')


if __name__ == '__main__':
    main()
