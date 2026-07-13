import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve as scipy_linalg_solve

# ---------------------------------------------------------
# 1. ROBOT DYNAMICS CLASS (LAGRANGIAN 2-DOF RR MANIPULATOR)
# ---------------------------------------------------------
class RobotDynamics:
    def __init__(self, m1: float, m2: float, l1: float, l2: float, g: float = 9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

        self.alpha = (l2**2) * m2 + (l1**2) * (m1 + m2)
        self.beta = l1 * l2 * m2
        self.delta = (l2**2) * m2

    def compute_acceleration(self, q: np.ndarray, qdot: np.ndarray, 
                             tau: np.ndarray, tau_ext: np.ndarray) -> np.ndarray:
        q1, q2 = q[0], q[1]
        qdot1, qdot2 = qdot[0], qdot[1]

        # Inertia Matrix M(q)
        M = np.zeros((2, 2))
        M[0, 0] = self.alpha + 2.0 * self.beta * np.cos(q2)
        M[0, 1] = self.delta + self.beta * np.cos(q2)
        M[1, 0] = M[0, 1]
        M[1, 1] = self.delta

        # Coriolis and Centrifugal Matrix C(q, qdot)
        C = np.zeros((2, 2))
        C[0, 0] = -2.0 * self.beta * np.sin(q2) * qdot2
        C[0, 1] = -self.beta * np.sin(q2) * qdot2
        C[1, 0] = self.beta * np.sin(q2) * qdot1
        C[1, 1] = 0.0

        # Gravity Vector g(q)
        g_vec = np.zeros(2)
        g_vec[0] = (1.0 / self.l2) * self.g * self.delta * np.cos(q1 + q2) + \
                   (1.0 / self.l1) * (self.alpha - self.delta) * self.g * np.cos(q1)
        g_vec[1] = (1.0 / self.l2) * self.g * self.delta * np.cos(q1 + q2)

        # RHS = tau + tau_ext - C*qdot - g
        rhs = tau + tau_ext - np.dot(C, qdot) - g_vec
        qddot = scipy_linalg_solve(M, rhs)
        
        return qddot

# ---------------------------------------------------------
# 2. DELAY INTERPOLATION HELPER
# ---------------------------------------------------------
def get_delayed_value(history_t, history_state, current_t, delay):
    target_t = current_t - delay
    
    if target_t <= history_t[0]:
        return history_state[0].copy()
    
    idx = np.searchsorted(history_t, target_t)
    
    if idx == len(history_t):
        return history_state[-1].copy()
    
    t_prev, t_next = history_t[idx - 1], history_t[idx]
    x_prev, x_next = history_state[idx - 1], history_state[idx]
    
    dt = t_next - t_prev
    if dt > 1e-9:
        weight = (target_t - t_prev) / dt
        return x_prev + weight * (x_next - x_prev)
    else:
        return x_prev.copy()

def run_teleoperation_simulation():
    dt = 0.001
    t_max = 35.0
    num_steps = int(t_max / dt)

    l1 = 0.38
    l2 = 0.38
    m1l, m2l = 3.9473, 0.6232
    m1r, m2r = 3.2409, 0.3185

    local_robot = RobotDynamics(m1l, m2l, l1, l2)
    remote_robot = RobotDynamics(m1r, m2r, l1, l2)

    local_state = np.array([-np.pi/3, np.pi/3, 0.0, 0.0])
    remote_state = np.array([0.0, 0.0, 0.0, 0.0])

    # Gains adjusted to lower values to prevent enormous torques and NaN errors
    Kl = np.array([20.0, 20.0])
    Bl = np.array([5.0, 5.0])
    Kr = np.array([20.0, 20.0])
    Br = np.array([5.0, 5.0])

    history_t = [0.0]
    history_local_q = [local_state[0:2].copy()]
    history_remote_q = [remote_state[0:2].copy()]

    plot_t = []
    plot_local_q = []
    plot_remote_q = []
    plot_tau_l = []
    plot_tau_r = []

    for step in range(num_steps):
        t = step * dt
        plot_t.append(t)

        # --- Operator Torque (Figure 6) ---
        tau_h = np.zeros(2)
        if 5.0 <= t < 15.0:
            tau_h[0] = 10.0
        elif 25.0 <= t < 35.0:
            tau_h[0] = 5.0
            
        if 5.0 <= t < 15.0:
            tau_h[1] = 15.0
        elif 15.0 <= t < 25.0:
            tau_h[1] = 20.0
        elif 25.0 <= t < 35.0:
            tau_h[1] = 10.0

        tau_ext_remote = np.zeros(2)

        # --- Variable Delays (Table 2) ---
        Tl = 0.2 + 0.1 * np.sin(5.0 * t) + 0.05 * np.sin(2.5 * t)
        Tr = 0.2 + 0.1 * np.sin(2.5 * t) + 0.05 * np.sin(5.0 * t)

        delayed_local_q = get_delayed_value(history_t, history_local_q, t, Tl)
        delayed_remote_q = get_delayed_value(history_t, history_remote_q, t, Tr)

        def system_ode(l_s, r_s):
            l_q, l_v = l_s[0:2], l_s[2:4]
            r_q, r_v = r_s[0:2], r_s[2:4]
            
            # ---------------------------------------------------------
            #  FIX: Corrected Local Controller based on PDF (q_l - q_r)
            #  Reversed the previous (q_r - q_l) which caused positive feedback blow-up.
            # ---------------------------------------------------------
            t_l = Kl * (l_q - delayed_remote_q) + Bl * l_v
            t_r = Kr * (delayed_local_q - r_q) - Br * r_v
            
            # Apply dynamic equation: M(q)ddq = tau_h - tau_l - C - g
            l_qddot = local_robot.compute_acceleration(l_q, l_v, -t_l, tau_h)
            r_qddot = remote_robot.compute_acceleration(r_q, r_v, t_r, tau_ext_remote)
            
            return np.concatenate([l_v, l_qddot]), np.concatenate([r_v, r_qddot]), t_l, t_r

        # RK4 Steps
        k1_l, k1_r, tl1, tr1 = system_ode(local_state, remote_state)
        k2_l, k2_r, _, _ = system_ode(local_state + 0.5 * dt * k1_l, remote_state + 0.5 * dt * k1_r)
        k3_l, k3_r, _, _ = system_ode(local_state + 0.5 * dt * k2_l, remote_state + 0.5 * dt * k2_r)
        k4_l, k4_r, _, _ = system_ode(local_state + dt * k3_l, remote_state + dt * k3_r)

        # Store torques
        plot_tau_l.append(tl1.copy())
        plot_tau_r.append(tr1.copy())

        # Update states using RK4
        local_state += (dt / 6.0) * (k1_l + 2 * k2_l + 2 * k3_l + k4_l)
        remote_state += (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

        # Update history
        history_t.append(t + dt)
        history_local_q.append(local_state[0:2].copy())
        history_remote_q.append(remote_state[0:2].copy())

        plot_local_q.append(local_state[0:2].copy())
        plot_remote_q.append(remote_state[0:2].copy())

    return (np.array(plot_t), np.array(plot_local_q), np.array(plot_remote_q), 
            np.array(plot_tau_l), np.array(plot_tau_r))
# ---------------------------------------------------------
# 4. PLOTTING AND SAVING GENERATION (White Background, PNG Output)
# ---------------------------------------------------------
if __name__ == "__main__":
    t, local_q, remote_q, tau_l, tau_r = run_teleoperation_simulation()
    
    # Calculate tracking errors
    tracking_error = local_q - remote_q

    # Use default matplotlib style for white background
    plt.style.use('default')

    # Figure 1: Position Tracking - Joint 1
    plt.figure(1, figsize=(12, 6))
    plt.plot(t, local_q[:, 0], 'b-', linewidth=1.5, label=r'Local Joint 1 ($q_{l1}$)')
    plt.plot(t, remote_q[:, 0], 'r--', linewidth=1.5, label=r'Remote Joint 1 ($q_{r1}$)')
    plt.title('Position Tracking Performance - Joint 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('position_tracking_joint1.png', dpi=300)

    # Figure 2: Position Tracking - Joint 2
    plt.figure(2, figsize=(12, 6))
    plt.plot(t, local_q[:, 1], 'b-', linewidth=1.5, label=r'Local Joint 2 ($q_{l2}$)')
    plt.plot(t, remote_q[:, 1], 'r--', linewidth=1.5, label=r'Remote Joint 2 ($q_{r2}$)')
    plt.title('Position Tracking Performance - Joint 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('position_tracking_joint2.png', dpi=300)

    # Figure 3: Tracking Errors
    plt.figure(3, figsize=(12, 6))
    plt.plot(t, tracking_error[:, 0], 'g-', linewidth=1.5, label=r'Tracking Error Joint 1 ($e_1 = q_{l1} - q_{r1}$)')
    plt.plot(t, tracking_error[:, 1], 'm-', linewidth=1.5, label=r'Tracking Error Joint 2 ($e_2 = q_{l2} - q_{r2}$)')
    plt.title('Teleoperation Tracking Errors vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tracking_errors.png', dpi=300)

    # Figure 4: Control Torques
    plt.figure(4, figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, tau_l[:, 0], 'b-', linewidth=1.5, label=r'Local Controller Torque $\tau_{l1}$')
    plt.plot(t, tau_r[:, 0], 'r--', linewidth=1.5, label=r'Remote Controller Torque $\tau_{r1}$')
    plt.title('Control Torques Generated by Controllers - Joint 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N.m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, tau_l[:, 1], 'b-', linewidth=1.5, label=r'Local Controller Torque $\tau_{l2}$')
    plt.plot(t, tau_r[:, 1], 'r--', linewidth=1.5, label=r'Remote Controller Torque $\tau_{r2}$')
    plt.title('Control Torques Generated by Controllers - Joint 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N.m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('control_torques.png', dpi=300)

    # Show all plots on screen
    plt.show()
