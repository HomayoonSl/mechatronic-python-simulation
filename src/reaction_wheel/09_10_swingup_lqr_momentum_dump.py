import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Physical parameters
Jr = 0.005
jf = 0.001
mf = 0.2
mr = 0.3
Lr = 0.1
L = 0.2
g = 9.81
b0 = 0.001
bm = 0.0005

# Motor electrical parameters
RA = 0.5
KT = 0.038
KE = 0.03915
V_MAX = 12.0
SPEED_MAX_RPM = 2550.0
SPEED_MAX_RAD = SPEED_MAX_RPM * (2.0 * np.pi / 60.0)

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# System Matrices
M_g = (mr * Lr + mf * L)
J_p = Jr + mf * (L ** 2)

M_mat = np.array([
    [J_p + jf, jf],
    [jf,       jf]
])
M_inv = np.linalg.inv(M_mat)

K_sub = np.array([
    [-M_g * g, 0.0],
    [0.0,      0.0]
])

C_sub = np.array([
    [b0, 0.0],
    [0.0, bm + (KT * KE / RA)]
])

H_sub = np.array([
    [0.0],
    [KT / RA]
])

Accel_x = -M_inv @ K_sub
Accel_v = -M_inv @ C_sub
B_accel = M_inv @ H_sub

A_lin = np.zeros((3, 3))
A_lin[0, 1] = 1.0                
A_lin[1, 0] = Accel_x[0, 0]      
A_lin[1, 1] = Accel_v[0, 0]      
A_lin[1, 2] = Accel_v[0, 1]      
A_lin[2, 0] = Accel_x[1, 0]      
A_lin[2, 1] = Accel_v[1, 0]      
A_lin[2, 2] = Accel_v[1, 1]      

B_lin = np.zeros((3, 1))
B_lin[1, 0] = B_accel[0, 0]
B_lin[2, 0] = B_accel[1, 0]

# LQR Design
Q = np.diag([2500.0, 100.0, 0.5]) 
R = np.array([[1.0]])
P = la.solve_continuous_are(A_lin, B_lin, Q, R)
K_lqr = (np.linalg.inv(R) @ B_lin.T @ P).flatten()

print("="*60)
print("EMBEDDED HYBRID CONTROLLER: SWING-UP & LQR")
print("="*60)
print(f"Computed LQR Gains: {np.round(K_lqr, 4)}")
print("="*60)

def get_derivatives(state, Va):
    theta, phi, theta_dot, phi_dot = state
    
    # Natural Back-EMF Saturation
    if phi_dot >= SPEED_MAX_RAD and Va > 0:
        Va = (RA / KT) * (bm * phi_dot) + KE * phi_dot
    elif phi_dot <= -SPEED_MAX_RAD and Va < 0:
        Va = (RA / KT) * (bm * phi_dot) + KE * phi_dot

    C_v = np.array([
        b0 * theta_dot,
        (bm + (KT * KE / RA)) * phi_dot
    ])
    
    G_theta = np.array([
        -M_g * g * np.sin(theta),
        0.0
    ])
    
    H_v = np.array([
        0.0,
        (KT / RA) * Va
    ])
    
    accel = M_inv @ (H_v - C_v - G_theta)
    return np.array([theta_dot, phi_dot, accel[0], accel[1]])

# Discrete-time simulation setup (200 Hz Sampling Rate)
dt = 0.005
t_eval = np.arange(0, 15.0, dt)
N = len(t_eval)

# State array
X = np.zeros((4, N))
X[:, 0] = [np.radians(179.9), 0.0, 0.0, 0.0]

V_applied = np.zeros(N)
modes = np.zeros(N) # 0: Swing-Up, 1: LQR

MODE_SWINGUP = 0
MODE_LQR = 1
current_mode = MODE_SWINGUP
k_pump = 10.0 # Energy injection gain

for i in range(N - 1):
    theta, phi, theta_dot, phi_dot = X[:, i]
    theta_wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    
    # 1. State Machine Logic (Hysteresis)
    if current_mode == MODE_SWINGUP:
        if abs(theta_wrapped) < 0.3 and abs(theta_dot) < 4.0:
            current_mode = MODE_LQR
    elif current_mode == MODE_LQR:
        if abs(theta_wrapped) > 0.6:
            current_mode = MODE_SWINGUP
            
    modes[i] = current_mode
    
    # 2. Control Law Calculation
    if current_mode == MODE_SWINGUP:
        # Energy-based Swing-Up (Astrom-Furuta inspired)
        E_pot = M_g * g * (np.cos(theta_wrapped) - 1.0)
        E_kin = 0.5 * J_p * (theta_dot ** 2)
        E_total = E_kin + E_pot
        
        # Inject energy by applying voltage proportional to energy error and velocity
        Va = k_pump * E_total * theta_dot 
    else:
        # LQR Stabilization & Momentum Dumping
        state_vec = np.array([theta_wrapped, theta_dot, phi_dot])
        Va = -np.dot(K_lqr, state_vec)
        
    Va = np.clip(Va, -V_MAX, V_MAX)
    V_applied[i] = Va
    
    # 3. Fixed-Step RK4 Integration (Microcontroller physics simulation)
    k1 = get_derivatives(X[:, i], Va)
    k2 = get_derivatives(X[:, i] + 0.5 * dt * k1, Va)
    k3 = get_derivatives(X[:, i] + 0.5 * dt * k2, Va)
    k4 = get_derivatives(X[:, i] + dt * k3, Va)
    
    X[:, i+1] = X[:, i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Plot formatting
theta_plot = (X[0, :] + np.pi) % (2 * np.pi) - np.pi
theta_deg = np.degrees(theta_plot)
wrap_idx = np.abs(np.diff(theta_deg)) > 300
theta_deg[:-1][wrap_idx] = np.nan

phi_dot_rpm = X[3, :] * (60.0 / (2.0 * np.pi))

switch_indices = np.where(np.diff(modes) > 0)[0]
first_catch = t_eval[switch_indices[0]] if len(switch_indices) > 0 else None

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(t_eval, theta_deg, "g-", linewidth=2, label="Pendulum Angle")
ax1.axhline(0, color="k", linestyle="-", alpha=0.3)
ax1.axhline(180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(-180, color="k", linestyle=":", alpha=0.5)
if first_catch:
    ax1.axvline(first_catch, color="r", linestyle="--", alpha=0.7, label="LQR Catch Region")
ax1.set_ylabel("Angle (deg)")
ax1.set_title("Embedded Hybrid Control: Swing-Up & Momentum Dumping")
ax1.grid(True)
ax1.set_ylim(-200, 200)
ax1.legend(loc="upper right")

ax2.plot(t_eval, phi_dot_rpm, "m-", linewidth=2.5, label="Flywheel Speed")
ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
if first_catch:
    ax2.axvline(first_catch, color="r", linestyle="--", alpha=0.7)
ax2.set_ylabel("Speed (RPM)")
ax2.grid(True)
ax2.legend(loc="upper right")

ax3.plot(t_eval, V_applied, "b-", linewidth=2, label="Motor Voltage")
ax3.axhline(V_MAX, color="k", linestyle=":", alpha=0.5)
ax3.axhline(-V_MAX, color="k", linestyle=":", alpha=0.5)
if first_catch:
    ax3.axvline(first_catch, color="r", linestyle="--", alpha=0.7)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Voltage (V)")
ax3.grid(True)
ax3.set_ylim(-13.5, 13.5)
ax3.legend(loc="upper right")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "04_embedded_hybrid_response.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
