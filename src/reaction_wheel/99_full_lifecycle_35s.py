import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ============================================================
# 1. Physical & Electrical Parameters (Unified)
# ============================================================
Jr = 0.005
jf = 0.001
mf = 0.2
mr = 0.3
Lr = 0.1
L = 0.2
g = 9.81
b0 = 0.001
bm = 0.0005

RA = 0.2       # Updated from PID analysis
KT = 0.038
KE = 0.03915
V_MAX = 12.0
SPEED_MAX_RPM = 2550.0
SPEED_MAX_RAD = SPEED_MAX_RPM * (2.0 * np.pi / 60.0)

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. System Matrices (Pre-computation)
# ============================================================
M_eq = mr * Lr + mf * L
J_p = Jr + mf * (L ** 2)

M_mat = np.array([[J_p + jf, jf],
                  [jf,       jf]])
M_inv = np.linalg.inv(M_mat)

# Linearization Matrices for LQR
K_sub = np.array([[-M_eq * g, 0.0], [0.0, 0.0]])
C_sub = np.array([[b0, 0.0], [0.0, bm + (KT * KE / RA)]])
H_sub = np.array([[0.0], [KT / RA]])

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

# Compute LQR Gains
Q = np.diag([2500.0, 100.0, 0.5]) 
R = np.array([[1.0]])
P = la.solve_continuous_are(A_lin, B_lin, Q, R)
K_lqr = (np.linalg.inv(R) @ B_lin.T @ P).flatten()

# PID Gains (From optimization)
Kp_pid = 300.0
Ki_pid = 150.0
Kd_pid = 0.1

print("="*60)
print("FULL LIFECYCLE SIMULATION: PID -> FAILURE -> SWING-UP -> LQR")
print("="*60)
print(f"LQR Gains: {np.round(K_lqr, 2)}")
print(f"PID Gains: Kp={Kp_pid}, Ki={Ki_pid}, Kd={Kd_pid}")
print("="*60)

# ============================================================
# 3. Dynamic Engine (Plant Physics)
# ============================================================
def get_derivatives(state, Va, dist_torque):
    theta, phi, theta_dot, phi_dot, int_theta = state
    
    # Actuator Limit / Back-EMF Physics
    if phi_dot >= SPEED_MAX_RAD and Va > 0:
        Va = (RA / KT) * (bm * phi_dot) + KE * phi_dot
    elif phi_dot <= -SPEED_MAX_RAD and Va < 0:
        Va = (RA / KT) * (bm * phi_dot) + KE * phi_dot

    C_v = np.array([b0 * theta_dot, (bm + (KT * KE / RA)) * phi_dot])
    G_theta = np.array([-M_eq * g * np.sin(theta), 0.0])
    H_v = np.array([0.0, (KT / RA) * Va])
    Dist = np.array([dist_torque, 0.0])
    
    accel = M_inv @ (H_v - C_v - G_theta + Dist)
    return np.array([theta_dot, phi_dot, accel[0], accel[1], 0.0])

# ============================================================
# 4. Simulation Setup & Supervisory State Machine
# ============================================================
dt = 0.005
T_TOTAL = 35.0
t_eval = np.arange(0, T_TOTAL, dt)
N = len(t_eval)

# States: [theta, phi, theta_dot, phi_dot, int_theta]
X = np.zeros((5, N))
X[:, 0] = [np.radians(10.0), 0.0, 0.0, 0.0, 0.0]

V_applied = np.zeros(N)
dist_array = np.zeros(N)
modes = np.zeros(N)

MODE_PID = 0
MODE_FALL = 1
MODE_SWINGUP = 2
MODE_LQR = 3

current_mode = MODE_PID
k_pump = 10.0 

# Time schedule parameters
T_DISTURBANCE = 5.0
T_START_SWINGUP = 30.0
DIST_MAGNITUDE = 0.05

for i in range(N - 1):
    t = t_eval[i]
    theta, phi, theta_dot, phi_dot, int_theta = X[:, i]
    theta_wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    
    # --- A. State Machine Transitions ---
    if current_mode == MODE_PID:
        # Fall condition: theta > 80 deg
        if abs(theta_wrapped) > np.radians(80.0):
            current_mode = MODE_FALL
            X[4, i] = 0.0 # Reset integrator
            
    elif current_mode == MODE_FALL:
        # Wait for system to settle naturally, start swingup at 30s
        if t >= T_START_SWINGUP:
            current_mode = MODE_SWINGUP
            
    elif current_mode == MODE_SWINGUP:
        # Catch condition
        if abs(theta_wrapped) < 0.3 and abs(theta_dot) < 4.0:
            current_mode = MODE_LQR
            
    elif current_mode == MODE_LQR:
        # Fail-safe drop
        if abs(theta_wrapped) > 0.6:
            current_mode = MODE_SWINGUP

    modes[i] = current_mode
    
    # --- B. Disturbance Logic ---
    current_dist = 0.0
    if current_mode == MODE_PID and t >= T_DISTURBANCE:
        current_dist = DIST_MAGNITUDE
    dist_array[i] = current_dist
    
    # --- C. Controllers ---
    Va = 0.0
    d_int = 0.0
    
    if current_mode == MODE_PID:
        V_cmd = Kp_pid * theta + Ki_pid * int_theta + Kd_pid * theta_dot
        Va = np.clip(V_cmd, -V_MAX, V_MAX)
        # Anti-Windup
        if abs(V_cmd) >= V_MAX and np.sign(theta) == np.sign(V_cmd):
            d_int = 0.0
        else:
            d_int = theta
            
    elif current_mode == MODE_FALL:
        Va = 0.0 
        d_int = 0.0
        
    elif current_mode == MODE_SWINGUP:
        E_pot = M_eq * g * (np.cos(theta_wrapped) - 1.0)
        E_kin = 0.5 * J_p * (theta_dot ** 2)
        E_total = E_kin + E_pot
        Va = k_pump * E_total * theta_dot 
        Va = np.clip(Va, -V_MAX, V_MAX)
        d_int = 0.0
        
    elif current_mode == MODE_LQR:
        state_vec = np.array([theta_wrapped, theta_dot, phi_dot])
        Va = -np.dot(K_lqr, state_vec)
        Va = np.clip(Va, -V_MAX, V_MAX)
        d_int = 0.0
        
    V_applied[i] = Va
    
    # --- D. RK4 Integration ---
    k1 = get_derivatives(X[:, i], Va, current_dist)
    k1[4] = d_int
    
    k2 = get_derivatives(X[:, i] + 0.5 * dt * k1, Va, current_dist)
    k2[4] = d_int
    
    k3 = get_derivatives(X[:, i] + 0.5 * dt * k2, Va, current_dist)
    k3[4] = d_int
    
    k4 = get_derivatives(X[:, i] + dt * k3, Va, current_dist)
    k4[4] = d_int
    
    X[:, i+1] = X[:, i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# 5. Visualization (4 separate, high-quality, extra-wide figures)
# ============================================================
modes[-1] = modes[-2]
V_applied[-1] = V_applied[-2]
dist_array[-1] = dist_array[-2]

theta_raw = X[0, :]
theta_wrapped = (theta_raw + np.pi) % (2 * np.pi) - np.pi
theta_deg = np.degrees(theta_wrapped)

# Remove wrapping artifacts for plot
theta_plot = np.copy(theta_deg)
wrap_idx = np.abs(np.diff(theta_plot)) > 300
theta_plot[:-1][wrap_idx] = np.nan

phi_dot_rpm = X[3, :] * (60.0 / (2.0 * np.pi))

# Extract transition times for plot lines
t_fall = t_eval[np.argmax(modes == MODE_FALL)] if np.any(modes == MODE_FALL) else None
t_swing = t_eval[np.argmax(modes == MODE_SWINGUP)] if np.any(modes == MODE_SWINGUP) else None
t_lqr = t_eval[np.argmax(modes == MODE_LQR)] if np.any(modes == MODE_LQR) else None

# Define regions for shading
regions = [
    (0, t_fall if t_fall else T_TOTAL, 'PID Control (Failed by Disturbance)', 'lightblue'),
    (t_fall, t_swing, 'Free Fall & Settle', 'lightgray'),
    (t_swing, t_lqr, 'Energy Swing-Up', 'lightyellow'),
    (t_lqr, T_TOTAL, 'LQR & Momentum Dump', 'lightgreen')
]

# Utility to add regions and annotations to any axis
def add_regions(ax, y_pos):
    for (start, end, label, color) in regions:
        if start is not None and end is not None and start < T_TOTAL:
            ax.axvspan(start, end, color=color, alpha=0.2)
            ax.text((start + end)/2, y_pos, label, ha='center', va='top', 
                    fontsize=14, fontweight='bold', color='black')

# Common font settings for better readability
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 1.5

# EXTRA-WIDE FIGURE SIZE (horizontal stretch)
FIG_WIDTH = 28   # inches
FIG_HEIGHT = 8   # inches

# 1. Pendulum Angle
fig1 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax1 = fig1.add_subplot(111)
ax1.plot(t_eval, theta_plot, "g-", linewidth=1.5, label='Pendulum angle')
ax1.axhline(0, color="k", linestyle="-", alpha=0.3)
ax1.axhline(180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(-180, color="k", linestyle=":", alpha=0.5)
ax1.set_ylabel("Angle (deg)", fontsize=16)
ax1.set_xlabel("Time (s)", fontsize=16)
ax1.set_title("Pendulum Angle – Full Lifecycle", fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(-190, 190)
add_regions(ax1, y_pos=180)
plt.tight_layout(pad=2.0)
fig1.savefig(os.path.join(OUTPUT_DIR, "05_angle_plot.png"), dpi=400, bbox_inches="tight")
plt.close(fig1)

# 2. Flywheel Speed (RPM)
fig2 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax2 = fig2.add_subplot(111)
ax2.plot(t_eval, phi_dot_rpm, "m-", linewidth=1.5, label='Flywheel speed')
ax2.axhline(SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.5, label='Speed limit')
ax2.axhline(-SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.5)
ax2.set_ylabel("Speed (RPM)", fontsize=16)
ax2.set_xlabel("Time (s)", fontsize=16)
ax2.set_title("Flywheel Speed – Full Lifecycle", fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(-1000, 2900)
add_regions(ax2, y_pos=2750)
ax2.legend(loc='upper right', fontsize=14)
plt.tight_layout(pad=2.0)
fig2.savefig(os.path.join(OUTPUT_DIR, "05_rpm_plot.png"), dpi=400, bbox_inches="tight")
plt.close(fig2)

# 3. Motor Voltage
fig3 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax3 = fig3.add_subplot(111)
ax3.plot(t_eval, V_applied, "b-", linewidth=1.5, label='Motor voltage')
ax3.axhline(V_MAX, color="k", linestyle=":", alpha=0.5, label='Voltage limit')
ax3.axhline(-V_MAX, color="k", linestyle=":", alpha=0.5)
ax3.set_ylabel("Voltage (V)", fontsize=16)
ax3.set_xlabel("Time (s)", fontsize=16)
ax3.set_title("Motor Voltage – Full Lifecycle", fontsize=18)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_ylim(-15, 15)
add_regions(ax3, y_pos=14)
ax3.legend(loc='upper right', fontsize=14)
plt.tight_layout(pad=2.0)
fig3.savefig(os.path.join(OUTPUT_DIR, "05_voltage_plot.png"), dpi=400, bbox_inches="tight")
plt.close(fig3)

# 4. Operational Mode & Disturbance
fig4 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax4 = fig4.add_subplot(111)
mode_labels = ['PID', 'Fall', 'Swing-Up', 'LQR']
mode_colors = ['blue', 'gray', 'orange', 'green']
for mode_val in range(4):
    idx = modes == mode_val
    if np.any(idx):
        ax4.fill_between(t_eval, mode_val-0.4, mode_val+0.4, where=idx, 
                         color=mode_colors[mode_val], alpha=0.6, label=mode_labels[mode_val])
ax4.plot(t_eval, dist_array*10, 'r--', linewidth=1, alpha=0.7, label='Disturbance (scaled x10)')
ax4.set_yticks([0,1,2,3])
ax4.set_yticklabels(mode_labels, fontsize=14)
ax4.set_ylabel("Control Mode", fontsize=16)
ax4.set_xlabel("Time (s)", fontsize=16)
ax4.set_title("Supervisory Mode and Disturbance", fontsize=18)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(loc='upper right', fontsize=14)
if t_fall: ax4.axvline(t_fall, color='k', linestyle='--', alpha=0.4)
if t_swing: ax4.axvline(t_swing, color='k', linestyle='--', alpha=0.4)
if t_lqr: ax4.axvline(t_lqr, color='k', linestyle='--', alpha=0.4)
plt.tight_layout(pad=2.0)
fig4.savefig(os.path.join(OUTPUT_DIR, "05_modes_disturbance_plot.png"), dpi=400, bbox_inches="tight")
plt.close(fig4)

print(f"All four extra‑wide plots saved to: {OUTPUT_DIR}")
