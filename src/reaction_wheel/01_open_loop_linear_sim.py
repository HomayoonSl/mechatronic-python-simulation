import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.integrate import solve_ivp

# System Geometry & Inertia Parameters
Jr = 0.005      
jf = 0.001      
mf = 0.2        
mr = 0.3        
Lr = 0.1        
L = 0.2         
g = 9.81        
b0 = 0.001      
bm = 0.0005     

# Exact Motor Parameters from KAG M63x40/1 Datasheet (12V variant)
Ra = 0.67       
kt = 0.067      
ke = 0.0764     

# Physical Actuator Limits
V_MAX = 12.0    
SPEED_MAX_RPM = 2550.0
SPEED_MAX_RAD = SPEED_MAX_RPM * (2 * np.pi / 60.0) # ~267 rad/s

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. State-Space Matrices Construction
M11 = Jr + mf * (L**2) + jf
M12 = jf
M21 = jf
M22 = jf

M = np.array([[M11, M12], [M21, M22]])
M_inv = np.linalg.inv(M)

G_mat = np.array([[-(mr * Lr + mf * L) * g, 0], [0, 0]])
C_mat = np.array([[b0, -(kt * ke / Ra)], [0, bm + (kt * ke / Ra)]])
H_mat = np.array([[-(kt / Ra)], [kt / Ra]])

A_lower_left = -M_inv @ G_mat
A_lower_right = -M_inv @ C_mat

A = np.block([
    [np.zeros((2, 2)), np.eye(2)],
    [A_lower_left, A_lower_right]
])

B = np.block([
    [np.zeros((2, 1))],
    [M_inv @ H_mat]
])

# Using Python-Control library for LTI Analysis
C_theta = np.array([[1, 0, 0, 0]])
C_phi_dot = np.array([[0, 0, 0, 1]])
D = np.array([[0]])

sys_theta = ct.ss(A, B, C_theta, D)
sys_phi_dot = ct.ss(A, B, C_phi_dot, D)

# 2. Terminal Analysis Printouts using control library
poles = ct.poles(sys_theta)
zeros_theta = ct.zeros(sys_theta)

print("="*50)
print("LINEAR SYSTEM ANALYSIS VIA CONTROL LIBRARY")
print("="*50)
print("System Poles (Eigenvalues):")
for idx, p in enumerate(poles):
    print(f"  Pole {idx+1}: {p:.4f}")

print("\nTransmission Zeros for Pendulum Angle (theta):")
if len(zeros_theta) == 0:
    print("  No zeros found for this output configuration.")
else:
    for idx, z in enumerate(zeros_theta):
        print(f"  Zero {idx+1}: {z:.4f}")
print("="*50)

# 3. Dynamic Simulator with Physical Saturation
def linear_physical_sat_dynamics(t, x, Va_step_value, t_step):
    theta, phi, theta_dot, phi_dot = x
    
    Va = Va_step_value if t >= t_step else 0.0
    Va = np.clip(Va, -V_MAX, V_MAX)
    
    if phi_dot >= SPEED_MAX_RAD and Va > 0:
        Va = (Ra / kt) * (bm * phi_dot) + ke * phi_dot
    elif phi_dot <= -SPEED_MAX_RAD and Va < 0:
        Va = (Ra / kt) * (bm * phi_dot) + ke * phi_dot

    state = np.array([[theta], [phi], [theta_dot], [phi_dot]])
    x_dot = A @ state + B * Va
    
    if abs(phi_dot) >= SPEED_MAX_RAD:
        x_dot[3, 0] = 0.0
        
    return [x_dot[0, 0], x_dot[1, 0], x_dot[2, 0], x_dot[3, 0]]

# Simulation Execution (10 Seconds)
t_end = 1.5
t_eval = np.linspace(0, t_end, 2000)
x0 = [np.radians(1.0), 0.0, 0.0, 0.0] 
voltage_amplitude = 12.0 
t_step_input = 0.5

sol = solve_ivp(
    linear_physical_sat_dynamics, 
    (0.0, t_end), 
    x0, 
    args=(voltage_amplitude, t_step_input), 
    t_eval=t_eval, 
    method="RK45"
)

theta_wrapped = (np.degrees(sol.y[0]) + 180) % 360 - 180
flywheel_velocity_rpm = sol.y[3] * (60.0 / (2 * np.pi))

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t, theta_wrapped, "g-", linewidth=2, label=r"Pendulum $\theta$")
ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
ax1.set_ylabel("Pendulum Angle (deg)")
ax1.set_ylim([-185, 185])
ax1.grid(True)
ax1.legend()
ax1.set_title("Linear Open-Loop Response with Hard Actuator & Velocity Saturation")

ax2.plot(sol.t, flywheel_velocity_rpm, "m-", linewidth=2, label=r"Flywheel $\dot{\phi}$")
ax2.axhline(SPEED_MAX_RPM, color="r", linestyle=":", label="Datasheet Limit (2550 RPM)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Flywheel Velocity (RPM)")
ax2.grid(True)
ax2.legend()

output_path = os.path.join(OUTPUT_DIR, "linear_open_loop_step_response.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Analysis completed. Plot saved to: {output_path}")
