import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters
Jr = 0.005      
jf = 0.001      
mf = 0.2        
mr = 0.3        
Lr = 0.1        
L = 0.2         
g = 9.81        
b0 = 0.001      
bm = 0.0005     
kt = 0.023      
ke = 0.023      
Ra = 2.5        

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. State-Space Matrices with Corrected Directional Signs
M11 = Jr + mf * (L**2) + jf
M12 = jf
M21 = jf
M22 = jf

M = np.array([[M11, M12], [M21, M22]])
M_inv = np.linalg.inv(M)

# G: Sign inversion to match Left = Positive (Counter-Clockwise convention adjustment)
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

eigenvalues = np.linalg.eigvals(A)
print("="*50)
print("CORRECTED LINEAR STATE-SPACE MATRICES & POLES")
print("="*50)
print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("\nSystem Eigenvalues (Poles):")
for idx, eig in enumerate(eigenvalues):
    print(f"Pole {idx+1}: {eig:.4f}")
print("="*50)

# 2. ODE Simulation handling Voltage Saturation [-12V, +12V]
def linear_sat_dynamics(t, x, Va_step_value, t_step):
    theta, phi, theta_dot, phi_dot = x
    
    # Step input with saturation bounds
    Va = Va_step_value if t >= t_step else 0.0
    Va = np.clip(Va, -12.0, 12.0)
    
    state = np.array([[theta], [phi], [theta_dot], [phi_dot]])
    x_dot = A @ state + B * Va
    
    return [x_dot[0, 0], x_dot[1, 0], x_dot[2, 0], x_dot[3, 0]]

# Simulation Setup (10 Seconds, Small initial tilt to the left: +1 degree)
t_end = 10.0
t_eval = np.linspace(0, t_end, 2000)
x0 = [np.radians(1.0), 0.0, 0.0, 0.0] 
voltage_amplitude = 5.0 
t_step_input = 1.0

sol = solve_ivp(
    linear_sat_dynamics, 
    (0.0, t_end), 
    x0, 
    args=(voltage_amplitude, t_step_input), 
    t_eval=t_eval, 
    method="RK45"
)

# Angle wrapping routine to wrap within [-180, +180] degrees
def wrap_angle(rad_angles):
    deg_angles = np.degrees(rad_angles)
    return (deg_angles + 180) % 360 - 180

theta_wrapped = wrap_angle(sol.y[0])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t, theta_wrapped, "g-", linewidth=2, label=r"Pendulum $\theta$")
ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
ax1.set_ylabel("Pendulum Angle (deg, Wrapped)")
ax1.set_ylim([-185, 185])
ax1.grid(True)
ax1.legend()
ax1.set_title("Linear Open-Loop Step Response with Voltage Saturation Constraints")

ax2.plot(sol.t, sol.y[3], "m-", linewidth=2, label=r"Flywheel $\dot{\phi}$")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Flywheel Velocity (rad/s)")
ax2.grid(True)
ax2.legend()

output_path = os.path.join(OUTPUT_DIR, "linear_open_loop_step_response.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
