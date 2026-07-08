import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters (Standard Sample Values)
Jr = 0.005      # Pendulum body inertia (kg*m^2)
jf = 0.001      # Flywheel inertia (kg*m^2)
mf = 0.2        # Flywheel mass (kg)
mr = 0.3        # Pendulum mass (kg)
Lr = 0.1        # Distance to pendulum COM (m)
L = 0.2         # Distance to flywheel center (m)
g = 9.81        # Gravity (m/s^2)
b0 = 0.001      # Pendulum joint friction
bm = 0.0005     # Motor shaft friction
kt = 0.023      # Motor torque constant (N*m/A)
ke = 0.023      # Motor Back-EMF constant (V*s/rad)
Ra = 2.5        # Armature resistance (Ohms)

# Output directory config
OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rwip_dynamics(t, x, Va_step_value, t_step):
    """
    State vector: x = [theta, phi, theta_dot, phi_dot]
    Input: Va (Motor voltage)
    """
    theta, phi, theta_dot, phi_dot = x
    
    # Step input logic
    Va = Va_step_value if t >= t_step else 0.0
    
    # Mass Matrix elements
    M11 = Jr + mf * (L**2) + jf
    M12 = jf
    M21 = jf
    M22 = jf
    
    # Right-hand side terms (Forces and Torques)
    # Electromagnetic damping term from motor: (kt * ke / Ra) * phi_dot
    # Motor torque term from voltage: (kt / Ra) * Va
    elec_damping = (kt * ke / Ra) * phi_dot
    voltage_torque = (kt / Ra) * Va
    
    G1 = (mr * Lr + mf * L) * g * np.sin(theta)
    
    rhs1 = G1 - b0 * theta_dot - voltage_torque + elec_damping
    rhs2 = voltage_torque - (bm + (kt * ke / Ra)) * phi_dot
    
    # Solve for accelerations: M * [theta_ddot, phi_ddot]^T = [rhs1, rhs2]^T
    # Cramer's rule for 2x2 matrix inversion
    det_M = M11 * M22 - M12 * M21
    
    theta_ddot = (rhs1 * M22 - rhs2 * M12) / det_M
    phi_ddot = (M11 * rhs2 - M21 * rhs1) / det_M
    
    return [theta_dot, phi_dot, theta_ddot, phi_ddot]

# Simulation Configuration
t_start = 0.0
t_end = 10.0
t_step_input = 0.5
voltage_amplitude = 5.0  # 5V Step Input
x0 = [0.01, 0.0, 0.0, 0.0]  # Small initial perturbation (10 mrad) to observe instability

t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, 1000)

# Run Numerical Integration
sol = solve_ivp(
    rwip_dynamics, 
    t_span, 
    x0, 
    args=(voltage_amplitude, t_step_input), 
    t_eval=t_eval, 
    method="RK45"
)

# Plotting Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Pendulum Angular Response
ax1.plot(sol.t, np.degrees(sol.y[0]), "r-", linewidth=2, label=r"Pendulum $\theta$")
ax1.axvline(t_step_input, color="k", linestyle="--", label="Step Input Time")
ax1.set_ylabel("Pendulum Angle (deg)")
ax1.grid(True)
ax1.legend()
ax1.set_title("Open-Loop nonlinear model Step Response (Reaction Wheel Inverted Pendulum)")

# Flywheel Angular Velocity Response
ax2.plot(sol.t, sol.y[3], "b-", linewidth=2, label=r"Flywheel $\dot{\phi}$")
ax2.axvline(t_step_input, color="k", linestyle="--")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Flywheel Velocity (rad/s)")
ax2.grid(True)
ax2.legend()

# Save output to specified directory layout
output_path = os.path.join(OUTPUT_DIR, "open_loop_step_nonlinear_response.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Simulation completed. Plot saved to: {output_path}")
