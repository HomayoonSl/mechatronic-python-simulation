import os
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

Jr = 0.005      
jf = 0.001      
mf = 0.2        
mr = 0.3        
Lr = 0.1        
L = 0.2         
g = 9.81        
b0 = 0.001      
bm = 0.0005     

Ra = 0.5        
kt = 0.038      
ke = 0.03915    

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

J_p = Jr + mf * (L**2)

M_mat = np.array([
    [J_p + jf, jf],
    [jf,       jf]
])
M_inv = np.linalg.inv(M_mat)

K_mat = np.array([
    [-(mr * Lr + mf * L) * g, 0.0],
    [0.0,                     0.0]
])

C_mat = np.array([
    [b0,  0.0],
    [0.0, bm + (kt * ke / Ra)]
])

H_mat = np.array([
    [0.0],
    [kt / Ra]
])

A_bl = -M_inv @ K_mat
A_br = -M_inv @ C_mat
B_b  = M_inv @ H_mat

A_sys = np.block([[np.zeros((2, 2)), np.eye(2)], [A_bl, A_br]])
B_sys = np.block([[np.zeros((2, 1))], [B_b]])

C_out = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
D_out = np.zeros((2, 1))

sys_lti = ct.ss(A_sys, B_sys, C_out, D_out)

motor_damping = bm + (kt * ke / Ra)
motor_gain = (kt / Ra)
num_motor = [motor_gain]
den_motor = [jf, motor_damping]
sys_motor_isolated = ct.tf(num_motor, den_motor)

print("="*50)
print("ISOLATED MOTOR SUBSYSTEM (TF: Speed / Voltage)")
print("="*50)
print(sys_motor_isolated)
print("="*50)

def nonlinear_dynamics(t, x, v_start, v_end, v_amplitude):
    theta, phi, theta_dot, phi_dot = x
    
    if v_start <= t < v_end:
        Va = v_amplitude
    else:
        Va = 0.0
        
    C_v = np.array([
        b0 * theta_dot,
        motor_damping * phi_dot
    ])
    
    G_theta = np.array([
        -(mr * Lr + mf * L) * g * np.sin(theta),
        0.0
    ])
    
    H_v = np.array([
        0.0,
        motor_gain * Va
    ])
    
    v_dot = M_inv @ (H_v - C_v - G_theta)
    
    return [theta_dot, phi_dot, v_dot[0], v_dot[1]]

t_span = (0.0, 10.0)
t_eval = np.linspace(0, 10.0, 3000)
V_START = 0.5
V_END = 5.5
V_AMPLITUDE = 5.0

x0 = [np.radians(0.0), 0.0, 0.0, 0.0]

sol = solve_ivp(nonlinear_dynamics, t_span, x0, 
                args=(V_START, V_END, V_AMPLITUDE), 
                t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.01)

theta_deg_raw = np.degrees(sol.y[0])
theta_deg_wrapped = (theta_deg_raw + 180) % 360 - 180

theta_plot = np.copy(theta_deg_wrapped)
wrap_points = np.abs(np.diff(theta_plot)) > 300
theta_plot[:-1][wrap_points] = np.nan

coupled_speed_rpm = sol.y[3] * (60.0 / (2 * np.pi))

U_array = np.zeros_like(t_eval)
U_array[(t_eval >= V_START) & (t_eval < V_END)] = V_AMPLITUDE

t_iso, isolated_speed_rad = ct.forced_response(sys_motor_isolated, T=t_eval, U=U_array)
isolated_speed_rpm = isolated_speed_rad * (60.0 / (2 * np.pi))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

ax1.plot(sol.t, theta_plot, "g-", linewidth=2)
ax1.axhline(180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(-180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(0, color="r", linestyle="-.", alpha=0.5)
ax1.set_ylabel("Pendulum Angle (deg)")
ax1.set_title("Nonlinear Open-Loop Response (5V Pulse: 0.5s to 5.5s)")
ax1.grid(True)
ax1.set_ylim(-200, 200)

ax2.plot(sol.t, coupled_speed_rpm, "m-", linewidth=2.5, label="Coupled Flywheel Speed (Actual)")
ax2.plot(t_eval, isolated_speed_rpm, "k--", linewidth=1.5, label="Isolated Motor Subsystem")
ax2.set_ylabel("Speed (RPM)")
ax2.grid(True)
ax2.legend(loc="upper right")

ax3.plot(t_eval, U_array, "b-", linewidth=2)
ax3.axhline(12, color="k", linestyle=":", alpha=0.5, label="Max Voltage (+12V)")
ax3.axhline(-12, color="k", linestyle=":", alpha=0.5, label="Min Voltage (-12V)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Motor Voltage (V)")
ax3.grid(True)
ax3.set_ylim(-13.5, 13.5)
ax3.legend(loc="upper right")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "02_open_loop_nonlinear_pulse.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
print("="*50)
