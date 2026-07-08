import os
import numpy as np
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

V_MAX = 12.0    
SPEED_MAX_RPM = 2550.0
SPEED_MAX_RAD = SPEED_MAX_RPM * (2 * np.pi / 60.0)

OUTPUT_DIR = os.path.join("outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

J_p = Jr + mf * (L**2)
M_mat = np.array([
    [J_p + jf, jf],
    [jf,       jf]
])
M_inv = np.linalg.inv(M_mat)

Kp = 60.0
Ki = 25.0
Kd = 12.0

print("="*50)
print("CLOSED-LOOP PID CONTROL ANALYSIS (CORRECTED POLARITY)")
print("="*50)
print(f"Controller Gains -> Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
print("="*50)

def pid_nonlinear_dynamics(t, x, t_dist, dist_torque):
    theta, phi, theta_dot, phi_dot, int_theta = x
    
    V_cmd = Kp * theta + Ki * int_theta + Kd * theta_dot
    Va = np.clip(V_cmd, -V_MAX, V_MAX)
    
    if abs(V_cmd) >= V_MAX and np.sign(theta) == np.sign(V_cmd):
        d_int = 0.0
    else:
        d_int = theta

    C_v = np.array([
        b0 * theta_dot,
        (bm + (kt * ke / Ra)) * phi_dot
    ])
    
    G_theta = np.array([
        -(mr * Lr + mf * L) * g * np.sin(theta),
        0.0
    ])
    
    H_v = np.array([
        0.0,
        (kt / Ra) * Va
    ])
    
    external_dist = np.array([0.0, 0.0])
    if t >= t_dist:
        external_dist[0] = dist_torque 

    accel = M_inv @ (H_v - C_v - G_theta + external_dist)
    
    return [theta_dot, phi_dot, accel[0], accel[1], d_int]

t_span = (0.0, 15.0)
t_eval = np.linspace(0, 15.0, 4000)

DIST_TIME = 5.0
DIST_TORQUE = 0.02  

x0 = [np.radians(10.0), 0.0, 0.0, 0.0, 0.0]

sol = solve_ivp(pid_nonlinear_dynamics, t_span, x0, 
                args=(DIST_TIME, DIST_TORQUE), 
                t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.01)

theta_deg_raw = np.degrees(sol.y[0])
theta_dot_rad = sol.y[2]
phi_dot_rpm = sol.y[3] * (60.0 / (2 * np.pi))
int_theta = sol.y[4]

V_cmd_raw = Kp * sol.y[0] + Ki * int_theta + Kd * theta_dot_rad
V_applied = np.clip(V_cmd_raw, -V_MAX, V_MAX)

theta_deg_wrapped = (theta_deg_raw + 180) % 360 - 180
theta_plot = np.copy(theta_deg_wrapped)
wrap_points = np.abs(np.diff(theta_plot)) > 300
theta_plot[:-1][wrap_points] = np.nan

pre_dist_mask = sol.t < DIST_TIME
if np.any(pre_dist_mask):
    theta_pre_dist = theta_deg_raw[pre_dist_mask]
    overshoot = np.max(theta_pre_dist) if np.max(theta_pre_dist) > 10.0 else 10.0
    steady_state_error_pre = theta_pre_dist[-1]
else:
    overshoot = 10.0
    steady_state_error_pre = 0.0

print("TRANSIENT RESPONSE (0s to 5s - Before Disturbance):")
print(f"  Initial Angle:      10.0 deg")
print(f"  Peak Overshoot:     {overshoot:.2f} deg")
print(f"  Steady-State Error: {steady_state_error_pre:.4f} deg")
print("-" * 50)
print("ROBUSTNESS ANALYSIS (After Step Disturbance):")
print(f"  Applied Disturbance: {DIST_TORQUE} Nm at t={DIST_TIME}s")
print(f"  Max Voltage Hit:     {np.max(np.abs(V_applied)):.2f} V")
print(f"  Max Flywheel Speed:  {np.max(np.abs(phi_dot_rpm)):.1f} RPM")

if np.max(np.abs(theta_deg_raw[sol.t > DIST_TIME])) > 90.0:
    print("  SYSTEM STATUS:       FAILED (Pendulum fell due to motor saturation)")
else:
    print("  SYSTEM STATUS:       STABLE (But speed is accumulating)")
print("="*50)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

ax1.plot(sol.t, theta_plot, "g-", linewidth=2, label="Pendulum Angle")
ax1.axvline(DIST_TIME, color="r", linestyle="--", alpha=0.6, label="Step Disturbance (0.02 Nm)")
ax1.axhline(180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(-180, color="k", linestyle=":", alpha=0.5)
ax1.axhline(0, color="k", linestyle="-", alpha=0.3)
ax1.set_ylabel("Angle (deg)")
ax1.set_title("Closed-Loop PID Performance under Step Disturbance")
ax1.grid(True)
ax1.set_ylim(-200, 200)
ax1.legend(loc="upper left")

ax2.plot(sol.t, phi_dot_rpm, "m-", linewidth=2.5, label="Flywheel Speed")
ax2.axhline(SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.5, label="Rated Max Speed")
ax2.axhline(-SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.5)
ax2.set_ylabel("Speed (RPM)")
ax2.grid(True)
ax2.legend(loc="upper left")

ax3.plot(sol.t, V_applied, "b-", linewidth=2, label="Motor Voltage")
ax3.axhline(V_MAX, color="k", linestyle=":", alpha=0.5)
ax3.axhline(-V_MAX, color="k", linestyle=":", alpha=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Voltage (V)")
ax3.grid(True)
ax3.set_ylim(-13.5, 13.5)
ax3.legend(loc="upper left")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "03_closed_loop_pid_response.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
