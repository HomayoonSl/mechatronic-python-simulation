import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp

# ============================================================
# 1. MESH GENERATION ENGINE (Industrial Gray)
# ============================================================
def create_box_mesh(width, height, depth):
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    vertices = [
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]
    ]
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
             [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]]
    return np.array(vertices), np.array(faces)

def create_cylinder_mesh_y_axis(radius, thickness, num_slices=40):
    vertices = []
    faces = []
    y_low, y_high = -thickness / 2.0, thickness / 2.0
    angles = np.linspace(0, 2 * np.pi, num_slices, endpoint=False)
    vertices.extend([[0.0, y_low, 0.0], [0.0, y_high, 0.0]])
    
    for alpha in angles:
        vertices.append([radius * np.cos(alpha), y_low, radius * np.sin(alpha)])
        vertices.append([radius * np.cos(alpha), y_high, radius * np.sin(alpha)])

    for i in range(num_slices):
        i_next = (i + 1) % num_slices
        b1, t1 = 2 + 2 * i, 2 + 2 * i + 1
        b2, t2 = 2 + 2 * i_next, 2 + 2 * i_next + 1
        faces.extend([[0, b1, b2], [1, t2, t1], [b1, t1, t2], [b1, t2, b2]])
    return np.array(vertices), np.array(faces)

v_rod, f_rod = create_box_mesh(width=0.015, height=0.01, depth=0.2)
v_fly, f_fly = create_cylinder_mesh_y_axis(radius=0.05, thickness=0.012)

# ============================================================
# 2. TRUE INVERTED PENDULUM DYNAMICS (Uncontrolled Open-Loop)
# ============================================================
Jr, jf, mf, mr, L, g = 0.005, 0.001, 0.2, 0.3, 0.2, 9.81
b0 = 0.015  # Joint friction
bm = 0.002  # Bearing drag

J_p = Jr + mf * (L ** 2)
M_mat = np.array([[J_p + jf, jf], [jf, jf]])
M_inv = np.linalg.inv(M_mat)

def inverted_pendulum_step_dynamics(t, x):
    """
    State: [theta, phi, theta_dot, phi_dot]
    theta = 0 is the unstable upright equilibrium pointing UP.
    """
    theta, phi, theta_dot, phi_dot = x
    
    # Gravitational torque for inverted pendulum: positive feedback when theta deviates from 0
    G_theta = (mr * (L/2) + mf * L) * g * np.sin(theta)
    
    # Pure theoretical Step Torque to perturb the system at t >= 0.5s
    tau_step = 0.01 if t >= 0.5 else 0.0
    
    H_v = np.array([0.0, tau_step])
    C_v = np.array([b0 * theta_dot + 0.005 * np.sign(theta_dot), bm * phi_dot])
    
    accel = M_inv @ (H_v - C_v + np.array([G_theta, 0.0]))
    return [theta_dot, phi_dot, accel[0], accel[1]]

# Total simulated time: 10 seconds. Initial state: Absolutely 0.0 (Perfect Upright)
t_total = 10.0
x0 = [0.0, 0.0, 0.0, 0.0]
num_frames = 250
t_eval = np.linspace(0.0, t_total, num_frames)
sol = solve_ivp(inverted_pendulum_step_dynamics, (0.0, t_total), x0, t_eval=t_eval, method='RK45')

# ============================================================
# 3. HIGH-SPEED ANIMATION & SIGNAL TRACKING GRID
# ============================================================
fig = plt.figure(figsize=(14, 7))

# 3D Render
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_xlim(-0.25, 0.25)
ax3d.set_ylim(-0.25, 0.25)
ax3d.set_zlim(-0.25, 0.25)
ax3d.view_init(elev=10, azim=55)
ax3d.set_title("Inverted Pendulum Open-Loop Step Response")

rod_collection = Poly3DCollection([], facecolors='#888888', edgecolors='#555555', alpha=0.9)
fly_collection = Poly3DCollection([], facecolors='#aaaaaa', edgecolors='#666666', alpha=0.9)
ax3d.add_collection3d(rod_collection)
ax3d.add_collection3d(fly_collection)
pointer_line, = ax3d.plot([], [], [], 'k-', linewidth=4, label="Freewheel Slip Marker")
ax3d.legend(loc="upper left")

# Real-time plots
ax_theta = fig.add_subplot(222)
ax_theta.set_xlim(0, t_total)
ax_theta.set_ylim(-200, 200)
ax_theta.set_ylabel("Angle (deg)")
ax_theta.grid(True)
line_theta, = ax_theta.plot([], [], 'r-', label=r"Pendulum Angle ($\theta$)")
ax_theta.axhline(0, color='black', linestyle=':', alpha=0.4, label="Unstable Upright")
ax_theta.axhline(180, color='blue', linestyle='--', alpha=0.5, label="+180 Hanging Bound")
ax_theta.axhline(-180, color='blue', linestyle='-.', alpha=0.5, label="-180 Hanging Bound")
ax_theta.legend(loc="lower left")

ax_torque = fig.add_subplot(224)
ax_torque.set_xlim(0, t_total)
ax_torque.set_ylim(-0.005, 0.02)
ax_torque.set_xlabel("Time (s)")
ax_torque.set_ylabel("Torque (Nm)")
ax_torque.grid(True)
line_torque, = ax_torque.plot([], [], 'b-', label="Theoretical Step Perturbation")
ax_torque.legend(loc="upper right")

def transform_planar_mesh(vertices, theta, phi, is_flywheel=False):
    # Pure kinematic mapping: theta=0 points straight UP, increasing clockwise around Y axis
    R_theta = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_phi = np.array([
        [np.cos(phi),    0, np.sin(phi)],
        [0,              1, 0],
        [-np.sin(phi),   0, np.cos(phi)]
    ])

    transformed = []
    for v in vertices:
        if is_flywheel:
            v_rot = R_phi @ np.array(v)
            v_shifted = v_rot + np.array([0.0, 0.0, L])  # Mounted at the top tip
            v_final = R_theta @ v_shifted
        else:
            v_shifted = np.array(v) + np.array([0.0, 0.0, L/2]) # Center of mass offset
            v_final = R_theta @ v_shifted
        transformed.append(v_final)
    return np.array(transformed)

t_data, theta_data, torque_data = [], [], []

def update_combined_graphics(frame):
    t = sol.t[frame]
    theta = sol.y[0][frame]
    phi = sol.y[1][frame]
    tau = 0.01 if t >= 0.5 else 0.0
    theta_wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
    theta_deg = np.degrees(theta_wrapped)

    t_data.append(t)
    theta_data.append(theta_deg)
    torque_data.append(tau)

    line_theta.set_data(t_data, theta_data)
    line_torque.set_data(t_data, torque_data)

    curr_v_rod = transform_planar_mesh(v_rod, theta, phi, is_flywheel=False)
    curr_v_fly = transform_planar_mesh(v_fly, theta, phi, is_flywheel=True)
    
    rod_collection.set_verts([[curr_v_rod[idx] for idx in face] for face in f_rod])
    fly_collection.set_verts([[curr_v_fly[idx] for idx in face] for face in f_fly])

    r_p = 0.045
    pt_local = np.array([r_p * np.cos(phi), 0.007, r_p * np.sin(phi)])
    R_theta = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    pt_world_start = R_theta @ np.array([0.0, 0.007, L])
    pt_world_end = R_theta @ (pt_local + np.array([0.0, 0.0, L]))
    
    pointer_line.set_data_3d(
        [pt_world_start[0], pt_world_end[0]],
        [pt_world_start[1], pt_world_end[1]],
        [pt_world_start[2], pt_world_end[2]]
    )
    
    return rod_collection, fly_collection, pointer_line, line_theta, line_torque

# Time-warped rapid visualization
anim = animation.FuncAnimation(fig, update_combined_graphics, frames=len(sol.t), interval=15, blit=False)
#plt.tight_layout()
#plt.show()


# تنظیم نرخ فریم بر ثانیه برای ویدئو (متناسب با گام‌های زمانی شبیه‌سازی)
fps = 15  

# تعریف نویسنده فیلم با استفاده از موتور ffmpeg
writer = animation.FFMpegWriter(
    fps=fps, 
    metadata=dict(artist='Me'), 
    codec='libx264', 
    extra_args=['-pix_fmt', 'yuv420p']
)

print("Inverting pendulum simulation: Writing to MP4 file using ffmpeg...")

# ذخیره انیمیشن در مسیر جاری با کیفیت بالا (DPI=150)
anim.save('inverted_pendulum_step_response.mp4', writer=writer, dpi=150)

print("Animation successfully saved as 'inverted_pendulum_step_response.mp4'")
