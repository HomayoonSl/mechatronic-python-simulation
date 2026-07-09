import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp

# ============================================================
# 1. MESH GENERATION ENGINE
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
# 2. PHYSICAL PARAMETERS & EXACT SATURATION DYNAMICS
# ============================================================
Jr, jf, mf, mr, Lr, L, g = 0.005, 0.001, 0.2, 0.3, 0.1, 0.2, 9.81
b0, bm = 0.001, 0.0005
RA, LA, KT, KE = 0.2, 0.0004, 0.038, 0.03915
I_MAX, V_MAX = 12.0, 12.0
SPEED_MAX_RPM = 2550.0
SPEED_MAX_RAD = SPEED_MAX_RPM * (2.0 * np.pi / 60.0)

# تنظیم گین‌های پلات ارسالی برای شبیه‌سازی دقیق رفتار نوسانی اشباع کادر
Kp, Ki, Kd = 240.0, 110.0, 14.5

J_p = Jr + mf * (L ** 2)
M_mat = np.array([[J_p + jf, jf], [jf, jf]])
M_inv = np.linalg.inv(M_mat)

def saturated_motor_dynamics(t, x):
    theta, phi, theta_dot, phi_dot, int_theta, i_motor = x

    # کنترلر با اشباع ولتاژ و آنتی‌وینداپ
    V_cmd = Kp * theta + Ki * int_theta + Kd * theta_dot
    Va = np.clip(V_cmd, -V_MAX, V_MAX)
    
    d_int = 0.0 if (abs(V_cmd) >= V_MAX and np.sign(theta) == np.sign(V_cmd)) else theta

    # دینامیک الکتریکی موتور
    di = (Va - RA * i_motor - KE * phi_dot) / LA
    if (i_motor >= I_MAX and di > 0) or (i_motor <= -I_MAX and di < 0):
        di = 0.0

    # اعمال محدودیت سخت افزاری سرعت فلایویل (اشباع سرعت فیزیکی)
    if abs(phi_dot) >= SPEED_MAX_RAD and np.sign(phi_dot) == np.sign(i_motor):
        tau_motor = 0.0  # موتور در سرعت ماکزیمم دیگر گشتاور شتاب‌دهنده تولید نمی‌کند
    else:
        tau_motor = KT * i_motor

    H_v = np.array([0.0, tau_motor])
    C_v = np.array([b0 * theta_dot, bm * phi_dot])
    G_theta = np.array([-(mr * Lr + mf * L) * g * np.sin(theta), 0.0])

    # اعمال اغتشاش پله‌ای ماندگار 0.02 نیوتون‌متر در ثانیه 5.0
    dist = np.array([0.0, 0.0])
    if t >= 5.0:
        dist[0] = 0.02

    accel = M_inv @ (H_v - C_v - G_theta + dist)
    return [theta_dot, phi_dot, accel[0], accel[1], d_int, di]

# تنظیمات شبیه‌سازی بیست ثانیه‌ای طبق درخواست شما
t_total = 20.0
fps_target = 15
num_frames = int(t_total * fps_target)
t_eval = np.linspace(0.0, t_total, num_frames)

x0 = [np.radians(10.0), 0.0, 0.0, 0.0, 0.0, 0.0]
sol = solve_ivp(saturated_motor_dynamics, (0.0, t_total), x0, t_eval=t_eval, method='RK45', rtol=1e-6)

# ============================================================
# 3. 4-SUBPLOT LAYOUT (Exactly Matching Your Image Format)
# ============================================================
fig = plt.figure(figsize=(16, 10))

# سمت چپ: انیمیشن سه بعدی صلب
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_xlim(-0.25, 0.25)
ax3d.set_ylim(-0.25, 0.25)
ax3d.set_zlim(-0.25, 0.25)
ax3d.view_init(elev=12, azim=55)
ax3d.set_title("Closed-Loop MultiBody Animation (Velocity Saturation)")

rod_collection = Poly3DCollection([], facecolors='#2ca02c', edgecolors='#1b5e20', alpha=0.8)
fly_collection = Poly3DCollection([], facecolors='#9467bd', edgecolors='#4a148c', alpha=0.8)
ax3d.add_collection3d(rod_collection)
ax3d.add_collection3d(fly_collection)
pointer_line, = ax3d.plot([], [], [], 'r-', linewidth=3, label="Flywheel Reference Pointer")

# سمت راست: نمودارهای ۴ گانه خطی فرستاده شده
ax_theta = fig.add_subplot(422)
ax_omega = fig.add_subplot(424)
ax_volt  = fig.add_subplot(426)
ax_curr  = fig.add_subplot(428)

# تنظیمات چارت زاویه پاندول
ax_theta.set_xlim(0, t_total)
ax_theta.set_ylim(-200, 200)
ax_theta.set_ylabel("Angle (deg)")
ax_theta.grid(True)
line_theta, = ax_theta.plot([], [], 'g-', linewidth=2, label="Pendulum Angle")
ax_theta.axvline(5.0, color="r", linestyle="--", alpha=0.6, label="Step Disturbance (0.02 Nm)")
ax_theta.legend(loc="upper left")

# تنظیمات چارت سرعت فلایویل
ax_omega.set_xlim(0, t_total)
ax_omega.set_ylim(-3000, 3000)
ax_omega.set_ylabel("Speed (RPM)")
ax_omega.grid(True)
line_omega, = ax_omega.plot([], [], 'm-', linewidth=2, label="Flywheel Speed")
ax_omega.axhline(SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.6, label="Rated Max Speed")
ax_omega.axhline(-SPEED_MAX_RPM, color="k", linestyle=":", alpha=0.6)
ax_omega.legend(loc="upper left")

# تنظیمات چارت ولتاژ موتور
ax_volt.set_xlim(0, t_total)
ax_volt.set_ylim(-14, 14)
ax_volt.set_ylabel("Voltage (V)")
ax_volt.grid(True)
line_volt, = ax_volt.plot([], [], 'b-', linewidth=2, label="Motor Voltage")
ax_volt.axhline(V_MAX, color="k", linestyle=":", alpha=0.5)
ax_volt.axhline(-V_MAX, color="k", linestyle=":", alpha=0.5)
ax_volt.legend(loc="upper left")

# تنظیمات چارت جریان موتور
ax_curr.set_xlim(0, t_total)
ax_curr.set_ylim(-14, 14)
ax_curr.set_xlabel("Time (s)")
ax_curr.set_ylabel("Current (A)")
ax_curr.grid(True)
line_curr, = ax_curr.plot([], [], 'c-', linewidth=2, label="Motor Current")
ax_curr.axhline(I_MAX, color="k", linestyle=":", alpha=0.5, label="Current Limit")
ax_curr.axhline(-I_MAX, color="k", linestyle=":", alpha=0.5)
ax_curr.legend(loc="upper left")

def transform_planar_mesh(vertices, theta, phi, is_flywheel=False):
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
            v_shifted = v_rot + np.array([0.0, 0.0, L])
            v_final = R_theta @ v_shifted
        else:
            v_shifted = np.array(v) + np.array([0.0, 0.0, L/2])
            v_final = R_theta @ v_shifted
        transformed.append(v_final)
    return np.array(transformed)

t_data, theta_data, omega_data, volt_data, curr_data = [], [], [], [], []

def update_frame(frame):
    t = sol.t[frame]
    theta = sol.y[0][frame]
    phi = sol.y[1][frame]
    theta_dot = sol.y[2][frame]
    phi_dot_rpm = sol.y[3][frame] * (60.0 / (2.0 * np.pi))
    int_theta = sol.y[4][frame]
    i_motor = sol.y[5][frame]

    V_cmd = Kp * theta + Ki * int_theta + Kd * theta_dot
    Va = np.clip(V_cmd, -V_MAX, V_MAX)

    # تبدیل امپلیمر دایره‌ای زاویه بین -180 تا 180 درجه برای رسم دقیق ناپایداری و سقوط
    theta_wrapped = (np.degrees(theta) + 180) % 360 - 180

    t_data.append(t)
    theta_data.append(theta_wrapped)
    omega_data.append(phi_dot_rpm)
    volt_data.append(Va)
    curr_data.append(i_motor)

    line_theta.set_data(t_data, theta_data)
    line_omega.set_data(t_data, omega_data)
    line_volt.set_data(t_data, volt_data)
    line_curr.set_data(t_data, curr_data)

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
    
    pointer_line.set_data_3d([pt_world_start[0], pt_world_end[0]], 
                             [pt_world_start[1], pt_world_end[1]], 
                             [pt_world_start[2], pt_world_end[2]])
    
    return rod_collection, fly_collection, pointer_line, line_theta, line_omega, line_volt, line_curr

# کامپایل فیلم با نرخ ۱۵ فریم بر ثانیه در فایل درخواستی پروژه
anim = animation.FuncAnimation(fig, update_frame, frames=len(sol.t), interval=1000/fps_target, blit=False)
plt.tight_layout()

# ذخیره خروجی نمودار استاتیک نهایی طبق نامگذاری اعلامی شما
plt.savefig('full_system_multibody_0.02.png', dpi=300, bbox_inches="tight")

output_filename = 'full_system_multibody_0.02.mp4'
writer = animation.FFMpegWriter(fps=fps_target, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])

print(f"Rendering 20s simulation video ({len(sol.t)} frames) with explicit velocity saturation limit...")
anim.save(output_filename, writer=writer, dpi=120)
print(f"Saved animation video as '{output_filename}' and structural plot as 'full_system_multibody_0.02.png'")
