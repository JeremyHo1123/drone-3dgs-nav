"""
Fly the drone in a 3DGS scene: take off, orbit a target, land,
and write a side-by-side video of the onboard view + top-down trajectory.

The purpose is to confirm three things by eye before formally hooking it into grad_nav training:
  1. The dynamics parameters are sane (failing to track the circle means thrust or gains are off)
  2. The camera intrinsics/extrinsics are correct (things appear where they should in the image)
  3. The scene can be loaded by grad_nav's renderer

The flight is not a geometric trajectory; it really runs QuadrotorSimulator: the controller
outputs only four numbers, "body rates x3 + thrust", exactly the same interface as the policy
output, and goes through exactly the same action processing as the env file (clip to [-1,1],
scaling, first-order delay).

Three easy-to-hit coordinate pitfalls, all handled by this tool:
  1. The docstring of quaternion_to_rotation_matrix in utils/gs_local.py says (w,x,y,z),
     but the code actually treats it as (x,y,z,w). Callers pass (x,y,z,w); the code is authoritative.
  2. The env negates y/z of the "position" but not of the "quaternion"; the two are inconsistent.
     Net result: camera position = torso_pos (the two negations cancel),
                 camera heading = (cos yaw, -sin yaw, 0), **yaw is negated**.
     So to face world azimuth h, set the torso yaw to -h.
  3. config.yml stores relative paths, so chdir to the workspace before loading.

Usage:
  python tools/fly_orbit_preview.py                       # default: orbit the box in scene04
  python tools/fly_orbit_preview.py --no-render           # trajectory tracking only, no rendering (fast)
  python tools/fly_orbit_preview.py --radius 2.5 --speed 0.5
"""
import argparse
import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*torch.cross without specifying.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAD_NAV = PROJECT_ROOT / "repos/grad_nav"
WORKSPACE = PROJECT_ROOT / "repos/SousVide/gsplats/workspace"

G = 9.81
THRUST_CMD_MAX = 0.5      # env clamps the thrust command to [0, 0.5]
BR_SCALE = 0.5            # env body rate scale and limit (rad/s)
_FONT = None              # status bar font (lazy-loaded)


def load_dynamics():
    """Load only the dynamics module itself, avoiding the torchvision import error in envs/__init__.py."""
    p = GRAD_NAV / "envs/assets/quadrotor_dynamics_advanced.py"
    spec = importlib.util.spec_from_file_location("quadrotor_dynamics", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.QuadrotorSimulator


def read_env_params(env_file="envs/drone_vla_multi_map.py"):
    """Parse the dynamics parameters directly from the env file, so the preview and training use the same numbers."""
    import re
    s = (GRAD_NAV / env_file).read_text()
    def num(n):
        return float(re.search(rf"self\.{n} = ([\d.]+)", s).group(1))
    def vec(n):
        return [float(v) for v in
                re.search(rf"self\.{n} = \[([^\]]+)\]", s).group(1).split(",")]
    return dict(
        mass=num("min_mass") + 0.5 * num("mass_range"),
        max_thrust=num("min_thrust") + 0.5 * num("thrust_range"),
        inertia=vec("init_inertia"), kp=vec("init_kp"), kd=vec("init_kd"),
        br_delay=num("br_delay_factor"), thrust_delay=num("thrust_delay_factor"),
    )


def quat_to_R(q):
    """q = (x, y, z, w) -> 3x3 rotation matrix (body -> world)."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class OrbitController:
    """Cascaded controller: position -> acceleration -> (thrust, desired attitude) -> body-rate command.

    Outputs normalized actions in [-1,1], the same interface as the policy output.
    """

    def __init__(self, mass, max_thrust, kp_pos=2.2, kd_pos=2.6, k_att=7.0):
        self.m, self.max_thrust = mass, max_thrust
        self.kp_pos, self.kd_pos, self.k_att = kp_pos, kd_pos, k_att

    def __call__(self, pos, vel, quat, p_ref, v_ref, yaw_ref, yawrate_ref):
        R = quat_to_R(quat)

        # --- Outer loop: position/velocity error -> desired acceleration -> desired thrust vector ---
        a_des = self.kp_pos * (p_ref - pos) + self.kd_pos * (v_ref - vel)
        f_des = self.m * (a_des + np.array([0.0, 0.0, G]))
        f_norm = np.linalg.norm(f_des)
        if f_norm < 1e-6:
            f_des, f_norm = np.array([0.0, 0.0, 1e-6]), 1e-6

        # Thrust is the projection onto the current body z axis (avoids overshoot when tilted)
        thrust_N = float(np.dot(f_des, R[:, 2]))
        thrust_norm = np.clip(thrust_N / self.max_thrust, 0.0, THRUST_CMD_MAX)

        # --- Inner loop: build the desired attitude from desired thrust direction and desired yaw ---
        b3 = f_des / f_norm
        b1_c = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0])
        b2 = np.cross(b3, b1_c)
        n2 = np.linalg.norm(b2)
        if n2 < 1e-6:                      # degenerate case when b1_c and b3 are nearly parallel
            b1_c = np.array([1.0, 0.0, 0.0]); b2 = np.cross(b3, b1_c); n2 = np.linalg.norm(b2)
        b2 /= n2
        b1 = np.cross(b2, b3)
        R_des = np.column_stack([b1, b2, b3])

        # Attitude error -> desired body rates (body frame)
        # ★ Do not use e_R = 0.5*vee(R_des^T R - R^T R_des): that formula gives zero when
        #   the error is exactly 180 degrees, and the drone gets stuck at an unstable equilibrium and does not turn at all.
        #   Use the rotation vector instead; scipy guarantees the shortest path, angle in [0, pi].
        from scipy.spatial.transform import Rotation
        rotvec = Rotation.from_matrix(R.T @ R_des).as_rotvec()   # body frame
        omega_des = self.k_att * rotvec
        omega_des += R.T @ np.array([0.0, 0.0, yawrate_ref])   # yaw rate feedforward

        # --- Convert to normalized [-1,1] actions (the env then rescales and applies the delay) ---
        a_br = np.clip(omega_des / BR_SCALE, -1.0, 1.0)
        a_th = np.clip(thrust_norm / 0.25 - 1.0, -1.0, 1.0)
        return np.concatenate([a_br, [a_th]])


def build_trajectory(center, radius, height, speed, dt, takeoff_time, start_z,
                     hover_time=1.0, landing_time=5.0):
    """Return a per-step sequence of (p_ref, v_ref, yaw_ref, yawrate_ref, phase).

    Four phases: takeoff -> one orbit -> hover in place -> landing.
    Takeoff and landing both use a cosine velocity profile; vertical velocity is 0 at both ends,
    so there are no acceleration jumps.
    Landing is deliberately slower than takeoff (default 5 s vs 4 s), matching real-flight practice.

    yaw_ref already includes the negation from coordinate pitfall 2 and can be fed straight to the controller.
    """
    steps_to, omega = int(takeoff_time / dt), speed / radius
    steps_orbit = int((2 * math.pi / omega) / dt)
    out = []
    p0 = np.array([center[0] + radius, center[1], start_z])

    for i in range(steps_to):                       # takeoff: smooth climb to orbit height
        s = (i + 1) / steps_to
        s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)          # cosine smoothing
        z = start_z + (height - start_z) * s_smooth
        vz = (height - start_z) * 0.5 * math.pi * math.sin(math.pi * s) / takeoff_time
        p = np.array([p0[0], p0[1], z])
        head = math.atan2(center[1] - p[1], center[0] - p[0])
        out.append((p, np.array([0.0, 0.0, vz]), -head, 0.0, "takeoff"))

    for i in range(steps_orbit):                    # orbit: camera always faces the center
        a = omega * dt * (i + 1)
        p = np.array([center[0] + radius * math.cos(a),
                      center[1] + radius * math.sin(a), height])
        v = np.array([-radius * omega * math.sin(a),
                      radius * omega * math.cos(a), 0.0])
        head = math.atan2(center[1] - p[1], center[0] - p[0])
        out.append((p, v, -head, -omega, "orbit"))   # both yaw and yawrate are negated

    # the orbit ends exactly back at the start point, where the landing happens; heading keeps facing the target
    p_end = out[-1][0].copy()
    head_end = math.atan2(center[1] - p_end[1], center[0] - p_end[0])
    zero = np.zeros(3)

    for _ in range(int(hover_time / dt)):        # hover: makes the phase change readable in the video
        out.append((p_end.copy(), zero.copy(), -head_end, 0.0, "hover"))

    steps_land = int(landing_time / dt)
    for i in range(steps_land):                  # landing: cosine descent, symmetric to takeoff
        s = (i + 1) / steps_land
        s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)
        z = height + (start_z - height) * s_smooth
        vz = (start_z - height) * 0.5 * math.pi * math.sin(math.pi * s) / landing_time
        out.append((np.array([p_end[0], p_end[1], z]),
                    np.array([0.0, 0.0, vz]), -head_end, 0.0, "landing"))

    # Settle for a while after touchdown. Without this, the video stops while the controller
    # has not yet converged (measured: the end point is about 7 cm below the target height).
    # The position loop has no integral term and converges in about 3 s, so this segment needs a full 2 s to show it settled.
    p_gnd = np.array([p_end[0], p_end[1], start_z])
    for _ in range(int(2.0 / dt)):
        out.append((p_gnd.copy(), zero.copy(), -head_end, 0.0, "touchdown"))
    return out


def make_map_background(ply_path, center, radius, extent, size):
    """Draw the point-cloud top view + orbit path into one static background image (done once)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    pts = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
    cols = np.asarray(o3d.io.read_point_cloud(str(ply_path)).colors)
    m = ((np.abs(pts[:, 0] - center[0]) < extent) &
         (np.abs(pts[:, 1] - center[1]) < extent) &
         (pts[:, 2] > 0.05) & (pts[:, 2] < 2.6))
    p, c = pts[m], (cols[m] if len(cols) else None)

    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#14161a")
    fig.patch.set_facecolor("#14161a")
    ax.scatter(p[:, 0], p[:, 1], s=1.6, c=c if c is not None else "0.6",
               linewidths=0, alpha=0.85)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(center[0] + radius * np.cos(th), center[1] + radius * np.sin(th),
            "--", color="#38b6ff", lw=1.8, alpha=0.95)
    ax.plot(*center, "x", color="#ff4040", ms=13, mew=3)
    # 1 m scale bar
    x0, y0 = center[0] - extent * 0.92, center[1] - extent * 0.90
    ax.plot([x0, x0 + 1.0], [y0, y0], "-", color="w", lw=2.5)
    ax.text(x0 + 0.5, y0 + extent * 0.03, "1 m", color="w", ha="center", fontsize=9)
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_aspect("equal"); ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def world_to_px(p, center, extent, size):
    """World coordinates -> background pixel coordinates (image y axis points down, so flip it)."""
    u = (p[0] - (center[0] - extent)) / (2 * extent) * size[0]
    v = size[1] - (p[1] - (center[1] - extent)) / (2 * extent) * size[1]
    return float(u), float(v)


def compose(fpv, bg, trail, pos, world_head, t, phase, extent, center, info):
    """Put the onboard view and top view side by side; draw the trail, the drone and the status bar."""
    from PIL import Image, ImageDraw
    mp = Image.fromarray(bg.copy())
    d = ImageDraw.Draw(mp)
    size = mp.size

    if len(trail) > 1:
        d.line([world_to_px(q, center, extent, size) for q in trail],
               fill=(255, 210, 0), width=2)

    u, v = world_to_px(pos, center, extent, size)
    # safety envelope (half wheelbase + prop radius = 0.465 m), drawn to scale as a circle
    r_px = 0.465 / (2 * extent) * size[0]
    d.ellipse([u - r_px, v - r_px, u + r_px, v + r_px], outline=(0, 255, 120), width=2)
    # heading arrow (world_head is the true world azimuth, not the torso yaw)
    L = r_px * 2.1
    d.line([(u, v), (u + L * math.cos(world_head), v - L * math.sin(world_head))],
           fill=(0, 255, 120), width=3)

    from PIL import ImageFont
    global _FONT
    if _FONT is None:
        try:                       # PIL default font has no CJK glyphs; they would all render as boxes
            _FONT = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
        except OSError:
            _FONT = ImageFont.load_default()

    fpv_img = Image.fromarray(fpv)
    W, H = fpv_img.width + size[0], max(fpv_img.height, size[1]) + 30
    canvas = Image.new("RGB", (W, H), (20, 22, 26))
    canvas.paste(fpv_img, (0, 0))
    canvas.paste(mp, (fpv_img.width, 0))
    ImageDraw.Draw(canvas).text(
        (10, H - 23),
        f"t={t:5.1f}s  {phase}  pos=({pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:4.2f})  "
        f"heading={math.degrees(world_head):6.1f}°  {info}",
        fill=(228, 228, 228), font=_FONT)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser(description="Orbit flight preview inside a 3DGS scene")
    ap.add_argument("--config", default=None, help="nerfstudio config.yml (relative to workspace)")
    ap.add_argument("--ply", default="exports/scene04_dense.ply")
    ap.add_argument("--center", type=float, nargs=2, default=[-0.77, -0.01],
                    help="x y of the orbit center (target object)")
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--height", type=float, default=1.4)
    ap.add_argument("--start-z", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=0.6, help="orbit linear speed m/s")
    ap.add_argument("--takeoff-time", type=float, default=4.0)
    ap.add_argument("--hover-time", type=float, default=1.0,
                    help="seconds to hover in place after the orbit before landing")
    ap.add_argument("--landing-time", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.05, help="outer-loop decision period (env uses 0.05)")
    ap.add_argument("--freq", type=float, default=200.0, help="inner-loop integration frequency")
    ap.add_argument("--extent", type=float, default=6.0, help="half side length of the top view, m")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="notes/scene04_orbit.mp4")
    ap.add_argument("--no-render", action="store_true", help="trajectory tracking only, no 3DGS rendering")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N steps (for debugging)")
    a = ap.parse_args()

    # ---------- dynamics ----------
    P = read_env_params()
    QuadrotorSimulator = load_dynamics()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = QuadrotorSimulator(
        mass=torch.full((1,), P["mass"]),
        inertia=torch.diag(torch.tensor(P["inertia"])).unsqueeze(0),
        link_length=0.30,
        Kp=torch.tensor(P["kp"]).unsqueeze(0), Kd=torch.tensor(P["kd"]).unsqueeze(0),
        freq=a.freq, max_thrust=torch.full((1,), P["max_thrust"]),
        total_time=a.dt, rotor_noise_std=0.0, br_noise_std=0.0,
        drag_coeff=0.5, cross_area=0.25)
    print(f"Dynamics parameters (read from envs/drone_vla_multi_map.py):")
    print(f"  mass {P['mass']:.2f} kg   max_thrust parameter {P['max_thrust']:.1f} N "
          f"(actual limit {0.5*P['max_thrust']:.1f} N, thrust-to-weight ratio "
          f"{0.5*P['max_thrust']/(P['mass']*G):.2f})")
    print(f"  inertia {P['inertia']}   Kp {P['kp']}")
    print(f"  delay br={P['br_delay']}  thrust={P['thrust_delay']}")

    ctrl = OrbitController(P["mass"], P["max_thrust"])
    traj = build_trajectory(np.array(a.center), a.radius, a.height,
                            a.speed, a.dt, a.takeoff_time, a.start_z,
                            a.hover_time, a.landing_time)
    if a.limit:
        traj = traj[:a.limit]
    print(f"\nTrajectory {len(traj)} steps = {len(traj)*a.dt:.1f} s "
          f"(takeoff {a.takeoff_time:.0f}s + one orbit {2*math.pi*a.radius/a.speed:.1f}s "
          f"+ hover {a.hover_time:.0f}s + landing {a.landing_time:.0f}s + touchdown 2s)")
    print(f"  orbit center ({a.center[0]:.2f}, {a.center[1]:.2f})  radius {a.radius} m  "
          f"height {a.height} m  linear speed {a.speed} m/s")
    print(f"  required yaw rate {a.speed/a.radius:.3f} rad/s  "
          f"(env limit {BR_SCALE} rad/s)")

    # ---------- 3DGS ----------
    gs = None
    if not a.no_render:
        sys.path.insert(0, str(GRAD_NAV))
        os.chdir(WORKSPACE)                      # config.yml stores relative paths
        from utils.gs_local import GS
        cfg = a.config or sorted(
            (WORKSPACE / "outputs/scene04/splatfacto").glob("*/config.yml"))[-1]
        cfg = Path(cfg)
        print(f"\nLoading 3DGS: {cfg}")
        gs = GS(cfg.relative_to(WORKSPACE) if cfg.is_absolute() else cfg,
                width=640, height=360, res=1.0)
        print(f"  Gaussians: {gs.pipeline.model.means.shape[0]:,}")

    bg = None
    if not a.no_render:
        bg = make_map_background(WORKSPACE / a.ply, np.array(a.center),
                                 a.radius, a.extent, (640, 360))

    # ---------- main loop ----------
    pos = np.array([a.center[0] + a.radius, a.center[1], a.start_z])
    vel = np.zeros(3)
    # The initial yaw is set directly to the trajectory's first reference (the drone already faces the target while on the ground).
    # Taking off from yaw=0 against yaw_ref=-180 degrees would give an attitude error of exactly 180 degrees,
    # which is an unstable equilibrium of the attitude controller and makes the orbit phase diverge.
    yaw0 = traj[0][2]
    quat = np.array([0.0, 0.0, math.sin(yaw0 / 2), math.cos(yaw0 / 2)])  # (x,y,z,w)
    omega = np.zeros(3)
    prev_br, prev_th = np.zeros(3), 0.0
    trail, frames, errs = [], [], []

    t_pos = torch.tensor([pos], dtype=torch.float32, device=dev)
    t_vel = torch.tensor([vel], dtype=torch.float32, device=dev)
    t_q = torch.tensor([[math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]],
                       dtype=torch.float32, device=dev)   # (w,x,y,z)
    t_w = torch.tensor([omega], dtype=torch.float32, device=dev)

    import time as _t
    t_start = _t.time()
    for k, (p_ref, v_ref, yaw_ref, yawrate_ref, phase) in enumerate(traj):
        act = ctrl(pos, vel, quat, p_ref, v_ref, yaw_ref, yawrate_ref)

        # ↓ exactly the same action processing as the env file (clip -> scale -> first-order delay -> clip again)
        br = P["br_delay"] * (np.clip(act[:3], -1, 1) * BR_SCALE) + \
             (1 - P["br_delay"]) * prev_br
        br = np.clip(br, -BR_SCALE, BR_SCALE)
        th = P["thrust_delay"] * ((np.clip(act[3], -1, 1) + 1) * 0.25) + \
             (1 - P["thrust_delay"]) * prev_th
        prev_br, prev_th = br, th

        with torch.no_grad():
            t_pos, t_vel, t_w, t_q, _, _ = sim.run_simulation(
                t_pos, t_vel, t_q, t_w,
                (torch.tensor([br], dtype=torch.float32, device=dev),
                 torch.tensor([th], dtype=torch.float32, device=dev)))
        pos = t_pos[0].cpu().numpy()
        vel = t_vel[0].cpu().numpy()
        omega = t_w[0].cpu().numpy()
        wxyz = t_q[0].cpu().numpy()
        quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])   # -> (x,y,z,w)

        errs.append(np.linalg.norm(pos - p_ref))
        trail.append(pos.copy())

        if gs is not None:
            gs_pos = torch.tensor([[pos[0], -pos[1], -pos[2]]],
                                  dtype=torch.float32, device=gs.device)
            gs_pose = torch.cat([gs_pos, torch.zeros([1, 3], device=gs.device),
                                 torch.tensor([quat], dtype=torch.float32,
                                              device=gs.device)], dim=-1)
            with torch.no_grad():
                _, img = gs.render(gs_pose)
            fpv = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            world_head = -math.atan2(2 * (quat[3]*quat[2] + quat[0]*quat[1]),
                                     1 - 2 * (quat[1]**2 + quat[2]**2))
            frames.append(compose(fpv, bg, trail, pos, world_head, k * a.dt, phase,
                                  a.extent, np.array(a.center),
                                  f"tracking error={errs[-1]*100:4.1f}cm"))
            if k % 40 == 0:
                print(f"  [{k:4d}/{len(traj)}] t={k*a.dt:5.1f}s {phase} "
                      f"error {errs[-1]*100:5.1f} cm  "
                      f"({_t.time()-t_start:.0f}s elapsed)", flush=True)

    errs = np.array(errs)
    phases = [t[4] for t in traj]
    print(f"\nTrajectory tracking error:")
    for name in ("takeoff", "orbit", "hover", "landing", "touchdown"):
        idx = [i for i, ph in enumerate(phases) if ph == name]
        if not idx:
            continue
        e = errs[idx]
        print(f"  {name}  {len(idx):>4} steps  mean {e.mean()*100:5.1f} cm   "
              f"max {e.max()*100:5.1f} cm")
    print(f"  overall    mean {errs.mean()*100:5.1f} cm   max {errs.max()*100:5.1f} cm")
    print(f"  final position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
          f"(start {traj[0][0][0]:.2f}, {traj[0][0][1]:.2f}, {a.start_z:.2f})")

    if frames:
        out = PROJECT_ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        imageio.mimwrite(out, frames, fps=a.fps, quality=8,
                         macro_block_size=1)
        print(f"\n✓ Video written: {out}")
        print(f"  {len(frames)} frames @ {a.fps} fps = {len(frames)/a.fps:.1f} s, "
              f"{frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
