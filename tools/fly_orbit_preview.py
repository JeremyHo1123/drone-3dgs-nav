"""
讓無人機在 3DGS 場景裡起飛、環繞目標、降落，
輸出機載視角 + 俯視軌跡的並排影片。

用途是在正式接進 grad_nav 訓練之前，肉眼確認三件事：
  1. 動力學參數合理（追不上圓周就代表推力或增益有問題）
  2. 相機內外參正確（畫面該有的東西在該在的位置）
  3. 場景能被 grad_nav 的渲染器載入

飛行不是幾何軌跡，是真的跑 QuadrotorSimulator：控制器只輸出
「角速度 x3 + 推力」四個數字，跟 policy 的輸出介面完全相同，
並且走與環境檔一模一樣的動作處理（clip 到 [-1,1]、縮放、一階延遲）。

三個容易踩的座標陷阱，本工具已處理：
  1. utils/gs_local.py 的 quaternion_to_rotation_matrix docstring 寫 (w,x,y,z)，
     但程式碼實際當成 (x,y,z,w)。呼叫端傳的是 (x,y,z,w)，以程式碼為準。
  2. 環境對「位置」做 y/z 反號、對「四元數」沒做，兩者處理不一致。
     淨結果是: 相機位置 = torso_pos（兩次反號抵消），
               相機朝向 = (cos yaw, -sin yaw, 0)，**yaw 被反號**。
     所以要朝向某個世界方位角 h，torso 的 yaw 要填 -h。
  3. config.yml 存的是相對路徑，載入前必須 chdir 到 workspace。

用法:
  python tools/fly_orbit_preview.py                       # 預設繞 scene04 的箱子
  python tools/fly_orbit_preview.py --no-render           # 只測軌跡追蹤，不渲染（快）
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
THRUST_CMD_MAX = 0.5      # env 把推力指令壓在 [0, 0.5]
BR_SCALE = 0.5            # env 的 body rate 縮放與上限 (rad/s)
_FONT = None              # 狀態列字型（延遲載入）


def load_dynamics():
    """只載入動力學模組本身，避開 envs/__init__.py 的 torchvision 匯入錯誤。"""
    p = GRAD_NAV / "envs/assets/quadrotor_dynamics_advanced.py"
    spec = importlib.util.spec_from_file_location("quadrotor_dynamics", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.QuadrotorSimulator


def read_env_params(env_file="envs/drone_vla_multi_map.py"):
    """直接從環境檔解析動力學參數，確保預覽與訓練用的是同一組數字。"""
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
    """q = (x, y, z, w) -> 3x3 旋轉矩陣（機體 -> 世界）。"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class OrbitController:
    """串級控制器: 位置 -> 加速度 -> (推力, 期望姿態) -> 角速度指令。

    輸出的是 [-1,1] 的正規化動作，與 policy 的輸出介面相同。
    """

    def __init__(self, mass, max_thrust, kp_pos=2.2, kd_pos=2.6, k_att=7.0):
        self.m, self.max_thrust = mass, max_thrust
        self.kp_pos, self.kd_pos, self.k_att = kp_pos, kd_pos, k_att

    def __call__(self, pos, vel, quat, p_ref, v_ref, yaw_ref, yawrate_ref):
        R = quat_to_R(quat)

        # --- 外環: 位置/速度誤差 -> 期望加速度 -> 期望推力向量 ---
        a_des = self.kp_pos * (p_ref - pos) + self.kd_pos * (v_ref - vel)
        f_des = self.m * (a_des + np.array([0.0, 0.0, G]))
        f_norm = np.linalg.norm(f_des)
        if f_norm < 1e-6:
            f_des, f_norm = np.array([0.0, 0.0, 1e-6]), 1e-6

        # 推力取在目前機體 z 軸上的投影（傾斜時才不會過衝）
        thrust_N = float(np.dot(f_des, R[:, 2]))
        thrust_norm = np.clip(thrust_N / self.max_thrust, 0.0, THRUST_CMD_MAX)

        # --- 內環: 由期望推力方向與期望 yaw 組出期望姿態 ---
        b3 = f_des / f_norm
        b1_c = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0])
        b2 = np.cross(b3, b1_c)
        n2 = np.linalg.norm(b2)
        if n2 < 1e-6:                      # b1_c 與 b3 幾乎平行時的退化情況
            b1_c = np.array([1.0, 0.0, 0.0]); b2 = np.cross(b3, b1_c); n2 = np.linalg.norm(b2)
        b2 /= n2
        b1 = np.cross(b2, b3)
        R_des = np.column_stack([b1, b2, b3])

        # 姿態誤差 -> 期望角速度（機體座標）
        # ★ 不能用 e_R = 0.5*vee(R_des^T R - R^T R_des)：那個公式在誤差
        #   剛好 180 度時算出來是零，無人機會卡在不穩定平衡上完全不轉。
        #   改用旋轉向量，scipy 保證取最短路徑、角度落在 [0, pi]。
        from scipy.spatial.transform import Rotation
        rotvec = Rotation.from_matrix(R.T @ R_des).as_rotvec()   # 機體座標
        omega_des = self.k_att * rotvec
        omega_des += R.T @ np.array([0.0, 0.0, yawrate_ref])   # yaw 速度前饋

        # --- 轉成 [-1,1] 正規化動作（env 之後會再縮放並套延遲）---
        a_br = np.clip(omega_des / BR_SCALE, -1.0, 1.0)
        a_th = np.clip(thrust_norm / 0.25 - 1.0, -1.0, 1.0)
        return np.concatenate([a_br, [a_th]])


def build_trajectory(center, radius, height, speed, dt, takeoff_time, start_z,
                     hover_time=1.0, landing_time=5.0):
    """回傳 (p_ref, v_ref, yaw_ref, yawrate_ref, phase) 逐步序列。

    四個階段: 起飛 -> 環繞一圈 -> 原地懸停 -> 降落。
    起飛與降落都用餘弦速度曲線，頭尾的垂直速度都是 0，不會有加速度突跳。
    降落刻意比起飛慢（預設 5 秒 vs 4 秒），與真機的操作習慣一致。

    yaw_ref 已含座標陷阱 2 的反號，可直接餵給控制器。
    """
    steps_to, omega = int(takeoff_time / dt), speed / radius
    steps_orbit = int((2 * math.pi / omega) / dt)
    out = []
    p0 = np.array([center[0] + radius, center[1], start_z])

    for i in range(steps_to):                       # 起飛: 平滑爬升到環繞高度
        s = (i + 1) / steps_to
        s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)          # 餘弦平滑
        z = start_z + (height - start_z) * s_smooth
        vz = (height - start_z) * 0.5 * math.pi * math.sin(math.pi * s) / takeoff_time
        p = np.array([p0[0], p0[1], z])
        head = math.atan2(center[1] - p[1], center[0] - p[0])
        out.append((p, np.array([0.0, 0.0, vz]), -head, 0.0, "起飛"))

    for i in range(steps_orbit):                    # 環繞: 相機始終對著中心
        a = omega * dt * (i + 1)
        p = np.array([center[0] + radius * math.cos(a),
                      center[1] + radius * math.sin(a), height])
        v = np.array([-radius * omega * math.sin(a),
                      radius * omega * math.cos(a), 0.0])
        head = math.atan2(center[1] - p[1], center[0] - p[0])
        out.append((p, v, -head, -omega, "環繞"))   # yaw 與 yawrate 都要反號

    # 環繞剛好繞回起點，降落就在該處進行；朝向維持對著目標物
    p_end = out[-1][0].copy()
    head_end = math.atan2(center[1] - p_end[1], center[0] - p_end[0])
    zero = np.zeros(3)

    for _ in range(int(hover_time / dt)):        # 懸停: 讓階段轉換在畫面上讀得出來
        out.append((p_end.copy(), zero.copy(), -head_end, 0.0, "懸停"))

    steps_land = int(landing_time / dt)
    for i in range(steps_land):                  # 降落: 與起飛對稱的餘弦下降
        s = (i + 1) / steps_land
        s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)
        z = height + (start_z - height) * s_smooth
        vz = (start_z - height) * 0.5 * math.pi * math.sin(math.pi * s) / landing_time
        out.append((np.array([p_end[0], p_end[1], z]),
                    np.array([0.0, 0.0, vz]), -head_end, 0.0, "降落"))

    # 觸地後再穩定一段。少了這段，影片會停在控制器還沒收斂的瞬間
    # （實測終點會低於目標高度約 7 cm）。位置環沒有積分項，
    # 收斂時間約 3 秒，所以這段要給滿 2 秒才看得出停穩。
    p_gnd = np.array([p_end[0], p_end[1], start_z])
    for _ in range(int(2.0 / dt)):
        out.append((p_gnd.copy(), zero.copy(), -head_end, 0.0, "觸地"))
    return out


def make_map_background(ply_path, center, radius, extent, size):
    """把點雲俯視圖 + 環繞路徑畫成一張靜態底圖（只做一次）。"""
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
    # 1 公尺比例尺
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
    """世界座標 -> 底圖像素座標（y 軸在影像中向下，所以要翻轉）。"""
    u = (p[0] - (center[0] - extent)) / (2 * extent) * size[0]
    v = size[1] - (p[1] - (center[1] - extent)) / (2 * extent) * size[1]
    return float(u), float(v)


def compose(fpv, bg, trail, pos, world_head, t, phase, extent, center, info):
    """把機載視角與俯視圖並排，畫上軌跡、機體與狀態列。"""
    from PIL import Image, ImageDraw
    mp = Image.fromarray(bg.copy())
    d = ImageDraw.Draw(mp)
    size = mp.size

    if len(trail) > 1:
        d.line([world_to_px(q, center, extent, size) for q in trail],
               fill=(255, 210, 0), width=2)

    u, v = world_to_px(pos, center, extent, size)
    # 安全外框（軸距一半 + 槳半徑 = 0.465 m），照比例畫成圓
    r_px = 0.465 / (2 * extent) * size[0]
    d.ellipse([u - r_px, v - r_px, u + r_px, v + r_px], outline=(0, 255, 120), width=2)
    # 朝向箭頭（world_head 是真實世界方位角，非 torso 的 yaw）
    L = r_px * 2.1
    d.line([(u, v), (u + L * math.cos(world_head), v - L * math.sin(world_head))],
           fill=(0, 255, 120), width=3)

    from PIL import ImageFont
    global _FONT
    if _FONT is None:
        try:                       # PIL 預設字型沒有中日韓字，會全部變成方框
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
        f"t={t:5.1f}s  {phase}  位置=({pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:4.2f})  "
        f"朝向={math.degrees(world_head):6.1f}°  {info}",
        fill=(228, 228, 228), font=_FONT)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser(description="3DGS 場景內的環繞飛行預覽")
    ap.add_argument("--config", default=None, help="nerfstudio config.yml（相對 workspace）")
    ap.add_argument("--ply", default="exports/scene04_dense.ply")
    ap.add_argument("--center", type=float, nargs=2, default=[-0.77, -0.01],
                    help="環繞中心（目標物）的 x y")
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--height", type=float, default=1.4)
    ap.add_argument("--start-z", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=0.6, help="環繞線速度 m/s")
    ap.add_argument("--takeoff-time", type=float, default=4.0)
    ap.add_argument("--hover-time", type=float, default=1.0,
                    help="環繞結束後原地懸停幾秒再降落")
    ap.add_argument("--landing-time", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.05, help="外圈決策週期（env 用 0.05）")
    ap.add_argument("--freq", type=float, default=200.0, help="內圈積分頻率")
    ap.add_argument("--extent", type=float, default=6.0, help="俯視圖半邊長 m")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="notes/scene04_orbit.mp4")
    ap.add_argument("--no-render", action="store_true", help="只測軌跡追蹤，不做 3DGS 渲染")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 步（除錯用）")
    a = ap.parse_args()

    # ---------- 動力學 ----------
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
    print(f"動力學參數（讀自 envs/drone_vla_multi_map.py）:")
    print(f"  質量 {P['mass']:.2f} kg   max_thrust 參數 {P['max_thrust']:.1f} N "
          f"(實際上限 {0.5*P['max_thrust']:.1f} N，推重比 "
          f"{0.5*P['max_thrust']/(P['mass']*G):.2f})")
    print(f"  慣量 {P['inertia']}   Kp {P['kp']}")
    print(f"  延遲 br={P['br_delay']}  thrust={P['thrust_delay']}")

    ctrl = OrbitController(P["mass"], P["max_thrust"])
    traj = build_trajectory(np.array(a.center), a.radius, a.height,
                            a.speed, a.dt, a.takeoff_time, a.start_z,
                            a.hover_time, a.landing_time)
    if a.limit:
        traj = traj[:a.limit]
    print(f"\n軌跡 {len(traj)} 步 = {len(traj)*a.dt:.1f} 秒 "
          f"(起飛 {a.takeoff_time:.0f}s + 環繞一圈 {2*math.pi*a.radius/a.speed:.1f}s "
          f"+ 懸停 {a.hover_time:.0f}s + 降落 {a.landing_time:.0f}s + 觸地 2s)")
    print(f"  環繞中心 ({a.center[0]:.2f}, {a.center[1]:.2f})  半徑 {a.radius} m  "
          f"高度 {a.height} m  線速度 {a.speed} m/s")
    print(f"  所需 yaw 速度 {a.speed/a.radius:.3f} rad/s  "
          f"(env 上限 {BR_SCALE} rad/s)")

    # ---------- 3DGS ----------
    gs = None
    if not a.no_render:
        sys.path.insert(0, str(GRAD_NAV))
        os.chdir(WORKSPACE)                      # config.yml 存相對路徑
        from utils.gs_local import GS
        cfg = a.config or sorted(
            (WORKSPACE / "outputs/scene04/splatfacto").glob("*/config.yml"))[-1]
        cfg = Path(cfg)
        print(f"\n載入 3DGS: {cfg}")
        gs = GS(cfg.relative_to(WORKSPACE) if cfg.is_absolute() else cfg,
                width=640, height=360, res=1.0)
        print(f"  高斯點數 {gs.pipeline.model.means.shape[0]:,}")

    bg = None
    if not a.no_render:
        bg = make_map_background(WORKSPACE / a.ply, np.array(a.center),
                                 a.radius, a.extent, (640, 360))

    # ---------- 主迴圈 ----------
    pos = np.array([a.center[0] + a.radius, a.center[1], a.start_z])
    vel = np.zeros(3)
    # 起始 yaw 直接對準軌跡的第一個參考值（無人機停在地面時就已朝向目標）。
    # 若從 yaw=0 起飛，對 yaw_ref=-180 度會產生剛好 180 度的姿態誤差，
    # 那是姿態控制的不穩定平衡點，會導致環繞段發散。
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

        # ↓ 與環境檔完全相同的動作處理（clip -> 縮放 -> 一階延遲 -> 再 clip）
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
                                  f"追蹤誤差={errs[-1]*100:4.1f}cm"))
            if k % 40 == 0:
                print(f"  [{k:4d}/{len(traj)}] t={k*a.dt:5.1f}s {phase} "
                      f"誤差 {errs[-1]*100:5.1f} cm  "
                      f"({_t.time()-t_start:.0f}s 已過)", flush=True)

    errs = np.array(errs)
    phases = [t[4] for t in traj]
    print(f"\n軌跡追蹤誤差:")
    for name in ("起飛", "環繞", "懸停", "降落", "觸地"):
        idx = [i for i, ph in enumerate(phases) if ph == name]
        if not idx:
            continue
        e = errs[idx]
        print(f"  {name}  {len(idx):>4} 步  平均 {e.mean()*100:5.1f} cm   "
              f"最大 {e.max()*100:5.1f} cm")
    print(f"  全程    平均 {errs.mean()*100:5.1f} cm   最大 {errs.max()*100:5.1f} cm")
    print(f"  最終位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  "
          f"（起點 {traj[0][0][0]:.2f}, {traj[0][0][1]:.2f}, {a.start_z:.2f}）")

    if frames:
        out = PROJECT_ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        imageio.mimwrite(out, frames, fps=a.fps, quality=8,
                         macro_block_size=1)
        print(f"\n✓ 影片已寫出: {out}")
        print(f"  {len(frames)} 幀 @ {a.fps} fps = {len(frames)/a.fps:.1f} 秒，"
              f"{frames[0].shape[1]}x{frames[0].shape[0]}")


if __name__ == "__main__":
    main()
