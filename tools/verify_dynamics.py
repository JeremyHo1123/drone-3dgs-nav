"""
驗證 grad_nav 的四旋翼動力學參數。

動力學參數填錯不會報錯，只會安靜地訓練出飛不起來的策略。
本工具在不跑訓練的前提下，直接對 QuadrotorSimulator 做三個測試。

它做兩件事：
  1. 換算報告——把你填的參數換算成有物理意義的量（推重比、懸停油門、
     角速度時間常數、馬達時間常數），並標出超出合理範圍的項目。
  2. 三個數值測試——懸停、滿油門爬升、角速度階躍，逐項 PASS/FAIL。

兩個上游程式碼的陷阱，本工具已經處理：
  - thrust 指令經過 (clip(a,-1,1)+1)*0.25 縮放，範圍是 [0, 0.5]。
    所以實際最大總推力 = 0.5 * max_thrust 參數，只有你填的一半。
  - rotor_noise_std=None 會讓 QuadrotorSimulator.update() 拋 NameError
    （第 119-121 行的 if 沒有 else 分支），要關噪聲得傳 0.0。

用法:
  python tools/verify_dynamics.py                         # 用原作 carl 的參數
  python tools/verify_dynamics.py --mass 0.7 --motor-thrust-g 700 \
      --arm-radius 0.11                                   # 只給機體規格，其餘自動推算
  python tools/verify_dynamics.py --mass 0.7 --max-thrust 55.0 \
      --inertia 0.0034 0.0034 0.0076 --kp 0.31 0.31 0.69  # 完全手動指定
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import warnings

import torch

# 上游 quadrotor_dynamics_advanced.py:132 用了沒指定 dim 的 torch.cross，
# 每步都會噴一次 deprecation warning。這是上游的事，這裡濾掉以免蓋住結果。
warnings.filterwarnings("ignore", message=".*torch.cross without specifying.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DYN_PATH = (PROJECT_ROOT
            / "repos/grad_nav/envs/assets/quadrotor_dynamics_advanced.py")

# 直接依檔案路徑載入。不能走 `from envs.assets...` 匯入，因為
# repos/grad_nav/envs/__init__.py 會連帶匯入全部環境檔，而 drone_long_traj.py
# 依賴已從 torchvision 移除的 torchvision.io.write_video，會在此炸掉。
# 動力學模組本身只依賴 torch，單獨載入沒有問題。
_spec = importlib.util.spec_from_file_location("quadrotor_dynamics", DYN_PATH)
_dyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dyn)
QuadrotorSimulator = _dyn.QuadrotorSimulator

G = 9.81

# 原作者那台無人機（代號 carl）的參數，出處：
#   envs/drone_vla_multi_map.py:93-104、envs/drone_long_traj.py:77-90
CARL = dict(
    mass=1.1,                             # min_mass 1.0 + 0.5 * mass_range 0.2
    max_thrust=26.0,                      # min_thrust 24.0 + 0.5 * thrust_range 4.0
    inertia=[0.01, 0.012, 0.025],
    kp=[1.0, 1.2, 2.5],
    kd=[0.001, 0.001, 0.002],
    br_delay=0.8,
    thrust_delay=0.7,
    br_limit=0.5,                         # rad/s，envs/drone_long_traj.py:84
    drag_coeff=0.5,                       # QuadrotorSimulator 預設值，env 從未傳入
    cross_area=0.1,
    freq=200.0,
    dt=0.05,
)

# 上游把 thrust 指令壓進 [0, 0.5]，見 drone_vla_multi_map.py:555
THRUST_CMD_MAX = 0.5


def build_args():
    p = argparse.ArgumentParser(
        description="驗證 grad_nav 四旋翼動力學參數",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mass", type=float, default=CARL["mass"],
                   help="總質量（含電池），kg")
    p.add_argument("--max-thrust", type=float, default=None,
                   help="填進 QuadrotorSimulator 的 max_thrust 參數（N）。"
                        "= 2 x 四顆馬達滿油門總推力。與 --motor-thrust-g 二選一")
    p.add_argument("--motor-thrust-g", type=float, default=None,
                   help="單顆馬達滿油門推力，公克。會自動換算成 max_thrust 參數")
    p.add_argument("--num-motors", type=int, default=4)
    p.add_argument("--arm-radius", type=float, default=None,
                   help="機體中心到馬達的距離，公尺（對角軸距的一半）。"
                        "給了就用經驗公式估慣量")
    p.add_argument("--inertia", type=float, nargs=3, default=None,
                   metavar=("IX", "IY", "IZ"), help="三軸慣量，kg*m^2")
    p.add_argument("--kp", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--kd", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--tau-rate", type=float, default=0.011,
                   help="想要的角速度反應時間常數，秒。沒給 --kp 時用它推算 Kp")
    p.add_argument("--br-delay", type=float, default=CARL["br_delay"])
    p.add_argument("--thrust-delay", type=float, default=CARL["thrust_delay"])
    p.add_argument("--br-limit", type=float, default=CARL["br_limit"],
                   help="policy 能要求的最大角速度，rad/s")
    p.add_argument("--drag-coeff", type=float, default=CARL["drag_coeff"])
    p.add_argument("--cross-area", type=float, default=CARL["cross_area"])
    p.add_argument("--freq", type=float, default=CARL["freq"], help="內圈積分頻率，Hz")
    p.add_argument("--dt", type=float, default=CARL["dt"], help="外圈 policy 決策週期，秒")
    p.add_argument("--cpu", action="store_true", help="強制在 CPU 上跑")
    return p.parse_args()


def resolve(a):
    """把使用者給的規格補齊成一組完整參數，並回報每個值是怎麼來的。"""
    src = {}

    if a.max_thrust is not None and a.motor_thrust_g is not None:
        sys.exit("錯誤：--max-thrust 與 --motor-thrust-g 只能擇一")

    if a.motor_thrust_g is not None:
        total_n = a.num_motors * a.motor_thrust_g / 1000.0 * G
        a.max_thrust = 2.0 * total_n
        src["max_thrust"] = (f"由 {a.num_motors} x {a.motor_thrust_g:.0f} g "
                             f"= {total_n:.2f} N 實際推力，x2 換算而來")
    elif a.max_thrust is None:
        a.max_thrust = CARL["max_thrust"]
        src["max_thrust"] = "原作 carl 的值"
    else:
        src["max_thrust"] = "手動指定"

    if a.inertia is None:
        if a.arm_radius is not None:
            # 係數出自 X 型四旋翼的逐元件模型：四顆馬達組件在半徑 L 上
            # （Ix=Iy=2*m_tip*L^2、Iz=4*m_tip*L^2），加機臂（桿繞端點 m*L^2/3）
            # 與中心質量（迴轉半徑約 0.07-0.08 m）。
            # 不要改用「從 carl 的 link_length=0.15 反推」的係數——
            # link_length 是上游宣告但從未使用的參數，不代表真實臂長。
            mL2 = a.mass * a.arm_radius ** 2
            a.inertia = [0.17 * mL2, 0.18 * mL2, 0.31 * mL2]
            src["inertia"] = (f"逐元件模型 Ix=0.17*m*L^2、Iy=0.18*m*L^2、"
                              f"Iz=0.31*m*L^2，L={a.arm_radius} m")
        else:
            a.inertia = list(CARL["inertia"])
            src["inertia"] = "原作 carl 的值（沒給 --arm-radius）"
    else:
        src["inertia"] = "手動指定"

    if a.kd is None:
        a.kd = [0.08 * i for i in a.inertia]
        src["kd"] = "Kd = 0.08 * I（由原作三軸的 Kd/I 比例反推）"
    else:
        src["kd"] = "手動指定"

    if a.kp is None:
        a.kp = [(i + d) / a.tau_rate for i, d in zip(a.inertia, a.kd)]
        src["kp"] = f"Kp = (I + Kd) / tau，tau = {a.tau_rate} s"
    else:
        src["kp"] = "手動指定"

    return a, src


def report_derived(a, src):
    """把參數換算成有物理意義的量。這一段比數值測試更常抓到錯誤。"""
    print("=" * 68)
    print("  換算報告")
    print("=" * 68)

    weight = a.mass * G
    real_max_thrust = THRUST_CMD_MAX * a.max_thrust
    twr = real_max_thrust / weight
    hover_norm = weight / a.max_thrust

    print(f"\n[ 推力 ]  {src['max_thrust']}")
    print(f"  max_thrust 參數          : {a.max_thrust:.2f} N")
    print(f"  實際可用最大總推力       : {real_max_thrust:.2f} N   (= 0.5 x 參數)")
    print(f"  機體重量                 : {weight:.2f} N   ({a.mass} kg)")
    print(f"  推重比                   : {twr:.2f}")
    print(f"  懸停時的正規化推力       : {hover_norm:.3f}   (指令上限 {THRUST_CMD_MAX})")

    warns = []
    if hover_norm >= THRUST_CMD_MAX:
        warns.append(f"懸停就已超過指令上限 {THRUST_CMD_MAX}，這台在模擬裡浮不起來。"
                     f"max_thrust 至少要 {weight / THRUST_CMD_MAX:.1f} N")
    elif hover_norm > 0.45:
        warns.append(f"懸停油門 {hover_norm:.3f} 太接近上限 {THRUST_CMD_MAX}，"
                     f"推重比只有 {twr:.2f}，幾乎沒有爬升餘裕")
    if twr > 4.0:
        warns.append(f"推重比 {twr:.2f} 偏高，懸停油門只有 {hover_norm:.3f}，"
                     f"policy 的輸出解析度會集中在很小的區間，可能不好訓")

    print(f"\n[ 姿態控制 ]  Kp: {src['kp']}   Kd: {src['kd']}")
    print(f"  {'軸':<8}{'I (kg*m^2)':>13}{'Kd':>12}{'Kp':>10}{'tau (s)':>11}")
    for name, i, kp, kd in zip(("roll x", "pitch y", "yaw z"), a.inertia, a.kp, a.kd):
        print(f"  {name:<8}{i:>13.5f}{kd:>12.5f}{kp:>10.3f}{(i + kd) / kp:>11.4f}")
    print(f"  慣量來源: {src['inertia']}")

    taus = [(i + d) / k for i, d, k in zip(a.inertia, a.kd, a.kp)]
    for name, t in zip(("roll", "pitch", "yaw"), taus):
        if t > a.dt:
            warns.append(f"{name} 的 tau={t:.4f} s 大於 policy 週期 {a.dt} s，"
                         f"姿態跟不上指令")
        elif t < 2.0 / a.freq:
            warns.append(f"{name} 的 tau={t:.4f} s 小於 2 個積分步 "
                         f"({2.0 / a.freq:.4f} s)，Euler 積分會發散")

    print(f"\n[ 指令延遲 ]  （一階低通，tau = -dt / ln(1 - a)）")
    for label, factor in (("角速度 br_delay_factor", a.br_delay),
                          ("推力   thrust_delay_factor", a.thrust_delay)):
        tau = -a.dt / torch.log(torch.tensor(1.0 - factor)).item()
        print(f"  {label:<28}= {factor:.2f}  ->  tau = {tau:.4f} s")

    print(f"\n[ 其他 ]")
    print(f"  角速度上限               : ±{a.br_limit:.2f} rad/s "
          f"(±{a.br_limit * 180 / 3.14159:.1f} 度/秒)")
    v_ref = 3.0
    f_drag = 0.5 * a.drag_coeff * a.cross_area * 1.225 * v_ref ** 2
    print(f"  {v_ref:.0f} m/s 時的空氣阻力  : {f_drag:.3f} N "
          f"(重量的 {100 * f_drag / weight:.1f}%)")
    print(f"  每個 policy 步的積分次數 : {int(a.dt * a.freq)}")

    if warns:
        print(f"\n[ 警告 ]")
        for w in warns:
            print(f"  ! {w}")
    return warns


def make_sim(a):
    """建一個關掉噪聲的 QuadrotorSimulator。噪聲要傳 0.0 不能傳 None。

    QuadrotorSimulator.__init__ 把 self.device 寫死成
    `cuda if torch.cuda.is_available() else cpu`，沒有參數可以覆蓋。
    要強制走 CPU 只能暫時遮蔽 cuda 偵測。
    """
    n = 1
    if a.cpu:
        _orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            return _build(a, n)
        finally:
            torch.cuda.is_available = _orig
    return _build(a, n)


def _build(a, n):
    return QuadrotorSimulator(
        mass=torch.full((n,), a.mass),
        inertia=torch.diag(torch.tensor(a.inertia)).unsqueeze(0).repeat(n, 1, 1),
        link_length=0.15,          # 上游從未使用此參數，填什麼都一樣
        Kp=torch.tensor(a.kp).unsqueeze(0).repeat(n, 1),
        Kd=torch.tensor(a.kd).unsqueeze(0).repeat(n, 1),
        freq=a.freq,
        max_thrust=torch.full((n,), a.max_thrust),
        total_time=a.dt,
        rotor_noise_std=0.0,
        br_noise_std=0.0,
        drag_coeff=a.drag_coeff,
        cross_area=a.cross_area,
    )


def initial_state(device):
    pos = torch.zeros(1, 3, device=device)
    vel = torch.zeros(1, 3, device=device)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)   # (w,x,y,z) 水平
    omega = torch.zeros(1, 3, device=device)
    return pos, vel, quat, omega


def test_hover(a, seconds=2.0):
    """測試 1：推力剛好等於重量時，高度應該不動。"""
    sim = make_sim(a)
    device = sim.device
    pos, vel, quat, omega = initial_state(device)

    # QuadrotorSimulator 本身不做夾限，夾限在環境檔（drone_vla_multi_map.py:555）。
    # 這裡要比照 env 夾，否則推力不足的機體會靠一個 env 根本下不出去的
    # 指令通過測試。
    want = a.mass * G / a.max_thrust
    cmd = min(want, THRUST_CMD_MAX)
    thrust_cmd = torch.full((1,), cmd, device=device)
    omega_des = torch.zeros(1, 3, device=device)

    n = int(seconds / a.dt)
    with torch.no_grad():
        for _ in range(n):
            pos, vel, omega, quat, _, _ = sim.run_simulation(
                pos, vel, quat, omega, (omega_des, thrust_cmd))

    drift = pos[0, 2].item()
    clipped = want > THRUST_CMD_MAX
    ok = (not clipped) and abs(drift) < 0.01
    print(f"\n  測試 1  懸停 {seconds:.0f} 秒")
    print(f"    懸停所需推力指令 = {want:.4f}" +
          (f"  -> 被夾到 {THRUST_CMD_MAX}" if clipped else ""))
    print(f"    高度漂移         = {drift * 1000:+.2f} mm   (門檻 ±10 mm)")
    print(f"    水平漂移         = {pos[0, 0].item() * 1000:+.2f}, "
          f"{pos[0, 1].item() * 1000:+.2f} mm")
    if clipped:
        print(f"    -> FAIL  懸停需求超過指令上限 {THRUST_CMD_MAX}，"
              f"這台在模擬裡浮不起來")
    else:
        print(f"    -> {'PASS' if ok else 'FAIL  質量或 max_thrust 換算有誤'}")
    return ok


def test_full_throttle(a):
    """測試 2：滿油門的第一步加速度，應等於 (0.5*max_thrust - mg)/m。"""
    sim = make_sim(a)
    device = sim.device
    pos, vel, quat, omega = initial_state(device)
    thrust_cmd = torch.full((1,), THRUST_CMD_MAX, device=device)
    omega_des = torch.zeros(1, 3, device=device)

    expect = (THRUST_CMD_MAX * a.max_thrust - a.mass * G) / a.mass
    with torch.no_grad():
        _, _, _, _, lin_acc, _ = sim.run_simulation(
            pos, vel, quat, omega, (omega_des, thrust_cmd))
    got = lin_acc[0, 2].item()

    # 一個 policy 步內速度已經上升，阻力會讓實測略低於理論值，容差放到 3%
    err = abs(got - expect) / max(abs(expect), 1e-6)
    ok = err < 0.03 and expect > 0
    print(f"\n  測試 2  滿油門垂直加速度")
    print(f"    理論值 = {expect:+.4f} m/s^2   (= (0.5 x {a.max_thrust:.2f} "
          f"- {a.mass} x 9.81) / {a.mass})")
    print(f"    實測值 = {got:+.4f} m/s^2   誤差 {err * 100:.2f}%  (門檻 3%)")
    if expect <= 0:
        print(f"    -> FAIL  滿油門仍在往下掉，推重比 "
              f"{THRUST_CMD_MAX * a.max_thrust / (a.mass * G):.2f} < 1")
    else:
        print(f"    -> {'PASS' if ok else 'FAIL  max_thrust 的 x2 換算可能搞錯了'}")
    return ok


def test_rate_step(a, axis=0, seconds=0.5):
    """測試 3：角速度階躍反應時間，應等於 tau = (I + Kd) / Kp。"""
    sim = make_sim(a)
    device = sim.device
    pos, vel, quat, omega = initial_state(device)
    thrust_cmd = torch.full((1,), a.mass * G / a.max_thrust, device=device)
    omega_des = torch.zeros(1, 3, device=device)
    omega_des[0, axis] = a.br_limit

    target = 0.632 * a.br_limit
    inner_dt = 1.0 / a.freq
    crossed_at = None
    t = 0.0

    with torch.no_grad():
        for _ in range(int(seconds / inner_dt)):
            # 直接呼叫 update() 才能看到每個內圈小步
            pos, vel, quat, omega, _, _, _ = sim.update(
                pos, vel, quat, omega, omega_des, thrust_cmd,
                torch.zeros(1, device=device))
            t += inner_dt
            if crossed_at is None and omega[0, axis].item() >= target:
                crossed_at = t

    name = ("roll x", "pitch y", "yaw z")[axis]
    expect = (a.inertia[axis] + a.kd[axis]) / a.kp[axis]
    print(f"\n  測試 3  {name} 角速度階躍（指令 {a.br_limit} rad/s）")
    print(f"    理論 tau = {expect:.4f} s   (= (I + Kd) / Kp)")
    if crossed_at is None:
        print(f"    實測     = 從未達到 63% ({target:.3f} rad/s)，"
              f"末值 {omega[0, axis].item():.4f}")
        print(f"    -> FAIL  Kp 太小，或角速度指令超出這組增益撐得住的範圍")
        return False
    err = abs(crossed_at - expect) / expect
    # 內圈解析度只有 1/freq 秒，tau 很小時單步就佔了不小比例
    tol = max(0.25, 1.5 * inner_dt / expect)
    ok = err < tol
    print(f"    實測     = {crossed_at:.4f} s   誤差 {err * 100:.1f}%  "
          f"(門檻 {tol * 100:.0f}%，受限於 {inner_dt * 1000:.1f} ms 積分解析度)")
    print(f"    -> {'PASS' if ok else 'FAIL  Kp / Kd / 慣量 有一項不一致'}")
    return ok


def main():
    a, src = resolve(build_args())
    warns = report_derived(a, src)

    device = make_sim(a).device
    print("\n" + "=" * 68)
    print(f"  數值測試  (device={device})")
    print("=" * 68)
    results = [
        test_hover(a),
        test_full_throttle(a),
        test_rate_step(a, axis=0),
        test_rate_step(a, axis=2),
    ]

    print("\n" + "=" * 68)
    n_pass = sum(results)
    print(f"  {n_pass}/{len(results)} 通過"
          + (f"，{len(warns)} 個警告" if warns else ""))
    print("=" * 68)

    print("\n  要套用這組參數，把以下幾行填進五個環境檔："
          "\n    envs/drone_long_traj.py, drone_multi_gate.py, drone_ppo.py,"
          "\n    drone_vla_long_task.py, drone_vla_multi_map.py\n")
    print(f"        self.min_mass  = {a.mass * 0.95:.4f}")
    print(f"        self.mass_range = {a.mass * 0.10:.4f}")
    print(f"        self.min_thrust = {a.max_thrust * 0.92:.4f}")
    print(f"        self.thrust_range = {a.max_thrust * 0.16:.4f}")
    print(f"        self.init_inertia = [{', '.join(f'{v:.5f}' for v in a.inertia)}]")
    print(f"        self.init_kp = [{', '.join(f'{v:.4f}' for v in a.kp)}]")
    print(f"        self.init_kd = [{', '.join(f'{v:.5f}' for v in a.kd)}]")
    print(f"        self.br_delay_factor = {a.br_delay}")
    print(f"        self.thrust_delay_factor = {a.thrust_delay}")
    print()

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
