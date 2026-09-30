"""
Verify grad_nav's quadrotor dynamics parameters.

Wrong dynamics parameters raise no error; they just silently train a policy that cannot fly.
Without running any training, this tool runs three tests directly on QuadrotorSimulator.

It does two things:
  1. Derived report: converts your parameters into physically meaningful quantities (thrust-to-weight ratio, hover throttle,
     angular-rate time constant, motor time constant) and flags anything outside a reasonable range.
  2. Three numerical tests: hover, full-throttle climb, angular-rate step, each reported as PASS/FAIL.

Two pitfalls in the upstream code, already handled by this tool:
  - The thrust command is scaled by (clip(a,-1,1)+1)*0.25, so its range is [0, 0.5].
    Therefore the actual maximum total thrust = 0.5 * the max_thrust parameter, only half of what you enter.
  - rotor_noise_std=None makes QuadrotorSimulator.update() raise NameError
    (the if on lines 119-121 has no else branch); to disable noise pass 0.0.

Usage:
  python tools/verify_dynamics.py                         # use the original carl parameters
  python tools/verify_dynamics.py --mass 0.7 --motor-thrust-g 700 \
      --arm-radius 0.11                                   # airframe specs only, the rest is derived
  python tools/verify_dynamics.py --mass 0.7 --max-thrust 55.0 \
      --inertia 0.0034 0.0034 0.0076 --kp 0.31 0.31 0.69  # fully manual
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import warnings

import torch

# Upstream quadrotor_dynamics_advanced.py:132 calls torch.cross without specifying dim,
# which emits a deprecation warning every step. That is upstream's problem; filter it here so it does not bury the results.
warnings.filterwarnings("ignore", message=".*torch.cross without specifying.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DYN_PATH = (PROJECT_ROOT
            / "repos/grad_nav/envs/assets/quadrotor_dynamics_advanced.py")

# Load directly by file path. Importing via `from envs.assets...` does not work, because
# repos/grad_nav/envs/__init__.py also imports every environment file, and drone_long_traj.py
# depends on torchvision.io.write_video, which has been removed from torchvision, so it blows up there.
# The dynamics module itself only depends on torch, so loading it alone is fine.
_spec = importlib.util.spec_from_file_location("quadrotor_dynamics", DYN_PATH)
_dyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dyn)
QuadrotorSimulator = _dyn.QuadrotorSimulator

G = 9.81

# Parameters of the original authors' drone (codename carl), from:
#   envs/drone_vla_multi_map.py:93-104, envs/drone_long_traj.py:77-90
CARL = dict(
    mass=1.1,                             # min_mass 1.0 + 0.5 * mass_range 0.2
    max_thrust=26.0,                      # min_thrust 24.0 + 0.5 * thrust_range 4.0
    inertia=[0.01, 0.012, 0.025],
    kp=[1.0, 1.2, 2.5],
    kd=[0.001, 0.001, 0.002],
    br_delay=0.8,
    thrust_delay=0.7,
    br_limit=0.5,                         # rad/s, envs/drone_long_traj.py:84
    drag_coeff=0.5,                       # QuadrotorSimulator default; the env never passes it
    cross_area=0.1,
    freq=200.0,
    dt=0.05,
)

# Upstream squashes the thrust command into [0, 0.5], see drone_vla_multi_map.py:555
THRUST_CMD_MAX = 0.5


def build_args():
    p = argparse.ArgumentParser(
        description="Verify grad_nav quadrotor dynamics parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mass", type=float, default=CARL["mass"],
                   help="Total mass (including battery), kg")
    p.add_argument("--max-thrust", type=float, default=None,
                   help="The max_thrust parameter passed to QuadrotorSimulator (N). "
                        "= 2 x total full-throttle thrust of the four motors. Mutually exclusive with --motor-thrust-g")
    p.add_argument("--motor-thrust-g", type=float, default=None,
                   help="Full-throttle thrust of one motor, grams. Converted to the max_thrust parameter automatically")
    p.add_argument("--num-motors", type=int, default=4)
    p.add_argument("--arm-radius", type=float, default=None,
                   help="Distance from the airframe center to a motor, meters (half the diagonal wheelbase). "
                        "If given, inertia is estimated with an empirical formula")
    p.add_argument("--inertia", type=float, nargs=3, default=None,
                   metavar=("IX", "IY", "IZ"), help="Inertia about the three axes, kg*m^2")
    p.add_argument("--kp", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--kd", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--tau-rate", type=float, default=0.011,
                   help="Desired angular-rate response time constant, seconds. Used to derive Kp when --kp is not given")
    p.add_argument("--br-delay", type=float, default=CARL["br_delay"])
    p.add_argument("--thrust-delay", type=float, default=CARL["thrust_delay"])
    p.add_argument("--br-limit", type=float, default=CARL["br_limit"],
                   help="Maximum angular rate the policy can command, rad/s")
    p.add_argument("--drag-coeff", type=float, default=CARL["drag_coeff"])
    p.add_argument("--cross-area", type=float, default=CARL["cross_area"])
    p.add_argument("--freq", type=float, default=CARL["freq"], help="Inner-loop integration frequency, Hz")
    p.add_argument("--dt", type=float, default=CARL["dt"], help="Outer-loop policy decision period, seconds")
    p.add_argument("--cpu", action="store_true", help="Force running on CPU")
    return p.parse_args()


def resolve(a):
    """Complete the user-given specs into a full parameter set, and report where each value came from."""
    src = {}

    if a.max_thrust is not None and a.motor_thrust_g is not None:
        sys.exit("Error: --max-thrust and --motor-thrust-g are mutually exclusive")

    if a.motor_thrust_g is not None:
        total_n = a.num_motors * a.motor_thrust_g / 1000.0 * G
        a.max_thrust = 2.0 * total_n
        src["max_thrust"] = (f"converted from {a.num_motors} x {a.motor_thrust_g:.0f} g "
                             f"= {total_n:.2f} N actual thrust, x2")
    elif a.max_thrust is None:
        a.max_thrust = CARL["max_thrust"]
        src["max_thrust"] = "original carl value"
    else:
        src["max_thrust"] = "manually specified"

    if a.inertia is None:
        if a.arm_radius is not None:
            # Coefficients come from a per-component model of an X quadrotor: four motor assemblies at radius L
            # (Ix=Iy=2*m_tip*L^2, Iz=4*m_tip*L^2), plus the arms (rod about its end, m*L^2/3)
            # and the central mass (radius of gyration about 0.07-0.08 m).
            # Do not switch to coefficients "back-derived from carl's link_length=0.15":
            # link_length is a parameter upstream declares but never uses; it is not the real arm length.
            mL2 = a.mass * a.arm_radius ** 2
            a.inertia = [0.17 * mL2, 0.18 * mL2, 0.31 * mL2]
            src["inertia"] = (f"per-component model Ix=0.17*m*L^2, Iy=0.18*m*L^2, "
                              f"Iz=0.31*m*L^2, L={a.arm_radius} m")
        else:
            a.inertia = list(CARL["inertia"])
            src["inertia"] = "original carl value (no --arm-radius given)"
    else:
        src["inertia"] = "manually specified"

    if a.kd is None:
        a.kd = [0.08 * i for i in a.inertia]
        src["kd"] = "Kd = 0.08 * I (back-derived from the original per-axis Kd/I ratio)"
    else:
        src["kd"] = "manually specified"

    if a.kp is None:
        a.kp = [(i + d) / a.tau_rate for i, d in zip(a.inertia, a.kd)]
        src["kp"] = f"Kp = (I + Kd) / tau, tau = {a.tau_rate} s"
    else:
        src["kp"] = "manually specified"

    return a, src


def report_derived(a, src):
    """Convert the parameters into physically meaningful quantities. This section catches errors more often than the numerical tests."""
    print("=" * 68)
    print("  Derived report")
    print("=" * 68)

    weight = a.mass * G
    real_max_thrust = THRUST_CMD_MAX * a.max_thrust
    twr = real_max_thrust / weight
    hover_norm = weight / a.max_thrust

    print(f"\n[ Thrust ]  {src['max_thrust']}")
    print(f"  max_thrust parameter       : {a.max_thrust:.2f} N")
    print(f"  usable max total thrust    : {real_max_thrust:.2f} N   (= 0.5 x parameter)")
    print(f"  airframe weight            : {weight:.2f} N   ({a.mass} kg)")
    print(f"  thrust-to-weight ratio     : {twr:.2f}")
    print(f"  normalized hover thrust    : {hover_norm:.3f}   (command limit {THRUST_CMD_MAX})")

    warns = []
    if hover_norm >= THRUST_CMD_MAX:
        warns.append(f"Hover alone already exceeds the command limit {THRUST_CMD_MAX}; this airframe cannot get airborne in the sim. "
                     f"max_thrust must be at least {weight / THRUST_CMD_MAX:.1f} N")
    elif hover_norm > 0.45:
        warns.append(f"Hover throttle {hover_norm:.3f} is too close to the limit {THRUST_CMD_MAX}; "
                     f"thrust-to-weight ratio is only {twr:.2f}, almost no climb margin")
    if twr > 4.0:
        warns.append(f"Thrust-to-weight ratio {twr:.2f} is high and hover throttle is only {hover_norm:.3f}; "
                     f"the policy's output resolution is squeezed into a small interval, which may be hard to train")

    print(f"\n[ Attitude control ]  Kp: {src['kp']}   Kd: {src['kd']}")
    print(f"  {'axis':<8}{'I (kg*m^2)':>13}{'Kd':>12}{'Kp':>10}{'tau (s)':>11}")
    for name, i, kp, kd in zip(("roll x", "pitch y", "yaw z"), a.inertia, a.kp, a.kd):
        print(f"  {name:<8}{i:>13.5f}{kd:>12.5f}{kp:>10.3f}{(i + kd) / kp:>11.4f}")
    print(f"  inertia source: {src['inertia']}")

    taus = [(i + d) / k for i, d, k in zip(a.inertia, a.kd, a.kp)]
    for name, t in zip(("roll", "pitch", "yaw"), taus):
        if t > a.dt:
            warns.append(f"{name} tau={t:.4f} s is larger than the policy period {a.dt} s; "
                         f"attitude cannot keep up with commands")
        elif t < 2.0 / a.freq:
            warns.append(f"{name} tau={t:.4f} s is smaller than 2 integration steps "
                         f"({2.0 / a.freq:.4f} s); Euler integration will diverge")

    print(f"\n[ Command delay ]  (first-order low-pass, tau = -dt / ln(1 - a))")
    for label, factor in (("rate   br_delay_factor", a.br_delay),
                          ("thrust thrust_delay_factor", a.thrust_delay)):
        tau = -a.dt / torch.log(torch.tensor(1.0 - factor)).item()
        print(f"  {label:<28}= {factor:.2f}  ->  tau = {tau:.4f} s")

    print(f"\n[ Other ]")
    print(f"  angular rate limit         : ±{a.br_limit:.2f} rad/s "
          f"(±{a.br_limit * 180 / 3.14159:.1f} deg/s)")
    v_ref = 3.0
    f_drag = 0.5 * a.drag_coeff * a.cross_area * 1.225 * v_ref ** 2
    print(f"  air drag at {v_ref:.0f} m/s          : {f_drag:.3f} N "
          f"({100 * f_drag / weight:.1f}% of weight)")
    print(f"  inner steps per policy step: {int(a.dt * a.freq)}")

    if warns:
        print(f"\n[ Warnings ]")
        for w in warns:
            print(f"  ! {w}")
    return warns


def make_sim(a):
    """Build a QuadrotorSimulator with noise disabled. Noise must be passed as 0.0, not None.

    QuadrotorSimulator.__init__ hardcodes self.device as
    `cuda if torch.cuda.is_available() else cpu`, with no parameter to override it.
    The only way to force CPU is to temporarily mask CUDA detection.
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
        link_length=0.15,          # never used upstream; any value works
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
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)   # (w,x,y,z) level
    omega = torch.zeros(1, 3, device=device)
    return pos, vel, quat, omega


def test_hover(a, seconds=2.0):
    """Test 1: when thrust exactly equals weight, altitude should not change."""
    sim = make_sim(a)
    device = sim.device
    pos, vel, quat, omega = initial_state(device)

    # QuadrotorSimulator itself does no clamping; clamping happens in the env file (drone_vla_multi_map.py:555).
    # Clamp the same way as the env here, otherwise an underpowered airframe would pass the test with a
    # command the env can never issue.
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
    print(f"\n  Test 1  hover {seconds:.0f} s")
    print(f"    hover thrust command = {want:.4f}" +
          (f"  -> clamped to {THRUST_CMD_MAX}" if clipped else ""))
    print(f"    altitude drift       = {drift * 1000:+.2f} mm   (threshold ±10 mm)")
    print(f"    horizontal drift     = {pos[0, 0].item() * 1000:+.2f}, "
          f"{pos[0, 1].item() * 1000:+.2f} mm")
    if clipped:
        print(f"    -> FAIL  hover requirement exceeds the command limit {THRUST_CMD_MAX}; "
              f"this airframe cannot get airborne in the sim")
    else:
        print(f"    -> {'PASS' if ok else 'FAIL  mass or max_thrust conversion is wrong'}")
    return ok


def test_full_throttle(a):
    """Test 2: first-step acceleration at full throttle should equal (0.5*max_thrust - mg)/m."""
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

    # Velocity already rises within one policy step, so drag makes the measurement slightly below theory; tolerance relaxed to 3%
    err = abs(got - expect) / max(abs(expect), 1e-6)
    ok = err < 0.03 and expect > 0
    print(f"\n  Test 2  full-throttle vertical acceleration")
    print(f"    theoretical = {expect:+.4f} m/s^2   (= (0.5 x {a.max_thrust:.2f} "
          f"- {a.mass} x 9.81) / {a.mass})")
    print(f"    measured    = {got:+.4f} m/s^2   error {err * 100:.2f}%  (threshold 3%)")
    if expect <= 0:
        print(f"    -> FAIL  still falling at full throttle, thrust-to-weight ratio "
              f"{THRUST_CMD_MAX * a.max_thrust / (a.mass * G):.2f} < 1")
    else:
        print(f"    -> {'PASS' if ok else 'FAIL  the x2 conversion of max_thrust is probably wrong'}")
    return ok


def test_rate_step(a, axis=0, seconds=0.5):
    """Test 3: angular-rate step response time should equal tau = (I + Kd) / Kp."""
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
            # Call update() directly to see every inner-loop substep
            pos, vel, quat, omega, _, _, _ = sim.update(
                pos, vel, quat, omega, omega_des, thrust_cmd,
                torch.zeros(1, device=device))
            t += inner_dt
            if crossed_at is None and omega[0, axis].item() >= target:
                crossed_at = t

    name = ("roll x", "pitch y", "yaw z")[axis]
    expect = (a.inertia[axis] + a.kd[axis]) / a.kp[axis]
    print(f"\n  Test 3  {name} angular-rate step (command {a.br_limit} rad/s)")
    print(f"    theoretical tau = {expect:.4f} s   (= (I + Kd) / Kp)")
    if crossed_at is None:
        print(f"    measured        = never reached 63% ({target:.3f} rad/s), "
              f"final value {omega[0, axis].item():.4f}")
        print(f"    -> FAIL  Kp too small, or the rate command exceeds what these gains can handle")
        return False
    err = abs(crossed_at - expect) / expect
    # Inner-loop resolution is only 1/freq s; when tau is small a single step is a sizable fraction of it
    tol = max(0.25, 1.5 * inner_dt / expect)
    ok = err < tol
    print(f"    measured        = {crossed_at:.4f} s   error {err * 100:.1f}%  "
          f"(threshold {tol * 100:.0f}%, limited by {inner_dt * 1000:.1f} ms integration resolution)")
    print(f"    -> {'PASS' if ok else 'FAIL  one of Kp / Kd / inertia is inconsistent'}")
    return ok


def main():
    a, src = resolve(build_args())
    warns = report_derived(a, src)

    device = make_sim(a).device
    print("\n" + "=" * 68)
    print(f"  Numerical tests  (device={device})")
    print("=" * 68)
    results = [
        test_hover(a),
        test_full_throttle(a),
        test_rate_step(a, axis=0),
        test_rate_step(a, axis=2),
    ]

    print("\n" + "=" * 68)
    n_pass = sum(results)
    print(f"  {n_pass}/{len(results)} passed"
          + (f", {len(warns)} warnings" if warns else ""))
    print("=" * 68)

    print("\n  To apply this parameter set, put the following lines into the five environment files:"
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
