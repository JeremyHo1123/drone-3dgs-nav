#!/usr/bin/env python3
"""
Fly the released orbit policy in the 3DGS simulator, closed loop.

The policy is the one that runs on the real drone: orbit_runtime.py loads the deployment bundle
(bundle_v4/weights.pt) and turns (state, camera image) into a body-rate + thrust command. This script
puts it in the simulator it was trained in -- grad_nav's DroneVLAOrbitEnv with the simulator patch
from configs/grad_nav/ -- and closes the loop:

    simulator renders the onboard camera (3DGS) + reports the state
        -> OrbitPolicy.observe()  -> OrbitPolicy.act()
        -> the same action goes into env.step(): command mixing, height integrator,
           command latency and quadrotor dynamics all run inside the simulator
        -> next state and image

Every step it also checks that the command the deployment code computed equals the command the
simulator computed from the same action (`cmd diff` in the output). The two are independent
implementations of the same formulas; a difference above about 1e-4 means the policy is not seeing
what it saw in training.

Each episode starts on the circle (radius 2.5 m, height 1.0 m), nose pointing at the box, at rest.
Task 0 orbits clockwise (seen from above), task 1 counter-clockwise. The policy only knows how to
orbit -- take-off, stopping after one lap and landing are done by the onboard program on the real
drone, so here the episode simply runs for --steps steps (20 Hz; 700 steps = 35 s, about 1.1 laps).

Usage (from the repository root, environment `droneenv` from the README):
    python orbit_policy/run_in_sim.py
    python orbit_policy/run_in_sim.py --start-deg 0 90 180 270 --dr --out-dir runs/orbit_sim --video
Options:
    --latency cfg|off   cfg (default): the command-latency and motor-response model the policy was
                        trained with (random per episode); off: commands act immediately
    --dr                randomise mass and max thrust per episode, as in training (default: nominal)
Output: one line per episode; with --out-dir, metrics.json, a top-down plot and (--video) the
onboard camera of each episode as MP4.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MAP_NAME = "sunny_baseball_field"
MM_CACHING_FREQUENCY = 16        # diff_env.MM_caching_frequency of the training config

ap = argparse.ArgumentParser(description="Fly the orbit policy in the 3DGS simulator")
ap.add_argument("--grad-nav", default=os.path.join(REPO, "repos", "grad_nav"),
                help="grad_nav checkout (commit 95218ab) with configs/grad_nav/grad_nav_simulator_v4.patch applied")
ap.add_argument("--bundle", default=os.path.join(HERE, "bundle_v4"), help="folder with weights.pt")
ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1], help="0 = clockwise, 1 = counter-clockwise")
ap.add_argument("--start-deg", type=float, nargs="+", default=[270.0],
                help="start bearings on the circle, degrees from the box centre (0 = +x); one episode per bearing and task")
ap.add_argument("--steps", type=int, default=700, help="steps per episode, 0.05 s each")
ap.add_argument("--latency", choices=["cfg", "off"], default="cfg")
ap.add_argument("--dr", action="store_true", help="mass / thrust randomisation as in training")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--video", action="store_true", help="with --out-dir: save the onboard camera as MP4")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
if args.video and not args.out_dir:
    ap.error("--video needs --out-dir")
bundle = os.path.abspath(args.bundle)
gn = os.path.abspath(args.grad_nav)
out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
if not os.path.exists(os.path.join(bundle, "weights.pt")):
    sys.exit(f"no weights.pt in {bundle} -- download it from the release (see orbit_policy/README.md)")
if not os.path.exists(os.path.join(gn, "envs", "drone_vla_orbit.py")):
    sys.exit(f"{gn} has no envs/drone_vla_orbit.py -- clone grad_nav and apply the simulator patch first")

from orbit_runtime import OrbitPolicy, SceneState  # noqa: E402  (this folder)

meta = torch.load(os.path.join(bundle, "weights.pt"), map_location="cpu", weights_only=True)["meta"]

# ---------------- simulator (grad_nav loads its scene and point cloud relative to the checkout)
sys.path.insert(0, gn)
os.chdir(gn)
import envs  # noqa: E402

dev = args.device
torch.manual_seed(args.seed)
env = envs.DroneVLAOrbitEnv(render=False, device=dev, num_envs=1, seed=args.seed,
                            episode_length=args.steps + 10, no_grad=True, stochastic_init=False,
                            MM_caching_frequency=MM_CACHING_FREQUENCY, map_name=MAP_NAME, eval_task_id=0,
                            env_hyper=dict(meta["env_hyper"]))
# utils/gs_local.py switches matmuls to TF32 on import; use full FP32 like the deployment code
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
env.obs_noise_level = 0.0               # the height integrator reads the observed height: no noise, as on the drone
env.domain_randomization = args.dr
if args.latency == "off":
    env.cmd_latency = None
hint = env.height_integrator
task_names = list(env.task_table.keys())
if task_names != list(meta["task_names"]):
    sys.exit(f"task strings differ between the bundle and the simulator:\n  {meta['task_names']}\n  {task_names}")
C = env.orbit_center.detach().cpu().numpy()
R, H, V = env.orbit_radius, env.orbit_height, env.orbit_speed

# ---------------- policy
pol = OrbitPolicy(bundle, instruction=0, device=dev, canon_quat=False,
                  T_hover=meta.get("t_hover_prog", 0.65), thrust_exponent=meta.get("thrust_exponent", 1.67))
print(f"bundle {bundle}: obs_frame {pol.obs_frame}, height integrator {'on' if pol.hint else 'off'}, start {pol.start_mode}")
print(f"simulator: latency model {'on' if env.cmd_latency is not None else 'off'}, "
      f"mass/thrust randomisation {'on' if args.dr else 'off'}, {args.steps} steps of {env.dt} s")

stash = {}
_process_gs = env.process_GS_data
_reset = env.reset


def _grab_image(depth_list, nerf_img):          # the rendered onboard camera, (144, 256, 3) in [0, 1]
    stash["img"] = nerf_img[0].detach().clone()
    return _process_gs(depth_list, nerf_img)


def _grab_reset(env_ids=None, force_reset=True):
    # step() resets an episode that ended; keep the last state before it is overwritten
    stash["pre_reset"] = (env.state_joint_q[0].clone(), env.state_joint_qd[0].clone())
    return _reset(env_ids, force_reset)


env.process_GS_data = _grab_image
env.reset = _grab_reset


def scene_state(q, qd):
    return SceneState(z=float(q[2]), quat=tuple(float(x) for x in q[3:7]), vel=tuple(float(x) for x in qd[3:6]))


def heading_xy(q):                              # body x axis projected on the ground (x, y, z, w)
    x, y, z, w = q
    h = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    return h / (np.linalg.norm(h) + 1e-9)


def end_reason(q, qd):
    r = math.hypot(q[0] - C[0], q[1] - C[1])
    if not env.min_height <= q[2] <= env.max_height:
        return f"height {q[2]:.2f} m outside {env.min_height}-{env.max_height} m"
    if not env.min_radius <= r <= env.max_radius:
        return f"distance to box {r:.2f} m outside {env.min_radius}-{env.max_radius} m"
    if np.linalg.norm(qd[0:3]) > env.body_rate_threshold:
        return "body rate too high"
    return "non-finite observation"


def run_episode(task, start_deg, idx):
    torch.manual_seed(args.seed + idx)          # same rotor noise / latency draw for the same episode
    env.reset()
    name = task_names[task]
    env.task_list[0] = name
    env.orbit_dir = torch.tensor([env.task_table[name]], device=dev)
    th0 = math.radians(start_deg)
    env.state_joint_q[0, 0] = float(C[0]) + R * math.cos(th0)
    env.state_joint_q[0, 1] = float(C[1]) + R * math.sin(th0)
    env.state_joint_q[0, 2] = H
    psi = th0 + math.pi                          # nose at the box
    env.state_joint_q[0, 3:7] = torch.tensor([0.0, 0.0, math.sin(psi / 2), math.cos(psi / 2)], device=dev)
    env.state_joint_qd[:] = 0
    env.prev_lin_vel[:] = 0
    env.calculateObservations()                  # render the start pose
    pol.instruction = task
    if pol.start_mode == "upstream":
        # v1/v2 training started each episode from this drone's own hover command (it differs from nominal with --dr)
        pol.start_action = env.actions[0].clone()
    pol.reset(scene_state(env.state_joint_q[0], env.state_joint_qd[0]), stash["img"])

    traj = [np.r_[env.state_joint_q[0].cpu().numpy(), env.state_joint_qd[0].cpu().numpy()]]
    frames = [stash["img"].cpu()] if args.video else None
    d_cmd, reason = 0.0, None
    t0 = time.time()
    with torch.no_grad():
        for t in range(args.steps):
            cmd = pol.act()
            env.step(pol.lpf.clone(), pol.last_z.unsqueeze(0))
            # the simulator mixed the same action itself; compare with the deployment code's command
            d = float((pol.actions - env.actions[0]).abs().max())
            if hint is not None:
                total = min(max(float(env.actions[0, 3]) + float(hint.f_I[0]), 0.0), 0.5)
                d = max(d, abs(cmd.sim_thrust_frac - total))
            d_cmd = max(d_cmd, d)
            if int(env.reset_buf[0]):
                q, qd = (x.cpu().numpy() for x in stash["pre_reset"])
                traj.append(np.r_[q, qd])
                reason = end_reason(q, qd)
                break
            q, qd = env.state_joint_q[0], env.state_joint_qd[0]
            traj.append(np.r_[q.cpu().numpy(), qd.cpu().numpy()])
            if frames is not None:
                frames.append(stash["img"].cpu())
            pol.observe(scene_state(q, qd), stash["img"])
    return np.array(traj), reason, d_cmd, frames, time.time() - t0


def metrics(traj, task):
    """Orbit quality. 'Steady' = after the first 2 s."""
    dt = env.dt
    sign = env.task_table[task_names[task]]      # +1 counter-clockwise, -1 clockwise
    p, q, v = traj[:, 0:3], traj[:, 3:7], traj[:, 10:13]
    t = np.arange(len(traj)) * dt
    d = p[:, :2] - C
    r = np.linalg.norm(d, axis=1)
    bearing = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    swept = np.degrees((bearing - bearing[0]) * sign)
    lap = np.nonzero(swept >= 360.0)[0]
    t_hat = np.stack([-d[:, 1], d[:, 0]], 1) / r[:, None] * sign
    look = np.array([math.degrees(math.acos(np.clip(heading_xy(qi) @ (-di / ri), -1, 1)))
                     for qi, di, ri in zip(q, d, r)])
    s = t >= 2.0
    return dict(
        flown_s=round(float(t[-1]), 2), swept_deg=round(float(swept[-1]), 1),
        lap_s=round(float(t[lap[0]]), 2) if len(lap) else None,
        radius_err_rms_m=round(float(np.sqrt(np.mean((r[s] - R) ** 2))), 4),
        radius_err_mean_m=round(float(np.mean(r[s] - R)), 4),
        height_err_rms_m=round(float(np.sqrt(np.mean((p[s, 2] - H) ** 2))), 4),
        height_err_mean_m=round(float(np.mean(p[s, 2] - H)), 4),
        nose_off_box_mean_deg=round(float(np.mean(look[s])), 2),
        speed_mean_mps=round(float(np.mean(np.sum(v[s, :2] * t_hat[s], 1))), 3),
        vz_abs_max_mps=round(float(np.max(np.abs(v[s, 2]))), 3),
    )


results = []
idx = 0
for start in args.start_deg:
    for task in args.tasks:
        traj, reason, d_cmd, frames, wall = run_episode(task, start, idx)
        m = metrics(traj, task)
        m.update(task=task, direction="clockwise" if task == 0 else "counter-clockwise", start_deg=start,
                 ended_early=reason, cmd_diff_max=float(f"{d_cmd:.3g}"))
        results.append(dict(m, trajectory_xyz=traj[:, 0:3].round(4).tolist()))
        lap = f"one lap in {m['lap_s']:.1f} s" if m["lap_s"] else f"swept {m['swept_deg']:.0f} deg (no full lap)"
        print(f"task {task} ({m['direction']}), start {start:.0f} deg: flew {m['flown_s']:.1f} s, {lap}; "
              f"radius err {m['radius_err_rms_m'] * 100:.1f} cm rms, height err {m['height_err_rms_m'] * 100:.1f} cm rms, "
              f"nose off box {m['nose_off_box_mean_deg']:.1f} deg, speed {m['speed_mean_mps']:.2f} m/s, "
              f"|vz| max {m['vz_abs_max_mps']:.2f} m/s; cmd diff {d_cmd:.1e}; {wall:.0f} s wall"
              + (f"; ENDED EARLY: {reason}" if reason else ""))
        if frames is not None:
            from utils.video_io import write_video        # grad_nav's imageio-based writer (simulator patch)
            name = f"task{task}_start{start:.0f}.mp4"
            write_video(os.path.join(out_dir, name), torch.stack(frames), fps=round(1 / env.dt))
        idx += 1

worst = max(r["cmd_diff_max"] for r in results)
print(f"largest command difference between deployment code and simulator: {worst:.1e}"
      + ("  (OK)" if worst < 1e-4 else "  (TOO LARGE: the policy is not getting the inputs it was trained on)"))

if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(dict(bundle=bundle, latency=args.latency, dr=args.dr, seed=args.seed, steps=args.steps,
                       orbit=dict(center=C.tolist(), radius=R, height=H, speed=V), episodes=results), f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    a = np.linspace(0, 2 * np.pi, 361)
    ax.plot(C[0] + R * np.cos(a), C[1] + R * np.sin(a), "k--", lw=1, label=f"target circle r = {R} m")
    ax.plot(*C, "s", color="saddlebrown", ms=12, label="box")
    for r_ in results:
        xyz = np.array(r_["trajectory_xyz"])
        ax.plot(xyz[:, 0], xyz[:, 1], lw=1.2, label=f"task {r_['task']}, start {r_['start_deg']:.0f} deg")
        ax.plot(xyz[0, 0], xyz[0, 1], "o", color=ax.lines[-1].get_color())
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Orbit policy in the 3DGS simulator (top view)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top_view.png"), dpi=120)
    print(f"wrote {out_dir}")
