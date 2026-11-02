# grad_nav simulator patch

`grad_nav_simulator_v4.patch` turns upstream [grad_nav](https://github.com/Qianzhong-Chen/grad_nav) into the simulator the [orbit policy](../../orbit_policy/README.md) was trained in: this project's drone, camera and scenes, the orbit-the-box task, and a model of the real drone's command latency.

- Base: upstream commit `95218ab` ("Merge pull request #11").
- Contains only the environment side (`envs/`, `utils/`). The changes to the training side (`algorithms/`, the training configs under `examples/cfg/`) are not published.
- Applying it to a clean `95218ab` gives exactly the files the v4 policy's acceptance runs used (only comments and messages were translated to English; checked token by token, and `run_in_sim.py` gives bit-identical episodes with both).

```bash
git clone https://github.com/Qianzhong-Chen/grad_nav.git repos/grad_nav
git -C repos/grad_nav checkout 95218ab
git -C repos/grad_nav apply --whitespace=nowarn ../../configs/grad_nav/grad_nav_simulator_v4.patch
```

The scene and the obstacle point cloud are not in the patch; [`orbit_policy/README.md`](../../orbit_policy/README.md), steps 2 and 3, sets them up.

## What it changes

| File | Change |
|---|---|
| `envs/drone_vla_orbit.py` (new) | The orbit-the-box task in `sunny_baseball_field`: start on the circle, orbit velocity field, reward, early termination, yaw-free observation (`ORBIT_OBS_FRAME: heading`), the episode start used on the drone (`ORBIT_START_MODE`). Its header lists every difference from upstream and why |
| `envs/assets/command_latency.py` (new) | Command delay (30–120 ms + jitter), first-order motor response with separate up/down time constants, and a throttle-to-thrust exponent, drawn per drone and episode. Enabled by `LATENCY_MODEL` in `env_hyper` |
| `envs/assets/height_integrator.py` (new) | Integral action on height, added to the thrust after the policy (`HEIGHT_INTEGRATOR`). The onboard code does the same |
| `utils/video_io.py` (new) | Replacement for `torchvision.io.write_video`, which torchvision 0.26 removed |
| `utils/gs_local.py` | Camera intrinsics of the Orbbec Gemini 2 (640×360) instead of the RealSense D435; this drone's camera mounting extrinsics; the two scenes in the `maps` dict |
| `envs/assets/quadrotor_dynamics_advanced.py` | Fixes a sign error in the quaternion-to-rotation-matrix formula (pure pitch was reversed); optional `thrust_force` input for the latency model |
| `envs/drone_vla_long_task.py` | This drone's airframe (mass, thrust, inertia, delays, safety radius — sources in the comments); the `scene04` route; latency model and height integrator hooks in `step()`; a fix for a hard-coded start point |
| `envs/drone_long_traj.py`, `drone_multi_gate.py`, `drone_ppo.py`, `drone_vla_multi_map.py` | Same airframe and video-writer changes; `scene04` in `drone_long_traj.py`; the same start-point fix in `drone_vla_multi_map.py` |
| `envs/__init__.py` | Registers `DroneVLAOrbitEnv` |

`orbit_policy/run_in_sim.py` builds the environment from the settings stored in the policy's `weights.pt` (`meta["env_hyper"]`), so no config file is needed to run the policy.

## License

grad_nav is under the NVIDIA Source Code License (non-commercial: research or evaluation only), and so is this patch as a derivative work. The full text is in [`orbit_policy/LICENSE-grad_nav.md`](../../orbit_policy/LICENSE-grad_nav.md).
