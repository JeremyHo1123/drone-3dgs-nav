# Custom drone visual navigation training environment

## Project goal

Build a **drone visual navigation training environment for our own scenes**, split as follows:

- **Environment building** follows the **SousVide / FiGS** route: reconstruct a **metric-scale** 3D Gaussian Splatting scene from a phone video + an ArUco tag
- **Training** follows **GRaD-Nav++**: differentiable RL, no human demonstrations, a language-conditioned VLA policy
- **The final goal is real-drone deployment** (PX4-family flight controller + Jetson onboard computer)

Both come from the same lab (Mac Schwager's Multi-Robot Systems Lab at Stanford) and are different generations of the same line of work; scene formats and vehicle settings are interchangeable. **But nobody has connected the two before, and this junction is the main engineering work of this project**.

## Why this combination

| | What we use | Reason |
|---|---|---|
| Scene reconstruction | SousVide's **FiGS** | It open-sources the whole "phone video -> metric 3DGS" path, including the ArUco scale solve. The GRaD-Nav series does not have this part itself (it directly consumes scenes produced by SousVide) |
| Training | **GRaD-Nav++** | Needs no MPC expert or demonstration data synthesis, trains in 3.5 hours (SousVide's IL takes 12 hours), and supports language instructions |

**Explicitly not used**: SousVide's MPC expert, data synthesis and SV-Net training pipeline are all unused. Therefore **acados does not need to be compiled**: it is only the solver SousVide uses to generate IL data. This removes the most painful part of the whole installation.

## Hardware and environment constraints

| Item | Spec | Impact |
|---|---|---|
| GPU | **RTX 5070 Ti (Blackwell, sm_120, 16GB)** | ⚠️ **The biggest technical risk of this project**, see below |
| OS | Ubuntu | conda and the NVIDIA driver are already installed |
| To install | CUDA toolchain inside a new conda environment | The driver layer does not need to change |
| Mapping camera | Phone | Needs our own checkerboard intrinsics calibration |
| Flight platform | PX4 family (PixRacer/Pixhawk) + Jetson | Same architecture as the paper; its deployment flow can be reused |

### ⚠️ Blackwell compatibility is the top risk

**SousVide's official `environment_x86.yml` cannot be used as is.** It is pinned to:

```yaml
- pytorch=2.1.2
- pytorch-cuda=11.8
- nvidia/label/cuda-11.8.0::cuda-toolkit
```

**CUDA 11.8 supports at most sm_90 (Hopper) and does not recognize sm_120 at all.** Running `conda env create -f environment_x86.yml` directly produces an environment that installs but does not run; the typical symptom is the runtime error `no kernel image is available for execution on the device`.

We must build our own environment: CUDA 12.8+, PyTorch 2.7+ (cu128), gsplat compiled from source. Details in `README.md` section 1 "Environment setup".

**This has not been tested on the target machine**; it is derived from Blackwell's architecture requirements. `README.md` section 3 "Verify the environment" provides verification commands; confirm each step passes before moving on.

## Staged goals and acceptance criteria

Do not skip ahead in the implementation order; every stage has a clearly **verifiable** output:

| Stage | Output | Acceptance criteria |
|---|---|---|
| **1. Environment** | A working conda env | `torch.cuda.is_available()` is True and matrix operations run on the GPU; gsplat imports and completes one render |
| **2. Run the official example** | Render images with their example gsplat | **First confirm the pipeline itself works, then switch to our own scene**. Skipping this step makes later debugging impossible to localize |
| **3. Camera calibration** | `configs/captures/<your phone>.json` | Reprojection error < 0.5 px |
| **4. Mapping** | Metric 3DGS scene + `transforms.json` | **Measure an object of known length in the scene: error < 2%** |
| **5. Point cloud** | `.ply` for the reward and A\* | No noise points in the middle of gates/passages; `validate_scene.py` start and goal point checks all pass |
| **6. Connect grad_nav** | Training runs in our own scene | Training curve does not diverge; correct first-person images are rendered |
| **7. Deployment** | Real-drone flight | In stages: first fly manually to verify the extrinsics, then enable the policy |

## Working rules for the AI assistant

- **Scale correctness is the lifeline of this project**. If the 3DGS scene is not metric, the dynamics, the obstacle-avoidance threshold (0.5 m) and the reward are all wrong as a consequence, and **nothing raises an error; it just performs poorly**. Verify every scale-related step on the spot; do not leave it for later
- **Do not skip verification steps**. Most failures in this pipeline are silent
- On version conflicts, **prioritize Blackwell compatibility** (CUDA 12.8+ / PyTorch cu128) first, then accommodate the upstream repos' version pins. The upstream pins were written in 2024, before Blackwell was released
- When modifying upstream repo code, **record what was changed and why**, because grad_nav has several hard-coded values written for their own drone (codename `carl`)
- When you need to verify upstream behavior, **read the source code directly**; do not guess. Neither repo is large

## Known hard-coding traps

grad_nav hard-codes the original authors' hardware and scenes in several places; these **must be changed** when switching to our own environment:

1. The camera intrinsics in `utils/gs_local.py`: `fx=462.956, fy=463.002, cx=323.076, cy=181.184` (640×360): this is a RealSense D435
2. The three constant matrices in `pose2nerf_transform()` in `utils/gs_local.py`, where `T_r2d` is the camera's **mounting extrinsics** on the vehicle (15.2 cm forward, tilted down 8°)
3. `utils/gs_local.py` and `envs/*.py` each have their own `maps` dict; **add** our scene **to both**
4. `gs_origin_offset = [-6.0, 0, 0]` in `envs/drone_long_traj.py`
5. The `task_table` in `envs/drone_vla_*.py`: instruction strings and their waypoints must be replaced with our own tasks

## Managing expectations

The following are **inherent limitations** of this approach, not implementation shortcomings:

- **The policy is scene-specific**. Its zero-shot transfer only bridges the rendering gap between "reconstruction vs real footage", **not across scenes**. A different room requires re-filming and retraining
- **Fragile to static scene changes**. SousVide measured this: when objects that were in the scene during training are removed, the policy consistently flies through where those objects used to be (success rate drops from 96% to 25%); in contrast, people walking around in the scene have almost no effect
- **No collision termination in simulation**; the drone can fly through walls. Obstacle avoidance relies only on a distance reward within 0.5 m, a soft constraint. The real drone will crash; the paper's real-world success rate is 6-7/10
- **Language instructions are not open-vocabulary**. GRaD-Nav++'s vocabulary is 12 combinations of 4 directions × 3 targets; the policy is only defined near the instructions it was trained on
- **Low-light failure**. When brightness drops below 40% of the original, SousVide's policy always drifts off course right from the start

## References

Links to the three papers are in the "References" section of `README.md`; the code download commands are in section 2 "Upstream repositories", and running them puts the code in `repos/`. Reading priority:

1. **SousVide** (arXiv 2412.16346): source of the mapping method; section III-A is FiGS
2. **GRaD-Nav++** (arXiv 2506.14009, RA-L): the training method
3. **GRaD-Nav** (arXiv 2503.03984): the predecessor, origin of the differentiable RL and CENet; read it before ++
