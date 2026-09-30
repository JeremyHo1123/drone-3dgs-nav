# Chapter 1 Environment installation log

Date: 2026-08-11
Environment name: `droneenv` (plus the helper env `sfmtools`)

## Final installed versions

| Package | Version | Notes |
|---|---|---|
| Python | 3.10.20 | |
| torch / torchvision | **2.11.0+cu128** / 0.26.0+cu128 | arch list includes sm_120 |
| gsplat | **1.4.0 (built from source, AOT)** | all 24 cubins are sm_120 |
| nerfstudio | 1.1.5 | |
| numpy | 1.26.4 | downgraded by nerfstudio, compatible with torch 2.11 |
| opencv-contrib-python | 4.10.0.84 | replaces nerfstudio's headless build |
| open3d | 0.19.0 | |
| gym | 0.26.2 | |
| transformers | **5.15.0** | ⚠️ see "To watch" |
| COLMAP | 4.0.4 (CUDA) | in the `sfmtools` env |
| ffmpeg | 8.1.2 | in the `sfmtools` env |
| nvcc | system 12.8.93 @ /usr/local/cuda | same version as torch cu128 |

## Deviations from the original plan (`implement.md`, not published), with reasons

### 1. No conda cuda-toolkit (originally section 1.4)
The system already has CUDA 12.8.93, and `torch.utils.cpp_extension.CUDA_HOME` automatically points to
`/usr/local/cuda`. The nvcc version **exactly matches** torch's cu128 build,
so there is no version skew. This saves about 3 GB and avoids two nvcc installs clashing.

### 2. MAX_JOBS 4 -> 12
This machine has 94 GB RAM / 24 cores; the plan's 4 was written for low-memory machines.

### 3. gsplat built at 1.4.0 rather than latest
nerfstudio 1.1.5 depends on `gsplat ==1.4.0` (exact pin).
Installing the latest version would leave a version conflict. The approach: **keep its version number, but build it ourselves**:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"; export MAX_JOBS=12; export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation --force-reinstall --no-deps \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
```

⚠️ **Order**: gsplat must be installed **after** nerfstudio, otherwise it gets overwritten by nerfstudio's
PyPI wheel.

⚠️ `gsplat-1.4.0` on PyPI is a `py3-none-any` **pure JIT wheel** (no .so).
It does not report `no kernel image` right away; instead it compiles with the local nvcc on the first call --
the risk is a stall halfway through training, and if the build fails the whole run is lost. AOT precompilation removes this risk.

### 4. opencv: remove headless, use contrib
nerfstudio pins `opencv-python-headless ==4.10.0.84`, but it **has no aruco**.
Installed `opencv-contrib-python==4.10.0.84` instead (same version, functionality is a superset).
pip leaves an unsatisfied-requirement warning, which is expected.

### 5. colmap / ffmpeg in a separate env `sfmtools`
sudo requires a password, so the apt route gets stuck. Installing via conda directly into droneenv would pull in
qt-main 5.15 + pango + nss + the whole xorg stack, and Qt conflicts would threaten chapter 7's
open3d visualization. Approach: a separate env + its `bin` **appended to the end of PATH**
(the end is deliberate: droneenv's python must take precedence).
conda executables find their dependencies via RPATH `$ORIGIN/../lib`, so that Qt does not leak in.

## Two extra pitfalls handled (not mentioned in the plan)

### A. ROS 2 Jazzy PYTHONPATH leak
The system globally sets `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`,
which is inserted at the **first position** of sys.path. That is a python3.12 directory, while droneenv is 3.10.
A scan found no direct name collisions so far, but the risk grows with every new package installed.

Fix: `droneenv/etc/conda/activate.d/zz_isolate_ros.sh` clears PYTHONPATH only inside this env,
and deactivate restores it. **The global ROS is unaffected** (needed by chapter 12).

### B. Incomplete dependency declarations in the conda-forge colmap package
Neither `colmap 4.1.1` nor `4.0.4` **declares a `faiss` dependency**; once installed, they die at runtime with
`undefined symbol: faiss::IndexIVFFlat::IndexIVFFlat(Index*, ulong, ulong, MetricType)`.
Newer faiss (1.12+) changed that constructor's signature.

Fix: pin `libfaiss 1.10.*` + `colmap 4.0.*`, written to
`sfmtools/conda-meta/pinned`, so later conda operations do not upgrade and break it again.

## Verification results (all passed)

```
A. Environment hygiene
  no ROS leak in sys.path
  python: droneenv/bin/python   colmap: sfmtools/bin/colmap   ffmpeg: sfmtools/bin/ffmpeg
B. PyTorch / Blackwell
  torch 2.11.0+cu128 | capability (12,0) | sm_120 in arch list: True | matmul ok
C. gsplat
  1.4.0 | compiled archs {'sm_120'} | forward render max 0.9347 | backward grad norm 16541.80
D. Dependencies
  cv2 4.10.0 (aruco ok) | open3d 0.19.0 | numpy 1.26.4 | gym 0.26.2 | transformers 5.15.0
  COLMAP 4.0.4 | ffmpeg 8.1.2
E. nerfstudio
  check_ffmpeg_installed + check_colmap_installed passed
  ns-train / ns-process-data / ns-export / ns-viewer all available, splatfacto exists
```

Checks beyond the plan (the plan only requires `import gsplat`, which is not enough):
- Used `cuobjdump --list-elf` to confirm the `.so` really contains sm_120 machine code
- Actually ran rasterization **forward + backward**. GRaD-Nav uses differentiable RL,
  so gradients must flow through the renderer; checking only the forward pass is not enough
- ArUco generate -> detect round trip, and confirmed `SOLVEPNP_IPPE_SQUARE` exists
- Called nerfstudio's `install_checks` directly; that is the real gate for chapter 6

## Early findings for chapter 6

`ColmapConverterToNerfstudioDataset.__post_init__` **unconditionally** calls
`check_ffmpeg_installed()` and `check_colmap_installed()`, and does `sys.exit(1)` on failure.
**It blocks even with `sfm_tool="hloc"`** (hloc uses pycolmap internally, not the executables).
This is why the colmap/ffmpeg executables must be installed.

The three scale-preserving flags belong to the **`nerfstudio-data` dataparser**, not the method,
and must be written after `nerfstudio-data`. Measured defaults:

| Flag | Default | Must be set to |
|---|---|---|
| `--orientation-method` | `up` | `none` |
| `--center-method` | `poses` | `none` |
| `--auto-scale-poses` | `True` | `False` |

All three defaults **happen to destroy the metric scale**.

## To watch

- **transformers 5.15.0**: the CLIP classes (CLIPModel/CLIPProcessor/...) are all still there,
  so imports do not fail. But 5.x is a major version jump and grad_nav was written against 4.x;
  processor defaults and return types may differ. **Verify when actually running CLIP in chapters 9/11**,
  and downgrade to 4.x if needed.
- gym 0.26.2 prints a "NumPy 2.0 not supported" warning. numpy is currently 1.26.4, so no impact.
