# drone-3dgs-nav

Building a **metric-scale** 3D Gaussian Splatting environment from a hand-held phone video, so a drone visual-navigation policy can be trained in it and then deployed on real hardware.

Two upstream projects are joined here, both from Stanford's Multi-Robot Systems Lab:

| Part | Source | Why |
|---|---|---|
| Scene reconstruction | **FiGS**, from [SousVide](https://github.com/StanfordMSL/SousVide) | The only open pipeline that goes phone video → metric 3DGS, with ArUco for scale |
| Policy training | [**GRaD-Nav++**](https://arxiv.org/abs/2506.14009) | Differentiable RL — no MPC expert, no demonstrations, 3.5 h of training, language-conditioned |

They share scene formats and vehicle configs, but **nobody has connected them before**. That junction is the main engineering work of this project.

Project background, rationale, and known limitations of this approach are in [`CLAUDE.md`](CLAUDE.md). Read that first if you intend to reuse this.

---

## Example scenes

Two complete scenes, each reconstructed from a single hand-held iPhone 12 video and checked against a tape measure.

<img src="scenes/scene04/figures/scene_map.png" width="49%"> <img src="scenes/sunny_baseball_field/figures/scene_map.png" width="49%">

| | [`scene04`](scenes/scene04/README.md) | [`sunny_baseball_field`](scenes/sunny_baseball_field/README.md) |
|---|---|---|
| Place | Tree-lined outdoor plaza | Grass baseball field, sunny day |
| Video | 5 min | 5.7 min (first 16 s skipped: camera operator's shadow) |
| Scale check | Box stack 172.5 cm vs tape 171.0 cm: **+0.87%** | Independent close-range tag cross-check: **0.65%**; box stack 170.7 cm vs tape 171.0 cm: −0.19% |
| Ground tilt, flatness | 1.14°, RMS 5.5 mm over 30 m | 0.85° (after leveling), RMS 5.7 mm |
| Dense point cloud (in this repo) | 978,285 points, 26 MB | 960,747 points, 25 MB |
| Trained checkpoint | [`scene04-v1` release](https://github.com/JeremyHo1123/drone-3dgs-nav/releases/tag/scene04-v1), 1.8 GB | [`sunny_baseball_field-v1` release](https://github.com/JeremyHo1123/drone-3dgs-nav/releases/tag/sunny_baseball_field-v1), 1.38 GB |

The dense point clouds are checked into this repo — **no GPU or setup needed to open them**:

```python
import open3d as o3d
o3d.visualization.draw_geometries([
    o3d.io.read_point_cloud("scenes/scene04/scene04_dense.ply")])
```

Both frames are metric and gravity-aligned: origin at the ArUco tag, +z up, distances in metres.

The trained checkpoints are too large for a git repo, so each one is attached to a GitHub release (linked above). How to download and load them, the file layout they need, and the three things that break loading are in each scene's README.

---

## What is in this repo

| | |
|---|---|
| [`tools/`](tools/) | 18 scripts — camera calibration, ArUco, capture pre-check, frame extraction and mapping, intrinsics hand-over, scale verification, export, evaluation, dynamics checks |
| [`notes/`](notes/) | Chapter-by-chapter implementation records, including every deviation from upstream and why |
| [`configs/`](configs/) | Camera intrinsics and capture configs |
| [`scenes/`](scenes/) | Two complete, scale-verified example scenes |
| [`CLAUDE.md`](CLAUDE.md) | Project goals, constraints, expectation management |

**Not in this repo:** the upstream repos, capture videos, extracted images, SfM intermediates, and training checkpoints — about 49 GB in total. All of it is either downloadable or reproducible. Each section below says how.

---

## 0. Prerequisites

| | Requirement | Check with |
|---|---|---|
| GPU | NVIDIA. **Blackwell (sm_120) recommended** — verified on RTX 5070 Ti and RTX PRO 6000 | `nvidia-smi` |
| CUDA toolkit | **12.8 or newer**, installed system-wide at `/usr/local/cuda` | `nvcc --version` |
| conda | Any distribution | `conda --version` |
| OS | Ubuntu 24.04 | others untested |

### ⚠️ Why the CUDA version matters

SousVide's official `environment_x86.yml` pins `pytorch-cuda=11.8`. **CUDA 11.8 tops out at sm_90 (Hopper) and does not recognise sm_120 (Blackwell) at all.**

Using that file gives you an environment that installs cleanly and then fails at runtime:

```
no kernel image is available for execution on the device
```

So the steps below **do not use the official yml**. Every version is pinned by hand.

---

## 1. Environment setup

Two conda environments. The reason they are separate is in section 1.3.

### 1.1 Main environment `droneenv`

```bash
conda create -n droneenv python=3.10 -y
conda activate droneenv
```

**The order below matters.** Each step explains why.

**Step 1 — PyTorch, cu128 build**

```bash
pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

**Step 2 — nerfstudio**

```bash
pip install nerfstudio==1.1.5
```

This downgrades numpy to `1.26.4`. **That is expected — do not undo it.** nerfstudio 1.1.5 is not numpy 2.x compatible, and torch 2.11 is fine with 1.26.4.

**Step 3 — gsplat, built from source, after nerfstudio**

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=12          # scale to your core count; use 4 on low-RAM machines
export CUDA_HOME=/usr/local/cuda

pip install --no-build-isolation --force-reinstall --no-deps \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
```

Three things to get right:

1. **Order.** gsplat must be installed *after* nerfstudio, or nerfstudio's dependency resolution overwrites your build with the PyPI wheel.
2. **Version 1.4.0, not latest.** nerfstudio 1.1.5 pins `gsplat ==1.4.0` exactly.
3. **Why build from source.** The PyPI `gsplat-1.4.0` is a `py3-none-any` JIT-only wheel with no compiled `.so`. It does not fail immediately — it compiles on the first render call using the local nvcc. That means a stall in the middle of training, and a lost run if compilation fails. Building ahead of time removes that risk entirely.

**Step 4 — swap OpenCV**

```bash
pip uninstall -y opencv-python-headless
pip install opencv-contrib-python==4.10.0.84
```

nerfstudio pins `opencv-python-headless==4.10.0.84`, which **has no ArUco module** — and ArUco is the only thing in this pipeline that knows how long a metre is. `opencv-contrib-python` is the same version with a superset of features.

pip will leave an unsatisfied-requirement warning afterwards. **That is expected.**

**Step 5 — remaining dependencies**

```bash
pip install open3d==0.19.0 gym==0.26.2
```

### 1.2 Isolate a leaked ROS 2 PYTHONPATH

If the machine sets `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` globally, it lands **first** on `sys.path`. That is a python 3.12 package directory and `droneenv` is 3.10 — incompatible ABI, and any name collision shadows the real package.

Clear it inside this environment only:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/zz_isolate_ros.sh << 'EOF'
export _DRONEENV_SAVED_PYTHONPATH="${PYTHONPATH:-}"
unset PYTHONPATH
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/zz_isolate_ros.sh << 'EOF'
export PYTHONPATH="${_DRONEENV_SAVED_PYTHONPATH:-}"
unset _DRONEENV_SAVED_PYTHONPATH
EOF
```

Scripts in `activate.d` run on `conda activate`; `deactivate.d` restores on exit. **The global ROS 2 install is untouched.**

### 1.3 Helper environment `sfmtools` (COLMAP + ffmpeg)

```bash
conda create -n sfmtools -y
conda activate sfmtools
conda install -c conda-forge "colmap=4.0.*" "libfaiss=1.10.*" ffmpeg -y

# pin, so a later conda operation cannot upgrade them and break things
printf 'libfaiss 1.10.*\ncolmap 4.0.*\n' > $CONDA_PREFIX/conda-meta/pinned
```

**Why a separate environment.** Installing colmap into `droneenv` pulls in qt-main 5.15, pango, nss and a full xorg stack. That Qt conflicts with Open3D's visualisation.

**Why libfaiss must be pinned.** The conda-forge colmap package **does not declare its faiss dependency**. Run it as-is and it dies with:

```
undefined symbol: faiss::IndexIVFFlat::IndexIVFFlat(Index*, ulong, ulong, MetricType)
```

faiss 1.12+ changed that constructor signature. Pinning to 1.10 avoids it.

**Expose colmap and ffmpeg to droneenv:**

```bash
conda activate droneenv

cat > $CONDA_PREFIX/etc/conda/activate.d/zz_sfmtools_path.sh << 'EOF'
export _DRONEENV_SAVED_PATH="$PATH"
export PATH="$PATH:$(conda info --base)/envs/sfmtools/bin"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/zz_sfmtools_path.sh << 'EOF'
export PATH="${_DRONEENV_SAVED_PATH:-$PATH}"
unset _DRONEENV_SAVED_PATH
EOF
```

**Appending rather than prepending is deliberate.** droneenv's own python must win; it must not be shadowed by sfmtools' python. conda-forge binaries resolve their own dependencies through `RPATH $ORIGIN/../lib`, so that Qt stack never leaks in.

**Why these two executables are mandatory:** nerfstudio's `ColmapConverterToNerfstudioDataset.__post_init__` calls `check_ffmpeg_installed()` and `check_colmap_installed()` unconditionally and `sys.exit(1)`s on failure — even with `sfm_tool="hloc"`, which uses pycolmap internally and needs neither executable.

---

## 2. Upstream repositories

```bash
cd <repo root>
mkdir -p repos && cd repos

git clone --recursive https://github.com/StanfordMSL/SousVide.git
git clone https://github.com/Qianzhong-Chen/grad_nav.git
```

`--recursive` is required — FiGS is a submodule of SousVide.

### ⚠️ acados is not needed

SousVide's install instructions tell you to build acados. **Skip it.** acados is only the MPC solver used to synthesise imitation-learning data, and this project takes the GRaD-Nav++ differentiable-RL route instead. That removes the most painful part of the whole install.

### Put the capture configs where the tools expect them

`tools/build_gsplat.py` reads configs from `repos/SousVide/configs/`. Copy the example configs there (the tools in section 4 write new configs there directly):

```bash
cd <repo root>
mkdir -p repos/SousVide/configs/captures repos/SousVide/configs/camera
cp configs/captures/*.json repos/SousVide/configs/captures/
cp configs/camera/*.json   repos/SousVide/configs/camera/
```

---

## 3. Verify the environment

**Failures in this pipeline are usually silent** — nothing errors, the result is just worse. Verify at every step.

```bash
conda activate droneenv
python - << 'EOF'
import torch, gsplat, cv2, open3d, numpy
print("torch     :", torch.__version__)
print("cuda ok   :", torch.cuda.is_available())
print("capability:", torch.cuda.get_device_capability())   # (12, 0) on Blackwell
print("arch list :", torch.cuda.get_arch_list())           # must contain 'sm_120'
print("gsplat    :", gsplat.__version__)                   # must be 1.4.0
print("cv2       :", cv2.__version__, "| aruco:", hasattr(cv2, "aruco"))
print("open3d    :", open3d.__version__)
print("numpy     :", numpy.__version__)                    # must be 1.26.x
EOF
```

**`import gsplat` alone is not enough.** This project uses differentiable RL, so gradients have to flow back through the rasteriser — exercise both directions, and confirm the compiled `.so` really contains sm_120 machine code rather than waiting to JIT:

```bash
python - << 'EOF'
import glob, os, gsplat
print("compiled .so:", glob.glob(os.path.join(os.path.dirname(gsplat.__file__), "*.so")))
EOF
cuobjdump --list-elf $(python -c "import glob,os,gsplat;print(glob.glob(os.path.join(os.path.dirname(gsplat.__file__),'*.so'))[0])") | head
```

Then confirm the nerfstudio executables and the two helper binaries resolve:

```bash
for c in ns-train ns-viewer ns-export ns-process-data; do
  command -v $c >/dev/null && echo "$c ok" || echo "$c MISSING"
done
colmap -h 2>&1 | head -1
ffmpeg -version | head -1
```

---

## 4. Building your own scene

The complete procedure for a new scene, in order. Replace `<phone>`, `<scene>` and `<config>` with your own names — for example `pixel8`, `lab01` and `pixel8_lab01`. Run everything from the repo root with `droneenv` active. The shell scripts find conda through `$CONDA_EXE`, which `conda init` sets; if they cannot find it, run `export CONDA_BASE=/path/to/your/conda` first.

| Step | What you do | Result |
|---|---|---|
| 4.1 | Calibrate the phone camera with a checkerboard (once per phone) | `repos/SousVide/configs/camera/<phone>.json` |
| 4.2 | Print the ArUco tag, create the scene's capture config | `repos/SousVide/configs/captures/<config>.json` |
| 4.3 | Film the scene | `repos/SousVide/gsplats/capture/<scene>.MOV` |
| 4.4 | Run frames, SfM, check | camera poses and COLMAP's own intrinsics |
| 4.5 | Copy COLMAP's intrinsics into the config | config matches the SfM |
| 4.6 | Run scale, train, export, verify | metric 3DGS scene and dense point cloud |
| 4.7 | Measure something with a tape | scale error < 2% |
| 4.8 | Prepare the point cloud for policy training | downsampled `.ply` |

### 4.1 Calibrate the phone camera (once per phone)

Camera **intrinsics** are the numbers that describe the lens: the focal length in pixels (`fx`, `fy`), the image center (`cx`, `cy`) and the lens distortion. The metric scale depends directly on `fx`: an `fx` that is 2% too large makes the whole scene 2% too large, and nothing reports an error. This step gives a first estimate and a reference. Step 4.5 later replaces it with values estimated from the scene video itself.

**Record the calibration video in exactly the video mode you will use for your scenes.** The intrinsics are only valid for that mode:

- The **1x wide lens**. Never the 0.5x ultra-wide (it is a different physical lens), and no zoom.
- The same orientation (portrait or landscape), resolution and frame rate as the scene videos, e.g. 1080p at 30 fps.
- On iPhone: Settings → Camera → Formats → **Most Compatible** (H.264). OpenCV may not be able to read HEVC.
- Copy the file to the computer **without re-encoding** (for example over a USB cable). The first calibration attempt for this repo arrived re-encoded at 2.1 Mbps instead of about 16 Mbps and was unusable.

**Print and film the checkerboard:**

```bash
python tools/make_checkerboard.py      # writes captures/checkerboard_9x6_A4.pdf (9x6 inner corners)
```

- Print the PDF at 100%. The exact square size does not matter, but the squares must stay square: the page carries a horizontal and a vertical ruler line, nominally 100 mm each, and the two must come out the same length.
- Glue the print onto a flat, rigid board. A wavy sheet taped to a wall gave a bad calibration here.
- Film it from many angles and distances, and **sweep it through every part of the image, especially the corners and edges**. If all the corners sit in the middle of the image, the distortion near the edges is left undetermined.

```bash
python tools/calibrate_camera.py --video <path/to/checkerboard.MOV> --name <phone>
```

It writes `repos/SousVide/configs/camera/<phone>.json` and prints these checks:

| Check | Pass | What it catches |
|---|---|---|
| Mean reprojection error | < 0.5 px | the lens model cannot explain the detected corners |
| Share of corners in the emptiest cell of a 3×3 grid over the image | ≥ 3% | corners bunched in the middle; distortion undetermined near the edges |
| (a) Distortion shift at the image corner | < 25 px | an ill-conditioned distortion solution |
| **(b) Change in `fx` when the distortion model is changed** | **< 1%** | **the one that matters for scale** — `fx` uncertainty turns directly into scale error |

If a check fails, the file is written as `<phone>.REJECTED.json` so that it cannot be used by accident. Re-record rather than override it. (`--force` writes the normal name anyway. The iPhone 12 calibration in this repo was accepted that way with only 1.9% in its emptiest cell — and step 4.5 later showed its `fx` to be about 2% too large.)

**A low reprojection error does not mean an accurate `fx`.** In a synthetic test, a calibration with one of the nine cells empty had a reprojection error of only 0.037 px, yet its `fx` was 1.1% off. The full story is in [`notes/03-calibration.md`](notes/03-calibration.md).

### 4.2 Print the ArUco tag and create the scene's capture config

The ArUco tag is a printed black-and-white square placed in the scene. **It is the only thing in the entire pipeline that knows how long a metre is.**

```bash
python tools/make_aruco.py --page 210 297        # A4; use 297 420 for A3. Writes to captures/
```

- ⚠️ **Measure the printed black square with a ruler** — the full side of the black area, including its outer black border but not the white margin — and use that value, in metres. Do not use the design value: printer scaling shifts it by a few percent, and that error passes straight through to your scene scale.
- Glue it flat onto a rigid board.

Each scene gets **its own capture config**, because step 4.5 writes that scene's own intrinsics into it. `make_capture_config.py --name <config>` reads the camera file with the **same name**, so copy your phone calibration to that name first:

```bash
cp repos/SousVide/configs/camera/<phone>.json repos/SousVide/configs/camera/<config>.json
python tools/make_capture_config.py --name <config> --marker-length 0.144 --num-images 600
```

This writes `repos/SousVide/configs/captures/<config>.json`, which has a `camera` block (the intrinsics) and an `extractor` block (`num_images`, `num_marked`, `marker_length`, `marker_id`).

Optional: if the start of your video will show you or your shadow, add `"skip_start_sec": <seconds>` to the `extractor` block. Frames before that time are never used.

To keep a config in git, copy it to `configs/captures/` (and its camera file to `configs/camera/`). That is where the configs of the two example scenes are.

### 4.3 Capture the scene

**Place the tag flat on level ground**, near the middle of the area you want to fly in. The tag defines the origin and the up direction (+z) of the scene, so a tag tilted by a few degrees tilts the whole scene. (That can be corrected in step 4.6 with `--level-ground`.)

**Record hand-held, in the same video mode as the calibration video:**

- **Film the tag from close up**, at several points along the way: at least 4 frames in which the tag is **at least 100 px wide** in the image. For the 14.4 cm tag and the iPhone 12 used here, that means the camera within about 2.4 m of the tag (distance ≈ `fx` × tag side ÷ 100 px). By default the scale solve uses only these close-range frames (`--min-tag-px 100`), because distant tags bias the scale.
- The tag must be clearly visible in at least `num_marked` frames (default 20) in total.
- **Keep yourself and your shadow out of the frame**, above all in the close-range tag shots. A moving shadow turns into stains or floaters in the 3DGS. (`sunny_baseball_field` had to drop its first 16 s for this reason, and most of its close-range tag shots with them.)
- **Walk a loop, and vary your height** — crouch, stand, reach up. A straight-line path produces a degenerate reconstruction that passes every SfM check and still fails on scale (see [`notes/07-scene-experiments.md`](notes/07-scene-experiments.md)).
- Move slowly; motion blur causes SfM registration failures.
- Keep the camera pointed at things with texture, not blank walls.

**Name the file** `<scene>.MOV` or `<scene>_<anything>.MOV` (for example `lab01_IMG_1234.MOV`) and put it in `repos/SousVide/gsplats/capture/`. Exactly one file there may match the scene name.

**Pre-check it** before spending hours on SfM:

```bash
python tools/preflight_capture.py --video repos/SousVide/gsplats/capture/<scene>.MOV --config <config>
```

It sorts every frame with the same logic the pipeline uses, and tells you whether there are enough frames with and without the tag.

### 4.4 Run the first half: frames, SfM, check

```bash
./tools/run_pipeline.sh --scene <scene> --config <config> --stages frames,sfm,check
```

| Stage | What it does | Time for 600 images |
|---|---|---|
| `frames` | Extract frames by 1-D farthest point sampling | ~6 min |
| `sfm` | hloc SuperPoint + SuperGlue, exhaustive matching | **~2 h 40 min** |
| `check` | Verify registration rate and tag-frame count — **changes nothing** | seconds |
| `scale` | Solve `Sim(3)` from ArUco, write metric `transforms.json` | ~40 s |
| `train` | `ns-train splatfacto`, 30k steps | ~20 min |
| `export` | Back-project a dense point cloud | ~27 min |
| `verify` | Automatic ground-plane and plane-distance checks | ~1 min |

**`check` is a separate stage on purpose.** If the tag-bearing frames are not all registered, the downstream step throws `Mismatched number of aruco and sfm transforms` — and by then SfM has already burned hours.

`--select sharp` makes the `frames` stage take the sharpest frame within each sampling bin, instead of the upstream `uniform` sampling spread over time.

Running `./tools/run_pipeline.sh --scene <scene> --config <config>` without `--stages` runs all seven stages in one go. For a new scene, don't: it skips step 4.5, so the scale solve would use your checkerboard `fx`.

**SfM cost is quadratic in image count.** Exhaustive matching compares every pair:

| Images | Pairs | SfM time |
|---|---|---|
| 300 | 44,850 | 30 min (measured) |
| 600 | 179,700 | 2 h 40 min (measured) |
| 1000 | 499,500 | ~5.5 h |
| 5000 | 12,497,500 | ~6 days |

More images is not better past a point. On the same video with the same path, 300 → 400 images changed eval PSNR from 21.12 to 20.86. **Trajectory shape dominates; image count does not.**

### 4.5 Put the SfM's own intrinsics into the config — do not skip this

During SfM, COLMAP self-calibrates the camera from your scene video, and both SfM and 3DGS training use those values. The scale solve, however, reads the `camera` block of your config. If the two disagree, the scene scale is off by the same percentage, silently (section 6.1). Copy them over:

```bash
python tools/use_sfm_intrinsics.py --scene <scene> --config <config> --dry-run   # only show the difference
python tools/use_sfm_intrinsics.py --scene <scene> --config <config>
```

It prints how far your config's `fx` was from COLMAP's. For the iPhone 12 used here, the checkerboard `fx` was 2.1–2.3% larger than COLMAP's in both example scenes; left in place, it would have made each scene about 2% too large. Treat a much larger difference as a sign that the calibration or the SfM went wrong.

It rewrites the `camera` block of `repos/SousVide/configs/captures/<config>.json`, and also `repos/SousVide/configs/camera/<config>.json` and the copies under `configs/` if they exist. The `extractor` block is left as it is.

### 4.6 Run the second half: scale, train, export, verify

```bash
./tools/run_pipeline.sh --scene <scene> --config <config> --stages scale,train,export,verify
```

Options for the `scale` stage (they can be combined):

| Option | Default | What it does, and when to use it |
|---|---|---|
| `--min-tag-px N` | `100` | Only tag observations at least N px wide are used for the scale solve; if fewer than 4 qualify, the 4 largest are used. `0` uses every observation, as upstream does (scene04 was built that way) |
| `--tag-rule present` | `exact` | For textured ground such as grass, where the detector reports small fake tags. `exact` (upstream) drops every image with more than one detection; `present` keeps an image whenever the real id is in it. The `scale` stage prints how many images had extra detections |
| `--level-ground R` | off | For a tag that was not lying flat: after the scale solve, the scene is rotated so that the ground within R m is horizontal (scale and origin unchanged). Use it when `verify` flags the floor as not level (tilt of 3° or more) on ground you know is level. `sunny_baseball_field` used `12` |

`--tag-rule` and `--level-ground` change `transforms.json`, which training reads. After changing either, re-run all four stages.

### 4.7 Verify the scale — do not skip this

```bash
./tools/open_pointcloud.sh <scene>
```

Shift + left-click two points, press Q, and the distance is printed. **Measure something you can physically reach with a tape. The error must be under 2%.** Both example scenes have a stack of cardboard boxes of known height next to the tag for exactly this.

Getting scale wrong does not raise an error. It silently corrupts the dynamics, the 0.5 m obstacle threshold, and the reward.

### 4.8 Prepare the cloud for policy training

```bash
python tools/prepare_pointcloud.py     # outlier removal, voxel downsample, validation
python tools/verify_dynamics.py        # quadrotor parameter sanity checks
python tools/fly_orbit_preview.py      # fly a real simulated orbit, render the view
```

Downsampling is not optional: grad_nav's `ObstacleDistanceCalculator` builds four `[num_envs, num_points, 3]` tensors at once. At 128 environments, scene04's 978k points would need about 4 GB on top of the Gaussians. Under 100k points keeps it near 0.4 GB.

For large outdoor scenes, `prepare_pointcloud.py` has two optional filters (both off by default): `--crop-center X Y --crop-radius R` keeps only the area around the task, and `--min-neighbors N --neighbor-radius 0.10` removes isolated 3DGS floaters, such as those left along the capture path. See `--help`.

---

## 5. Viewing a scene

```bash
./tools/open_pointcloud.sh scene04     # Open3D window, dense cloud, click to measure
./tools/open_gsplat.sh     scene04     # nerfstudio web viewer at localhost:7007
```

`open_gsplat.sh` picks the most recent training run automatically. **Stop it with Ctrl+C** — closing the browser tab leaves it holding about 2.7 GB of GPU memory.

## Moving a trained scene to another machine

Tested by copying only the files below into an empty directory and loading them with the same `eval_setup(config, test_mode="inference")` call grad_nav uses.

**Four files. The 620 MB of source images are not among them.**

```
workspace/                                   ← working directory
├── <scene>/
│   └── transforms.json                      520 KB
└── outputs/<scene>/splatfacto/<timestamp>/
    ├── config.yml                           8 KB
    ├── dataparser_transforms.json           310 B
    └── nerfstudio_models/step-*.ckpt        1.8 GB
```

`sparse_pc.ply` is optional — without it you get an Open3D warning and loading still succeeds. It is small, so bring it anyway.

```bash
rsync -avP other-host:.../<scene>/transforms.json                        <scene>/
rsync -avP other-host:.../outputs/<scene>/splatfacto/<timestamp>/  outputs/<scene>/splatfacto/<timestamp>/
```

**Three things break this:**

1. **Do not rename the timestamp directory.** The checkpoint path is rebuilt from the `timestamp` field inside `config.yml`, not resolved relative to the file's own location.
2. **Working directory must be the common parent** of `<scene>/` and `outputs/`, because `config.yml` stores relative paths.
3. **gsplat must be 1.4.0 on both machines.** The checkpoint holds raw parameter tensors.

If you only want to look at the scene, copy the dense `.ply` (26 MB) instead — no checkpoint, no nerfstudio.

---

## 6. Pitfalls

### 6.1 ⚠️ The two places intrinsics enter — and why they must agree

**This is the failure that cost a full rebuild of scene04.** It produced a 3.37% scale error and no warning of any kind.

Camera intrinsics are consumed at three points in this pipeline:

| Step | Source of intrinsics |
|---|---|
| SfM (`stage_sfm`) | COLMAP's own self-calibration — `build_gsplat.py` never passes `cfg["camera"]` to it |
| ArUco solve (`stage_scale`) | `cfg["camera"]`, i.e. your checkerboard calibration |
| 3DGS training (`stage_train`) | `transforms.json`, i.e. COLMAP's values again |

Two of three used COLMAP's values; only the scale step used the checkerboard's. Since `solvePnP` returns a distance **proportional to fx**, a focal length 2.32% too large yields a scene 2.32% too large.

| | Checkerboard fx = 1702.28 | COLMAP fx = 1663.73 |
|---|---|---|
| ArUco scale consistency | 3.5% | **2.6%** |
| RANSAC inliers | 9 / 20 | **11 / 20** |
| Median residual | 0.080 m | **0.044 m** |
| **Scale error vs tape** | **+3.37% — failed** | **+0.87% — passed** |

Every independent metric improved together — this was not a fit to the one measurement.

**The check to run before trusting a scene:**

```bash
python - << 'EOF'
import json
sfm = json.load(open("repos/SousVide/gsplats/workspace/<scene>/sfm/transforms.json"))
cfg = json.load(open("repos/SousVide/configs/captures/<config>.json"))["camera"]
fx_cfg = cfg["intrinsics_matrix"][0][0]
d = fx_cfg / sfm["fl_x"] - 1
print(f"config fx {fx_cfg:.2f} vs SfM fl_x {sfm['fl_x']:.2f}  ->  {d*100:+.2f}%")
print("OK" if abs(d) < 0.01 else "MISMATCH — this becomes your scale error")
EOF
```

[`tools/use_sfm_intrinsics.py`](tools/use_sfm_intrinsics.py) (step 4.5) prints the same comparison and copies COLMAP's values into the config.

When they disagree, **prefer COLMAP's**. It is self-calibrated from hundreds of images across the whole scene; a checkerboard video with poor corner coverage is far more weakly constrained, and its low reprojection error will not reveal the problem.

### 6.2 Silent failures to watch for

- **`orientation-method`, `center-method`, `auto-scale-poses`.** Defaults are `up`, `poses`, `True` — **all three destroy metric scale.** They belong to the `nerfstudio-data` dataparser and must appear *after* it on the command line. `build_gsplat.py` already sets them to `none`, `none`, `False`.
- **Every SfM metric can be green while scale is wrong.** scene02 registered 100% of images at 1.398 px and still had 106.7% ArUco scale dispersion, because the camera walked a straight line. Only an external metric reference catches this.
- **Laplacian variance is not a blur metric.** It responds to image content too. In scene02 the lowest-scoring frames were perfectly sharp photographs of a blank wall. Only compare within one scene.
- **A tag that is not lying flat tilts the whole scene.** The tag's normal becomes +z. In `sunny_baseball_field` the tag board, resting on grass, tilted the scene by about 3.5° — about 60 cm of ground-height change over 10 m of flight. `verify` flags tilts of 3° or more; `--level-ground` fixes them.
- **Distant tag observations bias the scale.** At twenty or thirty pixels, one or two pixels of corner error is several percent of distance, and the error is not zero-mean. On an earlier capture of the baseball field, all 20 tag observations gave a +2.66% scale error and the 4 closest gave +0.40%. Hence `--min-tag-px 100`.
- **Fake tags on grass.** Grass texture is detected as small ArUco tags with other ids. The upstream rule then throws away the whole image — often one of your best close-range views of the real tag. Use `--tag-rule present`.
- **Your own shadow.** A moving shadow in the frames becomes stains or floaters in the 3DGS. Keep it out of the video, or cut the start with `skip_start_sec`.

### 6.3 grad_nav hardcoded values

grad_nav has several values baked in for the authors' own drone (`carl`). **These must be changed:**

| File | Hardcoded | Change to |
|---|---|---|
| `utils/gs_local.py` | `fx=462.956, fy=463.002, cx=323.076, cy=181.184` at 640×360 | These are RealSense D435 values — use your own onboard camera's |
| `utils/gs_local.py` | three constant matrices in `pose2nerf_transform()` | `T_r2d` is the camera's **mounting extrinsics** (theirs: 15.2 cm forward, 8° down) |
| `utils/gs_local.py` **and** `envs/*.py` | a `maps` dict in each | **Add your scene to both** |
| `envs/drone_long_traj.py` | `gs_origin_offset = [-6.0, 0, 0]` | Match your scene's origin |
| `envs/drone_vla_*.py` | `task_table` | Instruction strings and their waypoints |

Record what you change and why.

### 6.4 Inherent limits of this approach

Not implementation defects — properties of the method:

- **Policies are per-scene.** Zero-shot transfer covers the render-vs-real gap only, not new rooms. A new space means new footage and new training.
- **Fragile to static scene changes.** SousVide measured a drop from 96% to 25% success when objects present during training were removed; the policy keeps flying through where they used to be. People walking around barely matter.
- **No collision termination in simulation.** Obstacle avoidance is a soft reward inside 0.5 m. Real hardware does collide — the paper reports 6–7 of 10.
- **Language instructions are not open-vocabulary.** GRaD-Nav++ covers 4 directions × 3 targets = 12 combinations.
- **Fails in low light.** Below 40% of original brightness, SousVide's policies drift off course from the start.

---

## Implementation notes

`notes/` holds the record of what was actually done, including every deviation from the upstream instructions and the reasoning. More useful than this README when something breaks.

The notes and some tool messages refer to "chapter N" of the original plan (`implement.md`, not published). The notes are numbered the same way: chapter 3 is `notes/03-calibration.md`, and so on.

| File | Topic |
|---|---|
| [`notes/00-prechecks.md`](notes/00-prechecks.md) | Pre-flight checks |
| [`notes/01-environment.md`](notes/01-environment.md) | Environment install, all deviations and why |
| [`notes/02-code-and-example.md`](notes/02-code-and-example.md) | Running the upstream example first |
| [`notes/03-calibration.md`](notes/03-calibration.md) | Camera calibration — including why the upstream tool is broken |
| [`notes/04-aruco.md`](notes/04-aruco.md) | ArUco and metric scale |
| [`notes/05-06-capture-and-gsplat.md`](notes/05-06-capture-and-gsplat.md) | Capture and mapping |
| [`notes/07-scene-experiments.md`](notes/07-scene-experiments.md) | Three-scene comparison — why trajectory shape beats image count |

## References

| Paper | Role |
|---|---|
| [SousVide](https://arxiv.org/abs/2412.16346) (arXiv 2412.16346) | Source of the mapping method; section III-A is FiGS |
| [GRaD-Nav++](https://arxiv.org/abs/2506.14009) (RA-L) | The training method |
| [GRaD-Nav](https://arxiv.org/abs/2503.03984) | Predecessor — differentiable RL and CENet; read before GRaD-Nav++ |

## License

The tools and notes in this repository are the author's own work. The upstream projects carry their own licenses — see [SousVide](https://github.com/StanfordMSL/SousVide) and [grad_nav](https://github.com/Qianzhong-Chen/grad_nav).
