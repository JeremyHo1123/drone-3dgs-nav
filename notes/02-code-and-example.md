# Chapter 2 Getting the code and running the official example

Date: 2026-08-11

## 2.1 Papers (downloaded to refs/)

| File | Size |
|---|---|
| 2024-12-20-sousvide.pdf | 9.5 MB |
| 2025-03-06-grad-nav.pdf | 3.1 MB, 8 pages |
| 2025-06-16-grad-nav-pp.pdf | 5.6 MB, 7 pages |

All three are valid PDFs, not error pages.

## 2.2 Code

```
repos/SousVide/            (includes the FiGS submodule)
repos/SousVide/FiGS/Hierarchical-Localization/   (hloc)
repos/SousVide/FiGS/acados/                      (empty directory, deliberately not initialized)
repos/grad_nav/
```

**Only the FiGS and hloc submodules were initialized; acados was not pulled.**
Confirmed that FiGS's `pyproject.toml` does not depend on acados; acados is imported only by the three files
`simulator.py`, `control/vehicle_rate_mpc.py` and `dynamics/quadcopter_rate_model.py`,
all on the MPC path we do not use.

FiGS **has no `__init__.py` at all** (PEP 420 namespace package),
so `import figs.render.capture_generation` does not trigger any parent-level import.
The concern in Appendix A (of the original plan, `implement.md`, not published) that "`import figs` fails as a knock-on effect of acados" does not materialize.

Key packages installed: `hloc 1.5`(editable), `figs 0.1.0`(editable),
`pycolmap 4.1.1`, `kornia 0.8.2`, `lightglue 0.0`.

### ⚠️ opencv got broken twice

- hloc's requirements want `opencv-python` (it actually installed **5.0.0.93**,
  which requires numpy>=2 and is incompatible with our numpy 1.26.4)
- FiGS depends on `albumentations`, which wants `opencv-python-headless>=4.9.0.80`

**All three distributions write into the same `cv2/` directory**, overwriting each other's files. And uninstalling
any one of them deletes shared files and leaves an orphaned directory.

Fix (redo this check if anything more is installed in chapter 6 or 9):

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
# confirm cv2/ inside site-packages has been emptied
pip install "opencv-contrib-python==4.10.0.84"
```

After cleaning up, confirmed: only one `cv2.abi3.so`, a single dist-info, `aruco: True`.

### hloc availability (a source of silent failure)

nerfstudio's `hloc_utils.py` wraps the hloc import in `try/except ImportError`;
a failure only sets `_HAS_HLOC = False` and **raises no error**. Measured result:

```
pycolmap: 4.1.1
hloc.extract_features / match_features / pairs_from_exhaustive
     / pairs_from_retrieval / reconstruction  all OK
_HAS_HLOC will be True -> the hloc path in chapter 6 is usable
```

## 2.3 Official example (the plan's acceptance method had to be replaced)

### Why the notebook was not run

`notebooks/figs_examples.ipynb` cannot run, for two independent reasons:

1. **cells 4 and 12 go through `VehicleRateMPC` -> import acados**,
   which directly conflicts with section 1.8, "acados does not need to be compiled". `figs/simulator.py` also imports acados.
2. The notebook defaults to `capture_name = "button"`, but in the downloaded data
   `gsplats/capture/` only has `backroom.MOV`, no button.

### Verification method used instead

Use `figs.render.gsplat.GSplat` directly -- that is the render path this project will really use
(grad_nav's `utils/gs_local.py` has the same structure), and it depends only on nerfstudio/torch/numpy.
Script logic: load the official backroom checkpoint -> take actual camera poses from the training set -> render -> save -> visual check.

⚠️ Coordinate detail: `render_rgb` internally computes `Tc2g = Tw2g @ T_c2w`,
and `Tw2g = diag(1,-1,-1,1)` is its own inverse. To render with dataset poses you must feed
`T_c2w = Tw2g @ Tc2g_dataset`, otherwise the image is upside down.

### Result (passed)

```
Number of Gaussians: 535,006      eval_dataset cameras: 30
Poses 0/1/2 all rendered (360,640,3) uint8 images
Brightness mean 137.1 / 85.4 / 86.7, distinct pixel values 227 / 241 / 236
```

Visual check: `notes/ch2_render/backroom_cam0.png` clearly shows **an ArUco tag lying flat
on the floor** (exactly the placement section 4.3 requires); `backroom_cam1.png` is a sharp
lab scene where furniture, cabinets and cardboard boxes are all recognizable. **The pipeline itself works.**

## Log of upstream code changes

### nerfstudio's torch.load (three places)

**Symptom**: `_pickle.UnpicklingError: Weights only load failed ...
Unsupported global: GLOBAL numpy.core.multiarray.scalar`

**Cause**: **Since PyTorch 2.6, the default of `weights_only` in `torch.load` changed from False to True**.
nerfstudio 1.1.5 was written in 2024 and does not pass this argument, while the checkpoint contains numpy scalars.
This project must use torch>=2.7 for Blackwell(sm_120), so the two inevitably conflict.

**What was changed** (each got `weights_only=False` plus an explanatory comment):

| File | Line | Affected functionality |
|---|---|---|
| `nerfstudio/utils/eval_utils.py` | 62 | `eval_setup` -> rendering, `ns-export` (chapter 8), `ns-viewer` |
| `nerfstudio/engine/trainer.py` | 432 | resume training from `load_dir` |
| `nerfstudio/engine/trainer.py` | 443 | load from a specified checkpoint |

(`scripts/downloads/download_data.py:517` has one more, which is not used and was not changed.)

⚠️ **This change is inside site-packages; reinstalling nerfstudio overwrites it.**
If you ever run `pip install --force-reinstall nerfstudio`, redo these three places.

⚠️ `weights_only=False` executes the pickle inside the checkpoint;
use it only on files you produced yourself or from a trusted source (here, the paper authors' official data).

## Disk

`repos/SousVide/gsplats.zip` is 4.4 GB, extracted into `gsplats/` at 5.0 GB.
The zip can be deleted (not deleted yet; left for you to decide).
