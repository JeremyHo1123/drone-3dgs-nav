# Chapter 0 Precheck log

Date: 2026-08-10

## Driver layer (required by chapter 0 of the original plan, `implement.md`, not published)

| Check | Requirement | Measured | Result |
|---|---|---|---|
| Driver Version | >= 570.xx | 580.126.09 | ✅ |
| nvidia-smi CUDA Version | >= 12.8 | 13.0 | ✅ |
| compute_cap | 12.0 (sm_120) | 12.0 | ✅ |
| GPU / VRAM | RTX 5070 Ti / 16 GB | RTX 5070 Ti / 16303 MiB | ✅ |

The desktop environment (Xorg + gnome-shell + browser) constantly uses about 661 MiB,
so about 15.6 GB is actually available. Subtract this when tuning `num_actors` in section 10.3.

## Additional environment inventory

- OS: Ubuntu 24.04.4 LTS (noble)
- conda 25.7.0 @ /home/jeremy/anaconda3
- Disk: 524 GB free on /, plenty
- gcc/g++ 13.3.0 - a version CUDA 12.8 accepts
- **The system already has CUDA toolkit 12.8.93 @ /usr/local/cuda** (nvcc available)
- ffmpeg **not installed** -> must be added before processing video in chapters 5 and 6

## Impact on chapter 1 (important)

Appendix C (of the original plan) marks chapter 1 as "not tested", but this machine already has an existing environment
`gaussian_splatting_128` that was tested and works:

```
torch: 2.9.0+cu128
cuda build: 12.8
arch list: ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
capability: (12, 0)
GPU matmul actually succeeded
```

-> Conclusion: **torch 2.9.0+cu128 is confirmed to actually run kernels on this machine's sm_120**,
so there is no need for a nightly build. The biggest uncertainty in section 1.3 is ruled out.

-> Section 1.4 (installing cuda-toolkit via conda) can use the system /usr/local/cuda 12.8 instead,
saving about 3 GB and avoiding version clashes between conda's and the system's nvcc.

## Directories created

~/drone/{repos,refs,gsplats,captures,notes}

## Path unification (2026-08-11, later change)

The working directory originally created per chapter 0 of `implement.md` was `~/drone-env`, separate from
`~/drone` where the project documents live, which gave two root directories. They have been merged into a single root **`~/drone`**.

Things that had to be handled as part of the move (do the same if it is ever moved again):

1. `mv ~/drone-env/* ~/drone/` -- same filesystem, so it is an instant metadata operation
2. **`hloc` and `figs` are editable installs; after the move they give `ModuleNotFoundError`**.
   They must be reinstalled, and **always with `--no-deps`**, otherwise
   `opencv-python` and `opencv-python-headless` get pulled back in and overwrite the contrib build:
   ```bash
   pip install -e ./FiGS/Hierarchical-Localization/ --no-deps --force-reinstall
   pip install -e ./FiGS/ --no-deps --force-reinstall
   ```
3. The absolute paths in `tools/calibrate_camera.py` were changed to a `PROJECT_ROOT` derived from
   `Path(__file__)`, so future moves need no code changes
4. The 13 occurrences of `~/drone-env` in `implement.md` were changed to `~/drone` (only path strings changed,
   technical content unchanged); a change note was added at the top of the file
5. The comment tag of the nerfstudio patch was changed from `PATCH(drone-env)` to `PATCH(drone)`

Regression checks after the move all passed: torch/gsplat/cv2(aruco)/colmap/ffmpeg work,
hloc and figs point to the new path, the chapter 2 render test re-ran successfully, and the calibration script's default paths are correct.
