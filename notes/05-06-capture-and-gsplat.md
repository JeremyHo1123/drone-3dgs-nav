# Chapter 5: Filming the scene + Chapter 6: Building a metric 3DGS

Date: 2026-08-13
Scene name: `scene01`
Outputs: `repos/SousVide/gsplats/workspace/scene01/`
      `outputs/scene01/splatfacto/2026-08-13_083816/`

## Chapter 5: Scene video

`captures/IMG_2121.MOV` (symlinked at `gsplats/capture/scene01.MOV`)

| Item | Value | Matches calibration video |
|---|---|---|
| Lens | Wide 26 mm f/1.6 (1x) | ✓ |
| Resolution | 1080 × 1920 (container 1920×1080 + rotation=-90) | ✓ |
| Frame rate | 29.98 fps | ✓ |
| Codec / bitrate | H.264 / 15.6 Mbps, native, not recompressed | ✓ |
| Length | 105.6 s / 3167 frames | The manual recommends 2-3 minutes; slightly short but sufficient |

### Preflight check (`tools/preflight_capture.py`)

Scans every frame with the same decision logic as `extract_frames()`:

```
with tag (exactly 1, id=0)          :  1148 frames   need >= 20  ✓
without tag                         :  2019 frames   need >= 280 ✓
2+ markers (false hit, reflection)  :    11 frames
wrong id (false hit, floor texture) :    22 frames
median marker side 171 px (89~224), estimated distance 1.43 m
```

⚠ **The tag must not be in view for the whole video**. `extract_frames` takes 280 images from the "without tag" pool;
if that pool is empty you get `IndexError`, and if it has fewer than 280 you get `TypeError` (`distribute_values` puts None
into the list). Both were confirmed by actually running it.

## Chapter 6: Building the map: four upstream problems

We run it in stages (`tools/build_gsplat.py`) instead of calling `generate_gsplat()` directly,
because the latter uses `subprocess.run(capture_output=True)`, which swallows the training output,
and there is no way to insert a check between SfM and training. It calls the same upstream functions.

### 1. hloc's third_party submodules not initialized

**Symptom**: `ModuleNotFoundError: No module named 'SuperGluePretrainedNetwork.models'`

**Cause**: in Chapter 2, to avoid the huge acados, we used
`git submodule update --init Hierarchical-Localization` (non-recursive),
which missed hloc's own `third_party/`. nerfstudio's hloc defaults are
`superpoint_aachen` + `superglue`, and both come from that submodule.

**Fix**:
```bash
cd FiGS/Hierarchical-Localization
git submodule update --init third_party/SuperGluePretrainedNetwork
```
Only this one is added. `d2net`/`r2d2` are alternative feature extractors and `deep-image-retrieval` is only needed for vocab_tree
retrieval; we use exhaustive matching and need none of them. hloc loads them lazily with `dynamic_load`.

### 2. pycolmap API incompatibility

**Symptom**: `TypeError: import_images(): incompatible function arguments`

**Cause**: hloc (a 2024 commit) uses `import_images(..., image_list=)`
and a dict-typed `incremental_mapping(options=...)`;
pycolmap **4.1.1** renamed it to `image_names=`, and options must now be of type
`IncrementalPipelineOptions`.

⚠ The Chapter 2 `import pycolmap` test passes anyway: **a successful import does not mean the function signatures are compatible**.

**Fix**: `pip install --no-deps pycolmap==0.6.1` (the version hloc was written against).
This is cleaner than patching the many upstream call sites one by one. Confirmed that 0.6.1 satisfies both hloc and nerfstudio
(`ImageReaderOptions` / `CameraMode` / `verify_matches` / `triangulate_points` are all present).

### 3. hloc moves the wrong reconstruction model (the most insidious: no error is raised)

**Symptom**: hloc logs `Largest model is #1 with 300 images` and
`num_reg_images = 300`, but nerfstudio reports
`COLMAP only found poses for 0.67% of the images`, and `transforms.json` has only 2 frames.

**Cause**: line 102 of `run_reconstruction` uses the **dict key** of `reconstructions`
as the folder name to move (`models_path / str(largest_index)`), but the dict keys returned by pycolmap 0.6.1
**do not match** the folder numbers it actually writes to disk. On disk, `models/0` is the
large 300-image model and `models/1` is a small 2-image model.

**Fix (two parts)**:
- Immediate fix: copy `models/0/*.bin` back into `sparse/0/`, then call
  `colmap_to_json(recon_dir=..., output_dir=..., image_rename_map=None,
  use_single_camera_mode=True)` directly to regenerate transforms.json.
  (Confirmed that the file name sets of `images/` and `sfm/images/` are identical -> rename_map is the identity)
- Permanent fix: patch `hloc/reconstruction.py` to **scan the model directories on disk and
  pick the one with the largest `pycolmap.Reconstruction(d).num_reg_images()`**, marked `PATCH(drone)`.
  ⚠ This change lives inside a git submodule; `git checkout` or a fresh clone will revert it.

### 4. Two count discrepancies (not errors, but worth understanding)

- **22 of the 300 extracted images contain the tag, not 20**: `extract_frames` first reads the bins sequentially
  and records millisecond timestamps, then jumps back with `cap.set(CAP_PROP_POS_MSEC)` to grab the frames.
  Millisecond seeking in H.264 is imprecise and lands on a nearby frame; 36% of the frames in your video contain the tag,
  so 2 timestamps originally classified as "no tag" landed on nearby frames containing the tag after seeking.
- **The check stage counts 20 images**: because it replicates the image-reading path of `extract_positions`
  (`imread(f)` then `cvtColor(BGR2GRAY)`), whose grayscale conversion differs slightly from calling `imread(f, IMREAD_GRAYSCALE)`
  directly, flipping 2 borderline cases. **20 is the number that counts**,
  and it exactly equals `num_marked`, so `extract_positions` passes.

## Results

```
SfM registration rate     300/300 = 100%
registered images w/ tag  20 (== num_marked ✓)
3D points                 47194, observations 305908, mean reprojection error 1.48 px
Sim(3)                    cs = 0.288849 (RANSAC: 17 of 20 points inliers, threshold 5 cm)
training                  30000 steps / 16 minutes / about 32 ms per step
number of Gaussians       1,401,340
training resolution       540x960 (images_2, nerfstudio auto-downscale, MAX_AUTO_RESOLUTION=1600)
```

A 100% registration rate means good capture quality: plenty of overlap, no motion blur, no coverage holes.

### Early signs about the scale (formal acceptance in Chapter 7)

The full bbox of the sparse point cloud is 14.82 × 29.09 × 11.05 m, **but it is dominated by outliers**;
only robust statistics are meaningful:

```
5~95 percentile range (m):  x 5.38   y 5.32   z 1.51
z 5/50/95 percentiles     : -0.04 / 0.48 / 1.48
9821 points within 1.5 m horizontally of the ArUco: median z = 0.0116 m
```

**The floor near the tag sits at z ≈ 1.2 cm, with the z axis pointing up** -> the origin is on the ArUco and
the world z axis is opposite to gravity, as flight requires.

### ⚠ To be decided in Chapter 7: the 2.68% intrinsics discrepancy

| Intrinsics | fx | cs | Median Sim(3) residual |
|---|---|---|---|
| Our calibration (Chapter 3) | 1702.28 | 0.288444 | **1.36 cm** |
| COLMAP self-estimate | 1664.67 | 0.280903 | 1.83 cm |

The `solvePnP` distance is proportional to fx, so **a 2.68% fx difference = a 2.68% scene scale difference**,
right at the edge of the Chapter 7 2% criterion.

The residuals **favor our calibration** (1.36 < 1.83 cm, and our residual is still smaller even though our larger cs scales distances up by 2.7%).
But COLMAP's distortion coefficients (k1=0.103, k2=-0.147) are much milder than ours
(0.220, -0.716), consistent with the Chapter 3 warning that "the distortion solution may be ill-conditioned".

**How to decide**: measure the length of a known object in Chapter 7.
- Matches -> keep things as they are
- About 2.7% too large -> set `camera` in `configs/captures/iphone12.json` to `null`
  (FiGS then falls back to the SfM intrinsics), and **only rerun `--stage scale`**.
  Scale only affects transforms.json and the point cloud; **no need to rerun SfM or retrain the 3DGS**.

Also noted: RANSAC is random; two runs gave cs = 0.288849 / 0.288444, a 0.14% difference.

## Render check

Three images in `notes/ch6_render/`, rendered from training camera poses:
the images are sharp and recognizable (the ArUco board, wooden floor, slippers, robot arm, tables and chairs, cables)
and match photos of the real place. The pipeline is correct.
