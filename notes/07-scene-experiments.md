# 3DGS quality experiments: comparing three scenes

Date: 2026-08-20
Purpose: identify the factors that affect reconstruction quality (this stage is experimental and has not been deployed to the drone yet)

## The three scenes

| | scene01 | scene02 | scene03 |
|---|---|---|---|
| Video | IMG_2121 (1'46", 3167 frames) | IMG_2154 (3'01", 5449 frames) | IMG_2121 (same as scene01) |
| Images | 300 | 400 | 400 |
| Frame selection | upstream extract_frames (millisecond seek) | uniform FPS (frame index) | uniform FPS (frame index) |
| config | iphone12 | iphone12_400 | iphone12_400 |

scene02 vs scene03 is the best-controlled pair: image count, frame selection, equipment and location are all identical;
**the only difference is how the camera operator walked while filming**.

## Results

| Metric | scene01 | scene02 | scene03 | Original authors' backroom |
|---|---|---|---|---|
| Trajectory two-dimensionality (PC2/PC1) | 0.29 | **0.08** | 0.27 | **0.43** |
| SfM registration rate | 100% | 100% | 100% | — |
| Reprojection error (px) | 1.479 | 1.398 | 1.502 | — |
| **ArUco scale consistency (IQR/median)** | 2.3% ✓ | **106.7% ✗** | 3.3% ✓ | — |
| ArUco vs SfM distance correlation | 0.9872 | **0.2876** | 0.9872 | — |
| RANSAC inliers | 17/20 | **5/20** | 17/20 | — |
| cs | 0.2888 | (invalid) | 0.3011 | — |
| Number of Gaussians | 1,401,340 | 1,242,877 | 1,388,962 | 535,006 |
| Training PSNR | 22.24 | 23.14 | 22.10 | **30.64** |
| eval PSNR | 21.12 | 21.15 | 20.86 | **27.79** |
| Training - eval gap | +1.12 | **+1.99** | +1.24 | +2.85 |

## Firm conclusions

### 1. Going from 300 to 400 images does not help
Same video, same trajectory (scene01 vs scene03): eval PSNR 21.12 -> 20.86.
**Image count is not the bottleneck.**

### 2. Trajectory shape is currently the most influential factor
scene02's camera path is almost a straight line (PC2/PC1 = 0.08, vs 0.27-0.29 for scene01/03).
The user confirmed: they walked back and forth along the same corridor, because that corridor happened to be empty.

Consequences:
- No similarity transform can satisfy all 20 ArUco points at once (dispersion 106.7%)
- At about 4x the trajectory width away from the capture path the floor is already smeared; at 7x the whole image is just color blobs
- Training PSNR is actually the highest (23.14) -> a sign of overfitting to the capture path

⚠ **Every one of SfM's own metrics is green** (100% registration, 1.398 px reprojection error, a single model);
only the external metric reference (ArUco) catches the problem. A low reprojection error only means "this solution explains the captured
images", not that it matches the real geometry.

### 3. These are not the main cause (ruled out one by one)
- **Motion blur**: the original backroom has a median Laplacian of 71, even lower than the user's
- **Exposure/white balance**: brightness variation 12.3% vs 12.5% in the original work; color temperature 4.32% vs 4.52%
- **Intrinsics**: switching to COLMAP's self-estimated intrinsics made scene02 even worse (106.7% -> 122.5%)
- **Quality of the images containing the tag**: keeping only the 8 images with marker>=150px and tilt<=45°, the dispersion is still 109%
- **The tag was moved**: the camera centroids of the different time segments are close, and the opening segment is already inconsistent internally

### 4. Laplacian variance is not a "blur metric"
It is also affected by image content. The 10 lowest-scoring images of scene02 **are not blurry at all; they are sharp shots of a blank wall**;
the lowest-scoring ones of scene03 are the ones with real motion blur. Comparisons are only meaningful within the same scene.

### 5. scene02 wastes 17% of its frames on blank walls

| Scene | Low-texture area share | Frames >50% blank | Frames >70% blank |
|---|---|---|---|
| scene01 | 24.6% | 4 | 0 |
| **scene02** | **37.8%** | **69** | **7** |
| scene03 | 24.9% | 7 | 0 |
| Original authors' backroom | 35.6% | **0** | **0** |

The original authors' total low-texture area is actually comparable (35.6%), but **not a single frame is dominated by blank area**:
their textureless regions are spread out (every frame has plain floor, but also has content).
scene02 has one long continuous stretch (frame_00360~00378) filming the wall.

## Still a 7 dB gap to the original authors; known partial causes

- **More complex scene content**: mean gradient 28-29 vs 20.6 in the original (about 40% higher).
  PSNR inherently penalizes high-frequency detail: scene03's training images are sharper (Laplacian 134 vs 106),
  yet its eval PSNR is lower, which is exactly this effect.
- **Trajectory still not two-dimensional enough**: the original is 0.43, our best (scene03) is 0.27.

## Concrete recommendations for the next capture

1. **Walk a loop**, circling the scene or the main objects, with a slightly different path each lap
2. **Vary the height**: do one pass crouching, one standing, and one with the phone held high. In a narrow space this is the least-effort way to turn
   a one-dimensional trajectory into a two-dimensional one
3. **Translate sideways past important objects**, so they enter the frame from different angles
4. **Always point the camera at something with content**; do not film blank walls for long stretches
5. **Switch to landscape**: the original authors used 1920×1080 landscape; in portrait a large part of the image is floor with repetitive texture,
   and the onboard camera (640×360) is also landscape
6. Criterion: draw the path you walked on a floor plan: **is it a line or an area?**

## Output files

```
review/scene0X_NNNimgs/          -> training images (symlinks)
review/sheets_scene0X/           -> contact sheets + the 10 sharpest / 10 blurriest
notes/eval_scene0X/              -> renders on-path vs off-path for comparison
notes/traj_compare.png           -> camera trajectory plot, scene01 vs scene02
tools/eval_gsplat.py             -> unified scene evaluation tool
tools/make_contact_sheet.py      -> contact sheet generator
```

## Tool changes

`tools/build_gsplat.py` gains `--select {uniform,sharp}`:
- `uniform` (default) = the original authors' 1-D farthest-point sampling. The upstream `distribute_values` is
  O(n·k²) pure Python (choosing 380 out of 3937 takes over ten minutes); it now uses numpy to maintain a minimum-distance array,
  **verified on 4 random test sets with gaps to select exactly the same values**, in 0.002 s.
- Frame grabbing now uses frame indices + a single sequential read of the whole video, avoiding the offset from upstream's millisecond seek
  (which is why scene01 got 2 extra images containing the tag). scene02/03 were measured to have exactly 20 each.
