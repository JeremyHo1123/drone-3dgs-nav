# Chapter 3 Camera intrinsics calibration (iPhone 12)

Date: 2026-08-13
Output: `repos/SousVide/configs/camera/iphone12.json`

## Upstream is broken; this chapter uses a self-written tool instead

`figs.render.capture_calibration.camera_calibration()` **cannot be used**, due to three independent problems:

1. Line 38 calls `ch.extract_frames(...)`, but `figs/utilities/capture_helper.py`
   **has no such function** -> `AttributeError`. A function with the same name exists in `capture_generation.py`,
   with a different signature and purpose (it writes files instead of returning arrays)
2. The default path is computed wrong: `Path(__file__).parent×3/'gsplats'` -> `FiGS/src/gsplats` (does not exist).
   `generate_gsplat` uses `parent×5`, which correctly reaches `SousVide/gsplats`
3. **The object-point index order is reversed**: `np.mgrid[0:rows, 0:cols].T`.
   `findChessboardCorners(patternSize=(cols,rows))` returns the corners in row-major order (rows rows
   of cols corners each), so the object points must be `mgrid[0:cols, 0:rows].T`
   (consistent with the official OpenCV tutorial). The wrong order produces completely wrong intrinsics --
   in a synthetic test, fx went from the true value 1310 to **58**

Replacement tool: `tools/calibrate_camera.py`, verified end to end on a synthetic video with known intrinsics
(fx error 0.10% at 9/9 coverage).

## Two things confirmed by testing that are safe to rely on

**`square_size` does not affect the calibration result.** With 25.0 / 3.0 / 0.025 the computed intrinsics and
distortion coefficients are identical to 4 decimal places. Scaling the object points only scales the translation vectors of the extrinsics proportionally.
-> The checkerboard does not need to be printed precisely, and the square side length does not need to be measured precisely.
The actual print came out at 22 mm (nominal 24 mm, 91.7%), with no impact.
The only thing to guard against is **non-uniform** scaling, which creates a spurious difference between fx and fy.

**A low reprojection error does not mean accurate intrinsics.** In the synthetic test at 8/9 coverage the error was only 0.037 px
(far below the 0.5 px criterion), yet fx was 1.1% off the true value; after filling in to 9/9 it dropped to 0.10%.
The error only says the model can explain the points captured, not that those points are enough to constrain the parameters.

## First recording (discarded)

`calibrate.mp4`: 720×1280, **2.1 Mbps** (native 1080p30 is about 17 Mbps
-> it was re-encoded and compressed in transit), checkerboard taped to a wall with a wavy paper surface.

Result: 67% of the corners were crowded in the center of the frame; the emptiest region had only **0.2%**.
The distortion became an ill-conditioned solution (`k1=+0.184, k2=-1.011`, one positive and one negative cancelling each other),
the radial displacement at the corners was **87 px**, and switching the distortion model **changed fx by 2.0%**.

⚠ Cross-validation (calibrating odd/even frames separately) showed only a 0.01% difference: **highly reproducible but not accurate** --
both halves share the same sampling bias, so this test cannot detect insufficient coverage. Do not use it as an acceptance check.

## Second recording (adopted)

`IMG_2119.MOV`: 1920×1080 container + `rotation=-90` metadata, **15.6 Mbps native**,
checkerboard mounted on stiff cardboard.

### ⚠ Rotation metadata

The container is 1920×1080, but OpenCV's `CAP_PROP_ORIENTATION_AUTO=1` automatically rotates it upright,
so it actually decodes as **1080×1920 portrait**. FiGS's `extract_frames` uses the same
`cv2.VideoCapture` and **applies exactly the same automatic rotation**, so the config's
`height=1920 / width=1080` is consistent with later processing.

### Result

```
Image size 1080 x 1920
fx=1702.2846  fy=1708.6379
cx=544.8192   cy=949.0888      (image center 540.0, 960.0)
Distortion k1=+0.220034 k2=-0.716124 p1=-0.000202 p2=+0.000909
Mean Reprojection Error: 0.0783 px
```

| Check | First | Second | Threshold | |
|---|---|---|---|---|
| Reprojection error | 0.041 px | 0.078 px | < 0.5 | ✓ |
| Share of corners in the emptiest region | 0.2% | 1.9% | >= 3% | ✗ |
| (a) Radial displacement at corners | 87 px | 33 px | < 25 px | ✗ |
| **(b) fx change after switching distortion model** | **2.0%** | **0.3%** | < 1% | **✓** |

**(b) is the metric that really determines scale** (solvePnP's distance solution is proportional to the focal length,
so the uncertainty in fx turns directly into scene scale error). It dropped from 2.0% to 0.3%,
so the 2% scale budget of chapter 7 is preserved.

The 25 px for (a) is an empirical threshold I set; 33 px corresponds to a 3.1% radial deviation at the corners,
which is actually within a reasonable range for a phone wide-angle lens; it is a borderline case.

### Independent sanity check (compared with the official bundled config files)

```
                       declared WxH      fx        fy       cx       cy   fx/short side   principal point
iphone15pro(official)  1080x1920     1276.5    1280.0    960.0    540.0           1.182   **inconsistent**
pixel8pro(official)    1080x1920     1761.0    1798.0    540.0    960.0           1.631   consistent
iphone12(yours)        1080x1920     1702.3    1708.6    544.8    949.1           1.576   consistent
```

- `fx/short side` 1.576 is the same order of magnitude as pixel8pro's 1.631, reasonable
- The principal point agrees with the image center (the official `iphone15pro.json` does not:
  cx=960/cy=540 corresponds to landscape 1920×1080, so that file has height/width swapped)
- fx and fy differ by 0.37%, no non-uniform scaling
- Field of view: 35.2° on the short side, 58.7° on the long side

## User decision

Corner distribution (1.9% vs 3%) and (a) did not meet their thresholds. The user judged that "the checkerboard is already as flat as it can get,
the rest is accepted as error, this is the limit for now", and decided to adopt it.
The official file name was produced with `--force` (that flag was added for this occasion; the check results are still printed in full).

**The residual risk will show up in chapter 7**: if the scene scale error exceeds 2%,
the intrinsics are the first thing to go back and check.

## Hard constraints for chapter 5

The scene video must use **exactly the same recording conditions** as the calibration video, otherwise these intrinsics are invalid:

- **1x wide-angle lens** (never the 0.5x ultra-wide, which is a different physical lens)
- **Handheld in portrait** (produces 1920×1080 + rotation=-90, which OpenCV reads as 1080×1920)
- 1080p / 30 fps
- **Transfer via USB cable or another method that does not re-encode** (the first video failed precisely because it was compressed to 2.1 Mbps)

## To do (after chapter 4 is finished)

`configs/captures/iphone12.json` has not been created yet. It consists of this file's camera block
plus an extractor block (`num_images` / `num_marked` / **`marker_length`** / `marker_id`).
`marker_length` must be the **measured side length** of the printed ArUco tag; that is the scale reference for the whole scene,
so it can only be assembled after chapter 4's measurement.
