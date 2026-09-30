# Chapter 4: ArUco tag

Date: 2026-08-13
Outputs: `captures/aruco_id0_A4_157mm.pdf`, `captures/aruco_id0_A3_222mm.pdf`
      `repos/SousVide/configs/captures/iphone12.json`

## FiGS hard requirements (confirmed by reading the source)

`capture_generation.py`:

| Item | Value | Changeable? |
|---|---|---|
| Dictionary | `DICT_4X4_50` (lines 166, 234) | **Hard-coded, cannot change** |
| `marker_id` | From `extractor_config["marker_id"]` | **Changeable** (the manual's claim that it is hard-coded is inaccurate) |
| `marker_length` | From config, used for `marker_points = ±L/2` | Required |
| Markers per frame | `len(ids) == 1` (line 262) | Hard rule; one extra marker and the frame is discarded |
| PnP | `cv2.SOLVEPNP_IPPE_SQUARE` | — |

`cv2.aruco` returns the corners of the **outer edge** of the black square ->
**`marker_length` = outer edge to outer edge of the outermost black border, excluding the white margin.**

## Deriving the maximum size

The DICT_4X4 pattern is 6x6 modules (4x4 data + a 1-module black border).
The detector needs a white margin around the black frame, by convention at least 1 module wide:

```
(page_width - S)/2 >= S/6   ->   S <= page_width * 3/4
A4 (210mm) -> 157mm      A3 (297mm) -> 222mm
```

The original authors used 34.1 cm and the manual recommends >= 25 cm; **a single sheet of paper cannot do that**.
⚠ Do not tile several A4 sheets: misaligned seams and unevenness directly contaminate the solvePnP pose solution.
For anything larger, have a print shop print A2 or bigger (`make_aruco.py --page 420 594`).

Self-verification after generation (render the PDF, then actually run detection):
both sizes give "exactly 1 marker, id=0, measured side length within 0.01% of nominal".

## Measured results

**Measured black-frame side length after printing: 14.4 cm in both width and height -> `marker_length = 0.144`**

144 / 157 = **91.7%**. **Exactly the same** as the Chapter 3 checkerboard ratio 22/24 = 91.7%.
Two different files with different nominal sizes but the same scale factor, so we can infer:

1. The printer applies "shrink to fit printable area", a fixed reduction to about 91.7%
2. **And the scaling is uniform**: this is exactly the risk we most needed to rule out (non-uniform scaling would create a spurious fx/fy difference)

Equal width and height also support uniform scaling. The larger white border on the paper is a result of shrinking the whole page uniformly,
which actually helps detection (the white border is the quiet zone).

## Scale error budget (marker_length = 0.144)

| Source | Contribution |
|---|---|
| marker_length measurement ±0.5 mm | 0.35% |
| Chapter 3 fx uncertainty | 0.30% |
| PnP pose noise (1 m, 245 px, 20 frames RANSAC) | 0.046% |
| **Total** | **0.46%** |

The Chapter 7 acceptance criterion is < 2%, leaving a margin of 1.54 percentage points.

⚠ The real cost of A4 is not resolution but **a stricter measurement accuracy requirement**:
measurement error converts 1:1 into scale error, and the denominator is the marker side length.
For the same ±1 mm measurement, the error for 144 mm (0.69%) is 2.4 times that of the original authors' 341 mm (0.29%).

Marker span in the image (fx=1702):
0.5 m -> 490 px, 1.0 m -> 245 px, 1.5 m -> 163 px, 2.0 m -> 123 px.
ArUco pose solutions are quite reliable above 100 px -> **when filming the tag in Chapter 5, keep a distance of 0.5-1.5 m**.

## capture config

`repos/SousVide/configs/captures/iphone12.json`

```
camera    : 1080x1920 (WxH)  fx=1702.28 fy=1708.64 cx=544.82 cy=949.09  distortion: 4 coefficients
extractor : num_images=300 num_marked=20 marker_id=0 marker_length=0.144
```

Verification method: load this config following exactly the code path of `extract_positions()`,
project the marker corners at known distances, then solve the distance back with `solvePnP(..., SOLVEPNP_IPPE_SQUARE)`.
0.5 / 1.0 / 1.5 / 2.0 m, each with 0° and 25° tilt: **error is 0.0000% in every case**
-> the config format, array shapes and marker_length convention are all correct.
(This verification covers only the numerical pipeline, not corner localization noise in real footage.)

## Placement requirements (when filming in Chapter 5)

- **Lay it flat on the floor**: the z axis of the reconstructed world frame is then opposite to gravity, which flight requires
- **Perfectly flat**: mount it on a rigid board or tape it down; wrinkles skew the pose solution
- **No second ArUco** anywhere in the scene
- ⚠ Watch out for **reflections in mirrors, glass and glossy floors**: a reflected tag is detected as a second marker,
  triggering the `len(ids) == 1` failure and discarding the frame. This is the easiest one to overlook
