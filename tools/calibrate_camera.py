"""
Camera intrinsics calibration (replaces FiGS's broken figs.render.capture_calibration.camera_calibration).

Why not use the upstream one:
  1. capture_calibration.py:38 calls ch.extract_frames(...), but
     figs/utilities/capture_helper.py has no such function -> AttributeError.
     A function with that name lives in capture_generation.py, with a completely different signature; it writes files instead of returning arrays.
  2. Its default path is computed wrong: gsplats_path = Path(__file__).parent x3/'gsplats'
     -> FiGS/src/gsplats (does not exist). generate_gsplat uses parent x5, which correctly lands on SousVide/gsplats.

The output format matches the "camera" block of SousVide/configs/captures/*.json:
  model / height / width / intrinsics_matrix / distortion_coefficients (4 values, flattened)

k3 is fixed at 0 in the distortion model (CALIB_FIX_K3), because the official FiGS config stores only 4 coefficients,
and k3 of a phone's wide-angle lens is usually negligible.

Usage:
  python calibrate_camera.py --video <path> --name iphone12 [--squares 9 6] [--max-frames 120]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# The project root is derived from this file's location (the parent of tools/), not a hard-coded absolute path,
# so the whole project can be moved elsewhere without changing the code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sample_frames(video_path: Path, max_frames: int, blur_reject: float):
    """Sample the video uniformly and drop clearly blurry frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}\n"
            "  An iPhone video encoded as HEVC (High Efficiency) may be unreadable.\n"
            "  Settings -> Camera -> Formats -> switch to \"Most Compatible\" (H.264) and record again."
        )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise RuntimeError("Failed to read the frame count; the file may be corrupted")

    idxs = np.linspace(0, total - 1, min(max_frames * 3, total)).astype(int)
    frames, sharp = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(f)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
    cap.release()

    if not frames:
        raise RuntimeError("Could not read a single frame")

    sharp = np.array(sharp)
    keep = sharp >= (np.median(sharp) * blur_reject)
    kept = [f for f, k in zip(frames, keep) if k][:max_frames]

    print(f"  Video: {total} frames @ {fps:.2f} fps, resolution {frames[0].shape[1]}x{frames[0].shape[0]} (WxH)")
    print(f"  Sampled {len(frames)} frames -> kept {len(kept)} frames after the sharpness filter"
          f" (median sharpness {np.median(sharp):.0f})")
    return kept


def detect_corners(frames, pattern, vis_dir: Path):
    """Detect checkerboard corners and refine them to sub-pixel accuracy."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # The scale of the object points does not affect the intrinsics (verified experimentally), so 1.0 units is fine here.
    #
    # ⚠ Index order: findChessboardCorners(patternSize=(cols,rows)) returns corners in
    #   row-major order, cols corners per row and rows rows in total, so the object points must be
    #   mgrid[0:cols, 0:rows].T -- consistent with the official OpenCV tutorial.
    #   FiGS's capture_calibration.py:70 writes mgrid[0:rows, 0:cols].T;
    #   the two orders do not match and produce completely wrong intrinsics (measured: fx went from 1310 to 58).
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    obj_pts, img_pts, used = [], [], []
    vis_dir.mkdir(parents=True, exist_ok=True)

    for n, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(n)
        if len(img_pts) <= 6:                      # save a few for visual inspection
            vis = f.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, ok)
            cv2.imwrite(str(vis_dir / f"detect_{len(img_pts):02d}.jpg"), vis)

    print(f"  Frames with a detected checkerboard: {len(img_pts)} / {len(frames)}")
    return obj_pts, img_pts


def coverage_report(img_pts, w, h):
    """
    Check how the corners are distributed across the image.

    ⚠ Only checking "is every cell > 0" is not enough: we have seen a case with 4014 points in the center
    and only 11 in a corner that still passed as 9/9; with such data the distortion coefficients near the image border are an extrapolation.
    So this checks "the share of the smallest cell in the total" instead. With a uniform spread each cell is about 11%.
    """
    grid = np.zeros((3, 3), int)
    for pts in img_pts:
        for p in pts.reshape(-1, 2):
            c = min(int(p[0] / w * 3), 2)
            r = min(int(p[1] / h * 3), 2)
            grid[r, c] += 1
    total = grid.sum()
    pct = grid / max(total, 1) * 100
    filled = int((grid > 0).sum())

    print("  Corner distribution over a 3x3 grid of the image (share in parentheses, about 11% each if uniform):")
    for r in range(3):
        print("    " + "  ".join(f"{grid[r, c]:6d}({pct[r, c]:4.1f}%)" for c in range(3)))
    print(f"  Cells with corners {filled}/9, smallest cell holds {pct.min():.1f}%")

    ok = True
    if filled < 9:
        print("  ✗ Some cells were never covered; the distortion coefficients are undefined there.")
        ok = False
    elif pct.min() < 3.0:
        print(f"  ✗ Distribution too concentrated (smallest cell only {pct.min():.1f}%, should be >= 3%)."
              "\n    The distortion coefficients near the image border are an extrapolation and unreliable."
              "\n    -> Make the checkerboard larger in frame and really move it into all four corners of the image.")
        ok = False
    else:
        print("  ✓ Distribution acceptable")
    return ok


def sanity_checks(obj_pts, img_pts, K, dist, w, h):
    """
    Two checks that the reprojection error alone cannot reveal.

    (a) How far distortion displaces the image corners. For a phone wide-angle lens this is usually < 20 px;
        tens of px usually means a degenerate solution where k1 and k2 have opposite signs and cancel out.
    (b) How much fx changes with a distortion model with fewer degrees of freedom. A large change
        means the data cannot distinguish these models, and fx itself carries that much uncertainty.
    """
    print("\n=== Advanced checks (what the reprojection error cannot show) ===")

    r2 = ((0 - K[0, 2]) / K[0, 0]) ** 2 + ((0 - K[1, 2]) / K[1, 1]) ** 2
    f = 1 + dist[0, 0] * r2 + dist[0, 1] * r2 ** 2
    px = abs(1 - f) * np.hypot(K[0, 2], K[1, 2])
    print(f"  (a) Radial displacement at the image corner {px:.0f} px ({abs(1-f)*100:.1f}% off ideal)", end="  ")
    ok_a = px < 25
    print("✓" if ok_a else "✗ too large, the distortion solution is likely degenerate")

    fxs = [K[0, 0]]
    for fl, name in [(cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2, "k1 only"),
                     (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K1
                      | cv2.CALIB_ZERO_TANGENT_DIST, "no distortion")]:
        _, K2, _, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=fl)
        fxs.append(K2[0, 0])
    spread = (max(fxs) - min(fxs)) / K[0, 0] * 100
    print(f"  (b) fx range across distortion models {min(fxs):.1f} ~ {max(fxs):.1f} ({spread:.1f}%)", end="  ")
    ok_b = spread < 1.0
    print("✓" if ok_b else
          f"✗ too large\n      The scene-scale acceptance criterion in chapter 7 is an error < 2%;"
          f" a {spread:.1f}% uncertainty in fx turns directly into scale error")
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--name", required=True, help="output file name, e.g. iphone12")
    ap.add_argument("--squares", nargs=2, type=int, default=[9, 6],
                    help="number of inner corners (cols rows), default 9 6")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--blur-reject", type=float, default=0.5,
                    help="drop frames whose sharpness is below this fraction of the median; 0 disables it")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "repos/SousVide/configs/camera")
    ap.add_argument("--force", action="store_true",
                    help="write the final file name (without the REJECTED suffix) even if checks fail. "
                         "Check results are still printed in full.")
    args = ap.parse_args()

    pattern = tuple(args.squares)
    print(f"=== Calibrating {args.name} ===")
    print(f"  Video: {args.video}")
    print(f"  Checkerboard inner corners: {pattern[0]}x{pattern[1]}")

    frames = sample_frames(args.video, args.max_frames, args.blur_reject)
    h, w = frames[0].shape[:2]

    vis_dir = PROJECT_ROOT / "notes" / f"calib_{args.name}"
    obj_pts, img_pts = detect_corners(frames, pattern, vis_dir)

    if len(img_pts) < 10:
        raise SystemExit(
            f"\n✗ Checkerboard detected in only {len(img_pts)} frames, too few (>= 20 recommended).\n"
            "  Common causes: checkerboard not flat, out of focus, wrong inner-corner count (9x6 is inner corners, not squares), too little light, or glare."
        )

    print()
    cov_ok = coverage_report(img_pts, w, h)

    # k3 fixed to 0: the FiGS config stores only 4 distortion coefficients
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (w, h), None, None, flags=cv2.CALIB_FIX_K3
    )
    if not ret:
        raise SystemExit("✗ Calibration failed")

    # Per-image reprojection error
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        errs.append(cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj))
    errs = np.array(errs)

    print()
    print("=== Results ===")
    print(f"  Image size (WxH): {w} x {h}")
    print(f"  fx={K[0,0]:.4f}  fy={K[1,1]:.4f}")
    print(f"  cx={K[0,2]:.4f}  cy={K[1,2]:.4f}   (image center is about {w/2:.1f}, {h/2:.1f})")
    print(f"  Distortion k1={dist[0,0]:+.6f} k2={dist[0,1]:+.6f} p1={dist[0,2]:+.6f} p2={dist[0,3]:+.6f}")
    print()
    print(f"  Mean Reprojection Error: {errs.mean():.4f} px      <- acceptance criterion < 0.5")
    print(f"    median {np.median(errs):.4f} / worst {errs.max():.4f} px")

    # Consistency checks
    warn = []
    if abs(K[0, 2] - w / 2) > 0.1 * w or abs(K[1, 2] - h / 2) > 0.1 * h:
        warn.append("Principal point (cx,cy) is more than 10% from the image center; usually the inner-corner count is swapped or the image is cropped")
    if abs(K[0, 0] - K[1, 1]) / K[0, 0] > 0.05:
        warn.append("fx and fy differ by more than 5%, which a phone lens should not do; check whether the video was scaled non-uniformly")
    for m in warn:
        print(f"  ⚠ {m}")

    print(f"  Corner detection previews: {vis_dir}")
    adv_ok = sanity_checks(obj_pts, img_pts, K, dist, w, h)

    print()
    print(f"  ⚠ The chapter 5 scene video must use EXACTLY the same recording settings ({w}x{h}, same lens, "
          f"same frame rate, same orientation),\n    otherwise these intrinsics will not match.")
    print()
    err_ok = errs.mean() < 0.5

    cam = {
        "model": "OPENCV",
        "height": h,
        "width": w,
        "intrinsics_matrix": K.tolist(),
        "distortion_coefficients": [float(dist[0, i]) for i in range(4)],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # On failure the file name gets a REJECTED suffix so chapter 6 does not use it by mistake.
    # Failures in this pipeline are mostly silent: a wrong scale raises no error, it just performs badly.
    # --force is for the case "known risk, decided to use it anyway".
    passed = err_ok and cov_ok and adv_ok
    out = args.out_dir / (f"{args.name}.json" if (passed or args.force)
                          else f"{args.name}.REJECTED.json")
    out.write_text(json.dumps(cam, indent=4))
    print(f"  Written: {out}")
    print()
    if err_ok and cov_ok and adv_ok:
        print("✅ PASSED (reprojection error, corner distribution and advanced checks all pass)")
    else:
        fails = []
        if not err_ok: fails.append(f"reprojection error {errs.mean():.3f} px")
        if not cov_ok: fails.append("corner distribution")
        if not adv_ok: fails.append("advanced checks")
        print(f"❌ FAILED: {', '.join(fails)}")
        print("   A low reprojection error does not mean accurate intrinsics -- it only says the model explains the points you captured,")
        print("   not that those points are enough to constrain the parameters. All three must pass.")


if __name__ == "__main__":
    main()
