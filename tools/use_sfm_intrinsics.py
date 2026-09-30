"""
Copy the camera intrinsics that COLMAP self-calibrated during the SfM stage into the
capture config, so the ArUco scale solve uses the same intrinsics as SfM and 3DGS training.

Why this step exists (README section 6.1): the pipeline consumes intrinsics in three places.
SfM self-calibrates its own, 3DGS training reads SfM's values from transforms.json, and only
the `scale` stage reads the `camera` block of the capture config. solvePnP distances are
proportional to fx, so if the config's fx is 2% off, the whole scene comes out 2% too large
or too small, and nothing warns you. scene04 failed exactly this way with a checkerboard fx
that was 2.32% too large.

Run it after `--stages frames,sfm,check` and before `--stages scale,...`:

  python tools/use_sfm_intrinsics.py --scene scene05 --config pixel8_scene05 --dry-run
  python tools/use_sfm_intrinsics.py --scene scene05 --config pixel8_scene05

It rewrites the `camera` block (the `extractor` block is kept as it is) in:
  repos/SousVide/configs/captures/<config>.json   the file build_gsplat.py reads (required)
  repos/SousVide/configs/camera/<config>.json     if it exists, so a later run of
                                                  make_capture_config.py cannot copy the
                                                  old intrinsics back in
  configs/captures/<config>.json                  if it exists (the copy tracked in git)
  configs/camera/<config>.json                    if it exists (the copy tracked in git)
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOUSVIDE = PROJECT_ROOT / "repos/SousVide"


def camera_from_sfm(t, src):
    """Build a capture-config camera block from nerfstudio's sfm/transforms.json."""
    if "fl_x" not in t:
        raise SystemExit(f"{src} has no top-level fl_x: SfM used more than one camera model "
                         "(per-frame intrinsics). This pipeline expects one camera for the whole video.")
    if t.get("camera_model") != "OPENCV":
        raise SystemExit(f"{src}: camera_model is {t.get('camera_model')!r}, expected 'OPENCV' "
                         "(k1, k2, p1, p2 distortion).")
    return {
        "model": "OPENCV",
        "height": t["h"],
        "width": t["w"],
        "_comment": ("Intrinsics COLMAP self-calibrated on this scene's own video, copied from "
                     f"{src.relative_to(PROJECT_ROOT)} by tools/use_sfm_intrinsics.py. They must match "
                     "what the SfM used: solvePnP distance is proportional to fx, so a wrong fx scales "
                     "the whole scene silently. See README section 6.1."),
        "intrinsics_matrix": [
            [t["fl_x"], 0.0, t["cx"]],
            [0.0, t["fl_y"], t["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [t["k1"], t["k2"], t["p1"], t["p2"]],
    }


def describe(cam):
    K = cam["intrinsics_matrix"]
    return (f"{cam['width']}x{cam['height']}  fx={K[0][0]:.2f} fy={K[1][1]:.2f} "
            f"cx={K[0][2]:.2f} cy={K[1][2]:.2f}  distortion={cam['distortion_coefficients']}")


def main():
    ap = argparse.ArgumentParser(
        description="Copy COLMAP's self-calibrated intrinsics from the SfM stage into a capture config.")
    ap.add_argument("--scene", required=True, help="scene name, as passed to run_pipeline.sh --scene")
    ap.add_argument("--config", required=True, help="capture config name, as passed to run_pipeline.sh --config")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    src = SOUSVIDE / "gsplats/workspace" / args.scene / "sfm/transforms.json"
    if not src.exists():
        raise SystemExit(f"{src} not found. Run the frames and sfm stages first:\n"
                         f"  ./tools/run_pipeline.sh --scene {args.scene} --config {args.config} "
                         "--stages frames,sfm,check")
    new_cam = camera_from_sfm(json.loads(src.read_text()), src)

    capture = SOUSVIDE / "configs/captures" / f"{args.config}.json"
    if not capture.exists():
        raise SystemExit(f"{capture} not found. Create it with tools/make_capture_config.py first.")
    old_cam = json.loads(capture.read_text())["camera"]

    print(f"SfM intrinsics : {describe(new_cam)}")
    if old_cam is None:
        print("config camera  : null")
    else:
        print(f"config camera  : {describe(old_cam)}")
        if (old_cam["width"], old_cam["height"]) != (new_cam["width"], new_cam["height"]):
            print("  !! the image size differs: the config was made for a different video mode")
        fx_old = old_cam["intrinsics_matrix"][0][0]
        d = fx_old / new_cam["intrinsics_matrix"][0][0] - 1
        print(f"  config fx vs SfM fx: {d * 100:+.2f}%  -> with the old values the scene scale "
              f"would have been off by about {d * 100:+.2f}%")

    targets = [capture] + [p for p in (SOUSVIDE / "configs/camera" / f"{args.config}.json",
                                       PROJECT_ROOT / "configs/captures" / f"{args.config}.json",
                                       PROJECT_ROOT / "configs/camera" / f"{args.config}.json")
                           if p.exists()]
    for p in targets:
        cfg = json.loads(p.read_text())
        if "extractor" in cfg:              # a capture config: replace only its camera block
            cfg["camera"] = new_cam
        else:                               # a camera file: replace the whole file
            cfg = new_cam
        if args.dry_run:
            print(f"would update {p.relative_to(PROJECT_ROOT)}")
        else:
            p.write_text(json.dumps(cfg, indent=4))
            print(f"updated {p.relative_to(PROJECT_ROOT)}")
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
