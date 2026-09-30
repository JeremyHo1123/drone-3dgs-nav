"""
Merge the camera intrinsics from Chapter 3 and the ArUco parameters from Chapter 4 into a FiGS capture config.

generate_gsplat() reads configs/captures/<name>.json, which contains two blocks:
  camera    -- produced by tools/calibrate_camera.py, written to configs/camera/<name>.json
  extractor -- frame extraction and ArUco parameters

⚠ marker_length is the **measured side length (meters) of the printed black square**, not the design value.
  It is the metric scale reference for the whole 3DGS scene; a wrong value raises no error, it just scales the whole scene wrong.

Usage:
  python make_capture_config.py --name iphone12 --marker-length 0.144
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG = PROJECT_ROOT / "repos/SousVide/configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--marker-length", required=True, type=float,
                    help="measured side length of the printed ArUco black square, in meters")
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--num-marked", type=int, default=20)
    ap.add_argument("--marker-id", type=int, default=0)
    args = ap.parse_args()

    cam_file = CFG / "camera" / f"{args.name}.json"
    if not cam_file.exists():
        raise SystemExit(f"Camera intrinsics not found: {cam_file}\n"
                         f"  Run tools/calibrate_camera.py first to generate it.")
    camera = json.loads(cam_file.read_text())

    if not (0.02 <= args.marker_length <= 2.0):
        raise SystemExit(f"marker_length={args.marker_length} does not look plausible. "
                         "The unit is meters: 14.4 cm should be entered as 0.144.")

    cfg = {
        "camera": camera,
        "extractor": {
            "num_images": args.num_images,
            "num_marked": args.num_marked,
            "marker_length": args.marker_length,
            "marker_id": args.marker_id,
        },
    }

    out_dir = CFG / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.json"
    out.write_text(json.dumps(cfg, indent=4))

    print(f"Wrote {out}\n")
    c = cfg["camera"]
    print(f"  camera    : {c['width']}x{c['height']} (WxH)  "
          f"fx={c['intrinsics_matrix'][0][0]:.2f} fy={c['intrinsics_matrix'][1][1]:.2f}")
    print(f"              cx={c['intrinsics_matrix'][0][2]:.2f} "
          f"cy={c['intrinsics_matrix'][1][2]:.2f}  distortion: {len(c['distortion_coefficients'])} coefficients")
    e = cfg["extractor"]
    print(f"  extractor : num_images={e['num_images']} num_marked={e['num_marked']} "
          f"marker_id={e['marker_id']}")
    print(f"              marker_length={e['marker_length']} m  ← scene scale reference")


if __name__ == "__main__":
    main()
