"""
把第 3 章的相機內參與第 4 章的 ArUco 參數合併成 FiGS 的 capture config。

generate_gsplat() 讀的是 configs/captures/<name>.json，內含兩個區塊：
  camera    —— 由 tools/calibrate_camera.py 產生，寫在 configs/camera/<name>.json
  extractor —— 抽幀與 ArUco 參數

⚠ marker_length 是**印出來的黑色方塊實測邊長（公尺）**，不是設計值。
  它就是整個 3DGS 場景的公制尺度基準，填錯不會報錯，只會讓場景整體縮放錯誤。

用法:
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
                    help="印出來的 ArUco 黑色方塊實測邊長，單位公尺")
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--num-marked", type=int, default=20)
    ap.add_argument("--marker-id", type=int, default=0)
    args = ap.parse_args()

    cam_file = CFG / "camera" / f"{args.name}.json"
    if not cam_file.exists():
        raise SystemExit(f"找不到相機內參 {cam_file}\n"
                         f"  先跑 tools/calibrate_camera.py 產生它。")
    camera = json.loads(cam_file.read_text())

    if not (0.02 <= args.marker_length <= 2.0):
        raise SystemExit(f"marker_length={args.marker_length} 看起來不合理。"
                         "單位是公尺，14.4 公分要填 0.144。")

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

    print(f"已寫入 {out}\n")
    c = cfg["camera"]
    print(f"  camera    : {c['width']}x{c['height']} (WxH)  "
          f"fx={c['intrinsics_matrix'][0][0]:.2f} fy={c['intrinsics_matrix'][1][1]:.2f}")
    print(f"              cx={c['intrinsics_matrix'][0][2]:.2f} "
          f"cy={c['intrinsics_matrix'][1][2]:.2f}  畸變 {len(c['distortion_coefficients'])} 個係數")
    e = cfg["extractor"]
    print(f"  extractor : num_images={e['num_images']} num_marked={e['num_marked']} "
          f"marker_id={e['marker_id']}")
    print(f"              marker_length={e['marker_length']} m  ← 場景尺度基準")


if __name__ == "__main__":
    main()
