"""
Pre-check before the chapter 6 scene reconstruction.

Scans every frame of the video with exactly the same decision logic as FiGS `extract_frames()`,
to confirm, before running the expensive SfM + 3DGS training, that the data itself will not make the pipeline fail.

Failure modes checked (all confirmed by reading the source):
  1. Frames with the tag < num_marked(20)
     -> extract_frames takes the warning branch, but the downstream extract_positions strictly requires
       the count to be EXACTLY num_marked, so it always raises
       ValueError: Mismatched number of aruco and sfm transforms
  2. Frames without the tag < num_images - num_marked(280)
     -> when distribute_values finds no candidate it puts None into the list ->
       TypeError: '>' not supported between 'float' and 'NoneType'
  3. No frames without the tag at all
     -> values[0] in distribute_values -> IndexError
  4. Reflections make a frame detect 2 or more markers
     -> len(ids)==1 fails, the frame goes into the "no tag" pool, and both sides lose

Usage:
  python preflight_capture.py --video <path> --config iphone12
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFG = PROJECT_ROOT / "repos/SousVide/configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--config", required=True, help="the name in configs/captures/<name>.json")
    ap.add_argument("--stride", type=int, default=1,
                    help="scan every Nth frame (>1 only for a quick preview; the verdict becomes inaccurate)")
    args = ap.parse_args()

    cfg = json.loads((CFG / "captures" / f"{args.config}.json").read_text())
    cam, ext = cfg["camera"], cfg["extractor"]
    Nimg, Narc = ext["num_images"], ext["num_marked"]
    mkr_id = ext["marker_id"]
    need_empty = Nimg - Narc

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"=== Pre-check {args.video.name} ===")
    print(f"  {w}x{h}, {total} frames @ {fps:.2f} fps = {total/fps:.1f} s")
    if (w, h) != (cam["width"], cam["height"]):
        print(f"  ✗ Resolution does not match the config's {cam['width']}x{cam['height']} -> intrinsics invalid")
    else:
        print(f"  ✓ Resolution matches the config")

    # Same detector settings as extract_frames
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    n_tag, n_empty, n_multi, n_wrongid = 0, 0, 0, 0
    tag_times, tag_px, sharp = [], [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.stride > 1 and idx % args.stride:
            idx += 1
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)

        if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
            n_tag += 1
            tag_times.append(t)
            p = corners[0].reshape(4, 2)
            tag_px.append(np.mean([np.linalg.norm(p[i] - p[(i+1) % 4]) for i in range(4)]))
            sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            n_empty += 1
            if ids is not None and len(ids) > 1:
                n_multi += 1
            elif ids is not None and len(ids) == 1:
                n_wrongid += 1
        idx += 1
    cap.release()

    print(f"\n=== Binning result per extract_frames ===")
    print(f"  With tag (exactly 1, id={mkr_id})       : {n_tag:5d} frames   need >= {Narc}",
          "✓" if n_tag >= Narc else "✗")
    print(f"  Without tag                      : {n_empty:5d} frames   need >= {need_empty}",
          "✓" if n_empty >= need_empty else "✗")
    if n_multi:
        print(f"  ⚠ Frames with 2+ markers         : {n_multi:5d} frames (reflections? these frames count for neither side)")
    if n_wrongid:
        print(f"  ⚠ Frames with a wrong id         : {n_wrongid:5d} frames")

    ok = n_tag >= Narc and n_empty >= need_empty

    if tag_px:
        a = np.array(tag_px)
        print(f"\n=== Quality of frames with the tag ===")
        print(f"  Marker side length in pixels: median {np.median(a):.0f} px"
              f" (min {a.min():.0f} / max {a.max():.0f})")
        n_good = int((a >= 100).sum())
        print(f"  Frames >= 100 px: {n_good} frames"
              f" (needed for a reliable pose solution; >= {Narc} is enough)",
              "✓" if n_good >= Narc else "✗")
        est_d = cam["intrinsics_matrix"][0][0] * ext["marker_length"] / np.median(a)
        print(f"  Camera distance estimated from the median: about {est_d:.2f} m")

        tt = np.array(tag_times)
        print(f"  Visible from: {tt.min():.1f}s ~ {tt.max():.1f}s"
              f" (video length {total/fps:.1f}s)")
        # Are the viewpoints spread out: use the variation of the marker pixel size as a rough proxy
        print(f"  Side-length coefficient of variation {a.std()/a.mean()*100:.0f}%"
              f" (larger means more varied distances/angles; a square marker seen from a single viewpoint has pose ambiguity)")

    print()
    if ok:
        print("✅ Pre-check passed, ready for the chapter 6 reconstruction")
    else:
        print("❌ Pre-check failed; reconstructing now will certainly fail. Re-shoot or adjust the config:")
        if n_tag < Narc:
            print(f"   - Only {n_tag} frames with the tag; lower num_marked to <= {n_tag}, "
                  f"or when re-shooting circle the tag for a few more seconds")
        if n_empty < need_empty:
            print(f"   - Only {n_empty} frames without the tag; lower num_images to <= {n_empty + Narc}, "
                  f"or when re-shooting keep the tag out of frame for longer")


if __name__ == "__main__":
    main()
