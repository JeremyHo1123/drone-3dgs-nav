"""
第 6 章建圖前的預檢。

用與 FiGS `extract_frames()` 完全相同的判定邏輯掃過影片每一幀，
在跑昂貴的 SfM + 3DGS 訓練之前，先確認資料本身不會讓管線失敗。

檢查的失敗模式（都已讀原始碼確認）：
  1. 含 tag 的幀 < num_marked(20)
     → extract_frames 走警告分支，但下游 extract_positions 硬性要求
       數量「剛好等於」num_marked，必然拋
       ValueError: Mismatched number of aruco and sfm transforms
  2. 不含 tag 的幀 < num_images - num_marked(280)
     → distribute_values 找不到候選時把 None 塞進清單 →
       TypeError: '>' not supported between 'float' and 'NoneType'
  3. 完全沒有不含 tag 的幀
     → distribute_values 的 values[0] → IndexError
  4. 反射造成一幀偵測到 2 個以上 marker
     → len(ids)==1 不成立，該幀被歸入「不含 tag」池，兩邊都吃虧

用法:
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
    ap.add_argument("--config", required=True, help="configs/captures/<name>.json 的 name")
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔幾幀掃一次（>1 只用於快速預覽，判定會失準）")
    args = ap.parse_args()

    cfg = json.loads((CFG / "captures" / f"{args.config}.json").read_text())
    cam, ext = cfg["camera"], cfg["extractor"]
    Nimg, Narc = ext["num_images"], ext["num_marked"]
    mkr_id = ext["marker_id"]
    need_empty = Nimg - Narc

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"=== 預檢 {args.video.name} ===")
    print(f"  {w}x{h}，{total} 幀 @ {fps:.2f} fps = {total/fps:.1f} 秒")
    if (w, h) != (cam["width"], cam["height"]):
        print(f"  ✗ 解析度與 config 的 {cam['width']}x{cam['height']} 不符 → 內參無效")
    else:
        print(f"  ✓ 解析度與 config 一致")

    # 與 extract_frames 相同的偵測器設定
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

    print(f"\n=== 依 extract_frames 的分箱結果 ===")
    print(f"  含 tag（剛好 1 個且 id={mkr_id}）: {n_tag:5d} 幀   需要 >= {Narc}",
          "✓" if n_tag >= Narc else "✗")
    print(f"  不含 tag                        : {n_empty:5d} 幀   需要 >= {need_empty}",
          "✓" if n_empty >= need_empty else "✗")
    if n_multi:
        print(f"  ⚠ 偵測到 2 個以上 marker 的幀   : {n_multi:5d} 幀（反射？這些幀兩邊都不算數）")
    if n_wrongid:
        print(f"  ⚠ 偵測到錯誤 id 的幀            : {n_wrongid:5d} 幀")

    ok = n_tag >= Narc and n_empty >= need_empty

    if tag_px:
        a = np.array(tag_px)
        print(f"\n=== 含 tag 幀的品質 ===")
        print(f"  marker 邊長像素: 中位數 {np.median(a):.0f} px"
              f"（最小 {a.min():.0f} / 最大 {a.max():.0f}）")
        n_good = int((a >= 100).sum())
        print(f"  >= 100 px 的幀: {n_good} 幀"
              f"（姿態解可靠所需；只要 >= {Narc} 就夠）",
              "✓" if n_good >= Narc else "✗")
        est_d = cam["intrinsics_matrix"][0][0] * ext["marker_length"] / np.median(a)
        print(f"  由中位數推估拍攝距離約 {est_d:.2f} m")

        tt = np.array(tag_times)
        print(f"  出現時段: {tt.min():.1f}s ~ {tt.max():.1f}s"
              f"（影片長 {total/fps:.1f}s）")
        # 視角是否分散：用 marker 像素大小的變異當粗略代理
        print(f"  邊長變異係數 {a.std()/a.mean()*100:.0f}%"
              f"（越大代表距離/角度越分散，單一視角的方形標記有姿態歧義）")

    print()
    if ok:
        print("✅ 預檢通過，可以進第 6 章建圖")
    else:
        print("❌ 預檢未通過，直接建圖必定失敗。重拍或調整 config：")
        if n_tag < Narc:
            print(f"   - 含 tag 的幀只有 {n_tag}，把 num_marked 降到 <= {n_tag}，"
                  f"或重拍時多繞 tag 拍幾秒")
        if n_empty < need_empty:
            print(f"   - 不含 tag 的幀只有 {n_empty}，把 num_images 降到 <= {n_empty + Narc}，"
                  f"或重拍時讓 tag 離開畫面久一點")


if __name__ == "__main__":
    main()
