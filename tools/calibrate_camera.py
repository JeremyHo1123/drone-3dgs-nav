"""
相機內參標定（取代 FiGS 壞掉的 figs.render.capture_calibration.camera_calibration）。

為什麼不用上游的：
  1. capture_calibration.py:38 呼叫 ch.extract_frames(...)，但
     figs/utilities/capture_helper.py 裡沒有這個函式 → AttributeError。
     同名函式在 capture_generation.py，簽章完全不同且用途是寫檔不是回傳陣列。
  2. 它的預設路徑算錯：gsplats_path = Path(__file__).parent×3/'gsplats'
     → FiGS/src/gsplats（不存在）。generate_gsplat 用的是 parent×5，才對到 SousVide/gsplats。

輸出格式與 SousVide/configs/captures/*.json 的 "camera" 區塊一致：
  model / height / width / intrinsics_matrix / distortion_coefficients(4個, 扁平)

畸變固定 k3=0（CALIB_FIX_K3），因為 FiGS 官方 config 只存 4 個係數，
且手機廣角鏡的 k3 通常可忽略。

用法:
  python calibrate_camera.py --video <path> --name iphone12 [--squares 9 6] [--max-frames 120]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 專案根目錄由本檔位置推導（tools/ 的上一層），不寫死絕對路徑，
# 這樣整個專案搬到別處也不用改程式。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sample_frames(video_path: Path, max_frames: int, blur_reject: float):
    """從影片均勻取樣，並丟掉明顯模糊的幀。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟影片: {video_path}\n"
            "  iPhone 若用 HEVC(High Efficiency) 編碼可能讀不到。\n"
            "  設定 → 相機 → 格式 → 改成「最相容」(H.264) 後重錄。"
        )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise RuntimeError("影片幀數讀取失敗，檔案可能損壞")

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
        raise RuntimeError("一幀都讀不到")

    sharp = np.array(sharp)
    keep = sharp >= (np.median(sharp) * blur_reject)
    kept = [f for f, k in zip(frames, keep) if k][:max_frames]

    print(f"  影片: {total} 幀 @ {fps:.2f} fps，解析度 {frames[0].shape[1]}x{frames[0].shape[0]} (WxH)")
    print(f"  取樣 {len(frames)} 幀 → 銳利度過濾後保留 {len(kept)} 幀"
          f"（清晰度中位數 {np.median(sharp):.0f}）")
    return kept


def detect_corners(frames, pattern, vis_dir: Path):
    """偵測棋盤格角點並做次像素細化。"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 物點的尺度不影響內參（已實測驗證），這裡用 1.0 單位即可。
    #
    # ⚠ 索引順序：findChessboardCorners(patternSize=(cols,rows)) 回傳的角點是
    #   每列 cols 個、共 rows 列的 row-major 順序，所以物點必須是
    #   mgrid[0:cols, 0:rows].T —— 與 OpenCV 官方教學一致。
    #   FiGS 的 capture_calibration.py:70 寫成 mgrid[0:rows, 0:cols].T，
    #   兩者順序對不起來，會標出完全錯誤的內參（實測 fx 由 1310 變成 58）。
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
        if len(img_pts) <= 6:                      # 存幾張供目視檢查
            vis = f.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, ok)
            cv2.imwrite(str(vis_dir / f"detect_{len(img_pts):02d}.jpg"), vis)

    print(f"  偵測到棋盤格的幀數: {len(img_pts)} / {len(frames)}")
    return obj_pts, img_pts


def coverage_report(img_pts, w, h):
    """
    檢查角點在畫面上的分佈。

    ⚠ 只看「每區有沒有 >0」是不夠的：實測遇過中央 4014 點、角落只有 11 點
    卻仍判定 9/9 通過的情況，那種資料的畸變係數在畫面周邊等同外插。
    因此這裡改看「最少的一區佔總數的比例」。均勻時每區應約 11%。
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

    print("  畫面 3x3 分區的角點分佈（括號為佔比，均勻時各約 11%）:")
    for r in range(3):
        print("    " + "  ".join(f"{grid[r, c]:6d}({pct[r, c]:4.1f}%)" for c in range(3)))
    print(f"  有角點的分區 {filled}/9，最少的一區佔 {pct.min():.1f}%")

    ok = True
    if filled < 9:
        print("  ✗ 有分區完全沒拍到，畸變係數在該處無定義。")
        ok = False
    elif pct.min() < 3.0:
        print(f"  ✗ 分佈過度集中（最少的一區僅 {pct.min():.1f}%，應 >= 3%）。"
              "\n    畸變係數在畫面周邊等同外插，不可靠。"
              "\n    → 把棋盤格拍大一點，並讓它確實移到畫面四個角落。")
        ok = False
    else:
        print("  ✓ 分佈可接受")
    return ok


def sanity_checks(obj_pts, img_pts, K, dist, w, h):
    """
    兩項只靠重投影誤差看不出來的檢查。

    (a) 畸變在畫面角落造成多大位移。手機廣角鏡通常 < 20 px；
        數十 px 多半是 k1 與 k2 一正一負互相抵消的病態解。
    (b) 換用自由度較低的畸變模型後 fx 變動多少。若變動大，
        表示資料無法分辨這些模型，fx 本身就帶著那麼多不確定性。
    """
    print("\n=== 進階檢查（重投影誤差看不出來的部分）===")

    r2 = ((0 - K[0, 2]) / K[0, 0]) ** 2 + ((0 - K[1, 2]) / K[1, 1]) ** 2
    f = 1 + dist[0, 0] * r2 + dist[0, 1] * r2 ** 2
    px = abs(1 - f) * np.hypot(K[0, 2], K[1, 2])
    print(f"  (a) 角落徑向位移 {px:.0f} px（偏離理想 {abs(1-f)*100:.1f}%）", end="  ")
    ok_a = px < 25
    print("✓" if ok_a else "✗ 過大，畸變解很可能是病態的")

    fxs = [K[0, 0]]
    for fl, name in [(cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2, "只解 k1"),
                     (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K1
                      | cv2.CALIB_ZERO_TANGENT_DIST, "不解畸變")]:
        _, K2, _, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=fl)
        fxs.append(K2[0, 0])
    spread = (max(fxs) - min(fxs)) / K[0, 0] * 100
    print(f"  (b) 換畸變模型後 fx 變動 {min(fxs):.1f} ~ {max(fxs):.1f}（{spread:.1f}%）", end="  ")
    ok_b = spread < 1.0
    print("✓" if ok_b else
          f"✗ 過大\n      第 7 章的場景尺度驗收標準是誤差 < 2%，"
          f"fx 的 {spread:.1f}% 不確定性會直接轉成尺度誤差")
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--name", required=True, help="輸出檔名，例如 iphone12")
    ap.add_argument("--squares", nargs=2, type=int, default=[9, 6],
                    help="內角點數 (cols rows)，預設 9 6")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--blur-reject", type=float, default=0.5,
                    help="低於中位數清晰度此倍率的幀丟棄，設 0 關閉")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "repos/SousVide/configs/camera")
    ap.add_argument("--force", action="store_true",
                    help="即使檢查未通過也寫出正式檔名（不加 REJECTED 後綴）。"
                         "檢查結果仍會完整印出。")
    args = ap.parse_args()

    pattern = tuple(args.squares)
    print(f"=== 標定 {args.name} ===")
    print(f"  影片: {args.video}")
    print(f"  棋盤格內角點: {pattern[0]}x{pattern[1]}")

    frames = sample_frames(args.video, args.max_frames, args.blur_reject)
    h, w = frames[0].shape[:2]

    vis_dir = PROJECT_ROOT / "notes" / f"calib_{args.name}"
    obj_pts, img_pts = detect_corners(frames, pattern, vis_dir)

    if len(img_pts) < 10:
        raise SystemExit(
            f"\n✗ 只偵測到 {len(img_pts)} 幀，太少（建議 >= 20）。\n"
            "  常見原因：棋盤格不平整、對焦不準、內角點數填錯（9x6 是內角點不是方格數）、光線不足或反光。"
        )

    print()
    cov_ok = coverage_report(img_pts, w, h)

    # k3 固定為 0：FiGS 的 config 只存 4 個畸變係數
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (w, h), None, None, flags=cv2.CALIB_FIX_K3
    )
    if not ret:
        raise SystemExit("✗ 標定失敗")

    # 逐張重投影誤差
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        errs.append(cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj))
    errs = np.array(errs)

    print()
    print("=== 結果 ===")
    print(f"  影像尺寸 (WxH): {w} x {h}")
    print(f"  fx={K[0,0]:.4f}  fy={K[1,1]:.4f}")
    print(f"  cx={K[0,2]:.4f}  cy={K[1,2]:.4f}   (畫面中心約為 {w/2:.1f}, {h/2:.1f})")
    print(f"  畸變 k1={dist[0,0]:+.6f} k2={dist[0,1]:+.6f} p1={dist[0,2]:+.6f} p2={dist[0,3]:+.6f}")
    print()
    print(f"  Mean Reprojection Error: {errs.mean():.4f} px      ← 驗收標準 < 0.5")
    print(f"    中位數 {np.median(errs):.4f} / 最差 {errs.max():.4f} px")

    # 一致性檢查
    warn = []
    if abs(K[0, 2] - w / 2) > 0.1 * w or abs(K[1, 2] - h / 2) > 0.1 * h:
        warn.append("主點 (cx,cy) 離畫面中心超過 10%，通常表示內角點數填反或畫面被裁切")
    if abs(K[0, 0] - K[1, 1]) / K[0, 0] > 0.05:
        warn.append("fx 與 fy 差異超過 5%，手機鏡頭不該如此，檢查影片是否被非等比縮放")
    for m in warn:
        print(f"  ⚠ {m}")

    print(f"  角點偵測示意圖: {vis_dir}")
    adv_ok = sanity_checks(obj_pts, img_pts, K, dist, w, h)

    print()
    print(f"  ⚠ 第 5 章的場景影片必須用「完全相同」的錄影設定（{w}x{h}、同一顆鏡頭、"
          f"同幀率、同方向），\n    否則這份內參對不上。")
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
    # 未通過時檔名加上 REJECTED，避免第 6 章誤用。
    # 這條管線的失敗多半是無聲的：尺度錯了不會報錯，只會表現差。
    # --force 用於「已知風險仍決定採用」的情況。
    passed = err_ok and cov_ok and adv_ok
    out = args.out_dir / (f"{args.name}.json" if (passed or args.force)
                          else f"{args.name}.REJECTED.json")
    out.write_text(json.dumps(cam, indent=4))
    print(f"  已寫入: {out}")
    print()
    if err_ok and cov_ok and adv_ok:
        print("✅ 通過（重投影誤差、角點分佈、進階檢查三項皆過）")
    else:
        fails = []
        if not err_ok: fails.append(f"重投影誤差 {errs.mean():.3f} px")
        if not cov_ok: fails.append("角點分佈")
        if not adv_ok: fails.append("進階檢查")
        print(f"❌ 未通過：{'、'.join(fails)}")
        print("   重投影誤差低不代表內參準——它只說明模型能解釋你拍到的那些點，")
        print("   不保證那些點足以約束參數。三項要全過。")


if __name__ == "__main__":
    main()
