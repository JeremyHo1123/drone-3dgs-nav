"""
第 7 章：驗證 3DGS 場景的公制尺度。

尺度錯了不會報錯，只會讓動力學、避障閾值(0.5 m)、reward 全部連帶錯誤。
必須當場驗證。

本工具做兩件事：
  1. 自動檢查——地面平面（高度、平整度、法向量），以及自動偵測到的
     牆面/平面之間的距離。你拿捲尺去核對這些數字即可。
  2. 互動量測——在 open3d 視窗中 Shift+左鍵選兩點，回報三維距離。

用法:
  python verify_scale.py --scene scene01              # 自動檢查
  python verify_scale.py --scene scene01 --pick       # 互動選點量測
  python verify_scale.py --scene scene01 --pcd <path> # 指定其他點雲（如第 8 章的稠密點雲）
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WS = PROJECT_ROOT / "repos/SousVide/gsplats/workspace"


def load(scene, pcd_path):
    p = Path(pcd_path) if pcd_path else WS / scene / "sparse_pc.ply"
    pcd = o3d.io.read_point_cloud(str(p))
    print(f"  點雲: {p}")
    print(f"  點數: {len(pcd.points):,}")
    return pcd


def robust_extent(P):
    print("\n=== 場景範圍 ===")
    print(f"  完整 bbox (m): {np.round(P.max(0) - P.min(0), 2)}  ← 被離群點主導，僅供參考")
    for lo, hi in [(1, 99), (5, 95)]:
        a, b = np.percentile(P, lo, axis=0), np.percentile(P, hi, axis=0)
        print(f"  {lo}~{hi} 百分位 (m): x {b[0]-a[0]:6.2f}  y {b[1]-a[1]:6.2f}  z {b[2]-a[2]:6.2f}")


def floor_check(pcd, P):
    """ArUco 平放地面 → 地面應為 z≈0 的水平面，法向量接近 (0,0,1)。"""
    print("\n=== 地面平面（驗證原點與 z 軸方向）===")
    # 只取 tag 周圍、接近 z=0 的點來擬合，避免抓到桌面
    r = np.linalg.norm(P[:, :2], axis=1)
    m = (r < 3.0) & (np.abs(P[:, 2]) < 0.25)
    if m.sum() < 100:
        print(f"  ⚠ 候選點僅 {m.sum()} 個，跳過")
        return
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(P[m])
    model, inliers = sub.segment_plane(distance_threshold=0.02,
                                       ransac_n=3, num_iterations=2000)
    a, b, c, d = model
    n = np.array([a, b, c]); n = n / np.linalg.norm(n) * np.sign(c if c else 1)
    pts = np.asarray(sub.points)[inliers]
    resid = np.abs(pts @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
    tilt = np.degrees(np.arccos(np.clip(n @ np.array([0, 0, 1.0]), -1, 1)))
    print(f"  擬合點數 {len(inliers)} / {m.sum()}")
    print(f"  法向量 {np.round(n,4)}   與 z 軸夾角 {tilt:.2f}°", end="  ")
    print("✓" if tilt < 3 else "⚠ 地面不水平，tag 可能沒平放或姿態解有偏差")
    z0 = -d / c if abs(c) > 1e-6 else float("nan")
    print(f"  平面高度 z = {z0*100:+.2f} cm（ArUco 平放地面時應接近 0）", end="  ")
    print("✓" if abs(z0) < 0.05 else "⚠")
    print(f"  平整度 RMS {resid.std()*1000:.1f} mm，最大 {resid.max()*1000:.1f} mm")


def plane_distances(pcd, P):
    """反覆 RANSAC 取出主要平面，回報平行平面之間的距離供實地核對。"""
    print("\n=== 自動偵測到的平面（拿捲尺核對這些距離）===")
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(P)
    work = work.voxel_down_sample(0.02)
    planes = []
    for i in range(8):
        if len(work.points) < 500:
            break
        model, inl = work.segment_plane(distance_threshold=0.03,
                                        ransac_n=3, num_iterations=1500)
        if len(inl) < 300:
            break
        pts = np.asarray(work.points)[inl]
        a, b, c, d = model
        n = np.array([a, b, c]); L = np.linalg.norm(n); n, d = n / L, d / L
        if n[2] < 0:
            n, d = -n, -d
        planes.append({"n": n, "d": d, "n_pts": len(inl),
                       "centroid": pts.mean(0), "extent": pts.max(0) - pts.min(0)})
        work = work.select_by_index(inl, invert=True)

    for i, p in enumerate(planes):
        kind = ("水平（地面/桌面/天花板）" if abs(p["n"][2]) > 0.9
                else "垂直（牆面）" if abs(p["n"][2]) < 0.3 else "傾斜")
        print(f"  平面 {i}: {kind:22} 點數 {p['n_pts']:6d}  "
              f"法向量 {np.round(p['n'],2)}  高度/位置 d={-p['d']:+.2f} m")
        print(f"           範圍 (m) {np.round(p['extent'],2)}  "
              f"中心 {np.round(p['centroid'],2)}")

    print("\n  平行平面之間的距離（夾角 < 10°）:")
    found = False
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            ni, nj = planes[i]["n"], planes[j]["n"]
            ang = np.degrees(np.arccos(np.clip(abs(ni @ nj), -1, 1)))
            if ang < 10:
                dist = abs(planes[i]["d"] - planes[j]["d"] * np.sign(ni @ nj))
                if dist > 0.3:
                    kind = "水平" if abs(ni[2]) > 0.9 else "垂直"
                    print(f"    平面 {i} ↔ {j} （{kind}）: {dist:.3f} m = {dist*100:.1f} cm")
                    found = True
    if not found:
        print("    （沒找到明顯的平行平面對，改用 --pick 手動量測）")


def pick(pcd):
    print("\n=== 互動量測 ===")
    print("  操作：Shift + 左鍵 選點（至少 2 個），選完按 Q 關閉視窗")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Shift+左鍵選點，完成後按 Q")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    P = np.asarray(pcd.points)
    if len(idx) < 2:
        print(f"  只選了 {len(idx)} 個點，需要至少 2 個")
        return
    print(f"  選了 {len(idx)} 個點:")
    for k, i in enumerate(idx):
        print(f"    P{k} = {np.round(P[i], 4)}")
    for k in range(len(idx) - 1):
        dv = P[idx[k + 1]] - P[idx[k]]
        print(f"  P{k}→P{k+1}: 距離 {np.linalg.norm(dv):.4f} m = {np.linalg.norm(dv)*100:.2f} cm"
              f"   (Δ {np.round(dv, 4)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pcd", default=None)
    ap.add_argument("--pick", action="store_true")
    args = ap.parse_args()

    pcd = load(args.scene, args.pcd)
    P = np.asarray(pcd.points)
    robust_extent(P)
    floor_check(pcd, P)
    plane_distances(pcd, P)
    if args.pick:
        pick(pcd)
    else:
        print("\n  要手動量測特定物件，加上 --pick")


if __name__ == "__main__":
    main()
