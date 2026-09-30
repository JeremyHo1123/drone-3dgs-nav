"""
Chapter 7: verify the metric scale of a 3DGS scene.

A wrong scale raises no error; it silently makes the dynamics, the obstacle-avoidance
threshold (0.5 m) and the reward all wrong along with it.
It must be verified on the spot.

This tool does two things:
  1. Automatic checks -- the floor plane (height, flatness, normal) and the distances
     between automatically detected walls/planes. Just check these numbers with a tape measure.
  2. Interactive measurement -- Shift+left-click two points in the open3d window to get
     their 3D distance.

Usage:
  python verify_scale.py --scene scene01              # automatic checks
  python verify_scale.py --scene scene01 --pick       # interactive point-picking measurement
  python verify_scale.py --scene scene01 --pcd <path> # use another point cloud (e.g. the chapter 8 dense point cloud)
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
    print(f"  Point cloud: {p}")
    print(f"  Points: {len(pcd.points):,}")
    return pcd


def robust_extent(P):
    print("\n=== Scene extent ===")
    print(f"  Full bbox (m): {np.round(P.max(0) - P.min(0), 2)}  <- dominated by outliers, for reference only")
    for lo, hi in [(1, 99), (5, 95)]:
        a, b = np.percentile(P, lo, axis=0), np.percentile(P, hi, axis=0)
        print(f"  {lo}~{hi} percentile (m): x {b[0]-a[0]:6.2f}  y {b[1]-a[1]:6.2f}  z {b[2]-a[2]:6.2f}")


def floor_check(pcd, P):
    """With the ArUco tag flat on the floor -> the floor should be a horizontal plane at z≈0 with normal close to (0,0,1)."""
    print("\n=== Floor plane (verifies origin and z-axis direction) ===")
    # Fit only points around the tag and near z=0, to avoid picking up table tops
    r = np.linalg.norm(P[:, :2], axis=1)
    m = (r < 3.0) & (np.abs(P[:, 2]) < 0.25)
    if m.sum() < 100:
        print(f"  ⚠ only {m.sum()} candidate points, skipping")
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
    print(f"  Fitted points {len(inliers)} / {m.sum()}")
    print(f"  Normal {np.round(n,4)}   angle to z axis {tilt:.2f}°", end="  ")
    print("✓" if tilt < 3 else "⚠ floor not level: the tag may not have been laid flat, or the pose solution is biased")
    z0 = -d / c if abs(c) > 1e-6 else float("nan")
    print(f"  Plane height z = {z0*100:+.2f} cm (should be close to 0 when the ArUco lies flat on the floor)", end="  ")
    print("✓" if abs(z0) < 0.05 else "⚠")
    print(f"  Flatness RMS {resid.std()*1000:.1f} mm, max {resid.max()*1000:.1f} mm")


def plane_distances(pcd, P):
    """Extract the main planes with repeated RANSAC and report distances between parallel planes for on-site checking."""
    print("\n=== Automatically detected planes (check these distances with a tape measure) ===")
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
        kind = ("horizontal (floor/table/ceiling)" if abs(p["n"][2]) > 0.9
                else "vertical (wall)" if abs(p["n"][2]) < 0.3 else "tilted")
        print(f"  Plane {i}: {kind:22} points {p['n_pts']:6d}  "
              f"normal {np.round(p['n'],2)}  height/position d={-p['d']:+.2f} m")
        print(f"           extent (m) {np.round(p['extent'],2)}  "
              f"center {np.round(p['centroid'],2)}")

    print("\n  Distances between parallel planes (angle < 10°):")
    found = False
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            ni, nj = planes[i]["n"], planes[j]["n"]
            ang = np.degrees(np.arccos(np.clip(abs(ni @ nj), -1, 1)))
            if ang < 10:
                dist = abs(planes[i]["d"] - planes[j]["d"] * np.sign(ni @ nj))
                if dist > 0.3:
                    kind = "horizontal" if abs(ni[2]) > 0.9 else "vertical"
                    print(f"    Plane {i} ↔ {j} ({kind}): {dist:.3f} m = {dist*100:.1f} cm")
                    found = True
    if not found:
        print("    (no clear pair of parallel planes found; measure manually with --pick instead)")


def pick(pcd):
    print("\n=== Interactive measurement ===")
    print("  Controls: Shift + left-click to pick points (at least 2), then press Q to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Shift+left-click to pick points, press Q when done")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    P = np.asarray(pcd.points)
    if len(idx) < 2:
        print(f"  Only {len(idx)} point(s) picked; at least 2 are needed")
        return
    print(f"  Picked {len(idx)} points:")
    for k, i in enumerate(idx):
        print(f"    P{k} = {np.round(P[i], 4)}")
    for k in range(len(idx) - 1):
        dv = P[idx[k + 1]] - P[idx[k]]
        print(f"  P{k}→P{k+1}: distance {np.linalg.norm(dv):.4f} m = {np.linalg.norm(dv)*100:.2f} cm"
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
        print("\n  To measure a specific object manually, add --pick")


if __name__ == "__main__":
    main()
