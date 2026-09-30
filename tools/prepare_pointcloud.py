"""
Prepare an exported dense point cloud as an obstacle point cloud usable by grad_nav.

Does three things: outlier removal, voxel downsampling, validation.

**Why downsampling is mandatory**: grad_nav's ObstacleDistanceCalculator
(utils/point_cloud_util.py:filter_points_in_fov) builds intermediate tensors of shape
[num_envs, num_points, 3], and four of them exist at once (vectors / vectors_norm / dot_products / angles),
about 4096 x N bytes in total (for 128 parallel environments).

  scene04 raw 978,285 points -> about 4.0 GB; together with 2.58M Gaussians and the network this blows past 16 GB.
  Below 100k points -> about 0.4 GB, safe.

This raises no error; it just hits CUDA out of memory partway through training, so deal with it beforehand.

**Why outlier removal is needed**: the reconstructed dense point cloud has stray points floating in mid-air. The obstacle
distance is measured to "the nearest point", so a single stray point acts as a ghost obstacle and silently corrupts the reward.

Usage:
  python tools/prepare_pointcloud.py --in <dense.ply> --out <target.ply>
  python tools/prepare_pointcloud.py --in ... --out ... --voxel 0.06 --num-envs 128
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# The four intermediate tensors that coexist in filter_points_in_fov, in bytes per "environment x point":
#   vectors [B,N,3] fp32      = 12
#   vectors_norm [B,N,3] fp32 = 12
#   dot_products [B,N] fp32   =  4
#   angles [B,N] fp32         =  4
BYTES_PER_ENV_POINT = 32


def mem_estimate(n_points, n_envs):
    """Return the size (GB) of filter_points_in_fov's intermediate tensors."""
    return n_envs * n_points * BYTES_PER_ENV_POINT / 1024 ** 3


def main():
    ap = argparse.ArgumentParser(description="Prepare the obstacle point cloud for grad_nav")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--voxel", type=float, default=0.08,
                    help="Voxel edge length in m. This is also the upper bound on the quantization error of the obstacle distance; "
                         "keep it well below obst_collision_limit (0.50 m in this project)")
    ap.add_argument("--nb-neighbors", type=int, default=20,
                    help="Statistical outlier removal: number of neighbors")
    ap.add_argument("--std-ratio", type=float, default=2.0,
                    help="Statistical outlier removal: std-dev multiplier; smaller removes more aggressively")
    ap.add_argument("--percentile", type=float, default=0.5,
                    help="First trim far outliers by percentile (trims this percentage on each axis)")
    ap.add_argument("--crop-center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="Keep only points within horizontal radius --crop-radius of (X, Y) (distant background outside "
                         "the task area never comes within obstacle-avoidance distance and only uses GPU memory)")
    ap.add_argument("--crop-radius", type=float, default=12.0)
    ap.add_argument("--min-neighbors", type=int, default=0,
                    help="Radius outlier removal: delete points with fewer neighbors than this within --neighbor-radius (0 = off). "
                         "Targets isolated 3DGS floaters left along the capture path, which statistical removal cannot clear in large scenes")
    ap.add_argument("--neighbor-radius", type=float, default=0.10)
    ap.add_argument("--num-envs", type=int, default=128,
                    help="Number of parallel environments used to estimate GPU memory")
    ap.add_argument("--budget-gb", type=float, default=1.5,
                    help="GPU memory budget for the intermediate tensors; a warning is printed if exceeded")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    pcd = o3d.io.read_point_cloud(str(src))
    pts = np.asarray(pcd.points)
    n0 = len(pts)
    print(f"Loaded {src}")
    print(f"  {n0:,} points   estimated GPU memory {mem_estimate(n0, a.num_envs):.2f} GB "
          f"({a.num_envs} environments)")
    print(f"  extent x[{pts[:,0].min():7.2f},{pts[:,0].max():7.2f}] "
          f"y[{pts[:,1].min():7.2f},{pts[:,1].max():7.2f}] "
          f"z[{pts[:,2].min():7.2f},{pts[:,2].max():7.2f}]")

    # 1) Percentile crop: first cut points very far from the main body, otherwise the voxel grid gets huge and sparse
    lo = np.percentile(pts, a.percentile, axis=0)
    hi = np.percentile(pts, 100 - a.percentile, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pcd = pcd.select_by_index(np.where(keep)[0])
    n1 = len(pcd.points)
    print(f"\n1) Percentile crop ({a.percentile}% ~ {100-a.percentile}%)      "
          f"{n0:,} -> {n1:,}  (-{100*(n0-n1)/n0:.1f}%)")
    p1 = np.asarray(pcd.points)
    print(f"     extent x[{p1[:,0].min():7.2f},{p1[:,0].max():7.2f}] "
          f"y[{p1[:,1].min():7.2f},{p1[:,1].max():7.2f}] "
          f"z[{p1[:,2].min():7.2f},{p1[:,2].max():7.2f}]")

    # 1b) Task-area crop (optional)
    if a.crop_center is not None:
        p = np.asarray(pcd.points)
        keep = np.linalg.norm(p[:, :2] - np.array(a.crop_center), axis=1) <= a.crop_radius
        pcd = pcd.select_by_index(np.where(keep)[0])
        print(f"1b) Task-area crop center {tuple(a.crop_center)} radius {a.crop_radius} m  "
              f"{n1:,} -> {len(pcd.points):,}")
        n1 = len(pcd.points)

    # 1c) Radius outlier removal (optional)
    #   2026-09-21 sunny_baseball_field: the route around the boxes had 157 floaters, with a median of only
    #   1 neighbor within 10 cm, and a median 0.48 m from the capture camera (typical 3DGS floaters near the camera). Statistical removal only cleared 27,
    #   because the distant trees across the whole scene are sparse to begin with, which widened the statistical threshold.
    if a.min_neighbors > 0:
        pcd, _ = pcd.remove_radius_outlier(nb_points=a.min_neighbors, radius=a.neighbor_radius)
        print(f"1c) Radius outlier removal (within {a.neighbor_radius} m: < {a.min_neighbors} neighbors)  "
              f"{n1:,} -> {len(pcd.points):,}")
        n1 = len(pcd.points)

    # 2) Statistical outlier removal: delete isolated points floating in mid-air with abnormally distant neighbors
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=a.nb_neighbors,
                                            std_ratio=a.std_ratio)
    n2 = len(pcd.points)
    print(f"2) Statistical outlier removal         {n1:,} -> {n2:,}  "
          f"(-{100*(n1-n2)/n1:.1f}%)")

    # 3) Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=a.voxel)
    n3 = len(pcd.points)
    print(f"3) Voxel downsampling ({a.voxel} m)         {n2:,} -> {n3:,}  "
          f"(-{100*(n2-n3)/n2:.1f}%)")

    gb = mem_estimate(n3, a.num_envs)
    print(f"\nFinal {n3:,} points   estimated GPU memory {gb:.2f} GB   "
          f"(total reduction {100*(n0-n3)/n0:.1f}%)")
    if gb > a.budget_gb:
        print(f"  ! Over the {a.budget_gb} GB budget. Two knobs to turn:"
              f"\n      increase --voxel (at the cost of larger obstacle-distance quantization error)"
              f"\n      decrease num_actors in the training config (GPU memory is proportional to it)")

    dst = Path(a.dst)
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(dst), pcd)
    print(f"\n✓ Wrote {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")

    # Clearance statistics at flight altitudes, to judge whether the airframe fits
    p = np.asarray(pcd.points)
    kd = o3d.geometry.KDTreeFlann(pcd)
    lo2, hi2 = p.min(0), p.max(0)
    zs = [z for z in (1.0, 1.4, 1.8) if lo2[2] < z < hi2[2]]
    print(f"\nClearance at flight altitudes (airframe safety radius 0.465 m):")
    for z in zs:
        g = [[x, y, z]
             for x in np.arange(lo2[0], hi2[0], 0.5)
             for y in np.arange(lo2[1], hi2[1], 0.5)]
        d = np.array([np.sqrt(kd.search_knn_vector_3d(np.array(c), 1)[2][0]) for c in g])
        print(f"  z={z:.1f} m: sampled {len(g):5d} points, "
              f"positions that fit the airframe {100*(d>=0.465).mean():5.1f}%, "
              f"median clearance {np.median(d):.2f} m")


if __name__ == "__main__":
    main()
