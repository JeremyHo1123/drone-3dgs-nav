"""
Export a surface point cloud from a splatfacto model (chapter 8).

Why not ns-export pointcloud:
  `ExportPointCloud.main()` in nerfstudio 1.1.5 is incompatible with splatfacto in two places:
    1. exporter.py:146 unconditionally accesses `pipeline.datamanager.train_pixel_sampler`,
       but the FullImageDatamanager used by splatfacto has no such attribute -> AttributeError
    2. exporter_utils.generate_point_cloud() line 130 does
       `assert isinstance(ray_bundle, RayBundle)`, while FullImageDatamanager's
       next_train() returns (Cameras, batch); splatfacto's get_outputs also takes a Camera
  Both are hard incompatibilities, not configuration problems.

This tool implements the approach described in chapter 8 of the original plan
(`implement.md`, not published; that description is correct):
  render depth from the training camera views, back-project to 3D points, and filter
  with accumulation(alpha) > threshold. Empty-air regions never build up enough opacity
  and get filtered out, so the space inside gates/passages is naturally empty.

⚠ Known cost: semi-transparent or sub-pixel structures such as glass, mesh screens and
  thin wires end up "missing points where there should be some".

Coordinate details:
  splatfacto uses render_mode="RGB+ED"; depth is the expected depth along the camera
  z axis, not the distance along the ray. So after getting world-frame origins/directions
  from nerfstudio's generate_rays, divide by the cosine of the angle between the ray and
  the camera principal axis to recover the true distance.

Usage:
  python export_pointcloud.py --config <config.yml> --out <out.ply> --num-points 1000000
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from nerfstudio.utils.eval_utils import eval_setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-points", type=int, default=1_000_000)
    ap.add_argument("--alpha-thresh", type=float, default=0.5,
                    help="accumulation below this value is treated as empty air and filtered out")
    ap.add_argument("--remove-outliers", action="store_true", default=True)
    ap.add_argument("--std-ratio", type=float, default=2.0)
    ap.add_argument("--nb-neighbors", type=int, default=20)
    args = ap.parse_args()

    cfg, pipeline, _, _ = eval_setup(args.config, test_mode="inference")
    model = pipeline.model
    dm = pipeline.datamanager
    cams = dm.train_dataset.cameras
    N = len(cams)
    per_cam = max(1, args.num_points // N)
    print(f"  {N} training cameras, target of {per_cam:,} sampled points per camera")

    P, C = [], []
    rng = np.random.default_rng(0)
    for i in range(N):
        cam = cams[i:i + 1].to(model.device)
        with torch.no_grad():
            out = model.get_outputs_for_camera(cam)
        depth = out["depth"][..., 0]                  # (H,W) along camera z axis
        acc = out["accumulation"][..., 0]             # (H,W)
        rgb = out["rgb"]                              # (H,W,3)

        rb = cams.generate_rays(camera_indices=i).to(model.device)
        o = rb.origins.squeeze()                      # (H,W,3) world frame
        d = rb.directions.squeeze()                   # (H,W,3) normalized

        # camera principal axis (nerfstudio/OpenGL convention: camera looks along -z)
        c2w = cams[i].camera_to_worlds.to(model.device)
        fwd = -c2w[:3, 2]
        cos = (d @ fwd).clamp(min=1e-6)               # cosine of angle between ray and principal axis
        pts = o + d * (depth / cos).unsqueeze(-1)

        m = acc > args.alpha_thresh
        pts, cols = pts[m], rgb[m]
        n = pts.shape[0]
        if n == 0:
            continue
        if n > per_cam:
            sel = torch.from_numpy(rng.choice(n, per_cam, replace=False)).to(pts.device)
            pts, cols = pts[sel], cols[sel]
        P.append(pts.cpu().numpy())
        C.append(cols.cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N} cameras, {sum(len(x) for x in P):,} points so far")

    P = np.concatenate(P); C = np.concatenate(C)
    print(f"  {len(P):,} points after back-projection")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1).astype(np.float64))

    if args.remove_outliers:
        pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors,
                                                   std_ratio=args.std_ratio)
        print(f"  {len(pcd.points):,} points after statistical outlier removal"
              f" (removed {len(P)-len(pcd.points):,})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pcd)
    Q = np.asarray(pcd.points)
    print(f"  Wrote {args.out}")
    print(f"  Range x [{Q[:,0].min():.2f},{Q[:,0].max():.2f}] "
          f"y [{Q[:,1].min():.2f},{Q[:,1].max():.2f}] "
          f"z [{Q[:,2].min():.2f},{Q[:,2].max():.2f}]")

    # Sanity check: the floor should be a horizontal plane at z≈0
    r = np.linalg.norm(Q[:, :2], axis=1)
    m = (r < 3.0) & (np.abs(Q[:, 2]) < 0.25)
    if m.sum() > 500:
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(Q[m])
        (a, b, c, d0), inl = sub.segment_plane(0.02, 3, 2000)
        n = np.array([a, b, c]); n = n / np.linalg.norm(n) * np.sign(c or 1)
        tilt = np.degrees(np.arccos(np.clip(n @ np.array([0, 0, 1.0]), -1, 1)))
        print(f"  Floor check: normal {np.round(n,4)}, angle to z axis {tilt:.2f}°, "
              f"height z={-d0/c*100:+.2f} cm", end="  ")
        print("✓ back-projection correct" if tilt < 3 else "✗ back-projection may have a coordinate error")


if __name__ == "__main__":
    main()
