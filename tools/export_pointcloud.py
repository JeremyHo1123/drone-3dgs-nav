"""
從 splatfacto 模型導出表面點雲（第 8 章）。

為什麼不用 ns-export pointcloud：
  nerfstudio 1.1.5 的 `ExportPointCloud.main()` 有兩處與 splatfacto 不相容：
    1. exporter.py:146 無條件存取 `pipeline.datamanager.train_pixel_sampler`，
       但 splatfacto 用的 FullImageDatamanager 沒有這個屬性 → AttributeError
    2. exporter_utils.generate_point_cloud() 第 130 行
       `assert isinstance(ray_bundle, RayBundle)`，而 FullImageDatamanager 的
       next_train() 回傳 (Cameras, batch)；splatfacto 的 get_outputs 也是吃 Camera
  兩者都是硬性不相容，不是設定問題。

本工具照 implement.md 第 8 章描述的原理實作（該描述是正確的）：
  從訓練相機視角渲染 depth，反投影成 3D 點，用 accumulation(alpha) > 門檻過濾。
  空氣區域累積不透明度上不來會被濾掉，所以閘門/通道中間天然是空的。

⚠ 已知代價：玻璃、紗網、細鐵絲這類半透明或亞像素結構會「該有點卻沒有點」。

座標細節：
  splatfacto 用 render_mode="RGB+ED"，depth 是沿相機 z 軸的期望深度，
  不是沿射線的距離。因此用 nerfstudio 的 generate_rays 取得世界座標下的
  origins/directions 後，需除以射線與相機主軸的夾角餘弦才能還原真實距離。

用法:
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
                    help="accumulation 低於此值視為空氣，直接濾掉")
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
    print(f"  訓練相機 {N} 個，每個目標取樣 {per_cam:,} 點")

    P, C = [], []
    rng = np.random.default_rng(0)
    for i in range(N):
        cam = cams[i:i + 1].to(model.device)
        with torch.no_grad():
            out = model.get_outputs_for_camera(cam)
        depth = out["depth"][..., 0]                  # (H,W) 沿相機 z 軸
        acc = out["accumulation"][..., 0]             # (H,W)
        rgb = out["rgb"]                              # (H,W,3)

        rb = cams.generate_rays(camera_indices=i).to(model.device)
        o = rb.origins.squeeze()                      # (H,W,3) 世界座標
        d = rb.directions.squeeze()                   # (H,W,3) 已正規化

        # 相機主軸（nerfstudio/OpenGL 慣例：相機看向 -z）
        c2w = cams[i].camera_to_worlds.to(model.device)
        fwd = -c2w[:3, 2]
        cos = (d @ fwd).clamp(min=1e-6)               # 射線與主軸夾角餘弦
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
            print(f"    {i+1}/{N} 台相機，累計 {sum(len(x) for x in P):,} 點")

    P = np.concatenate(P); C = np.concatenate(C)
    print(f"  反投影後 {len(P):,} 點")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1).astype(np.float64))

    if args.remove_outliers:
        pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors,
                                                   std_ratio=args.std_ratio)
        print(f"  統計離群濾波後 {len(pcd.points):,} 點"
              f"（移除 {len(P)-len(pcd.points):,}）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pcd)
    Q = np.asarray(pcd.points)
    print(f"  已寫入 {args.out}")
    print(f"  範圍 x [{Q[:,0].min():.2f},{Q[:,0].max():.2f}] "
          f"y [{Q[:,1].min():.2f},{Q[:,1].max():.2f}] "
          f"z [{Q[:,2].min():.2f},{Q[:,2].max():.2f}]")

    # 正確性檢查：地面應為 z≈0 的水平面
    r = np.linalg.norm(Q[:, :2], axis=1)
    m = (r < 3.0) & (np.abs(Q[:, 2]) < 0.25)
    if m.sum() > 500:
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(Q[m])
        (a, b, c, d0), inl = sub.segment_plane(0.02, 3, 2000)
        n = np.array([a, b, c]); n = n / np.linalg.norm(n) * np.sign(c or 1)
        tilt = np.degrees(np.arccos(np.clip(n @ np.array([0, 0, 1.0]), -1, 1)))
        print(f"  地面檢查: 法向量 {np.round(n,4)}，與 z 軸夾角 {tilt:.2f}°，"
              f"高度 z={-d0/c*100:+.2f} cm", end="  ")
        print("✓ 反投影正確" if tilt < 3 else "✗ 反投影可能有座標錯誤")


if __name__ == "__main__":
    main()
