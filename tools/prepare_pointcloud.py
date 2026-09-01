"""
把匯出的稠密點雲整備成 grad_nav 可用的障礙物點雲。

做三件事：去離群點、體素降採樣、驗證。

**為什麼一定要降採樣**：grad_nav 的 ObstacleDistanceCalculator
（utils/point_cloud_util.py:filter_points_in_fov）會建出 [環境數, 點數, 3]
的中間張量，而且同時存在四個（vectors / vectors_norm / dot_products / angles），
合計約 4096 x N bytes（以 128 個並行環境計）。

  scene04 原始 978,285 點 -> 約 4.0 GB，加上 2.58M 高斯點與網路會撐爆 16 GB。
  降到 10 萬點以下 -> 約 0.4 GB，安全。

這不會報錯，只會在訓練途中 CUDA out of memory，所以要事前處理掉。

**為什麼要去離群點**：重建的稠密點雲會有飄在半空中的雜點。障礙距離是取
「最近的點」，一顆雜點就等於一個幽靈障礙物，會讓 reward 無聲地算錯。

用法:
  python tools/prepare_pointcloud.py --in <dense.ply> --out <target.ply>
  python tools/prepare_pointcloud.py --in ... --out ... --voxel 0.06 --num-envs 128
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# filter_points_in_fov 同時存在的四個中間張量，換算成每個「環境 x 點」的位元組：
#   vectors [B,N,3] fp32      = 12
#   vectors_norm [B,N,3] fp32 = 12
#   dot_products [B,N] fp32   =  4
#   angles [B,N] fp32         =  4
BYTES_PER_ENV_POINT = 32


def mem_estimate(n_points, n_envs):
    """回傳 filter_points_in_fov 的中間張量大小（GB）。"""
    return n_envs * n_points * BYTES_PER_ENV_POINT / 1024 ** 3


def main():
    ap = argparse.ArgumentParser(description="整備 grad_nav 用的障礙物點雲")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--voxel", type=float, default=0.08,
                    help="體素邊長 m。這也是障礙距離的量化誤差上界，"
                         "要明顯小於 obst_collision_limit（本專案 0.50 m）")
    ap.add_argument("--nb-neighbors", type=int, default=20,
                    help="統計式離群點移除：鄰居數")
    ap.add_argument("--std-ratio", type=float, default=2.0,
                    help="統計式離群點移除：標準差倍率，越小刪越兇")
    ap.add_argument("--percentile", type=float, default=0.5,
                    help="先用百分位數裁掉遠端離群點（每軸各裁掉這個百分比）")
    ap.add_argument("--num-envs", type=int, default=128,
                    help="用來估算顯存的並行環境數")
    ap.add_argument("--budget-gb", type=float, default=1.5,
                    help="中間張量的顯存預算，超過會出聲警告")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    pcd = o3d.io.read_point_cloud(str(src))
    pts = np.asarray(pcd.points)
    n0 = len(pts)
    print(f"讀入 {src}")
    print(f"  {n0:,} 點   估計顯存 {mem_estimate(n0, a.num_envs):.2f} GB "
          f"（{a.num_envs} 個環境）")
    print(f"  範圍 x[{pts[:,0].min():7.2f},{pts[:,0].max():7.2f}] "
          f"y[{pts[:,1].min():7.2f},{pts[:,1].max():7.2f}] "
          f"z[{pts[:,2].min():7.2f},{pts[:,2].max():7.2f}]")

    # 1) 百分位裁切：先砍掉離主體極遠的點，否則體素格會被撐得很大很稀
    lo = np.percentile(pts, a.percentile, axis=0)
    hi = np.percentile(pts, 100 - a.percentile, axis=0)
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    pcd = pcd.select_by_index(np.where(keep)[0])
    n1 = len(pcd.points)
    print(f"\n1) 百分位裁切 ({a.percentile}% ~ {100-a.percentile}%)  "
          f"{n0:,} -> {n1:,}  (-{100*(n0-n1)/n0:.1f}%)")
    p1 = np.asarray(pcd.points)
    print(f"     範圍 x[{p1[:,0].min():7.2f},{p1[:,0].max():7.2f}] "
          f"y[{p1[:,1].min():7.2f},{p1[:,1].max():7.2f}] "
          f"z[{p1[:,2].min():7.2f},{p1[:,2].max():7.2f}]")

    # 2) 統計式離群點移除：刪掉飄在半空、鄰居異常遠的孤點
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=a.nb_neighbors,
                                            std_ratio=a.std_ratio)
    n2 = len(pcd.points)
    print(f"2) 統計式離群點移除              {n1:,} -> {n2:,}  "
          f"(-{100*(n1-n2)/n1:.1f}%)")

    # 3) 體素降採樣
    pcd = pcd.voxel_down_sample(voxel_size=a.voxel)
    n3 = len(pcd.points)
    print(f"3) 體素降採樣 ({a.voxel} m)         {n2:,} -> {n3:,}  "
          f"(-{100*(n2-n3)/n2:.1f}%)")

    gb = mem_estimate(n3, a.num_envs)
    print(f"\n最終 {n3:,} 點   估計顯存 {gb:.2f} GB   "
          f"（總縮減 {100*(n0-n3)/n0:.1f}%）")
    if gb > a.budget_gb:
        print(f"  ! 超過預算 {a.budget_gb} GB。兩個可調的旋鈕："
              f"\n      --voxel 調大（代價是障礙距離的量化誤差變大）"
              f"\n      訓練設定的 num_actors 調小（顯存與它成正比）")

    dst = Path(a.dst)
    if not dst.is_absolute():
        dst = PROJECT_ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(dst), pcd)
    print(f"\n✓ 已寫出 {dst}  ({dst.stat().st_size/1024**2:.1f} MB)")

    # 飛行高度層的淨空統計，用來判斷機體放不放得下
    p = np.asarray(pcd.points)
    kd = o3d.geometry.KDTreeFlann(pcd)
    lo2, hi2 = p.min(0), p.max(0)
    zs = [z for z in (1.0, 1.4, 1.8) if lo2[2] < z < hi2[2]]
    print(f"\n飛行高度的淨空（機體安全半徑 0.465 m）:")
    for z in zs:
        g = [[x, y, z]
             for x in np.arange(lo2[0], hi2[0], 0.5)
             for y in np.arange(lo2[1], hi2[1], 0.5)]
        d = np.array([np.sqrt(kd.search_knn_vector_3d(np.array(c), 1)[2][0]) for c in g])
        print(f"  z={z:.1f} m: 取樣 {len(g):5d} 點，"
              f"能容納機體的位置 {100*(d>=0.465).mean():5.1f}%，"
              f"中位淨空 {np.median(d):.2f} m")


if __name__ == "__main__":
    main()
