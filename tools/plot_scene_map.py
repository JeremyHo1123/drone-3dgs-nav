"""
把點雲畫成帶公尺格線的俯視圖與側視圖，方便對照實際場地量測。

座標系：原點在 ArUco tag，z 軸向上（重力反方向）。
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WS = PROJECT_ROOT / "repos/SousVide/gsplats/workspace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pcd", default=None)
    ap.add_argument("--clip", type=float, default=99.0,
                    help="裁掉離群點的百分位（預設 99）")
    args = ap.parse_args()

    p = Path(args.pcd) if args.pcd else WS / args.scene / "sparse_pc.ply"
    pcd = o3d.io.read_point_cloud(str(p))
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"  載入 {len(P):,} 點")

    # 去離群：統計濾波 + 百分位裁切
    pcd2, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    P, C = P[keep], (C[keep] if C is not None else None)
    lo, hi = np.percentile(P, 100 - args.clip, axis=0), np.percentile(P, args.clip, axis=0)
    m = np.all((P >= lo) & (P <= hi), axis=1)
    P, C = P[m], (C[m] if C is not None else None)
    print(f"  去離群後 {len(P):,} 點")
    print(f"  範圍 x [{P[:,0].min():.2f}, {P[:,0].max():.2f}]  "
          f"y [{P[:,1].min():.2f}, {P[:,1].max():.2f}]  "
          f"z [{P[:,2].min():.2f}, {P[:,2].max():.2f}]")

    out = PROJECT_ROOT / "notes" / f"ch7_map_{args.scene}.png"
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    views = [
        ("TOP VIEW (x-y), z toward viewer", 0, 1, "x [m]", "y [m]"),
        ("SIDE VIEW (x-z)", 0, 2, "x [m]", "z [m]"),
        ("SIDE VIEW (y-z)", 1, 2, "y [m]", "z [m]"),
    ]
    for ax, (title, i, j, xl, yl) in zip(axes, views):
        ax.scatter(P[:, i], P[:, j], s=0.4,
                   c=(C if C is not None else "steelblue"), linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.grid(True, which="major", lw=0.8, alpha=0.5)
        ax.grid(True, which="minor", lw=0.3, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        # 標出 ArUco 原點
        ax.plot(0, 0, "r+", ms=16, mew=2.5)
        ax.annotate("ArUco (0,0)", (0, 0), xytext=(6, 6),
                    textcoords="offset points", color="red", fontsize=9)

    fig.suptitle(f"{args.scene}  |  major grid 1 m, minor 0.5 m  |  origin = ArUco tag, z = up",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"  已存 {out}")


if __name__ == "__main__":
    main()
