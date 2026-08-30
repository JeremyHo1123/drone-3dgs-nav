"""
統一評估已訓練的 splatfacto 場景，供多場景橫向比較。

輸出：
  高斯數量、訓練/eval 視角 PSNR、以及「沿拍攝路徑」與「偏離拍攝路徑」的渲染對照。

為什麼要看偏離路徑的視角：
  相機軌跡若退化成一維，垂直於路徑方向的幾何缺乏約束。
  沿路徑的視角看起來會正常，偏離後才會暴露問題。
  無人機飛的是三維空間，不是拍攝時走過的那條線。

⚠ PSNR 量測時必須手動套用相機優化器：splatfacto 的 get_outputs 只在
  self.training=True 時才呼叫 camera_optimizer.apply_to_camera()，
  評估模式下用的是原始位姿（models/splatfacto.py:543-547）。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from nerfstudio.utils.eval_utils import eval_setup


def psnr_set(model, ds, idxs, apply_opt=True):
    cams = ds.cameras
    out = []
    for i in idxs:
        cam = cams[i:i + 1].to(model.device)
        cam.metadata = {"cam_idx": int(i)}
        if apply_opt:
            try:
                cam.camera_to_worlds = model.camera_optimizer.apply_to_camera(cam)
            except Exception:
                pass
        with torch.no_grad():
            o = model.get_outputs_for_camera(cam)
        pred = o["rgb"].cpu().numpy()
        gt = ds.get_image_float32(int(i)).numpy()
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        mse = float(((pred - gt) ** 2).mean())
        out.append(10 * np.log10(1 / max(mse, 1e-12)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="輸出檔名前綴，例如 scene02")
    ap.add_argument("--n-psnr", type=int, default=12)
    ap.add_argument("--offsets", type=float, nargs="*", default=[0.0, 0.3, 0.6, 1.0],
                    help="偏離拍攝路徑的距離（SfM 單位），沿軌跡的法線方向")
    args = ap.parse_args()

    cfg, pipe, _, _ = eval_setup(args.config, test_mode="val")
    model = pipe.model
    tr, ev = pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset
    print(f"  高斯數量: {model.means.shape[0]:,}")
    print(f"  train {len(tr)} 張 / eval {len(ev)} 張，"
          f"渲染 {int(tr.cameras[0].width)}x{int(tr.cameras[0].height)}")

    ti = np.linspace(0, len(tr) - 1, min(args.n_psnr, len(tr))).astype(int)
    ei = np.linspace(0, len(ev) - 1, min(args.n_psnr, len(ev))).astype(int)
    ptr = psnr_set(model, tr, ti)
    pev = psnr_set(model, ev, ei)
    print(f"  訓練視角 PSNR {ptr.mean():.2f} dB (n={len(ptr)})")
    print(f"  eval 視角 PSNR {pev.mean():.2f} dB (n={len(ei)})")
    print(f"  兩者差 {ptr.mean()-pev.mean():+.2f} dB"
          f"   （差距大=視角覆蓋不足；同樣低=資料或容量限制）")

    # 軌跡的法線方向（第三主軸）
    P = tr.cameras.camera_to_worlds[:, :3, 3].cpu().numpy()
    c = P.mean(0)
    _, s, Vt = np.linalg.svd(P - c)
    normal = Vt[2]
    print(f"  軌跡主軸尺度 {np.round(s/np.sqrt(len(P)),3)}  → 法線 {np.round(normal,3)}")

    out = Path("/home/jeremy/drone/notes") / f"eval_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    mid = len(tr) // 2
    for d in args.offsets:
        cam = tr.cameras[mid:mid + 1].to(model.device)
        c2w = cam.camera_to_worlds.clone()
        c2w[0, :3, 3] += torch.tensor(normal * d, dtype=c2w.dtype, device=c2w.device)
        cam.camera_to_worlds = c2w
        with torch.no_grad():
            o = model.get_outputs_for_camera(cam)
        img = (o["rgb"].cpu().numpy() * 255).astype(np.uint8)
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        p = out / f"offpath_{d:.1f}.png"
        cv2.imwrite(str(p), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        print(f"    偏離 {d:.1f} 單位: 亮度 mean {img.mean():.1f}，相異值 {len(np.unique(img))} → {p.name}")
    print(f"  影像存於 {out}")


if __name__ == "__main__":
    main()
