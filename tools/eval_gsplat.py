"""
Evaluate trained splatfacto scenes in a uniform way, for side-by-side comparison across scenes.

Outputs:
  Gaussian count, train/eval view PSNR, and renders "along the capture path" vs "off the capture path".

Why look at off-path views:
  If the camera trajectory degenerates into a 1-D line, geometry perpendicular to the path is unconstrained.
  Views along the path look fine; problems only show up once you move off it.
  The drone flies in 3-D space, not along the line walked during capture.

⚠ When measuring PSNR the camera optimizer must be applied by hand: splatfacto's get_outputs only
  calls camera_optimizer.apply_to_camera() when self.training=True;
  in eval mode it uses the original poses (models/splatfacto.py:543-547).
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
    ap.add_argument("--tag", required=True, help="Output file name prefix, e.g. scene02")
    ap.add_argument("--n-psnr", type=int, default=12)
    ap.add_argument("--offsets", type=float, nargs="*", default=[0.0, 0.3, 0.6, 1.0],
                    help="Distances off the capture path (SfM units), along the trajectory's normal direction")
    args = ap.parse_args()

    cfg, pipe, _, _ = eval_setup(args.config, test_mode="val")
    model = pipe.model
    tr, ev = pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset
    print(f"  Gaussians: {model.means.shape[0]:,}")
    print(f"  train {len(tr)} images / eval {len(ev)} images, "
          f"render {int(tr.cameras[0].width)}x{int(tr.cameras[0].height)}")

    ti = np.linspace(0, len(tr) - 1, min(args.n_psnr, len(tr))).astype(int)
    ei = np.linspace(0, len(ev) - 1, min(args.n_psnr, len(ev))).astype(int)
    ptr = psnr_set(model, tr, ti)
    pev = psnr_set(model, ev, ei)
    print(f"  train-view PSNR {ptr.mean():.2f} dB (n={len(ptr)})")
    print(f"  eval-view PSNR {pev.mean():.2f} dB (n={len(ei)})")
    print(f"  difference {ptr.mean()-pev.mean():+.2f} dB"
          f"   (large gap = insufficient view coverage; both low = data or capacity limit)")

    # Normal direction of the trajectory (third principal axis)
    P = tr.cameras.camera_to_worlds[:, :3, 3].cpu().numpy()
    c = P.mean(0)
    _, s, Vt = np.linalg.svd(P - c)
    normal = Vt[2]
    print(f"  trajectory principal-axis scales {np.round(s/np.sqrt(len(P)),3)}  → normal {np.round(normal,3)}")

    out = Path(__file__).resolve().parent.parent / "notes" / f"eval_{args.tag}"
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
        print(f"    offset {d:.1f} units: brightness mean {img.mean():.1f}, distinct values {len(np.unique(img))} → {p.name}")
    print(f"  images saved to {out}")


if __name__ == "__main__":
    main()
