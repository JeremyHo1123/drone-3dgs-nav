"""
把訓練用的影像做成可快速掃視的接觸表，並標出每張的銳利度。

用途：肉眼檢查訓練資料品質——哪些幀糊、哪些區域重複、覆蓋是否均勻。
銳利度用 Laplacian 變異數，數值越大越銳利（此指標同時受畫面內容影響，
同一場景內比較才有意義）。
"""
import argparse, glob
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--cols", type=int, default=10)
ap.add_argument("--rows", type=int, default=10)
ap.add_argument("--thumb-w", type=int, default=170)
a = ap.parse_args()

fs = sorted(glob.glob(f"{a.dir}/*.png"))
out = Path("/home/jeremy/drone/review") / f"sheets_{a.tag}"
out.mkdir(parents=True, exist_ok=True)
print(f"  {a.tag}: {len(fs)} 張")

sharp = []
thumbs = []
for f in fs:
    g = cv2.imread(f, 0)
    sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
    im = cv2.imread(f)
    h = int(a.thumb_w * im.shape[0] / im.shape[1])
    thumbs.append(cv2.resize(im, (a.thumb_w, h)))
sharp = np.array(sharp)
th = thumbs[0].shape[0]

def label(t, txt, color=(0, 255, 255)):
    t = t.copy()
    cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(t, txt, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return t

per = a.cols * a.rows
n_sheet = int(np.ceil(len(fs) / per))
for s in range(n_sheet):
    sheet = np.full((a.rows * (th + 4), a.cols * (a.thumb_w + 4), 3), 30, np.uint8)
    for k in range(per):
        i = s * per + k
        if i >= len(fs): break
        r, c = divmod(k, a.cols)
        col = (0, 165, 255) if sharp[i] < 60 else (0, 255, 255)
        stem = Path(fs[i]).stem
        nm = stem.split('_')[-1] if '_' in stem else stem
        t = label(thumbs[i], f"{nm} s{sharp[i]:.0f}", col)
        sheet[r*(th+4):r*(th+4)+th, c*(a.thumb_w+4):c*(a.thumb_w+4)+a.thumb_w] = t
    p = out / f"sheet_{s+1:02d}.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"    {p.name}  ({min(per,len(fs)-s*per)} 張)")

# 最銳利 / 最模糊各 10 張
for name, idx in [("sharpest", np.argsort(-sharp)[:10]), ("blurriest", np.argsort(sharp)[:10])]:
    W = 300
    tw = [cv2.resize(cv2.imread(fs[i]), (W, int(W*cv2.imread(fs[i]).shape[0]/cv2.imread(fs[i]).shape[1]))) for i in idx]
    hh = tw[0].shape[0]
    grid = np.full((2*(hh+4), 5*(W+4), 3), 30, np.uint8)
    for k, (i, t) in enumerate(zip(idx, tw)):
        r, c = divmod(k, 5)
        t = label(t, f"{Path(fs[i]).stem} s={sharp[i]:.0f}")
        grid[r*(hh+4):r*(hh+4)+hh, c*(W+4):c*(W+4)+W] = t
    p = out / f"_{name}10.jpg"
    cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    {p.name}")

print(f"  銳利度: 中位數 {np.median(sharp):.0f}, "
      f"5/95 百分位 {np.percentile(sharp,5):.0f}/{np.percentile(sharp,95):.0f}, "
      f"<60 的有 {(sharp<60).sum()} 張")
