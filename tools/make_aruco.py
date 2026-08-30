"""
產生 FiGS 定尺度用的 ArUco tag PDF。

FiGS 的硬性要求（已讀原始碼確認，capture_generation.py:166/234）：
  - 字典必須是 DICT_4X4_50（寫死，不可改）
  - marker_id 由 config 的 extractor_config["marker_id"] 指定（預設 0）
  - solvePnP 的 marker_points 是 ±marker_length/2，而 cv2.aruco 回傳的是
    黑色方塊「外緣」角點 → marker_length 就是黑色方塊的整個邊長
    （含最外圈黑邊，不含白色留白）

尺寸選擇：
  DICT_4X4 的圖案是 6x6 個模組（4x4 資料 + 最外圈 1 模組黑邊）。
  偵測器需要黑色方塊四周有白色留白，慣例是至少 1 個模組寬。
  因此邊長 S 需滿足 (頁寬 - S)/2 >= S/6，即 S <= 頁寬 * 3/4。
    A4 (210mm) -> S <= 157.5mm
    A3 (297mm) -> S <= 222.8mm
  原作用 34.1 cm，手冊建議至少 25 cm；單張 A4/A3 達不到，
  要更大需送影印店印 A2 以上（本工具用 --page 可指定任意尺寸）。
  ⚠ 不建議用多張 A4 拼貼——接縫的錯位與不平整會直接汙染 solvePnP 的姿態解。

用法:
  python make_aruco.py                 # 同時產生 A4 與 A3
  python make_aruco.py --page 420 594  # 自訂頁面 (寬 高, mm)，例如 A2
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM_PER_INCH = 25.4
MARKER_ID = 0
MODULES = 6          # DICT_4X4 = 4x4 資料 + 1 模組黑邊 => 6x6
SAFE_MM = 10.0       # 印表機不可列印邊界的安全值

PAGES = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}


def build_pdf(page_name, page_w, page_h, out_dir: Path):
    # 取 6x6 的模組點陣，一個像素就是一個模組，避免重採樣造成的幾何誤差
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = cv2.aruco.generateImageMarker(d, MARKER_ID, MODULES)   # 6x6, 0/255

    # 白邊 >= 1 模組，且 >= 安全邊界
    s_by_quiet = page_w * MODULES / (MODULES + 2)     # (page_w - S)/2 >= S/6
    s_by_safe = page_w - 2 * SAFE_MM
    S = np.floor(min(s_by_quiet, s_by_safe))          # 取整數 mm
    mod = S / MODULES
    ox = (page_w - S) / 2
    oy = (page_h - S) / 2 + 8                          # 略上移，下方放說明

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w); ax.set_ylim(0, page_h)
    ax.set_aspect("equal"); ax.axis("off")

    for r in range(MODULES):
        for c in range(MODULES):
            if bits[r, c] == 0:                        # 黑模組
                ax.add_patch(Rectangle(
                    (ox + c * mod, oy + (MODULES - 1 - r) * mod),
                    mod, mod, facecolor="black", edgecolor="none"))

    # 標示「要量的就是這一段」——對齊黑色方塊外緣
    ay = oy - 7
    ax.annotate("", xy=(ox, ay), xytext=(ox + S, ay),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color="black"))
    for x in (ox, ox + S):
        ax.plot([x, x], [ay - 2, oy], color="black", lw=0.5, ls=":")
    ax.text(ox + S / 2, ay - 4,
            f"MEASURE THIS EDGE (outer black square). Nominal {S:.0f} mm.",
            ha="center", va="top", fontsize=9)
    ax.text(ox + S / 2, ay - 11,
            "measured = ________ mm   ->  marker_length = ______ m",
            ha="center", va="top", fontsize=9)

    ax.text(ox, oy + S + 5,
            f"ArUco DICT_4X4_50  id={MARKER_ID}   |   {page_name}   |   "
            f"nominal side {S:.0f} mm",
            fontsize=9, va="bottom")
    ax.text(ox, oy + S + 1,
            "Print scale need NOT be exact - but you MUST measure the printed "
            "black edge. That number is the metric scale of the whole scene.",
            fontsize=7, va="bottom")

    out = out_dir / f"aruco_id{MARKER_ID}_{page_name}_{S:.0f}mm.pdf"
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out, S


def verify(pdf: Path, S_nominal: float):
    """渲染 PDF 後實際跑一次偵測，並量出黑色方塊邊長。"""
    import subprocess, tempfile, glob, os
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-png", "-r", "300", str(pdf),
                        os.path.join(td, "p")], check=True)
        f = sorted(glob.glob(os.path.join(td, "p*.png")))[0]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)

    n = 0 if ids is None else len(ids)
    ok = (n == 1 and int(ids[0][0]) == MARKER_ID)
    msg = f"偵測到 {n} 個 marker"
    if n >= 1:
        msg += f"，id={[int(i) for i in ids.ravel()]}"
    if ok:
        p = corners[0].reshape(4, 2)
        sides = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
        mm = 25.4 / 300
        meas = np.mean(sides) * mm
        msg += f"，量得黑框邊長 {meas:.2f} mm（標稱 {S_nominal:.0f}，"
        msg += f"偏差 {abs(meas - S_nominal) / S_nominal * 100:.2f}%）"
    return ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", nargs=2, type=float, metavar=("W_MM", "H_MM"),
                    help="自訂頁面尺寸(mm)，不給則產生 A4 與 A3")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "captures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = ([("custom", args.page[0], args.page[1])] if args.page
               else [(k, v[0], v[1]) for k, v in PAGES.items()])

    for name, w, h in targets:
        pdf, S = build_pdf(name, w, h, args.out_dir)
        ok, msg = verify(pdf, S)
        print(f"{name} ({w:.0f}x{h:.0f} mm)")
        print(f"  {pdf.name}")
        print(f"  黑色方塊標稱邊長 {S:.0f} mm，白邊 {(w - S) / 2:.1f} mm "
              f"（= {(w - S) / 2 / (S / MODULES):.2f} 個模組，需 >= 1）")
        print(f"  驗證: {'通過' if ok else '**失敗**'} — {msg}")
        print()


if __name__ == "__main__":
    main()
