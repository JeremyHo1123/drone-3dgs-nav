"""
Generate the ArUco tag PDF used by the FiGS scale solve.

Hard requirements of FiGS (confirmed by reading the source, capture_generation.py:166/234):
  - The dictionary must be DICT_4X4_50 (hard-coded, cannot be changed)
  - marker_id is set by the config's extractor_config["marker_id"] (default 0)
  - solvePnP's marker_points are +/-marker_length/2, and cv2.aruco returns the
    corners of the black square's OUTER edge -> marker_length is the full side length of the black square
    (including the outermost black border, excluding the white quiet zone)

Choosing the size:
  A DICT_4X4 pattern is 6x6 modules (4x4 data + a 1-module black border).
  The detector needs a white quiet zone around the black square, by convention at least 1 module wide.
  So the side length S must satisfy (page width - S)/2 >= S/6, i.e. S <= page width * 3/4.
    A4 (210mm) -> S <= 157.5mm
    A3 (297mm) -> S <= 222.8mm
  The original authors used 34.1 cm and the manual recommends at least 25 cm; a single A4/A3 sheet cannot reach that,
  so for larger tags have a print shop print A2 or bigger (this tool accepts any size via --page).
  ⚠ Tiling several A4 sheets is not recommended -- misalignment and unevenness at the seams directly corrupt the solvePnP pose solution.

Usage:
  python make_aruco.py                 # generate both A4 and A3
  python make_aruco.py --page 420 594  # custom page (width height, mm), e.g. A2
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
MODULES = 6          # DICT_4X4 = 4x4 data + 1-module black border => 6x6
SAFE_MM = 10.0       # safe value for the printer's non-printable margin

PAGES = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}


def build_pdf(page_name, page_w, page_h, out_dir: Path):
    # Take the 6x6 module bitmap, one pixel per module, to avoid geometric error from resampling
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = cv2.aruco.generateImageMarker(d, MARKER_ID, MODULES)   # 6x6, 0/255

    # White margin >= 1 module, and >= the safe margin
    s_by_quiet = page_w * MODULES / (MODULES + 2)     # (page_w - S)/2 >= S/6
    s_by_safe = page_w - 2 * SAFE_MM
    S = np.floor(min(s_by_quiet, s_by_safe))          # round down to whole mm
    mod = S / MODULES
    ox = (page_w - S) / 2
    oy = (page_h - S) / 2 + 8                          # shifted up slightly; the notes go below

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w); ax.set_ylim(0, page_h)
    ax.set_aspect("equal"); ax.axis("off")

    for r in range(MODULES):
        for c in range(MODULES):
            if bits[r, c] == 0:                        # black module
                ax.add_patch(Rectangle(
                    (ox + c * mod, oy + (MODULES - 1 - r) * mod),
                    mod, mod, facecolor="black", edgecolor="none"))

    # Mark "this is the segment to measure" -- aligned with the black square's outer edge
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
    """Render the PDF, actually run detection once, and measure the black square's side length."""
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
    msg = f"detected {n} marker(s)"
    if n >= 1:
        msg += f", id={[int(i) for i in ids.ravel()]}"
    if ok:
        p = corners[0].reshape(4, 2)
        sides = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
        mm = 25.4 / 300
        meas = np.mean(sides) * mm
        msg += f", measured black square side {meas:.2f} mm (nominal {S_nominal:.0f}, "
        msg += f"deviation {abs(meas - S_nominal) / S_nominal * 100:.2f}%)"
    return ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", nargs=2, type=float, metavar=("W_MM", "H_MM"),
                    help="custom page size (mm); if omitted, generate A4 and A3")
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
        print(f"  Black square nominal side {S:.0f} mm, white margin {(w - S) / 2:.1f} mm "
              f"(= {(w - S) / 2 / (S / MODULES):.2f} modules, need >= 1)")
        print(f"  Verification: {'PASSED' if ok else '**FAILED**'} - {msg}")
        print()


if __name__ == "__main__":
    main()
