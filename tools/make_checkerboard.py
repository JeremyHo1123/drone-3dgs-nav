"""
Generate a print-ready checkerboard (9x6 inner corners = 10x7 squares), A4.

Two output formats:
  - PDF: the page size is a physical A4, so printing is unambiguous. **Use this one for printing (recommended).**
  - PNG: for the script's self-check (cv2 writes no DPI metadata, so a viewer can only guess the size when printing)

About print scaling:
  A calibration checkerboard **does not need an exact print scale**. Tests verified that square_size does not affect
  the calibrated intrinsics or distortion coefficients (scaling the object points only scales the translation vectors of the extrinsics proportionally).
  The only thing that breaks it is "non-uniform scaling" -- squares turning into rectangles create a fake fx/fy difference.
  So the paper carries two scale lines with a nominal 100 mm length (one horizontal, one vertical):
  if both measure the same length the scaling is uniform; they do not have to be exactly 100 mm.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

MM_PER_INCH = 25.4
SQUARE_MM = 24.0                    # see the SAFE_MM note below
INNER_CORNERS = (9, 6)              # (cols, rows) for cv2.findChessboardCorners
SQ_COLS = INNER_CORNERS[0] + 1      # 10 squares (long side)
SQ_ROWS = INNER_CORNERS[1] + 1      # 7 squares (short side)
A4_W_MM, A4_H_MM = 210.0, 297.0

# Typical laser/inkjet printers have a 4~5 mm non-printable margin; 12 mm is safe.
# With 25 mm squares the board is 250 mm tall, and after the safe margins there is no room for the labels,
# which get clipped or trigger the printer's "shrink to fit printable area". Changed to 24 mm:
# the board is 168 x 240 mm, leaving room for labels on every side.
# (The square size does not affect the calibration result, so shrinking costs nothing.)
SAFE_MM = 12.0

out_dir = Path(__file__).resolve().parent.parent / "captures"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- PDF (for printing)
board_w = SQ_ROWS * SQUARE_MM        # 175 mm
board_h = SQ_COLS * SQUARE_MM        # 250 mm
ox = (A4_W_MM - board_w) / 2 - 3     # slightly left of center; the right side is for the vertical label
oy = (A4_H_MM - board_h) / 2         # top and bottom margins split evenly for the label text

fig = plt.figure(figsize=(A4_W_MM / MM_PER_INCH, A4_H_MM / MM_PER_INCH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, A4_W_MM)
ax.set_ylim(0, A4_H_MM)
ax.set_aspect("equal")
ax.axis("off")

for r in range(SQ_COLS):
    for c in range(SQ_ROWS):
        if (r + c) % 2 == 0:
            continue                  # draw black squares only
        ax.add_patch(Rectangle(
            (ox + c * SQUARE_MM, oy + r * SQUARE_MM),
            SQUARE_MM, SQUARE_MM,
            facecolor="black", edgecolor="none"))

# The uniform-scale check uses "the span of the board itself" -- far more precise than measuring one square
# (on a 24 mm square a 0.5 mm ruler error is 2%; on the 240 mm span it is only 0.2%).
ax.annotate("", xy=(ox, oy - 6), xytext=(ox + board_w, oy - 6),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"))
ax.text(ox + board_w / 2, oy - 9,
        f"W = {SQ_ROWS} squares = {board_w:.0f} mm nominal",
        ha="center", va="top", fontsize=8)

ax.annotate("", xy=(ox + board_w + 6, oy), xytext=(ox + board_w + 6, oy + board_h),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color="black"))
ax.text(ox + board_w + 9, oy + board_h / 2,
        f"H = {SQ_COLS} squares = {board_h:.0f} mm nominal",
        ha="left", va="center", fontsize=8, rotation=90)

# Three lines of text stacked bottom-up; the lowest line is 1.5 mm above the board's top edge and must not overlap the board
ax.text(ox, oy + board_h + 1.5,
        f"CHECK: measured H / W must be {board_h / board_w:.3f} "
        f"(= {board_h:.0f}/{board_w:.0f}). Mount FLAT on rigid board.",
        fontsize=7, va="bottom")
ax.text(ox, oy + board_h + 6.0,
        "Print scale need NOT be 100%. Uniform scaling is harmless.",
        fontsize=7, va="bottom")
ax.text(ox, oy + board_h + 10.5,
        f"{INNER_CORNERS[0]}x{INNER_CORNERS[1]} inner corners  |  "
        f"{SQ_ROWS}x{SQ_COLS} squares @ {SQUARE_MM:.0f} mm nominal",
        fontsize=9, va="bottom")

pdf = out_dir / "checkerboard_9x6_A4.pdf"
fig.savefig(pdf, format="pdf")
plt.close(fig)

# ---------------------------------------------------------------- PNG (for the self-check)
DPI = 300
sq_px = SQUARE_MM / MM_PER_INCH * DPI
bw, bh = int(round(SQ_ROWS * sq_px)), int(round(SQ_COLS * sq_px))
board = np.zeros((bh, bw), np.uint8)
for r in range(SQ_COLS):
    for c in range(SQ_ROWS):
        if (r + c) % 2 == 0:
            board[int(round(r * sq_px)):int(round((r + 1) * sq_px)),
                  int(round(c * sq_px)):int(round((c + 1) * sq_px))] = 255
pad = int(sq_px * 0.8)
png_img = cv2.copyMakeBorder(board, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
png = out_dir / "checkerboard_9x6_A4.png"
cv2.imwrite(str(png), png_img)

# ---------------------------------------------------------------- self-check
ok, corners = cv2.findChessboardCorners(png_img, INNER_CORNERS, None)
n = 0 if corners is None else len(corners)

print(f"Generated:")
print(f"  {pdf}   <- for printing")
print(f"  {png}   <- for the script's self-check")
print(f"  Board {SQ_ROWS}x{SQ_COLS} squares = {board_w:.0f} x {board_h:.0f} mm, inner corners {INNER_CORNERS[0]}x{INNER_CORNERS[1]}")
print(f"  A4 page {A4_W_MM:.0f} x {A4_H_MM:.0f} mm")
print()
print(f"  cv2.findChessboardCorners self-check: {'PASSED' if ok else '**FAILED**'}"
      f" (detected {n} corners, expected {INNER_CORNERS[0] * INNER_CORNERS[1]})")
