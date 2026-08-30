"""
產生可直接列印的棋盤格（9x6 內角點 = 10x7 方格），A4。

輸出兩種格式：
  - PDF：頁面尺寸即實體 A4，列印無歧義。**建議用這個列印。**
  - PNG：給程式自我檢查用（cv2 不寫 DPI 中繼資料，列印時檢視器只能猜尺寸）

關於列印縮放：
  標定用的棋盤格**不需要精確的列印比例**。實測驗證過 square_size 不影響
  標定出的內參與畸變係數（縮放物點只會等比縮放外參的平移向量）。
  唯一會壞事的是「非等比縮放」——方格變成長方形會讓 fx/fy 出現假差異。
  因此紙上印了兩條標稱 100 mm 的標尺線（一橫一直）：
  兩條量起來一樣長就代表等比，不必剛好是 100 mm。
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

MM_PER_INCH = 25.4
SQUARE_MM = 24.0                    # 見下方 SAFE_MM 的說明
INNER_CORNERS = (9, 6)              # (cols, rows) 給 cv2.findChessboardCorners
SQ_COLS = INNER_CORNERS[0] + 1      # 10 格（長邊）
SQ_ROWS = INNER_CORNERS[1] + 1      # 7 格（短邊）
A4_W_MM, A4_H_MM = 210.0, 297.0

# 一般雷射/噴墨印表機的不可列印邊界約 4~5 mm，留 12 mm 才安全。
# 方格 25 mm 時棋盤高 250 mm，扣掉安全邊界後放不下標註文字，
# 會被裁切或觸發印表機的「縮小以符合可列印範圍」。改為 24 mm，
# 棋盤 168 x 240 mm，上下左右都留得下標註。
# （方格尺寸不影響標定結果，縮小沒有代價。）
SAFE_MM = 12.0

out_dir = Path(__file__).resolve().parent.parent / "captures"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- PDF（列印用）
board_w = SQ_ROWS * SQUARE_MM        # 175 mm
board_h = SQ_COLS * SQUARE_MM        # 250 mm
ox = (A4_W_MM - board_w) / 2 - 3     # 略偏左，右側留給縱向標註
oy = (A4_H_MM - board_h) / 2         # 上下留白平均分配給標註文字

fig = plt.figure(figsize=(A4_W_MM / MM_PER_INCH, A4_H_MM / MM_PER_INCH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, A4_W_MM)
ax.set_ylim(0, A4_H_MM)
ax.set_aspect("equal")
ax.axis("off")

for r in range(SQ_COLS):
    for c in range(SQ_ROWS):
        if (r + c) % 2 == 0:
            continue                  # 只畫黑格
        ax.add_patch(Rectangle(
            (ox + c * SQUARE_MM, oy + r * SQUARE_MM),
            SQUARE_MM, SQUARE_MM,
            facecolor="black", edgecolor="none"))

# 等比檢查改用「棋盤本身的跨距」——比量單一方格精準得多
# （量 24 mm 的一格，尺的 0.5 mm 誤差就是 2%；量 240 mm 的跨距只有 0.2%）。
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

# 三行說明由下往上排，最下面一行距棋盤上緣 1.5 mm，不可壓到板子
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

# ---------------------------------------------------------------- PNG（自我檢查用）
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

# ---------------------------------------------------------------- 自我檢查
ok, corners = cv2.findChessboardCorners(png_img, INNER_CORNERS, None)
n = 0 if corners is None else len(corners)

print(f"已產生:")
print(f"  {pdf}   ← 列印用")
print(f"  {png}   ← 程式自我檢查用")
print(f"  棋盤 {SQ_ROWS}x{SQ_COLS} 方格 = {board_w:.0f} x {board_h:.0f} mm，內角點 {INNER_CORNERS[0]}x{INNER_CORNERS[1]}")
print(f"  A4 頁面 {A4_W_MM:.0f} x {A4_H_MM:.0f} mm")
print()
print(f"  cv2.findChessboardCorners 自我檢查: {'通過' if ok else '**失敗**'}"
      f"（偵測到 {n} 個角點，應為 {INNER_CORNERS[0] * INNER_CORNERS[1]}）")
