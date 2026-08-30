# 第 4 章 ArUco tag

日期：2026-08-13
產出：`captures/aruco_id0_A4_157mm.pdf`、`captures/aruco_id0_A3_222mm.pdf`
      `repos/SousVide/configs/captures/iphone12.json`

## FiGS 的硬性要求（已讀原始碼確認）

`capture_generation.py`：

| 項目 | 值 | 可否更改 |
|---|---|---|
| 字典 | `DICT_4X4_50`（第 166、234 行） | **寫死，不可改** |
| `marker_id` | 來自 `extractor_config["marker_id"]` | **可改**（手冊說寫死是不準確的） |
| `marker_length` | 來自 config，用於 `marker_points = ±L/2` | 必填 |
| 每幀 marker 數 | `len(ids) == 1`（第 262 行） | 硬性，多一個該幀就作廢 |
| PnP | `cv2.SOLVEPNP_IPPE_SQUARE` | — |

`cv2.aruco` 回傳的是黑色方塊**外緣**角點 →
**`marker_length` = 最外圈黑邊的外緣到外緣，不含白色留白。**

## 尺寸上限的推導

DICT_4X4 的圖案是 6x6 模組（4x4 資料 + 1 模組黑邊）。
偵測器需要黑框四周有白色留白，慣例至少 1 個模組寬：

```
(頁寬 - S)/2 >= S/6   →   S <= 頁寬 * 3/4
A4 (210mm) -> 157mm      A3 (297mm) -> 222mm
```

原作用 34.1 cm、手冊建議 >= 25 cm，**單張紙做不到**。
⚠ 不要用多張 A4 拼貼——接縫錯位與不平整會直接汙染 solvePnP 的姿態解。
要更大請送影印店印 A2 以上（`make_aruco.py --page 420 594`）。

產生後的自我驗證（渲染 PDF 再實跑偵測）：
兩種尺寸都是「剛好 1 個 marker、id=0、量得邊長與標稱差 0.01%」。

## 實測結果

**列印後實測黑框邊長：長寬皆 14.4 cm → `marker_length = 0.144`**

144 / 157 = **91.7%**。與第 3 章棋盤格的 22/24 = 91.7% **完全一致**。
兩份不同檔案、不同標稱尺寸卻是同一縮放比，可推論：

1. 印表機套用「縮小以符合可列印範圍」，固定縮到約 91.7%
2. **且為等比縮放**——這正是最需要排除的風險（非等比會讓 fx/fy 出現假差異）

長寬相等亦佐證等比。紙上白邊變多是整頁等比縮小的結果，
對偵測反而有利（白邊即 quiet zone）。

## 尺度誤差預算（marker_length = 0.144）

| 來源 | 貢獻 |
|---|---|
| marker_length 量測 ±0.5 mm | 0.35% |
| 第 3 章 fx 不確定性 | 0.30% |
| PnP 姿態噪聲（1 m、245 px、20 幀 RANSAC） | 0.046% |
| **合計** | **0.46%** |

第 7 章驗收標準 < 2%，餘裕 1.54 個百分點。

⚠ A4 的真正代價不是解析度，是**量測精度要求變嚴格**：
量測誤差 1:1 轉成尺度誤差，而分母是 marker 邊長。
同樣量到 ±1 mm，144 mm 的誤差（0.69%）是原作 341 mm（0.29%）的 2.4 倍。

marker 在畫面上的跨距（fx=1702）：
0.5 m → 490 px、1.0 m → 245 px、1.5 m → 163 px、2.0 m → 123 px。
ArUco 姿態解在 100 px 以上相當可靠 → **第 5 章拍 tag 時距離 0.5~1.5 m**。

## capture config

`repos/SousVide/configs/captures/iphone12.json`

```
camera    : 1080x1920 (WxH)  fx=1702.28 fy=1708.64 cx=544.82 cy=949.09  畸變 4 係數
extractor : num_images=300 num_marked=20 marker_id=0 marker_length=0.144
```

驗證方式：完全照抄 `extract_positions()` 的程式碼路徑載入此 config，
以已知距離投影 marker 角點再 `solvePnP(..., SOLVEPNP_IPPE_SQUARE)` 解回距離。
0.5 / 1.0 / 1.5 / 2.0 m，各含 0° 與 25° 傾角，**誤差皆 0.0000%**
→ config 格式、陣列形狀、marker_length 慣例都正確。
（此驗證只涵蓋數值管線，不含實拍的角點定位噪聲。）

## 擺放要求（第 5 章拍攝時）

- **平放地面**：重建出的世界座標 z 軸即重力反方向，飛行必要
- **完全平整**，貼硬板或用膠帶壓平；皺摺會讓姿態解歪掉
- 場地內**不可有第二張 ArUco**
- ⚠ 注意**鏡子、玻璃、光亮地板的反射**——反射的 tag 會被偵測成第二個 marker，
  觸發 `len(ids) == 1` 失敗，該幀作廢。這點最容易忽略
