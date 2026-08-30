# 第 5 章 拍攝場景 + 第 6 章 建立公制 3DGS

日期：2026-08-13
場景名稱：`scene01`
產出：`repos/SousVide/gsplats/workspace/scene01/`
      `outputs/scene01/splatfacto/2026-08-13_083816/`

## 第 5 章 場景影片

`captures/IMG_2121.MOV`（在 `gsplats/capture/scene01.MOV` 建了 symlink）

| 項目 | 值 | 與標定影片一致 |
|---|---|---|
| 鏡頭 | 廣角 26 mm f/1.6（1x） | ✓ |
| 解析度 | 1080 × 1920（容器 1920×1080 + rotation=-90） | ✓ |
| 幀率 | 29.98 fps | ✓ |
| 編碼 / 位元率 | H.264 / 15.6 Mbps 原生未壓縮 | ✓ |
| 長度 | 105.6 秒 / 3167 幀 | 手冊建議 2–3 分鐘，略短但足夠 |

### 預檢（`tools/preflight_capture.py`）

用與 `extract_frames()` 相同的判定邏輯掃過每一幀：

```
含 tag（剛好 1 個且 id=0）:  1148 幀   需要 >= 20  ✓
不含 tag                    :  2019 幀   需要 >= 280 ✓
2 個以上 marker（反射誤判）  :    11 幀
錯誤 id（地板紋理誤判）      :    22 幀
marker 邊長中位數 171 px（89~224），推估距離 1.43 m
```

⚠ **tag 不可全程在畫面中**。`extract_frames` 從「不含 tag」池取 280 張，
該池為空會 `IndexError`，不足 280 會 `TypeError`（`distribute_values` 把 None
塞進清單）。兩者皆已實測確認。

## 第 6 章 建圖：四個上游問題

分階段執行（`tools/build_gsplat.py`）而非直接呼叫 `generate_gsplat()`，
原因是後者用 `subprocess.run(capture_output=True)` 吞掉訓練輸出，
且無法在 SfM 與訓練之間插入檢查點。呼叫的是上游同一批函式。

### 1. hloc 的 third_party 子模組未初始化

**症狀**：`ModuleNotFoundError: No module named 'SuperGluePretrainedNetwork.models'`

**原因**：第 2 章為避開龐大的 acados，用
`git submodule update --init Hierarchical-Localization`（未遞迴），
漏了 hloc 自己的 `third_party/`。nerfstudio 的 hloc 預設是
`superpoint_aachen` + `superglue`，兩者都來自該子模組。

**處理**：
```bash
cd FiGS/Hierarchical-Localization
git submodule update --init third_party/SuperGluePretrainedNetwork
```
只補這一個。`d2net`/`r2d2` 是替代特徵器、`deep-image-retrieval` 只有 vocab_tree
檢索才需要，我們走 exhaustive 都用不到；hloc 用 `dynamic_load` 延遲載入。

### 2. pycolmap API 不相容

**症狀**：`TypeError: import_images(): incompatible function arguments`

**原因**：hloc（2024 年的 commit）用 `import_images(..., image_list=)`
與 dict 型別的 `incremental_mapping(options=...)`；
pycolmap **4.1.1** 已改名為 `image_names=`，options 也改成必須是
`IncrementalPipelineOptions` 型別。

⚠ 第 2 章的 `import pycolmap` 測試會通過——**import 成功不代表函式簽章相容**。

**處理**：`pip install --no-deps pycolmap==0.6.1`（hloc 當初對應的版本）。
比逐一修改上游多處呼叫乾淨。已確認 0.6.1 同時滿足 hloc 與 nerfstudio
（`ImageReaderOptions` / `CameraMode` / `verify_matches` / `triangulate_points` 皆在）。

### 3. hloc 搬錯重建模型（最陰險，不會報錯）

**症狀**：hloc 記錄 `Largest model is #1 with 300 images`、
`num_reg_images = 300`，但 nerfstudio 報
`COLMAP only found poses for 0.67% of the images`，`transforms.json` 只有 2 個 frame。

**原因**：`run_reconstruction` 第 102 行用 `reconstructions` 的**字典 key**
當資料夾名去搬（`models_path / str(largest_index)`），但 pycolmap 0.6.1 回傳的
字典 key 與它實際寫到磁碟的資料夾編號**對不起來**。磁碟上 `models/0` 才是
300 張的大模型，`models/1` 是 2 張的小模型。

**處理（兩件事）**：
- 立即修正：把 `models/0/*.bin` 複製回 `sparse/0/`，再直接呼叫
  `colmap_to_json(recon_dir=..., output_dir=..., image_rename_map=None,
  use_single_camera_mode=True)` 重新產生 transforms.json。
  （確認過 `images/` 與 `sfm/images/` 檔名集合完全相同 → rename_map 是 identity）
- 永久修正：patch `hloc/reconstruction.py`，改為**掃描磁碟上的模型目錄、
  用 `pycolmap.Reconstruction(d).num_reg_images()` 取最大者**，標記為 `PATCH(drone)`。
  ⚠ 這是改在 git submodule 內，`git checkout`/重新 clone 會被還原。

### 4. 兩個計數差異（非錯誤，但要理解）

- **抽出的 300 張裡含 tag 的是 22 張而非 20**：`extract_frames` 先循序讀取分箱
  並記錄毫秒時間戳，之後用 `cap.set(CAP_PROP_POS_MSEC)` 跳回去取幀。
  H.264 的毫秒 seek 不精確，會落在鄰近幀；你的影片 36% 的幀含 tag，
  於是 2 個原本歸類為「無 tag」的時間戳 seek 後落到含 tag 的鄰近幀。
- **check 階段數到 20 張**：因為它複製了 `extract_positions` 的讀圖路徑
  （`imread(f)` 再 `cvtColor(BGR2GRAY)`），與直接 `imread(f, IMREAD_GRAYSCALE)`
  的灰階轉換有微小差異，翻轉 2 張邊緣個案。**20 才是有效數字**，
  且恰好等於 `num_marked`，所以 `extract_positions` 通過。

## 結果

```
SfM 註冊率                300/300 = 100%
已註冊影像中含 tag        20 張（== num_marked ✓）
3D 點                     47194，觀測 305908，平均重投影誤差 1.48 px
Sim(3)                    cs = 0.288849（RANSAC 20 點中 17 inlier，門檻 5 cm）
訓練                      30000 步 / 16 分鐘 / 約 32 ms per step
高斯數量                  1,401,340
訓練解析度                540×960（images_2，nerfstudio 自動降尺度，MAX_AUTO_RESOLUTION=1600）
```

100% 註冊率代表拍攝品質好：重疊充足、無動態模糊、無覆蓋空洞。

### 尺度的初步跡象（第 7 章正式驗收）

稀疏點雲的完整 bbox 是 14.82 × 29.09 × 11.05 m，**但那被離群點主導**，
穩健統計才有意義：

```
5~95 百分位範圍 (m):  x 5.38   y 5.32   z 1.51
z 的 5/50/95 百分位 : -0.04 / 0.48 / 1.48
距 ArUco 水平 1.5 m 內的 9821 個點，z 中位數 = 0.0116 m
```

**tag 附近地板落在 z ≈ 1.2 cm、z 軸向上** → 原點在 ArUco 上、
世界 z 軸為重力反方向，符合飛行需求。

### ⚠ 待第 7 章裁決：內參的 2.68% 落差

| 內參 | fx | cs | Sim(3) 殘差中位數 |
|---|---|---|---|
| 我們的標定（第 3 章） | 1702.28 | 0.288444 | **1.36 cm** |
| COLMAP 自估 | 1664.67 | 0.280903 | 1.83 cm |

`solvePnP` 的距離正比於 fx，故 **2.68% 的 fx 差異 = 2.68% 的場景尺度差異**，
剛好踩在第 7 章 2% 標準的邊緣。

殘差**支持我們的標定**（1.36 < 1.83 cm，且我們的 cs 較大、距離放大 2.7%
的情況下殘差仍較小）。但 COLMAP 的畸變係數（k1=0.103, k2=-0.147）比我們的
（0.220, -0.716）溫和許多，與第 3 章「畸變解可能是病態的」警告一致。

**判讀方式**：第 7 章量已知物體長度。
- 相符 → 維持現狀
- 偏大約 2.7% → 把 `configs/captures/iphone12.json` 的 `camera` 設為 `null`
  （FiGS 會 fallback 到 SfM 內參），**只需重跑 `--stage scale`**。
  尺度只影響 transforms.json 與點雲，**不需重跑 SfM，也不需重新訓練 3DGS**。

另記：RANSAC 有隨機性，兩次執行的 cs 為 0.288849 / 0.288444，差 0.14%。

## 渲染驗證

`notes/ch6_render/` 三張，從訓練相機位姿渲染：
畫面清晰可辨——ArUco 紙板、木地板、拖鞋、機械手臂、桌椅、線材，
與實地照片一致。管線正確。
