# 第 1 章 環境安裝紀錄

日期：2026-08-11
環境名稱：`droneenv`（另有輔助 env `sfmtools`）

## 最終落地版本

| 套件 | 版本 | 備註 |
|---|---|---|
| Python | 3.10.20 | |
| torch / torchvision | **2.11.0+cu128** / 0.26.0+cu128 | arch list 含 sm_120 |
| gsplat | **1.4.0（自行源碼編譯，AOT）** | 24 個 cubin 全為 sm_120 |
| nerfstudio | 1.1.5 | |
| numpy | 1.26.4 | 被 nerfstudio 降版，torch 2.11 相容 |
| opencv-contrib-python | 4.10.0.84 | 取代 nerfstudio 的 headless 版 |
| open3d | 0.19.0 | |
| gym | 0.26.2 | |
| transformers | **5.15.0** | ⚠️ 見「待觀察」 |
| COLMAP | 4.0.4 (CUDA) | 在 `sfmtools` env |
| ffmpeg | 8.1.2 | 在 `sfmtools` env |
| nvcc | 系統 12.8.93 @ /usr/local/cuda | 與 torch cu128 同版 |

## 相對 implement.md 的偏離（連同理由）

### 1. 不裝 conda 的 cuda-toolkit（原 1.4 節）
系統已有 CUDA 12.8.93，且 `torch.utils.cpp_extension.CUDA_HOME` 自動指向
`/usr/local/cuda`。nvcc 版本與 torch 的 cu128 build **完全一致**，
沒有 version skew。省下約 3 GB，也避免兩套 nvcc 打架。

### 2. MAX_JOBS 4 → 12
本機 94 GB RAM / 24 核，手冊的 4 是為小記憶體機器寫的。

### 3. gsplat 編 1.4.0 而非最新版
nerfstudio 1.1.5 的依賴是 `gsplat ==1.4.0`（精確鎖定）。
若裝最新版會留下版本衝突。做法是**照它的版本號、但自己編譯**：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"; export MAX_JOBS=12; export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation --force-reinstall --no-deps \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
```

⚠️ **順序**：gsplat 必須在 nerfstudio **之後**裝，否則會被 nerfstudio 的
PyPI wheel 覆蓋。

⚠️ PyPI 上的 `gsplat-1.4.0` 是 `py3-none-any` 的**純 JIT wheel**（無 .so）。
它不會立刻報 `no kernel image`，而是在首次呼叫時才用本機 nvcc 現編——
風險是訓練跑到一半停頓，且編譯失敗就整場報銷。AOT 預編可消除此風險。

### 4. opencv：移除 headless 換 contrib
nerfstudio 鎖 `opencv-python-headless ==4.10.0.84`，但它**沒有 aruco**。
改裝 `opencv-contrib-python==4.10.0.84`（版本對齊，功能是超集）。
pip 會留下一個未滿足的版本宣告警告，屬預期。

### 5. colmap / ffmpeg 放獨立 env `sfmtools`
sudo 需要密碼，apt 路線會卡住。conda 直接裝進 droneenv 會拉進
qt-main 5.15 + pango + nss + 整套 xorg，Qt 衝突會威脅第 7 章的
open3d 視覺化。做法：獨立 env + 把其 `bin` **附加到 PATH 尾端**
（尾端是刻意的，droneenv 的 python 必須優先）。
conda 的執行檔用 RPATH `$ORIGIN/../lib` 找相依，所以那套 Qt 不會滲進來。

## 額外處理的兩個坑（手冊未提及）

### A. ROS 2 Jazzy 的 PYTHONPATH 洩漏
系統全域設 `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`，
會插在 sys.path **第一順位**。那是 python3.12 的目錄，droneenv 是 3.10。
目前掃過無直接撞名，但每裝一個新套件風險就多一分。

處理：`droneenv/etc/conda/activate.d/zz_isolate_ros.sh` 只在本 env 內
清除 PYTHONPATH，deactivate 還原。**全域 ROS 不受影響**（第 12 章要用）。

### B. conda-forge colmap 套件相依宣告不完整
`colmap 4.1.1` 與 `4.0.4` 都**沒有宣告 `faiss` 相依**，裝完執行會死在
`undefined symbol: faiss::IndexIVFFlat::IndexIVFFlat(Index*, ulong, ulong, MetricType)`。
新版 faiss（1.12+）改了該建構子簽章。

處理：釘 `libfaiss 1.10.*` + `colmap 4.0.*`，寫在
`sfmtools/conda-meta/pinned`，避免日後 conda 操作又升級弄壞。

## 驗證結果（全數通過）

```
A. 環境衛生
  sys.path 無 ROS 洩漏
  python: droneenv/bin/python   colmap: sfmtools/bin/colmap   ffmpeg: sfmtools/bin/ffmpeg
B. PyTorch / Blackwell
  torch 2.11.0+cu128 | capability (12,0) | sm_120 in arch list: True | matmul ok
C. gsplat
  1.4.0 | 編譯架構 {'sm_120'} | 前向 render max 0.9347 | 反向 grad norm 16541.80
D. 依賴
  cv2 4.10.0 (aruco ok) | open3d 0.19.0 | numpy 1.26.4 | gym 0.26.2 | transformers 5.15.0
  COLMAP 4.0.4 | ffmpeg 8.1.2
E. nerfstudio
  check_ffmpeg_installed + check_colmap_installed 通過
  ns-train / ns-process-data / ns-export / ns-viewer 皆可用，splatfacto 存在
```

比手冊多驗的項目（手冊只要求 `import gsplat`，不足）：
- 用 `cuobjdump --list-elf` 確認 `.so` 內真的是 sm_120 機器碼
- 實跑 rasterization **前向 + 反向**。GRaD-Nav 走可微分 RL，
  梯度必須穿得過渲染器，只驗前向不夠
- ArUco 做生成→偵測往返，並確認 `SOLVEPNP_IPPE_SQUARE` 存在
- 直接呼叫 nerfstudio 的 `install_checks`，那才是第 6 章真正的門檻

## 給第 6 章的預先發現

`ColmapConverterToNerfstudioDataset.__post_init__` **無條件**呼叫
`check_ffmpeg_installed()` 與 `check_colmap_installed()`，失敗即 `sys.exit(1)`。
**即使 `sfm_tool="hloc"` 也會擋**（hloc 內部用 pycolmap，不用執行檔）。
這就是為什麼 colmap/ffmpeg 執行檔非裝不可。

保尺度的三個旗標屬於 **`nerfstudio-data` dataparser**，不是 method，
必須寫在 `nerfstudio-data` 之後。實測預設值：

| 旗標 | 預設 | 需設為 |
|---|---|---|
| `--orientation-method` | `up` | `none` |
| `--center-method` | `poses` | `none` |
| `--auto-scale-poses` | `True` | `False` |

三個預設值**正好都會摧毀公制尺度**。

## 待觀察

- **transformers 5.15.0**：CLIP 類別（CLIPModel/CLIPProcessor/…）都還在，
  不會 import 失敗。但 5.x 是主版本跳躍，grad_nav 是對 4.x 寫的，
  processor 預設值與回傳型別可能有差。**第 9/11 章實跑 CLIP 時要驗**，
  必要時降到 4.x。
- gym 0.26.2 會印「不支援 NumPy 2.0」的警告。目前 numpy 是 1.26.4，無影響。
