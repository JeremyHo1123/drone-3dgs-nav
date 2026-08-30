# 客製化無人機視覺導航訓練環境

用手機錄影重建**公制尺度**的 3D Gaussian Splatting 場景，餵給可微分 RL 訓練無人機視覺導航策略，最終部署到真機。

專案的完整背景、技術選型理由、預期管理與已知限制寫在 [`CLAUDE.md`](CLAUDE.md)。**開始動手之前先讀那一份**，這份 README 只講「怎麼把環境裝起來、怎麼跑」。

---

## 這個 repo 裡有什麼、沒有什麼

| | 內容 |
|---|---|
| **有** | 自己寫的 11 個腳本（`tools/`）、逐章實作筆記（`notes/*.md`）、相機標定與拍攝設定（`configs/`）、專案說明（`CLAUDE.md`、`implement.md`） |
| **沒有** | 上游 repo（SousVide、grad_nav）、拍攝影片、抽出的影像、SfM 中間產物、訓練好的 checkpoint、點雲 |

沒有的那些合計約 49 GB，都能重新取得或重新產生，所以不放進 git。下面每一節會說明怎麼補齊。

---

## 0. 前置需求

| 項目 | 需求 | 說明 |
|---|---|---|
| GPU | NVIDIA，**建議 Blackwell（sm_120）** | 本專案在 RTX 5070 Ti 與 RTX PRO 6000 上驗證過 |
| CUDA toolkit | **12.8 以上**，裝在系統層（`/usr/local/cuda`） | 用 `nvcc --version` 確認 |
| conda | 任一版本（Anaconda / Miniconda 皆可） | |
| OS | Ubuntu 24.04 | 其他版本未測 |

### ⚠️ 為什麼 CUDA 版本這麼要緊

SousVide 官方的 `environment_x86.yml` 鎖死 `pytorch-cuda=11.8`。**CUDA 11.8 最高只支援 sm_90（Hopper），完全不認得 sm_120（Blackwell）。**

直接用官方的 yml 會裝出一個「能安裝但跑不動」的環境。症狀是執行到一半才報：

```
no kernel image is available for execution on the device
```

所以下面的安裝步驟**不使用官方 yml**，全部手動指定版本。

---

## 1. 安裝環境

總共要建**兩個** conda 環境。分開的理由寫在第 1.3 節。

### 1.1 主環境 `droneenv`

```bash
conda create -n droneenv python=3.10 -y
conda activate droneenv
```

**接下來的安裝順序不能調換**，理由寫在每一步下面。

**第一步：PyTorch（cu128 版）**

```bash
pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

**第二步：nerfstudio**

```bash
pip install nerfstudio==1.1.5
```

這一步會把 numpy 降到 `1.26.4`。**這是預期行為，不要去升回來** —— nerfstudio 1.1.5 不相容 numpy 2.x，而 torch 2.11 和 1.26.4 相處得很好。

**第三步：gsplat（必須從源碼編譯，且必須在 nerfstudio 之後）**

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=12          # 依機器核心數調整，記憶體小的機器改成 4
export CUDA_HOME=/usr/local/cuda

pip install --no-build-isolation --force-reinstall --no-deps \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"
```

三個要注意的地方：

1. **順序**：gsplat 必須裝在 nerfstudio **之後**。反過來的話，nerfstudio 的相依會把你編好的版本用 PyPI wheel 蓋掉
2. **版本號是 1.4.0，不是最新版**：nerfstudio 1.1.5 精確鎖定 `gsplat ==1.4.0`。裝最新版會留下版本衝突
3. **為什麼要自己編**：PyPI 上的 `gsplat-1.4.0` 是 `py3-none-any` 的純 JIT wheel，裡面沒有預編好的 `.so`。它不會馬上報錯，而是等到第一次呼叫渲染時才用本機 nvcc 現場編譯 —— 風險是訓練跑到一半停頓，編譯失敗就整場報銷。自己編是 AOT（ahead-of-time，事先編好），可以完全避開這個風險

**第四步：換掉 opencv**

```bash
pip uninstall -y opencv-python-headless
pip install opencv-contrib-python==4.10.0.84
```

nerfstudio 鎖定 `opencv-python-headless==4.10.0.84`，但那個版本**沒有 ArUco 模組**，而本專案的公制定尺度完全靠 ArUco。`opencv-contrib-python` 是同版本的功能超集。

pip 之後會留下一行「未滿足的版本宣告」警告，**那是預期的，可以忽略**。

**第五步：其他相依**

```bash
pip install open3d==0.19.0 gym==0.26.2
```

### 1.2 隔離 ROS 2 的 PYTHONPATH（如果機器上有裝 ROS 2）

如果系統全域設了 `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`，它會插在 `sys.path` 的**第一順位**。那是 python 3.12 的目錄，而 `droneenv` 是 3.10 —— ABI 不同，任何撞名的套件都會蓋掉環境內的正版。

處理方式是只在這個 env 內清掉它：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/zz_isolate_ros.sh << 'EOF'
export _DRONEENV_SAVED_PYTHONPATH="${PYTHONPATH:-}"
unset PYTHONPATH
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/zz_isolate_ros.sh << 'EOF'
export PYTHONPATH="${_DRONEENV_SAVED_PYTHONPATH:-}"
unset _DRONEENV_SAVED_PYTHONPATH
EOF
```

`activate.d` 裡的腳本會在 `conda activate` 時自動執行，`deactivate.d` 則在離開時還原。**全域的 ROS 2 不受影響。**

### 1.3 輔助環境 `sfmtools`（COLMAP + ffmpeg）

```bash
conda create -n sfmtools -y
conda activate sfmtools
conda install -c conda-forge "colmap=4.0.*" "libfaiss=1.10.*" ffmpeg -y

# 釘住版本，避免日後 conda 操作把它們升級弄壞
printf 'libfaiss 1.10.*\ncolmap 4.0.*\n' > $CONDA_PREFIX/conda-meta/pinned
```

**為什麼要獨立一個 env：** 直接把 colmap 裝進 `droneenv` 會拉進 qt-main 5.15 + pango + nss + 整套 xorg。那組 Qt 會和 open3d 的視覺化打架。

**為什麼要釘 libfaiss 1.10：** conda-forge 的 colmap 套件**沒有宣告 faiss 相依**。裝完直接執行會死在：

```
undefined symbol: faiss::IndexIVFFlat::IndexIVFFlat(Index*, ulong, ulong, MetricType)
```

新版 faiss（1.12 以上）改了那個建構子的簽章。釘在 1.10 就沒事。

**把 colmap 與 ffmpeg 接到 droneenv：**

```bash
conda activate droneenv

cat > $CONDA_PREFIX/etc/conda/activate.d/zz_sfmtools_path.sh << 'EOF'
export _DRONEENV_SAVED_PATH="$PATH"
export PATH="$PATH:$(conda info --base)/envs/sfmtools/bin"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/zz_sfmtools_path.sh << 'EOF'
export PATH="${_DRONEENV_SAVED_PATH:-$PATH}"
unset _DRONEENV_SAVED_PATH
EOF
```

**`$PATH` 後面而不是前面是刻意的。** `droneenv` 自己的 python 必須優先，不能被 `sfmtools` 的 python 蓋掉。conda-forge 的執行檔用 RPATH `$ORIGIN/../lib` 找自己的相依，所以那套 Qt 不會滲進來。

**為什麼非裝這兩個執行檔不可：** nerfstudio 的 `ColmapConverterToNerfstudioDataset.__post_init__` **無條件**呼叫 `check_ffmpeg_installed()` 與 `check_colmap_installed()`，失敗就 `sys.exit(1)`。即使我們用 `sfm_tool="hloc"`（hloc 內部用 pycolmap，根本不需要執行檔）也照樣被擋。

---

## 2. 取得上游 repo

```bash
cd <這個 repo 的根目錄>
mkdir -p repos && cd repos

git clone --recursive https://github.com/StanfordMSL/SousVide.git
git clone https://github.com/Qianzhong-Chen/grad_nav.git
```

`--recursive` 不能省，FiGS 是 SousVide 的 submodule。

### ⚠️ 不需要編譯 acados

SousVide 的安裝說明會叫你編 acados。**本專案不需要。** acados 只是它做 imitation learning 資料合成時的 MPC 求解器，而我們走 GRaD-Nav++ 的可微分 RL 路線，完全不用那條管線。這省掉整個安裝流程裡最痛的一段。

### 把自己的設定檔放回上游 repo 的位置

`tools/build_gsplat.py` 到 `repos/SousVide/configs/` 底下讀設定，所以要複製過去：

```bash
cd <這個 repo 的根目錄>
mkdir -p repos/SousVide/configs/captures repos/SousVide/configs/camera
cp configs/captures/*.json repos/SousVide/configs/captures/
cp configs/camera/*.json   repos/SousVide/configs/camera/
```

---

## 3. 驗證環境（不要跳過）

這條管線的失敗**多半是無聲的** —— 不報錯，只是結果變差。所以每一步都要當場驗。

```bash
conda activate droneenv
python - << 'EOF'
import torch, gsplat, cv2, open3d, numpy
print("torch      :", torch.__version__)
print("CUDA 可用  :", torch.cuda.is_available())
print("GPU 能力   :", torch.cuda.get_device_capability())   # Blackwell 要是 (12, 0)
print("arch list  :", torch.cuda.get_arch_list())           # 要含 'sm_120'
print("gsplat     :", gsplat.__version__)                   # 要是 1.4.0
print("cv2        :", cv2.__version__, "| aruco:", hasattr(cv2, "aruco"))
print("open3d     :", open3d.__version__)
print("numpy      :", numpy.__version__)                    # 要是 1.26.x
EOF
```

**只驗 `import gsplat` 是不夠的。** 本專案走可微分 RL，梯度必須穿得過渲染器，所以前向和反向都要實跑：

```bash
python - << 'EOF'
import torch, gsplat
# 確認 .so 內真的是 sm_120 機器碼，而不是等著 JIT 現編
import glob, subprocess, os
so = glob.glob(os.path.join(os.path.dirname(gsplat.__file__), "*.so"))
print("編出來的 .so:", so)
EOF
```

再確認 nerfstudio 的四個執行檔都在，且 colmap / ffmpeg 找得到：

```bash
ns-train --help  > /dev/null && echo "ns-train  ok"
ns-viewer --help > /dev/null && echo "ns-viewer ok"
ns-export --help > /dev/null && echo "ns-export ok"
colmap -h 2>&1 | head -1
ffmpeg -version | head -1
```

---

## 4. 完整使用流程

### 4.1 相機標定（每支手機做一次）

用手機錄一段**棋盤格**的影片，各種角度和距離都要有。

```bash
# 產生棋盤格 PDF，印出來貼在硬板上（不能有皺摺）
python tools/make_checkerboard.py

# 標定
python tools/calibrate_camera.py --video captures/<你的影片>.MOV --name <手機名稱>
```

**驗收標準：重投影誤差要小於 0.5 px。** 沒過就重錄，不要將就 —— 內參錯了，後面每一步都跟著錯。

輸出會落在 `configs/camera/<手機名稱>.json`。

### 4.2 產生 ArUco tag 與拍攝設定

ArUco tag 是場景裡的一張印出來的黑白方塊圖案，用來給場景定出**公制尺度**。它是整條管線裡唯一知道「1 公尺有多長」的東西。

```bash
# 產生 tag（A4 或 A3），印出來後量實際邊長
python tools/make_aruco.py --page 210 297

# 建立拍攝設定，marker-length 填「你量到的實際邊長，單位是公尺」
python tools/make_capture_config.py \
  --name <設定名稱> \
  --marker-length 0.144 \
  --num-images 300
```

⚠️ **`--marker-length` 一定要量印出來的實體，不要用檔名上的數字。** 印表機的縮放設定會讓實際尺寸和設計尺寸差幾個百分比，而這個誤差會**原封不動變成整個場景的尺度誤差**。

### 4.3 拍攝

手持手機繞著場景走，要求：

- ArUco tag 至少要在 20 張影像裡清楚可見（對應設定裡的 `num_marked`）
- 覆蓋要完整，避免只從一個方向拍
- 動作要慢，動態模糊會讓 SfM 註冊失敗

把影片放到：

```
repos/SousVide/gsplats/capture/<檔名裡要含場景名稱>.MOV
```

**檔名必須含場景名稱，且該名稱在 `capture/` 底下只能配到一個檔案**，否則腳本會直接停下來。

拍完先做預檢：

```bash
python tools/preflight_capture.py --video <影片路徑> --config <設定名稱>
```

### 4.4 建圖

分五個階段，**建議一階段一階段跑**，不要一次 `--stage all`：

```bash
cd <這個 repo 的根目錄>
conda activate droneenv

python tools/build_gsplat.py --scene <場景名> --config <設定名稱> --stage frames
python tools/build_gsplat.py --scene <場景名> --config <設定名稱> --stage sfm
python tools/build_gsplat.py --scene <場景名> --config <設定名稱> --stage check   # ← 檢查點
python tools/build_gsplat.py --scene <場景名> --config <設定名稱> --stage scale
python tools/build_gsplat.py --scene <場景名> --config <設定名稱> --stage train
```

| 階段 | 做什麼 |
|---|---|
| `frames` | 從影片抽幀 |
| `sfm` | 用 hloc 做 Structure-from-Motion，算出每張影像的相機位置 |
| `check` | **只檢查、不改任何東西**：確認含 tag 的影像有沒有全部被 SfM 註冊 |
| `scale` | 用 ArUco 定出公制尺度，寫出 `transforms.json` 與 `sparse_pc.ply` |
| `train` | 跑 `ns-train splatfacto`，輸出直接串流到畫面 |

**`check` 階段為什麼獨立出來：** 如果含 tag 的影像有幾張沒被 SfM 註冊，下游會拋 `Mismatched number of aruco and sfm transforms`，而那時 SfM 已經白跑一小時了。

`--select` 可以選抽幀策略：`uniform`（原作者作法，預設）或 `sharp`（在時間箱內挑最銳利的一張）。

### ⚠️ 三個保尺度旗標

`--stage train` 內部跑的是：

```bash
ns-train splatfacto ... nerfstudio-data \
  --orientation-method none \
  --center-method none \
  --auto-scale-poses False
```

這三個旗標的 nerfstudio 預設值分別是 `up` / `poses` / `True`，而**三個預設值正好都會摧毀公制尺度**。它們屬於 `nerfstudio-data` 這個 dataparser，必須寫在 `nerfstudio-data` 之後才有效。

`build_gsplat.py` 已經處理好了。手動跑 `ns-train` 時要自己記得加。

### 4.5 驗證尺度（這是生命線）

```bash
python tools/verify_scale.py --scene <場景名> --pick
```

`--pick` 會開一個互動視窗讓你點兩個點，量出它們的距離。

**驗收標準：量場景中已知長度的物體，誤差要小於 2%。**

尺度錯了不會報錯，只會讓後面的動力學、避障閾值（0.5 m）、reward 全部連帶錯誤，而且表現差得莫名其妙。**這一步不能跳。**

### 4.6 匯出點雲與評估

```bash
# 匯出稠密點雲，給 reward 計算與 A* 路徑規劃用
python tools/export_pointcloud.py \
  --config repos/SousVide/gsplats/workspace/outputs/<場景名>/splatfacto/<時間戳>/config.yml \
  --out repos/SousVide/gsplats/workspace/exports/<場景名>_dense.ply

# 渲染品質評估（PSNR + 離軌偏移測試）
python tools/eval_gsplat.py --config <同上的 config.yml> --tag <場景名>

# 畫場景俯視圖，確認閘門/通道中間沒有雜訊點
python tools/plot_scene_map.py --scene <場景名>
```

---

## 5. 場景在機器之間的搬移

這一節是**實測結果**，不是推測。測試方式：只複製下面列出的檔案到一個全新的空目錄，用 grad_nav 相同的 `eval_setup(config_path, test_mode="inference")` 載入，確認成功並讀出 1,401,340 個高斯點。

### 最小必要檔案：只有 4 個

以場景 `scene01`、訓練時間戳 `2026-08-13_083816` 為例：

```
workspace/                                   ← 執行指令時的工作目錄
├── scene01/
│   └── transforms.json                      ← 260 KB
└── outputs/scene01/splatfacto/2026-08-13_083816/
    ├── config.yml                           ← 12 KB
    ├── dataparser_transforms.json           ← 1 KB
    └── nerfstudio_models/
        └── step-000029999.ckpt              ← 956 MB
```

總共約 **956 MB**，幾乎全是那個 checkpoint。

### 不需要搬的東西

| 不用搬 | 原本大小 | 說明 |
|---|---|---|
| `scene01/images/` | 620 MB | **實測確認不需要**。載入時只讀 `transforms.json` 拿相機位姿，不開影像檔 |
| `scene01/sfm/` | 1.3 GB | SfM 中間產物，`transforms.json` 產出後就沒用了 |
| `scene01/sparse_pc.ply` | 1.3 MB | 沒有它只會印 open3d 的警告，載入照樣成功。但它很小，建議還是帶著 |

原本 1.9 GB 的場景，搬移只需要 956 MB。

### 三個會讓搬移失敗的陷阱

**1. 資料夾名稱必須和 config.yml 裡的時間戳一模一樣**

checkpoint 的路徑是從 `config.yml` 裡的 `timestamp` 欄位**重新組出來**的，不是相對於 config.yml 自己的位置。我測試時把資料夾改名成 `T`，結果就報：

```
No checkpoint directory found at outputs/scene01/splatfacto/2026-08-13_083816/nerfstudio_models
```

**不要改任何一層資料夾的名字。**

**2. 路徑是相對的，工作目錄必須正確**

`config.yml` 裡存的是相對路徑：

```yaml
data: scene01
output_dir: outputs
```

所以執行指令時的**工作目錄必須是 `scene01/` 和 `outputs/` 的共同上層**（也就是 `workspace/`）。在別的地方跑會找不到檔案。

**3. 兩台機器的 gsplat 版本要一致**

checkpoint 存的是高斯參數張量。載入端的 gsplat 若不是 1.4.0，張量結構對不上就會出錯。兩台都照第 1 節裝就沒問題。

### 搬移指令

在**接收端**這台跑（把 `<場景名>` 和 `<時間戳>` 換成實際值）：

```bash
W=repos/SousVide/gsplats/workspace
SCENE=scene01
TS=2026-08-13_083816

mkdir -p $W/$SCENE $W/outputs/$SCENE/splatfacto/$TS/nerfstudio_models

rsync -avP pro_6000:drone/$W/$SCENE/transforms.json      $W/$SCENE/
rsync -avP pro_6000:drone/$W/$SCENE/sparse_pc.ply        $W/$SCENE/
rsync -avP pro_6000:drone/$W/outputs/$SCENE/splatfacto/$TS/  \
           $W/outputs/$SCENE/splatfacto/$TS/
```

`rsync -avP` 的 `-P` 會顯示進度並支援中斷續傳，傳 956 MB 時很有用。

### 只是想「看看」場景長什麼樣

那更簡單，只要一個 `.ply` 檔（約 26 MB），不需要 checkpoint 也不需要 nerfstudio：

```bash
rsync -avP pro_6000:drone/repos/SousVide/gsplats/workspace/exports/<場景名>_dense.ply .
```

用任何 3DGS 檢視器打開就好。

---

## 6. 已知的硬編碼陷阱

`grad_nav` 有數處寫死了原作者的硬體與場景（他們的無人機代號是 `carl`）。換成自己的環境時**必定要改**：

| 檔案 | 硬編碼內容 | 要改成 |
|---|---|---|
| `utils/gs_local.py` | 相機內參 `fx=462.956, fy=463.002, cx=323.076, cy=181.184`（640×360） | 這是 RealSense D435 的值，換成你機上相機的內參 |
| `utils/gs_local.py` | `pose2nerf_transform()` 內的三個常數矩陣 | 其中 `T_r2d` 是相機在機體上的**安裝外參**（原作者是前 15.2 cm、下傾 8°） |
| `utils/gs_local.py` 與 `envs/*.py` | 各有一份 `maps` 字典 | **兩邊都要加**自己的場景名稱 |
| `envs/drone_long_traj.py` | `gs_origin_offset = [-6.0, 0, 0]` | 依自己場景的原點調整 |
| `envs/drone_vla_*.py` | `task_table` | 指令字串與對應 waypoint，換成自己的任務 |

修改上游程式碼時**要記錄改了哪裡與為什麼**。

---

## 7. 給 AI 助手的工作規範

- **對話與文件一律使用繁體中文**，專有名詞（3DGS、ArUco、splatfacto、VLA 等）保留英文
- **尺度正確性是本專案的生命線**。任何涉及尺度的步驟都要當場驗證，不要留到後面
- **不要跳過驗證步驟**，這條管線的失敗多半是無聲的
- 遇到版本衝突時，**優先保 Blackwell 相容性**（CUDA 12.8+ / PyTorch cu128），再去遷就上游 repo 的版本鎖。上游的版本鎖寫於 2024 年，早於 Blackwell 發布
- 需要查證上游行為時**直接讀原始碼**，不要臆測。兩個 repo 都不大

---

## 8. 逐章實作筆記

`notes/` 底下是實際做過的紀錄，含每一步的偏離與理由。遇到問題時比這份 README 更有參考價值：

| 檔案 | 內容 |
|---|---|
| [`notes/00-prechecks.md`](notes/00-prechecks.md) | 動手前的檢查 |
| [`notes/01-environment.md`](notes/01-environment.md) | 環境安裝的完整紀錄與所有偏離理由 |
| [`notes/02-code-and-example.md`](notes/02-code-and-example.md) | 跑通官方範例 |
| [`notes/03-calibration.md`](notes/03-calibration.md) | 相機標定 |
| [`notes/04-aruco.md`](notes/04-aruco.md) | ArUco 定尺度 |
| [`notes/05-06-capture-and-gsplat.md`](notes/05-06-capture-and-gsplat.md) | 拍攝與建圖 |
| [`notes/07-scene-experiments.md`](notes/07-scene-experiments.md) | 場景實驗比較 |

筆記裡引用的截圖沒有放進 repo（約 21 MB），留在原始機器上。
