# 實作步驟：從手機影片到可訓練的公制 3DGS 無人機環境

> 目標與約束見 `CLAUDE.md`。本文件是可執行的操作手冊。
> **每一章結尾都有驗證步驟，通過才進下一章。** 這條管線的失敗多半是無聲的。
>
> **路徑異動（2026-08-11）**：原本的工作目錄 `~/drone-env` 已與專案文件目錄合併，
> 統一為 **`~/drone`**。本文件中所有路徑已同步更新，技術內容未變動。
> 實作過程中相對本文件的偏離與上游錯誤，逐章記錄在 `notes/`。

---

## 0. 前置檢查

已知目標機器：Ubuntu、conda 已裝、NVIDIA 驅動已裝、**RTX 5070 Ti（Blackwell, sm_120, 16GB）**。

先確認驅動層符合 Blackwell 需求：

```bash
nvidia-smi
# 需要：Driver Version >= 570.xx，右上角 CUDA Version 顯示 >= 12.8
# 若驅動版本低於 570，先升級驅動再繼續，否則後面全部白做

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# 預期輸出類似：NVIDIA GeForce RTX 5070 Ti, 12.0, 16376 MiB
```

**`compute_cap` 必須是 `12.0`（即 sm_120）。** 這個數字決定了後面所有編譯選項。

```bash
# 建立工作目錄
mkdir -p ~/drone/{repos,refs,gsplats,captures,notes}
cd ~/drone
```

---

## 1. 環境安裝（Blackwell 專章）

### 1.1 為什麼不能用官方環境檔

SousVide 的 `environment_x86.yml` 鎖死 `pytorch=2.1.2` + `pytorch-cuda=11.8` + `cuda-11.8.0::cuda-toolkit`。**CUDA 11.8 最高支援 sm_90，不認得 sm_120。** 照做會裝出一個能安裝但執行時報 `no kernel image is available for execution on the device` 的環境。

**以下自建流程未在 Blackwell 上實測過**，是依架構需求推導的。逐步驗證，不要一次跑完。

### 1.2 建立環境

```bash
conda create -n droneenv python=3.10 -y
conda activate droneenv
```

Python 用 3.10：FiGS 要求 `>=3.10`，而 nerfstudio 生態對 3.11+ 的支援較不穩。

### 1.3 PyTorch（Blackwell 必須 cu128）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Blackwell 的正式支援從 PyTorch 2.7 開始。** 若上面裝到的版本低於 2.7，改用 nightly：

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**✅ 驗證（這一步不過就不要往下走）：**

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))   # 必須是 (12, 0)
print("arch list:", torch.cuda.get_arch_list())             # 必須包含 sm_120
# 真的在 GPU 上算一次——available=True 不代表 kernel 能跑
x = torch.randn(4096, 4096, device='cuda')
print("matmul ok:", (x @ x).sum().item() is not None)
EOF
```

`get_arch_list()` **沒有 `sm_120` 就是裝錯版本**，回頭換 nightly。最後那個 matmul 是關鍵——很多相容性問題只在實際跑 kernel 時才爆。

### 1.4 CUDA toolkit（編譯用）

驅動已有 runtime，但編譯 gsplat 需要 `nvcc`：

```bash
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit cuda-nvcc -y
nvcc --version   # 需 >= 12.8
```

若 conda 頻道沒有 12.8.0，改用系統套件或 `conda install -c nvidia cuda-nvcc` 取最新版，只要 `nvcc --version` >= 12.8 即可。

### 1.5 gsplat（要為 sm_120 編譯）

**預編譯 wheel 沒有 sm_120 的 kernel**，必須從源碼編譯並明確指定架構：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4          # 記憶體不足時降到 2
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

編譯需要數分鐘到十幾分鐘。

**✅ 驗證：**

```bash
python -c "import gsplat; print('gsplat', gsplat.__version__)"
```

### 1.6 nerfstudio

```bash
pip install nerfstudio
```

**不要安裝 tiny-cuda-nn。** 官方環境檔裡有它，但那是 `nerfacto` 的 hash grid 用的——**我們只用 `splatfacto`，完全不需要**。它是整個安裝流程裡最容易在新架構上編譯失敗的套件，跳過它省下大量時間。

### 1.7 SfM 與其他依賴

```bash
# COLMAP（hloc 的底層，也是 fallback）
conda install -c conda-forge colmap -y

# 影像處理與點雲
pip install "opencv-contrib-python>=4.10" open3d gdown imageio[ffmpeg]

# grad_nav 需要
pip install gym transformers
```

⚠️ **必須是 `opencv-contrib-python`**，不是 `opencv-python`——ArUco 模組在 contrib 裡。若已裝過 `opencv-python` 要先移除，兩者會衝突。

**✅ 驗證：**

```bash
python - <<'EOF'
import cv2, open3d, gym, transformers
print("cv2:", cv2.__version__)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)   # ArUco 可用
print("aruco ok, open3d:", open3d.__version__)
EOF
colmap -h | head -3
ns-train --help | head -3
```

### 1.8 本章不需要做的事

- ❌ **acados 不用編譯**。它是 SousVide 的 MPC 專家求解器，我們走 RL 不走 IL
- ❌ **dflex 不用安裝**。grad_nav 的 `envs/dflex_env.py` 裡 `import dflex as df` 是**被註解掉的**，實際動力學用它自寫的 `QuadrotorSimulator`
- ❌ **tiny-cuda-nn 不用安裝**（見 1.6）

---

## 2. 取得程式碼與論文

**論文與程式碼全部由本文件的指令下載，不需要事先準備。**

### 2.1 論文

```bash
cd ~/drone/refs

curl -L -o 2024-12-20-sousvide.pdf      https://arxiv.org/pdf/2412.16346
curl -L -o 2025-03-06-grad-nav.pdf      https://arxiv.org/pdf/2503.03984
curl -L -o 2025-06-16-grad-nav-pp.pdf   https://arxiv.org/pdf/2506.14009

# 驗證都不是空檔或錯誤頁（每個應為數 MB）
ls -lh *.pdf
```

| 論文 | arXiv | 讀什麼 |
|---|---|---|
| **SOUS VIDE** | [2412.16346](https://arxiv.org/abs/2412.16346) | 第 III-A 節 FiGS：場景重建與 ArUco 定尺度 |
| **GRaD-Nav** | [2503.03984](https://arxiv.org/abs/2503.03984) | 可微分 RL、CENet、reward 設計。**讀 ++ 前先讀這篇** |
| **GRaD-Nav++** | [2506.14009](https://arxiv.org/abs/2506.14009) | 訓練主體：CLIP 條件、MoE 動作頭 |

### 2.2 程式碼

```bash
cd ~/drone/repos

# 建圖：SousVide + FiGS（FiGS 是 submodule）
git clone https://github.com/StanfordMSL/SousVide.git
cd SousVide
git submodule update --recursive --init
cd ..

# 訓練：GRaD-Nav / GRaD-Nav++（同一個 repo）
git clone https://github.com/Qianzhong-Chen/grad_nav.git
```

`git submodule` 會拉下 `FiGS/`，其中含 `acados/` 與 `Hierarchical-Localization/` 兩個子模組。**acados 不用編譯**（見 1.8），但 hloc 要裝：

```bash
cd ~/drone/repos/SousVide
pip install -e ./FiGS/Hierarchical-Localization/
pip install -e ./FiGS/
```

⚠️ **不要執行 `pip install -e .`（SousVide 本體）**，它的依賴會拉進整套 IL 訓練管線與 acados。我們只需要 FiGS 的 render 模組。

**✅ 驗證：**

```bash
python -c "from figs.render.capture_generation import generate_gsplat; print('FiGS render ok')"
```

若這行報錯提到 acados，表示 FiGS 的 `__init__` 有連帶 import，此時改為直接呼叫模組檔而非透過 package（見附錄 A）。

### 2.3 先跑通官方範例（不要跳過）

```bash
cd ~/drone/repos/SousVide
# 下載他們的 example gsplats，解壓到 gsplats/
gdown 1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP
unzip <下載到的檔名>.zip -d gsplats/
```

然後跑 `notebooks/figs_examples.ipynb`。

**這一步的意義：先確認「環境 + 管線」本身沒問題。** 如果直接換自己的場景才發現渲染不出來，你無法判斷是環境裝壞了還是場景建壞了。

**✅ 驗證：** notebook 能渲染出第一人稱影像。

---

## 3. 相機內參標定

手機拍攝**必須自己標定**。FiGS 附的 `iphone15pro.json` 只是參考，不同機型甚至同機型不同錄影模式的內參都不同，而 ArUco 定尺度用的 `solvePnP` 精度**直接取決於內參準不準**。

### 3.1 準備棋盤格

印一張 9×6 內角點的棋盤格（即 10×7 個方格），**貼在硬板上保持完全平整**。量出實際方格邊長（mm）。

### 3.2 錄標定影片

用**與拍攝場景完全相同的錄影設定**（解析度、幀率、鏡頭、對焦模式）錄一段 30-60 秒：

- 棋盤格在畫面中**各個位置**都出現過（中心、四角、邊緣）
- 各種角度傾斜（不要只有正對）
- 各種距離
- ⚠️ **關閉自動對焦**，否則焦距會變，內參就不是常數

### 3.3 執行標定

```python
from figs.render.capture_calibration import camera_calibration

camera_calibration(
    calibration_file_name="calib.mov",     # 放在 gsplats/capture/
    camera_name="myphone",                 # 輸出 configs/camera/myphone.json
    checkerboard_size=(9, 6),              # 內角點數 (rows, cols)
    square_size=25.0,                      # 你量到的方格邊長 (mm)
    max_images=100,
)
```

**✅ 驗證：** 輸出會印 `Mean Reprojection Error`，**應小於 0.5 px**。大於 1.0 px 表示標定不可靠，重錄影片。

### 3.4 組成 capture config

把標定結果與 ArUco 參數合併成 `configs/captures/myphone.json`：

```json
{
    "camera": {
        "model": "OPENCV",
        "height": 1920,
        "width": 1080,
        "intrinsics_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "distortion_coefficients": [k1, k2, p1, p2]
    },
    "extractor": {
        "num_images": 300,
        "num_marked": 20,
        "marker_length": 0.300,
        "marker_id": 0
    }
}
```

⚠️ `height`/`width` 要對應**實際錄影方向**。原作是直式（1920×1080，height 在前）。橫式拍就對調。

---

## 4. 準備 ArUco tag

### 4.1 生成

字典必須是 **`DICT_4X4_50`**、ID **0**（FiGS 寫死）：

```python
import cv2, numpy as np
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
img = cv2.aruco.generateImageMarker(d, 0, 2000)     # id=0, 2000px
img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=255)  # 白邊必要
cv2.imwrite("aruco_id0.png", img)
```

**白邊不可省略**——偵測器靠黑白邊界定位，沒有白邊會偵測不到。

### 4.2 列印與量測

- 印越大越好。原作用 **34.1 cm**，建議至少 **25 cm**（A3 可印到約 28 cm）
- **印完拿尺量實際黑色方塊的邊長**，填進 `marker_length`（單位公尺）
- ❌ 不要填設計值——印表機縮放常有 1-3% 誤差，而**這個數字就是整個場景的尺度基準**

### 4.3 擺放

- 平放地面（推薦）或平貼牆面
- **必須完全平整**，皺摺會讓 `solvePnP` 的姿態解歪掉
- 平放地面時，重建出的世界座標系 z 軸就是重力反方向——這對飛行是必要的
- ⚠️ 場地內**不可有第二張 ArUco tag**，且注意鏡面/玻璃反射造成的重複偵測。FiGS 只接受每幀**剛好偵測到一個** marker 的畫面（`len(ids) == 1`）

---

## 5. 拍攝場景

```bash
# 影片放到
~/drone/repos/SousVide/gsplats/capture/myscene.mov
```

### 拍攝規範

| 項目 | 要求 | 為什麼 |
|---|---|---|
| 長度 | **2-3 分鐘** | 原作規格，會抽出 300 幀 |
| 開場 | **先對著 ArUco tag 拍一段**，從**不同角度與距離** | 需要 ≥20 幀含 tag。單一視角的方形標記有姿態歧義，角度要分散 |
| 走查 | 繞著場景走環路，相鄰畫面重疊 ≥60% | SfM 的基本要求；純原地旋轉解不出位移 |
| 速度 | 緩慢平穩 | 動態模糊會嚴重傷害重建品質 |
| 曝光/白平衡 | **鎖定** | 自動曝光會讓同一表面在不同幀顏色不一致 |
| 對焦 | **鎖定**（與標定時相同） | 焦距變動會讓內參失效 |
| 光線 | 均勻、充足 | 低光是這條路線的已知弱點 |
| 覆蓋 | 無人機**會飛到的每個區域**都要從多角度拍到 | 沒拍到的地方會長出 floater，變成點雲雜訊 |

### 要飛的關鍵物件

如果要做過閘門這類任務，**閘門/障礙物在拍攝時就必須在場景裡**，並且要繞著它多拍幾圈。策略學到的是這個場景的視覺結構——SousVide 實測，把訓練時在場的物件搬走，成功率從 96% 掉到 25%。

---

## 6. 建立公制 3DGS

```python
from figs.render.capture_generation import generate_gsplat

generate_gsplat(
    scene_file_name="myscene",
    capture_cfg_name="myphone",     # 第 3.4 節建立的 config
)
```

### 這一步內部做了什麼

1. **分堆抽幀**：掃過每一幀跑 ArUco 偵測，從有 tag 的取 20 張、其餘取 280 張
2. **SfM**：`ImagesToNerfstudioDataset(sfm_tool="hloc", matching_method="exhaustive")`
3. **PnP**：對每張含 tag 的影像跑 `cv2.solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)`，得到相機在 **ArUco 座標系（公制）** 的位置
4. **解 Sim(3)**：`compute_ransac_transform(Psfm, Parc)` → `cs`(scale), `Rs`(rotation), `ts`(translation)
5. **套用變換**到所有相機位姿與稀疏點雲：
   ```python
   Tc2w[:3,:3] = Rs @ Tc2s[:3,:3]
   Tc2w[:3,3]  = cs * Rs @ Tc2s[:3,3] + ts
   ```
6. **訓練**：`ns-train splatfacto` 帶 `--orientation-method none --center-method none --auto-scale-poses False`

第 6 步那三個旗標是**保住公制尺度的關鍵**——nerfstudio 預設會把場景正規化進 ±1 的 bounding box，那會毀掉前五步的成果。

### 常見失敗

**`ValueError: Error: Mismatched number of aruco and sfm transforms.`**

`extract_positions()` 硬性要求含 tag 的影像數**剛好等於** `num_marked`。抽幀時保證選了 20 張，但 SfM 可能 register 失敗其中幾張。處理方式：

1. 先確認影片裡含 tag 的幀夠多且清晰
2. 仍失敗就把 `num_marked` 調低（如 15）重跑
3. 或放寬該檢查為 `if len(TTarc) < 5:`（至少要 5 點才解得出 Sim(3)，越多越穩）

---

## 7. 驗證尺度（最重要的一步）

**尺度錯了不會報錯，只會讓後面全部表現差。** 一定要當場驗證。

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud(
    "gsplats/workspace/myscene/sparse_pc.ply")
print(pcd.get_axis_aligned_bounding_box())   # 尺寸應該符合實際房間大小（公尺）
o3d.visualization.draw_geometries([pcd])
```

在 open3d 視窗中按 **Shift + 左鍵**選點可量距離。

**驗收標準：**

| 檢查 | 標準 |
|---|---|
| 房間 bounding box | 與實際尺寸相符（例如 8 m × 6 m × 3 m） |
| 已知物體長度 | 量門框寬度、桌子長度，**誤差 < 2%** |
| z 軸方向 | 地面應該是水平的（tag 平放地面時） |
| 天花板高度 | 符合實際 |

誤差超過 2%，回頭檢查：`marker_length` 填的是實際量測值嗎？tag 是否平整？標定的重投影誤差是否夠小？

---

## 8. 導出點雲（給 reward 與 A\* 用）

```bash
cd ~/drone/repos/SousVide/gsplats/workspace
ns-export pointcloud \
  --load-config outputs/myscene/splatfacto/<timestamp>/config.yml \
  --output-dir ./exports/ \
  --num-points 60000 \
  --remove-outliers True \
  --normal-method open3d
```

⚠️ **`--num-points` 一定要降下來。** 預設 1,000,000 會讓 grad_nav 爆顯存——它的 `ObstacleDistanceCalculator` 每步都要對所有點做 `[B, N, 3]` 的張量運算，128 環境 × 100 萬點會產生數 GB 的中間張量。**建議 3-6 萬點**（16 GB VRAM 更要保守）。

### 為什麼導出的點雲會乾淨

`ns-export pointcloud` 不是導出高斯中心，而是**從訓練相機視角渲染 depth 再反投影**，並用 `alpha > 0.5` 過濾。空氣區域累積不透明度上不來，直接被濾掉——所以閘門中間天然是空的。它產生的是**表面點雲**。

反面代價：玻璃、紗網、細鐵絲這類半透明或亞像素結構**該有點卻沒有**，規劃時要注意。

**✅ 驗證：**

```bash
python ~/drone/repos/grad_nav/tools/validate_scene.py \
  --pcd exports/point_cloud.ply \
  --start <x> <y> <z> --dest <x> <y> <z> --radius 0.3
```

起訖點必須都在自由空間，**所有 FAIL 都要修正**。座標從第 7 步的視覺化裡讀取。

---

## 9. 接到 grad_nav

這是本專案的主要工程工作——**沒有人接過這兩套**。

### 9.1 放檔案

```bash
cd ~/drone/repos/grad_nav
mkdir -p envs/assets/gs_data envs/assets/point_cloud

# 3DGS checkpoint 目錄（含 config.yml）
cp -r ~/drone/repos/SousVide/gsplats/workspace/outputs/myscene \
      envs/assets/gs_data/myscene

# 點雲
cp ~/drone/repos/SousVide/gsplats/workspace/exports/point_cloud.ply \
   envs/assets/point_cloud/myscene.ply
```

### 9.2 註冊場景（**兩個地方都要改**）

`utils/gs_local.py` 的 `get_gs()`：

```python
maps = {
    "gate_left": "sv_917_3_left_nerfstudio",
    ...
    "myscene": "myscene",          # ← 加這行
}
```

`envs/drone_vla_long_task.py`（GRaD-Nav++ 用的 env）裡另有一份 `maps` 字典，同樣要加。

### 9.3 換相機內參

`utils/gs_local.py` 的 `generate_output_camera()` 寫死了 RealSense D435：

```python
fx, fy = 462.956, 463.002
cx, cy = 323.076, 181.184
```

**改成你飛行相機的標定值**（注意：不是建圖用的手機，是機載相機）。同時檢查 `GS.__init__` 的 `width=640, height=360` 與 `resolution_quality`。

### 9.4 換機體-相機外參

`utils/gs_local.py` 的 `pose2nerf_transform()` 有三個常數矩陣：

| 矩陣 | 意義 | 要不要改 |
|---|---|---|
| `T_c2r` | 光學系 → 機體系的軸重排 | 通常不用改 |
| `T_r2d` | **相機在機體上的安裝位姿** | **必須改** |
| `T_f2n` | ROS → nerfstudio 座標系翻轉 | 不用改 |

原作的 `T_r2d` 是前 15.2 cm、側 3.1 cm、下 1.2 cm、下傾 8.05°。

**量測方法**：從飛控/IMU 中心量到相機光心的三軸距離，鏡頭傾角用水平儀量。想更準就做 hand-eye calibration。

參考 SousVide 的 `configs/frames/carl.json` 的 `camera_to_body_transform` 格式（那是同一台機的另一種寫法）。

### 9.5 座標原點偏移

`envs/drone_vla_long_task.py`：

```python
self.gs_origin_offset = torch.tensor([[-6.0, 0., 0.]] * self.num_envs, ...)
```

這是「訓練座標原點」與「3DGS 場景原點」的偏移。你的場景原點在 ArUco tag 上，**依照你希望的起飛點設定這個值**。

---

## 10. 設定自己的任務

### 10.1 修改 task_table

`envs/drone_vla_long_task.py` 的 `task_table` 是**指令字串 → (waypoints, A\* 參考軌跡)** 的對照表：

```python
self.task_table = {
    "[Task 0] Go straight through the gate then STOP over MONITOR": [
        torch.tensor([[x1,y1,z1], [x2,y2,z2], ...]),   # reward waypoints
    ],
    ...
}
```

換成自己的任務時：

- **waypoint 座標**在點雲 frame 裡量（第 7 步的視覺化），再減去 `gs_origin_offset` 轉到訓練 frame
- 指令字串會餵給 CLIP，**用英文、句式與原作接近**（policy 學的是向量鄰域，不是語意）
- 每個任務的參考軌跡由 `traj_planner_vla` 用 A\* 自動規劃，不用手寫

### 10.2 場景數量的限制

GRaD-Nav++ 有兩種訓練模式：

| 模式 | 需要幾個場景 | 說明 |
|---|---|---|
| `drone_vla_long_task` | **1 個** | 單場景多任務。**先做這個** |
| `drone_vla_multi_map` | **2 個** | 跨環境輪替訓練，MoE 抗遺忘的實驗需要 |

**只建一個場景就只能跑 long_task。** 想複現 MoE 抗遺忘的效果，要拍兩個不同的場景（不同閘門位置與干擾物）。

### 10.3 依 16 GB VRAM 調參

`examples/cfg/gradnav_vla_moe/drone_long_task.yaml`：

```yaml
num_actors: 128        # ← 16GB 可能不夠，先降到 32 或 64
steps_num: 32          # 短視界長度，先別動
max_epochs: 600
```

`envs/drone_vla_long_task.py`：

```python
resolution_quality = 0.4      # 渲染解析度倍率，OOM 時降到 0.25
```

顯存壓力主要來自三處：3DGS 渲染的 batch、點雲距離計算的 `[B, N, 3]` 張量、CLIP 推論。**先用 `num_actors: 16` 跑通再逐步加大。**

---

## 11. 訓練

```bash
cd ~/drone/repos/grad_nav
python examples/train_gradnav_vla_moe.py \
  --cfg examples/cfg/gradnav_vla_moe/drone_long_task.yaml \
  --logdir examples/logs/DroneVLALongTaskEnv/gradnav_MoE
```

首次執行會從 HuggingFace 下載 CLIP（`openai/clip-vit-base-patch16`，注意實際程式碼用的是 **patch16**，論文寫 patch32）。

### 測試

```bash
python examples/train_gradnav_vla_moe.py \
  --cfg examples/cfg/gradnav_vla_moe/drone_long_task.yaml \
  --checkpoint examples/logs/.../final_policy.pt --play --render
```

### 訓練中要看什麼

| 訊號 | 健康的樣子 |
|---|---|
| reward 曲線 | 穩定上升，不發散 |
| 渲染畫面 | `--render` 存下的影像應該像真實場景，不是糊的或全黑 |
| 每步耗時 | 3DGS 渲染應佔約 55%，點雲檢查約 11% |
| 顯存 | 穩定，不持續增長 |

**渲染畫面不對就立刻停下**——那表示座標轉換錯了，再訓練也是白費。

---

## 12. 真機部署（PX4 + Jetson）

### 12.1 部署前的必要檢查

| 檢查 | 方法 |
|---|---|
| 相機內參一致 | 模擬用的內參 = 機載相機實際標定值 |
| 影像前處理一致 | 解析度、裁切、長寬比、正規化，**模擬與真機必須完全相同** |
| 外參正確 | 手動飛行時，比對機載畫面與同位姿的 3DGS 渲染圖 |
| 控制頻率 | policy 輸出頻率與飛控期望一致（原作 25 Hz） |
| 動作定義 | body rate 的軸向與正負號、推力的正規化範圍 |

⚠️ **影像前處理一致性最常被忽略。** grad_nav 用 `AdaptiveAvgPool2d((224,224))` 把 256×144 拉成方形——**長寬比是被壓扁的**。真機端必須做完全相同的變形，否則 policy 看到的畫面分佈就不同了。

### 12.2 分階段上機

1. **地面測試**：無人機固定在地面，跑 policy，只看輸出是否合理（不解鎖馬達）
2. **手動飛 + 記錄**：手動飛一趟，錄下機載影像與位姿，離線比對 3DGS 渲染圖——**這是驗證外參最有效的方法**
3. **繫繩測試**：低空、有防護、隨時可切手動
4. **完整飛行**

### 12.3 參考資源

Stanford MSL 的 TrajBridge（PX4 ↔ 機載電腦的橋接）：
<https://github.com/StanfordMSL/TrajBridge>

其硬體說明頁列有與論文相同的機體配置，可作為選型參考。

### 12.4 預期表現

論文的真機成功率是 **6-7/10**（GRaD-Nav）與 **67%**（GRaD-Nav++）。**失敗多半是撞上去**——模擬中沒有碰撞終止，無人機可以穿牆，避障只是 0.5 m 內的軟性距離獎勵。第一次飛請預期會撞。

---

## 附錄 A：疑難排解

| 症狀 | 原因與處理 |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch/gsplat 沒有 sm_120 kernel。回第 1.3、1.5 節重裝 |
| `import figs` 連帶 import acados 失敗 | 直接複製 `FiGS/src/figs/render/capture_generation.py` 與 `utilities/capture_helper.py` 出來單獨使用，它們只依賴 nerfstudio/cv2/open3d |
| `cv2.aruco` 不存在 | 裝成 `opencv-python` 了。移除後改裝 `opencv-contrib-python` |
| `Mismatched number of aruco and sfm transforms` | 見第 6 節「常見失敗」 |
| A\* 回傳 `None` / `Could not infer dtype of NoneType` | 起訖點在障礙物內。用 `tools/validate_scene.py` 重選 |
| 場景尺寸明顯不對 | `marker_length` 填錯，或 nerfstudio 的三個正規化旗標沒關 |
| 訓練 OOM | 依序降低 `num_actors` → `resolution_quality` → 點雲 `--num-points` |
| 渲染畫面全黑或錯位 | 座標轉換錯。檢查 `gs_origin_offset` 與 `pose2nerf_transform` |
| 點雲有空中雜訊團 | 重建品質不足（floater）。`remove_statistical_outlier` 殺不掉成團的 floater，要重拍該區域 |

## 附錄 B：關鍵參數速查

| 參數 | 位置 | 原作值 | 說明 |
|---|---|---|---|
| `marker_length` | `configs/captures/*.json` | 0.341 | **ArUco 實際邊長（公尺），尺度基準** |
| `marker_id` / 字典 | FiGS 寫死 | 0 / `DICT_4X4_50` | 不可改 |
| `num_images` / `num_marked` | `configs/captures/*.json` | 300 / 20 | 總幀數 / 含 tag 幀數 |
| `fx, fy, cx, cy` | `utils/gs_local.py` | 462.956, 463.002, 323.076, 181.184 | D435，**要換** |
| `T_r2d` | `utils/gs_local.py` | 前 0.152 m、下傾 8.05° | 安裝外參，**要換** |
| `gs_origin_offset` | `envs/drone_vla_*.py` | [-6, 0, 0] | 訓練原點偏移 |
| `obst_threshold` | `envs/drone_vla_*.py` | 0.5 | 避障 reward 觸發距離（公尺） |
| `obst_collision_limit` | `envs/drone_vla_*.py` | 0.20 | 碰撞判定距離。⚠️ `collision_penalty_coef = 0.0`，**預設是關閉的** |
| `num_actors` | cfg yaml | 128 | 並行環境數，16 GB 要降 |
| `resolution_quality` | `envs/drone_vla_*.py` | 0.4 | 渲染解析度倍率 |
| `steps_num` | cfg yaml | 32 | SHAC 短視界長度 |

## 附錄 C：本文件的把握程度

誠實標示，避免在錯誤的地方浪費時間：

| 內容 | 依據 |
|---|---|
| FiGS 的 ArUco 流程、參數、Sim(3) 求解 | **已讀原始碼確認**（`capture_generation.py`） |
| nerfstudio 三個正規化旗標 | **已讀原始碼確認**（FiGS 呼叫 `ns-train` 的參數） |
| grad_nav 的硬編碼位置、reward、termination | **已讀原始碼確認** |
| `collision_penalty_coef = 0.0` | **已讀原始碼確認**（兩支 VLA env 皆是） |
| 官方環境檔鎖 CUDA 11.8 | **已讀 `environment_x86.yml` 確認** |
| **Blackwell 安裝流程（第 1 章）** | ⚠️ **未實測**，依架構需求推導。逐步驗證 |
| **grad_nav 與 FiGS 場景的接合（第 9 章）** | ⚠️ **推導**，兩者格式相容但無人實際接過 |
| 16 GB VRAM 的建議參數 | ⚠️ **估計值**，需實測調整 |
