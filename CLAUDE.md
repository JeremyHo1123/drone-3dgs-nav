# 客製化無人機視覺導航訓練環境

## 專案目標

建立一套**自己場景的無人機視覺導航訓練環境**，分工如下：

- **環境建立**走 **SousVide / FiGS** 的路線：用手機錄影 + ArUco tag 重建出**公制尺度**的 3D Gaussian Splatting 場景
- **訓練**走 **GRaD-Nav++**：可微分 RL（differentiable RL），無人類示範，語言條件的 VLA policy
- **最終目標是真機部署**（PX4 系飛控 + Jetson 機載電腦）

這兩套來自同一個實驗室（Stanford Mac Schwager 的 Multi-Robot Systems Lab），是同一條技術線的不同世代，場景格式與機體設定可以互通——**但沒有人把兩者接起來過，這個接合點是本專案的主要工程工作**。

## 為什麼是這個組合

| | 用誰 | 理由 |
|---|---|---|
| 場景重建 | SousVide 的 **FiGS** | 它把「手機影片 → 公制 3DGS」整條做成開源，含 ArUco 定尺度。GRaD-Nav 系列自己沒有這段（它直接吃 SousVide 產出的場景） |
| 訓練 | **GRaD-Nav++** | 不需要 MPC 專家與示範資料合成，訓練 3.5 小時（SousVide 的 IL 要 12 小時），且支援語言指令 |

**明確不採用的部分**：SousVide 的 MPC 專家、資料合成、SV-Net 訓練管線全部不用。因此 **acados 不需要編譯**——它只是 SousVide 做 IL 資料時的求解器。這省掉整個安裝流程裡最痛的一段。

## 硬體與環境約束

| 項目 | 規格 | 影響 |
|---|---|---|
| GPU | **RTX 5070 Ti（Blackwell, sm_120, 16GB）** | ⚠️ **本專案最大的技術風險**，見下 |
| OS | Ubuntu | conda 與 NVIDIA 驅動已安裝完成 |
| 待裝 | 新 conda 環境內的 CUDA 工具鏈 | 驅動層不用動 |
| 建圖相機 | 手機 | 需自行做棋盤格內參標定 |
| 飛行平台 | PX4 系（PixRacer/Pixhawk）+ Jetson | 與論文架構相同，可沿用其部署流程 |

### ⚠️ Blackwell 相容性是首要風險

**SousVide 官方的 `environment_x86.yml` 不能直接使用。** 它鎖死在：

```yaml
- pytorch=2.1.2
- pytorch-cuda=11.8
- nvidia/label/cuda-11.8.0::cuda-toolkit
```

**CUDA 11.8 最高只支援 sm_90（Hopper），完全不認得 sm_120。** 直接 `conda env create -f environment_x86.yml` 會裝出一個能安裝但跑不動的環境，典型症狀是執行時報 `no kernel image is available for execution on the device`。

必須自建環境：CUDA 12.8+、PyTorch 2.7+（cu128）、gsplat 從源碼編譯。細節見 `implement.md` 第 1 章。

**這一點沒有在目標機器上實測過**，是根據 Blackwell 的架構需求推導的。第 1 章附有逐步驗證指令，每一步都要確認通過再往下走。

## 分階段目標與驗收標準

實作順序不可跳，每階段有明確的**可驗證**產出：

| 階段 | 產出 | 驗收標準 |
|---|---|---|
| **1. 環境** | 可用的 conda env | `torch.cuda.is_available()` 為 True 且能在 GPU 上做矩陣運算；gsplat 能 import 且跑通一次渲染 |
| **2. 跑通官方範例** | 用他們的 example gsplat 渲染出圖 | **先確認管線本身沒問題，再換自己的場景**。跳過這步會讓後面的除錯無法定位 |
| **3. 相機標定** | `configs/captures/<你的手機>.json` | 重投影誤差 < 0.5 px |
| **4. 建圖** | 公制 3DGS 場景 + `transforms.json` | **量場景中已知長度的物體，誤差 < 2%** |
| **5. 點雲** | `.ply` 給 reward 與 A\* 用 | 閘門/通道中間無雜訊點；`validate_scene.py` 起訖點檢查全過 |
| **6. 接上 grad_nav** | 能在自己場景裡跑起訓練 | 訓練曲線不發散，能渲染出正確的第一人稱畫面 |
| **7. 部署** | 真機飛行 | 分階段：先手動飛驗證外參，再開 policy |

## 給 AI 助手的工作規範

- **對話與文件一律使用繁體中文**；專有名詞（3DGS、ArUco、splatfacto、VLA 等）保留英文
- **尺度正確性是本專案的生命線**。3DGS 場景若不是公制，動力學、避障閾值（0.5 m）、reward 全部連帶錯誤，而且**不會報錯，只會表現差**。任何涉及尺度的步驟都要當場驗證，不要留到後面
- **不要跳過驗證步驟**。這條管線的失敗多半是無聲的
- 遇到版本衝突時，**優先保 Blackwell 相容性**（CUDA 12.8+ / PyTorch cu128），再去遷就上游 repo 的版本鎖。上游的版本鎖是 2024 年寫的，早於 Blackwell 發布
- 修改上游 repo 的程式碼時，**記錄改了哪裡與為什麼**，因為 grad_nav 有多處硬編碼是為他們自己那台無人機（代號 `carl`）寫的
- 需要查證上游行為時**直接讀原始碼**，不要臆測。兩個 repo 都不大

## 已知的硬編碼陷阱

grad_nav 有數處寫死了原作者的硬體與場景，換成自己的環境時**必定要改**：

1. `utils/gs_local.py` 的相機內參 `fx=462.956, fy=463.002, cx=323.076, cy=181.184`（640×360）——這是 RealSense D435
2. `utils/gs_local.py` 的 `pose2nerf_transform()` 內三個常數矩陣，其中 `T_r2d` 是相機在機體上的**安裝外參**（前 15.2 cm、下傾 8°）
3. `utils/gs_local.py` 與 `envs/*.py` 各有一份 `maps` 字典，**兩邊都要加**自己的場景
4. `envs/drone_long_traj.py` 的 `gs_origin_offset = [-6.0, 0, 0]`
5. `envs/drone_vla_*.py` 的 `task_table`——指令字串與對應 waypoint，要換成自己的任務

## 重要的預期管理

以下是這條技術路線的**固有限制**，不是實作沒做好：

- **策略是逐場景特化的**。它的 zero-shot 只跨「重建 vs 實拍」的渲染落差，**不跨場景**。換一個房間就要重新拍攝、重新訓練
- **對靜態場景變化脆弱**。SousVide 實測：把訓練時就在場景裡的物件搬走，策略會穩定地飛過那些物件本該在的位置（成功率從 96% 掉到 25%）；反而是有人在場景中走動幾乎不影響
- **模擬中沒有碰撞終止**，無人機可以穿牆。避障只靠 0.5 m 內的距離獎勵，是軟約束。真機會撞，論文的真機成功率是 6-7/10
- **語言指令不是 open-vocabulary**。GRaD-Nav++ 的語彙是 4 個方向 × 3 個目標的 12 種組合，policy 只在訓練過的指令附近有定義
- **低光失效**。亮度降到原始 40% 以下，SousVide 的策略一律從起點就飄離航線

## 參考資料

三篇論文與程式碼的下載指令寫在 `implement.md` 第 2 章，執行後會落到 `refs/` 與 `repos/`。閱讀優先順序：

1. **SousVide**（arXiv 2412.16346）— 建圖方法的來源，第 III-A 節是 FiGS
2. **GRaD-Nav++**（arXiv 2506.14009, RA-L）— 訓練方法
3. **GRaD-Nav**（arXiv 2503.03984）— 前作，可微分 RL 與 CENet 的出處，讀 ++ 前先讀它
