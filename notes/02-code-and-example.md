# 第 2 章 取得程式碼與跑通官方範例

日期：2026-08-11

## 2.1 論文（已下載到 refs/）

| 檔案 | 大小 |
|---|---|
| 2024-12-20-sousvide.pdf | 9.5 MB |
| 2025-03-06-grad-nav.pdf | 3.1 MB, 8 頁 |
| 2025-06-16-grad-nav-pp.pdf | 5.6 MB, 7 頁 |

三個都是有效 PDF，不是錯誤頁。

## 2.2 程式碼

```
repos/SousVide/            (含 FiGS submodule)
repos/SousVide/FiGS/Hierarchical-Localization/   (hloc)
repos/SousVide/FiGS/acados/                      (空目錄，刻意不初始化)
repos/grad_nav/
```

**只初始化了 FiGS 與 hloc 兩個 submodule，沒有拉 acados。**
確認過 FiGS 的 `pyproject.toml` 不依賴 acados，acados 只被
`simulator.py`、`control/vehicle_rate_mpc.py`、`dynamics/quadcopter_rate_model.py`
三個檔案 import，全是我們不走的 MPC 路徑。

FiGS **沒有任何 `__init__.py`**（PEP 420 namespace package），
所以 `import figs.render.capture_generation` 不會觸發任何父層 import。
附錄 A 擔心的「import figs 連帶 acados 失敗」不會發生。

安裝的關鍵套件：`hloc 1.5`(editable)、`figs 0.1.0`(editable)、
`pycolmap 4.1.1`、`kornia 0.8.2`、`lightglue 0.0`。

### ⚠️ opencv 被弄壞了兩次

- hloc 的 requirements 要 `opencv-python`（實際裝到 **5.0.0.93**，
  它要求 numpy>=2，與我們的 numpy 1.26.4 不相容）
- FiGS 依賴 `albumentations`，後者要 `opencv-python-headless>=4.9.0.80`

**三個發行版都寫進同一個 `cv2/` 目錄**，檔案互相覆蓋。而且解除安裝其中
任何一個都會刪掉共用檔案、留下孤兒目錄。

處理方式（第 6、9 章若再裝東西要重做一次這個檢查）：

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
# 確認 site-packages 內 cv2/ 已清空
pip install "opencv-contrib-python==4.10.0.84"
```

清乾淨後確認：只剩一個 `cv2.abi3.so`、單一 dist-info、`aruco: True`。

### hloc 可用性（無聲失敗的來源）

nerfstudio 的 `hloc_utils.py` 把 hloc 的 import 包在 `try/except ImportError`，
失敗只是把 `_HAS_HLOC = False`，**不會報錯**。實測結果：

```
pycolmap: 4.1.1
hloc.extract_features / match_features / pairs_from_exhaustive
     / pairs_from_retrieval / reconstruction  皆 OK
_HAS_HLOC 會是 True → 第 6 章的 hloc 路徑可用
```

## 2.3 官方範例（手冊驗收方式需替換）

### 為什麼沒跑 notebook

`notebooks/figs_examples.ipynb` 跑不起來，有兩個獨立原因：

1. **cell 4 與 cell 12 走 `VehicleRateMPC` → import acados**，
   與第 1.8 節「acados 不用編譯」直接衝突。`figs/simulator.py` 也 import acados。
2. notebook 預設 `capture_name = "button"`，但下載的資料裡
   `gsplats/capture/` 只有 `backroom.MOV`，沒有 button。

### 改用的驗證方式

直接用 `figs.render.gsplat.GSplat`——那是本專案真正會用到的渲染路徑
（grad_nav 的 `utils/gs_local.py` 與它同構），只依賴 nerfstudio/torch/numpy。
腳本邏輯：載入官方 backroom checkpoint → 取訓練集實際相機位姿 → 渲染 → 存檔 → 目視確認。

⚠️ 座標細節：`render_rgb` 內部做 `Tc2g = Tw2g @ T_c2w`，
而 `Tw2g = diag(1,-1,-1,1)` 是自逆的。要用資料集位姿渲染必須餵
`T_c2w = Tw2g @ Tc2g_dataset`，否則畫面上下顛倒。

### 結果（通過）

```
高斯數量: 535,006      eval_dataset 相機數: 30
位姿 0/1/2 皆渲染出 (360,640,3) uint8 影像
亮度 mean 137.1 / 85.4 / 86.7，相異像素值 227 / 241 / 236
```

目視確認：`notes/ch2_render/backroom_cam0.png` 清楚可見**平放地面的
ArUco tag**（正是第 4.3 節要求的擺法）；`backroom_cam1.png` 是清晰的
實驗室場景，家具、櫃子、紙箱都可辨識。**管線本身沒問題。**

## 修改上游程式碼的紀錄

### nerfstudio 的 torch.load（三處）

**症狀**：`_pickle.UnpicklingError: Weights only load failed ...
Unsupported global: GLOBAL numpy.core.multiarray.scalar`

**原因**：PyTorch **2.6 起 `torch.load` 的 `weights_only` 預設由 False 改為 True**。
nerfstudio 1.1.5 寫於 2024 年，呼叫時沒帶這個參數，而 checkpoint 內含 numpy scalar。
本專案為了 Blackwell(sm_120) 必須用 torch>=2.7，兩者必然衝突。

**改了哪裡**（都加上 `weights_only=False` 與說明註解）：

| 檔案 | 行 | 影響的功能 |
|---|---|---|
| `nerfstudio/utils/eval_utils.py` | 62 | `eval_setup` → 渲染、`ns-export`（第 8 章）、`ns-viewer` |
| `nerfstudio/engine/trainer.py` | 432 | 從 `load_dir` 續訓 |
| `nerfstudio/engine/trainer.py` | 443 | 從指定 checkpoint 載入 |

（`scripts/downloads/download_data.py:517` 也有一處，用不到，未改。）

⚠️ **這是改在 site-packages 裡，重裝 nerfstudio 會被覆蓋。**
若日後 `pip install --force-reinstall nerfstudio`，要重做這三處。

⚠️ `weights_only=False` 會執行 checkpoint 內的 pickle，
只對自己產生或信任來源（此處為論文作者的官方資料）的檔案使用。

## 磁碟

`repos/SousVide/gsplats.zip` 4.4 GB，已解壓成 `gsplats/` 5.0 GB。
zip 可刪（尚未刪，留給你決定）。
