# 第 0 章 前置檢查紀錄

日期：2026-08-10

## 驅動層（implement.md 第 0 章要求）

| 檢查項 | 要求 | 實測 | 結果 |
|---|---|---|---|
| Driver Version | >= 570.xx | 580.126.09 | ✅ |
| nvidia-smi CUDA Version | >= 12.8 | 13.0 | ✅ |
| compute_cap | 12.0 (sm_120) | 12.0 | ✅ |
| GPU / VRAM | RTX 5070 Ti / 16 GB | RTX 5070 Ti / 16303 MiB | ✅ |

桌面環境（Xorg + gnome-shell + 瀏覽器）常駐佔用約 661 MiB，
實際可用約 15.6 GB。第 10.3 節調 `num_actors` 時要把這扣掉。

## 額外環境盤點

- OS: Ubuntu 24.04.4 LTS (noble)
- conda 25.7.0 @ /home/jeremy/anaconda3
- 磁碟：/ 剩餘 524 GB，充裕
- gcc/g++ 13.3.0 — CUDA 12.8 可接受的版本
- **系統已有 CUDA toolkit 12.8.93 @ /usr/local/cuda**（nvcc 可用）
- ffmpeg **未安裝** → 第 5、6 章處理影片前要補

## 對第 1 章的影響（重要）

附錄 C 標示第 1 章「未實測」，但這台機器已有既存環境
`gaussian_splatting_128` 實測可用：

```
torch: 2.9.0+cu128
cuda build: 12.8
arch list: ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
capability: (12, 0)
實際 GPU matmul 成功
```

→ 結論：**torch 2.9.0+cu128 在本機的 sm_120 上確認能實際執行 kernel**，
不需要走 nightly。第 1.3 節的最大不確定性已排除。

→ 第 1.4 節（conda 裝 cuda-toolkit）可改用系統的 /usr/local/cuda 12.8，
省下約 3 GB 且避免 conda 與系統 nvcc 版本打架。

## 建立的目錄

~/drone/{repos,refs,gsplats,captures,notes}

## 路徑統一（2026-08-11，事後異動）

原先依 implement.md 第 0 章建立的工作目錄是 `~/drone-env`，與專案文件所在的
`~/drone` 分開，造成兩個根目錄。已合併為單一根目錄 **`~/drone`**。

搬移時必須連帶處理的事（下次若再搬要照做）：

1. `mv ~/drone-env/* ~/drone/` —— 同一檔案系統，是瞬間完成的中繼資料操作
2. **`hloc` 與 `figs` 是 editable 安裝，搬完會 `ModuleNotFoundError`**。
   必須重裝，且**一定要加 `--no-deps`**，否則會再次把
   `opencv-python` 與 `opencv-python-headless` 拉回來蓋掉 contrib 版：
   ```bash
   pip install -e ./FiGS/Hierarchical-Localization/ --no-deps --force-reinstall
   pip install -e ./FiGS/ --no-deps --force-reinstall
   ```
3. `tools/calibrate_camera.py` 的絕對路徑改成由 `Path(__file__)` 推導的
   `PROJECT_ROOT`，之後再搬就不用改程式
4. `implement.md` 內 13 處 `~/drone-env` 已改為 `~/drone`（只動路徑字串，
   技術內容未變），檔案開頭有異動說明
5. nerfstudio 的 patch 註解標籤由 `PATCH(drone-env)` 改為 `PATCH(drone)`

搬移後回歸驗證全過：torch/gsplat/cv2(aruco)/colmap/ffmpeg 正常、
hloc 與 figs 指向新路徑、第 2 章渲染測試重跑成功、標定腳本預設路徑正確。
