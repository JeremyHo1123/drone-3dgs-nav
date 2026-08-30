"""
第 6 章建圖：分階段執行 FiGS 的 generate_gsplat。

為什麼不直接呼叫 generate_gsplat()：
  1. 它第 143 行用 subprocess.run(..., capture_output=True) 跑 ns-train，
     訓練過程的輸出全被吞掉，跑一小時看不到任何進度，失敗也只在最後才知道
  2. 它是單一函式一路跑到底，沒有機會在 SfM 之後、訓練之前檢查註冊率。
     若含 tag 的影像有幾張沒被 SfM 註冊，下游 extract_positions 會拋
     "Mismatched number of aruco and sfm transforms"，那時 SfM 已白跑

本腳本呼叫的是上游同一批函式（extract_frames / extract_positions /
compute_ransac_transform），沒有重寫任何數學，只是拆開並加上檢查點。

階段:
  --stage frames   抽幀
  --stage sfm      hloc SfM
  --stage check    檢查註冊率（不改任何東西）
  --stage scale    ArUco 定尺度，寫出 transforms.json 與 sparse_pc.ply
  --stage train    ns-train splatfacto（輸出直接串流到 stdout）
  --stage all      依序全部
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOUSVIDE = PROJECT_ROOT / "repos/SousVide"
GSPLATS = SOUSVIDE / "gsplats"
CONFIGS = SOUSVIDE / "configs"
SELECT_MODE = "uniform"   # uniform = 原作者作法；sharp = 箱內取最銳利


def paths(scene):
    ws = GSPLATS / "workspace"
    proc = ws / scene
    return {
        "workspace": ws,
        "process": proc,
        "images": proc / "images",
        "spc": proc / "sparse_pc.ply",
        "tfm": proc / "transforms.json",
        "sfm": proc / "sfm",
        "sfm_spc": proc / "sfm" / "sparse_pc.ply",
        "sfm_tfm": proc / "sfm" / "transforms.json",
        "outputs": ws / "outputs",
    }



def _fps(values, k):
    """
    一維最遠點取樣，與上游 capture_helper.distribute_values 等價。

    上游是 O(n·k^2) 的純 Python 迴圈（3937 選 380 約需十幾分鐘）；
    這裡用 numpy 維護「到已選集合的最小距離」陣列，複雜度 O(n·k)，
    3937 選 380 只要 0.002 秒。已用 4 組含空洞的隨機測資驗證選出的值完全相同。
    """
    v = np.asarray(values, dtype=float)
    sel = [0]                                   # 與上游一致，從 values[0] 開始
    mind = np.abs(v - v[0])
    for _ in range(k - 1):
        j = int(np.argmax(mind))
        sel.append(j)
        np.minimum(mind, np.abs(v - v[j]), out=mind)
    return sorted(sel)


def extract_frames_custom(video_path, images_path, ext_cfg, mode="uniform"):
    """
    抽幀。分箱準則與上游 extract_frames 完全一致：
      含 tag = 剛好偵測到 1 個且 id 相符；其餘全歸「不含 tag」池。
      再從含 tag 池取 num_marked 張、不含 tag 池取 num_images - num_marked 張。

    mode="uniform"（預設，等同原作者作法）
        兩池各自做一維最遠點取樣，純粹依時間分散，不看畫面銳利度。
    mode="sharp"
        兩池各自做時間分箱、箱內取最銳利的一幀。

    與上游唯一的實作差異：上游先記毫秒時間戳，之後用
    cap.set(CAP_PROP_POS_MSEC) 跳回去取幀，而 H.264 的毫秒 seek 不精確、
    會落在鄰近幀（實測導致「應取 20 張含 tag、實際抽出 22 張」）。
    這裡改用幀索引 + 全程循序讀取，選出的幀與選取準則不變，只是取得方式不會偏移。
    """
    import cv2
    Nimg = ext_cfg["num_images"]
    Narc = ext_cfg["num_marked"]
    mkr_id = ext_cfg["marker_id"]

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟 {video_path}")
    tag_i, tag_s, emp_i, emp_s = [], [], [], []
    i = 0
    print(f"[frames] 第 1 遍：逐幀 ArUco 分箱（mode={mode}）")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sv = cv2.Laplacian(g, cv2.CV_64F).var() if mode == "sharp" else 0.0
        _, ids, _ = det.detectMarkers(g)
        if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
            tag_i.append(i); tag_s.append(sv)
        else:
            emp_i.append(i); emp_s.append(sv)
        i += 1
    cap.release()
    print(f"[frames]   總幀數 {i}，含 tag {len(tag_i)}，不含 tag {len(emp_i)}")

    if len(tag_i) < Narc:
        raise SystemExit(f"含 tag 的幀只有 {len(tag_i)}，少於 num_marked={Narc}")
    if len(emp_i) < Nimg - Narc:
        raise SystemExit(f"不含 tag 的幀只有 {len(emp_i)}，少於 {Nimg - Narc}")

    def choose(idxs, svals, k):
        idxs = np.asarray(idxs)
        if mode == "sharp":
            svals = np.asarray(svals)
            return [int(idxs[b[np.argmax(svals[b])]])
                    for b in np.array_split(np.arange(len(idxs)), k) if len(b)]
        return [int(idxs[j]) for j in _fps(idxs, k)]

    sel_tag = choose(tag_i, tag_s, Narc)
    sel_emp = choose(emp_i, emp_s, Nimg - Narc)
    sel = sorted(set(sel_tag) | set(sel_emp))
    print(f"[frames]   選出 含tag {len(sel_tag)} + 不含tag {len(sel_emp)} = {len(sel)} 張")
    if len(sel) != Nimg:
        print(f"[frames]   ⚠ 兩池有重疊，實際 {len(sel)} 張（應為 {Nimg}）")

    print("[frames] 第 2 遍：循序讀取寫出選中的幀（不 seek）")
    want = set(sel)
    cap = cv2.VideoCapture(str(video_path))
    i = 0; k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            k += 1
            cv2.imwrite(str(images_path / f"frame_{k:05d}.png"), f)
        i += 1
    cap.release()
    print(f"[frames]   寫出 {k} 張")


def stage_frames(scene, cfg, P):
    caps = list((GSPLATS / "capture").glob(f"*{scene}*"))
    if len(caps) != 1:
        raise SystemExit(f"capture/ 內配到 {len(caps)} 個檔案，必須剛好 1 個: {caps}")
    P["process"].mkdir(parents=True, exist_ok=True)
    P["images"].mkdir(parents=True, exist_ok=True)
    print(f"[frames] 來源 {caps[0].name}")
    extract_frames_custom(caps[0], P["images"], cfg["extractor"], mode=SELECT_MODE)
    n = len(list(P["images"].glob("*.png")))
    print(f"[frames] 完成，抽出 {n} 張 → {P['images']}")
    if n != cfg["extractor"]["num_images"]:
        print(f"[frames] ⚠ 與 num_images={cfg['extractor']['num_images']} 不符")


def stage_sfm(scene, cfg, P):
    from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
    print(f"[sfm] hloc exhaustive，{len(list(P['images'].glob('*.png')))} 張影像"
          f"（兩兩配對，這是最耗時的一段）")
    ns = ImagesToNerfstudioDataset(
        data=P["images"], output_dir=P["sfm"],
        camera_type="perspective", matching_method="exhaustive",
        sfm_tool="hloc", gpu=True,
    )
    ns.main()
    print(f"[sfm] 完成 → {P['sfm']}")


def stage_check(scene, cfg, P):
    """SfM 註冊率 + 含 tag 影像是否足夠，兩者都會決定下游成敗。"""
    import cv2
    n_in = len(list(P["images"].glob("*.png")))
    tfm = json.loads(P["sfm_tfm"].read_text())
    n_reg = len(tfm["frames"])
    rate = n_reg / n_in * 100
    print(f"[check] SfM 註冊 {n_reg}/{n_in} 張 = {rate:.1f}%", end="  ")
    print("✓" if rate >= 90 else "⚠ 偏低，可能是重疊不足或動態模糊")

    ext = cfg["extractor"]
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    n_tag = 0
    for fr in tfm["frames"]:
        img = cv2.imread(str(P["sfm"].parent / fr["file_path"]))
        if img is None:
            continue
        _, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is not None and len(ids) == 1 and ids[0] == ext["marker_id"]:
            n_tag += 1
    need = ext["num_marked"]
    print(f"[check] 已註冊影像中含 tag 的有 {n_tag} 張，"
          f"extract_positions 要求剛好 == num_marked({need})", end="  ")
    if n_tag == need:
        print("✓")
    else:
        print("✗")
        print(f"[check]   → 下一階段會拋 'Mismatched number of aruco and sfm transforms'")
        print(f"[check]   → 解法：把 configs/captures/{scene} 的 num_marked 改成 {n_tag}"
              f"（至少要 5 才解得出 Sim(3)）")
    return rate, n_tag


def stage_scale(scene, cfg, P):
    """ArUco 定尺度。完全沿用上游的 extract_positions 與 compute_ransac_transform。"""
    from figs.render.capture_generation import extract_positions
    import figs.utilities.capture_helper as ch

    tfm_data = json.loads(P["sfm_tfm"].read_text())
    sparse = o3d.io.read_point_cloud(P["sfm_spc"].as_posix())

    camera_config = cfg["camera"]
    if camera_config is None:
        raise SystemExit("camera 區塊是 null，本專案應使用自行標定的內參")

    Psfm, Parc = extract_positions(P["sfm"], cfg["extractor"], camera_config)
    cs, Rs, ts = ch.compute_ransac_transform(Psfm, Parc)
    print(f"[scale] Sim(3) 解出 scale cs = {cs:.6f}")
    print(f"[scale] 旋轉 Rs=\n{Rs}")
    print(f"[scale] 平移 ts = {ts.ravel()}")

    for frame in tfm_data["frames"]:
        Tc2s = np.array(frame["transform_matrix"])
        Tc2w = np.eye(4)
        Tc2w[:3, :3] = Rs @ Tc2s[:3, :3]
        Tc2w[:3, 3] = cs * Rs @ Tc2s[:3, 3] + ts
        frame["transform_matrix"] = Tc2w.tolist()

    pts = np.asarray(sparse.points)
    for i, p in enumerate(pts):
        pts[i, :] = cs * Rs @ p + ts
    sparse.points = o3d.utility.Vector3dVector(pts)

    P["tfm"].write_text(json.dumps(tfm_data, indent=4))
    o3d.io.write_point_cloud(P["spc"].as_posix(), sparse)

    bb = sparse.get_axis_aligned_bounding_box()
    ext_mm = bb.get_extent()
    print(f"[scale] 寫出 {P['tfm'].name} 與 {P['spc'].name}")
    print(f"[scale] 稀疏點雲 bounding box (公尺): "
          f"{ext_mm[0]:.2f} x {ext_mm[1]:.2f} x {ext_mm[2]:.2f}")
    print(f"[scale] ⚠ 這是第 7 章要驗的數字，先對照一下實際房間大小是否合理")


def stage_train(scene, cfg, P):
    """ns-train，輸出直接串流。三個保尺度旗標與上游完全一致。"""
    cmd = [
        "ns-train", "splatfacto",
        "--data", scene,
        "--viewer.quit-on-train-completion", "True",
        "--output-dir", "outputs",
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        "nerfstudio-data",
        "--orientation-method", "none",
        "--center-method", "none",
        "--auto-scale-poses", "False",
    ]
    print(f"[train] cwd = {P['workspace']}")
    print(f"[train] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=P["workspace"].as_posix())
    print(f"[train] returncode = {r.returncode}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--select", choices=["uniform", "sharp"], default="uniform",
                    help="uniform=原作者的最遠點取樣（預設）；sharp=箱內取最銳利")
    ap.add_argument("--stage", required=True,
                    choices=["frames", "sfm", "check", "scale", "train", "all"])
    args = ap.parse_args()
    global SELECT_MODE
    SELECT_MODE = args.select

    cfg = json.loads((CONFIGS / "captures" / f"{args.config}.json").read_text())
    P = paths(args.scene)

    stages = (["frames", "sfm", "check", "scale", "train"]
              if args.stage == "all" else [args.stage])
    for s in stages:
        print(f"\n{'='*70}\n=== stage: {s}\n{'='*70}")
        {"frames": stage_frames, "sfm": stage_sfm, "check": stage_check,
         "scale": stage_scale, "train": stage_train}[s](args.scene, cfg, P)


if __name__ == "__main__":
    main()
