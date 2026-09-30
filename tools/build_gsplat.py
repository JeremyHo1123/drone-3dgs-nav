"""
Chapter 6 mapping: run FiGS's generate_gsplat in separate stages.

Why not just call generate_gsplat():
  1. Its line 143 runs ns-train with subprocess.run(..., capture_output=True),
     so all training output is swallowed: an hour of training shows no progress, and a failure only shows up at the very end
  2. It is a single function that runs straight through, with no chance to check the registration rate after SfM and before training.
     If a few tag images are not registered by SfM, the downstream extract_positions raises
     "Mismatched number of aruco and sfm transforms", and by then the SfM run has been wasted

This script calls the same upstream functions (extract_frames / extract_positions /
compute_ransac_transform) and rewrites none of the math; it only splits them up and adds checkpoints.

Stages:
  --stage frames   frame extraction
  --stage sfm      hloc SfM
  --stage check    check the registration rate (changes nothing)
  --stage scale    ArUco scale solve, writes transforms.json and sparse_pc.ply
  --stage train    ns-train splatfacto (output streamed straight to stdout)
  --stage all      all of the above, in order
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
SELECT_MODE = "uniform"   # uniform = original authors' method; sharp = sharpest frame in each bin
MIN_TAG_PX = 100          # scale solve only trusts observations with tag side >= this; 0 = no filter (upstream behavior)
MIN_TAG_OBS = 4           # keep at least this many after filtering; if fewer, take the largest ones instead
RANSAC_RESTARTS = 30      # number of fixed-seed reruns; the solution with the smallest residual wins, so results are reproducible
TAG_RULE = "exact"        # exact = upstream "exactly one tag"; present = accept whenever marker_id is present (see _tag_observations_present)
LEVEL_R = 0.0             # >0: after the scale solve, level using the ground within this radius (meters); 0 = no leveling (upstream behavior, see _ground_level_rotation)


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
    1-D farthest point sampling, equivalent to upstream capture_helper.distribute_values.

    Upstream is an O(n*k^2) pure-Python loop (picking 380 of 3937 takes over ten minutes);
    here numpy maintains a "minimum distance to the selected set" array, complexity O(n*k),
    and picking 380 of 3937 takes only 0.002 s. Verified on 4 random test sets with gaps: the selected values are identical.
    """
    v = np.asarray(values, dtype=float)
    sel = [0]                                   # same as upstream: start from values[0]
    mind = np.abs(v - v[0])
    for _ in range(k - 1):
        j = int(np.argmax(mind))
        sel.append(j)
        np.minimum(mind, np.abs(v - v[j]), out=mind)
    return sorted(sel)


def extract_frames_custom(video_path, images_path, ext_cfg, mode="uniform"):
    """
    Frame extraction. The binning rule is exactly the same as upstream extract_frames:
      tag frame = exactly 1 marker detected and its id matches; every other frame goes to the "no tag" pool.
      Then take num_marked frames from the tag pool and num_images - num_marked from the no-tag pool.

    mode="uniform" (default, same as the original authors' method)
        1-D farthest point sampling in each pool, spread purely over time, ignoring image sharpness.
    mode="sharp"
        Time-bin each pool and take the sharpest frame in each bin.

    The only implementation difference from upstream: upstream records millisecond timestamps and later
    seeks back with cap.set(CAP_PROP_POS_MSEC) to grab frames, but millisecond seeking in H.264 is imprecise
    and lands on neighboring frames (measured: "should take 20 tag frames, actually extracted 22").
    Here we use frame indices + a full sequential read; the selected frames and selection rule are unchanged, only the retrieval no longer drifts.

    ext_cfg["skip_start_sec"] (optional, default 0)
        Frames in the first this-many seconds of the video go into neither pool; sampling of the remaining frames is unchanged.
        Use it to exclude segments where the camera operator is in frame (e.g. the first 16 s of sunny_baseball_field
        show the camera operator's shadow) - a moving shadow becomes stains or floaters on the ground in 3DGS.
        Upstream extract_positions only reads num_marked / marker_length / marker_id,
        so this extra field does not affect it.
    """
    import cv2
    Nimg = ext_cfg["num_images"]
    Narc = ext_cfg["num_marked"]
    mkr_id = ext_cfg["marker_id"]
    skip_sec = float(ext_cfg.get("skip_start_sec", 0))

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")
    skip_n = int(np.ceil(skip_sec * cap.get(cv2.CAP_PROP_FPS))) if skip_sec > 0 else 0
    if skip_n:
        print(f"[frames] skipping the first {skip_sec:g} s = first {skip_n} frames (frames 0-{skip_n - 1})")
    tag_i, tag_s, emp_i, emp_s = [], [], [], []
    i = 0
    print(f"[frames] pass 1: per-frame ArUco binning (mode={mode})")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i < skip_n:
            i += 1
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sv = cv2.Laplacian(g, cv2.CV_64F).var() if mode == "sharp" else 0.0
        _, ids, _ = det.detectMarkers(g)
        if ids is not None and len(ids) == 1 and ids[0] == mkr_id:
            tag_i.append(i); tag_s.append(sv)
        else:
            emp_i.append(i); emp_s.append(sv)
        i += 1
    cap.release()
    print(f"[frames]   total frames {i} (skipped {skip_n}), with tag {len(tag_i)}, without tag {len(emp_i)}")

    if len(tag_i) < Narc:
        raise SystemExit(f"Only {len(tag_i)} frames contain the tag, fewer than num_marked={Narc}")
    if len(emp_i) < Nimg - Narc:
        raise SystemExit(f"Only {len(emp_i)} frames without the tag, fewer than {Nimg - Narc}")

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
    print(f"[frames]   selected with-tag {len(sel_tag)} + without-tag {len(sel_emp)} = {len(sel)} frames")
    if len(sel) != Nimg:
        print(f"[frames]   ⚠ the two pools overlap: actually {len(sel)} frames (expected {Nimg})")

    print("[frames] pass 2: sequential read, writing the selected frames (no seek)")
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
    print(f"[frames]   wrote {k} frames")


def stage_frames(scene, cfg, P):
    # The file name must be <scene>.MOV or <scene>_*.MOV. Not *{scene}*, because
    # sunny_baseball_field_IMG_2527.MOV would also match baseball_field
    caps = [p for p in (GSPLATS / "capture").iterdir()
            if p.stem == scene or p.stem.startswith(scene + "_")]
    if len(caps) != 1:
        raise SystemExit(f"{len(caps)} files matched in capture/, must be exactly 1: {caps}")
    P["process"].mkdir(parents=True, exist_ok=True)
    P["images"].mkdir(parents=True, exist_ok=True)
    print(f"[frames] source {caps[0].name}")
    extract_frames_custom(caps[0], P["images"], cfg["extractor"], mode=SELECT_MODE)
    n = len(list(P["images"].glob("*.png")))
    print(f"[frames] done, extracted {n} frames → {P['images']}")
    if n != cfg["extractor"]["num_images"]:
        print(f"[frames] ⚠ does not match num_images={cfg['extractor']['num_images']}")


def stage_sfm(scene, cfg, P):
    from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
    print(f"[sfm] hloc exhaustive, {len(list(P['images'].glob('*.png')))} images"
          f" (all-pairs matching; this is the slowest part)")
    ns = ImagesToNerfstudioDataset(
        data=P["images"], output_dir=P["sfm"],
        camera_type="perspective", matching_method="exhaustive",
        sfm_tool="hloc", gpu=True,
    )
    ns.main()
    print(f"[sfm] done → {P['sfm']}")


def stage_check(scene, cfg, P):
    """SfM registration rate + whether there are enough tag images; both decide downstream success."""
    import cv2
    n_in = len(list(P["images"].glob("*.png")))
    tfm = json.loads(P["sfm_tfm"].read_text())
    n_reg = len(tfm["frames"])
    rate = n_reg / n_in * 100
    print(f"[check] SfM registered {n_reg}/{n_in} images = {rate:.1f}%", end="  ")
    print("✓" if rate >= 90 else "⚠ low; likely insufficient overlap or motion blur")

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
    print(f"[check] {n_tag} registered images contain the tag; "
          f"extract_positions requires exactly == num_marked({need})", end="  ")
    if n_tag == need:
        print("✓")
    else:
        print("✗")
        print(f"[check]   → the next stage will raise 'Mismatched number of aruco and sfm transforms'")
        print(f"[check]   → fix: change num_marked in configs/captures/{scene} to {n_tag}"
              f" (at least 5 are needed to solve Sim(3))")
    return rate, n_tag


def _tag_pixel_sizes(P, cfg):
    """
    Measure the mean ArUco side length (pixels) in each registered image.

    The iteration order is exactly the same as extract_positions (the same iteration over transforms["frames"],
    the same "exactly one marker detected and its id matches" condition), so the returned array can index
    the Psfm / Parc returned by extract_positions directly.
    """
    import cv2
    ext = cfg["extractor"]
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    out = []
    for fr in json.loads((P["sfm"] / "transforms.json").read_text())["frames"]:
        img = cv2.imread(str(P["sfm"].parent / fr["file_path"]))
        if img is None:
            continue
        c, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is not None and len(ids) == 1 and ids[0] == ext["marker_id"]:
            q = c[0].reshape(4, 2)
            out.append(float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)])))
    return np.array(out)


def _tag_observations_present(P, cfg, camera_config):
    """
    Used by TAG_RULE="present": accept an image whenever marker_id is in it, ignoring other ids.

    -- Why (2026-09-21, found while mapping sunny_baseball_field) ----------------
    Upstream extract_positions requires "exactly one marker detected and its id matches". Grass texture is often
    mistaken for fake tags of ten to twenty pixels (id 17, 23, 37, 49...); the real tag is detected, yet the whole
    image is discarded because of one extra false detection. In sunny_baseball_field the discarded images were exactly
    the ones with the largest tags (129, 108, 95 px), and close-range observations are what the scale solve needs most:

        rule      largest 4 sides (px)    cs         median resid  ground tilt
        exact     178, 93, 77, 73         1.764460   0.89 cm       4.85°
        present   178, 129, 108, 95       1.771382   0.23 cm       3.30°

    The math is identical to upstream (same marker_points, SOLVEPNP_IPPE_SQUARE, same image reading
    and grayscale conversion); only "which images count" changes. If marker_id appears twice in one image there is
    no way to tell which one is real, so the whole image is skipped.

    Returns Psfm, Parc (3xN) and the tag side length (px) of each observation.
    """
    import cv2
    ext = cfg["extractor"]
    L = ext["marker_length"]
    K = np.array(camera_config["intrinsics_matrix"])
    D = np.array(camera_config["distortion_coefficients"])
    M = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]])
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    Psfm, Parc, px = [], [], []
    n_extra = 0
    for fr in json.loads((P["sfm"] / "transforms.json").read_text())["frames"]:
        img = cv2.imread(str(P["sfm"].parent / fr["file_path"]))
        if img is None:
            raise SystemExit(f"Cannot read {fr['file_path']}")
        c, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        hit = [k for k, i in enumerate(ids.ravel()) if i == ext["marker_id"]]
        if len(hit) != 1:
            continue
        n_extra += len(ids) > 1
        q = c[hit[0]]
        ok, rvec, tvec = cv2.solvePnP(M, q, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        Tw2c = np.eye(4)
        Tw2c[:3, :3], Tw2c[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        Parc.append(np.linalg.inv(Tw2c)[:3, 3])
        Psfm.append(np.array(fr["transform_matrix"])[:3, 3])
        q = q.reshape(4, 2)
        px.append(float(np.mean([np.linalg.norm(q[j] - q[(j + 1) % 4]) for j in range(4)])))
    print(f"[scale] tag rule present: {len(px)} images contain id {ext['marker_id']}, "
          f"{n_extra} of them also have false detections of other ids (the exact rule would drop the whole image)")
    return np.array(Psfm).T, np.array(Parc).T, np.array(px)


def _ground_level_rotation(pts, rmax):
    """
    Return the smallest rotation Rl (about the origin) that makes the ground horizontal, and the tilt before rotation (degrees).

    -- Why (2026-09-21, found while mapping sunny_baseball_field) ----------------
    The world frame is the tag frame, z axis = tag normal. The tag is taped to cardboard and the cardboard sits on grass;
    if one edge of the cardboard is raised a few centimeters by the grass, the whole scene tilts a few degrees, with no error raised.
    In sunny_baseball_field the ground is consistently tilted by about 3.5°, in the same direction, at every fit radius:

        fit radius  2 m     3 m     5 m     8 m     12 m    15 m
        sunny     4.11°   4.29°   4.07°   3.56°   3.59°   3.37°
        baseball  1.95°   1.60°   1.35°   1.08°   0.71°   0.66°

    The same field is close to level in baseball_field over a large fit radius, so the tag is tilted, not the field.
    "The angle between the tag plane and the ground" is unaffected by Sim(3) (triangulating the four tag corners from
    multiple views also gives 3.99°), so it is not a wrong scale solve. A 3.5° slope means the relative height of the ground changes by about 60 cm over 10 m of forward flight.

    Method: fit from the inside out, ring by ring (2 m, 3 m, 5 m... up to rmax); each ring takes a ±25 cm band of points
    around the previous ring's plane, so that under tilt the distant ground does not fall out of a |z|<0.25 window.
    Rotation only, no translation or scaling: the origin stays at the tag and the scale is unchanged.
    """
    n, d = np.array([0.0, 0.0, 1.0]), 0.0
    r_xy = np.linalg.norm(pts[:, :2], axis=1)
    o3d.utility.random.seed(0)
    for r in [x for x in (2, 3, 5, 8, 12, 15, 20, 30) if x < rmax] + [rmax]:
        m = (r_xy < r) & (np.abs(pts @ n + d) < 0.25)
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[m])
        (a, b, c, dd), inl = sub.segment_plane(0.03, 3, 3000)
        nn = np.array([a, b, c]); k = np.linalg.norm(nn)
        n, d = nn / k * np.sign(c), dd / k * np.sign(c)
        print(f"[level]   r<{r:g} m: tilt {np.degrees(np.arccos(n[2])):.2f}°, "
              f"inlier {len(inl)}/{m.sum()}")
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, z); s_ = np.linalg.norm(axis); c_ = float(n @ z)
    if s_ < 1e-12:
        return np.eye(3), 0.0
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / s_
    ang = np.arctan2(s_, c_)
    Rl = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx     # Rodrigues, Rl @ n = z
    return Rl, float(np.degrees(ang))


def stage_scale(scene, cfg, P):
    """
    ArUco scale solve. Reuses upstream extract_positions and compute_ransac_transform,
    but before feeding RANSAC it filters out long-range observations by the tag's pixel size.

    -- Why filter (2026-09-10, found while mapping baseball_field) -------------------
    The distance recovered by solvePnP is proportional to fx*L/px, where px is the tag's side length in the image. When the tag is only
    twenty or thirty pixels, a corner localization error of one or two pixels causes a distance error of several percent, and because
    of IPPE_SQUARE's pose ambiguity for a planar square, this error is not zero-mean: it has a systematic bias.

    Upstream frame extraction does farthest point sampling over time on the "tag frames", ignoring how big the tag is,
    so most of the 20 selected frames are taken from far away. Measured (cross-validated on two scenes, re-solved with the
    k observations with the largest tags, box height checked against the tape-measured 171.0 cm):

        top k        baseball_field      scene04
           4           +0.40%            -0.48%
           5           +0.40%            -0.41%
           6           +0.81%            +1.68%
           7           +1.37%            +1.86%
        20 (upstream)  +2.66%            +0.73%

    Two different videos, shot on different dates: in both, "use only the closest 4-5" is the most accurate. scene04 lands at +0.73%
    with all 20, which is a coincidence of errors canceling out, not a correct method - the same approach
    degrades to +2.83% on baseball_field.

    After switching to min_tag_px=100 (camera within about 2.4 m), all three metrics of baseball_field
    improved at once; the ground tilt is completely independent of the tape measure, so it serves as independent evidence:

        median residual  15.1 cm  →  0.49 cm
        ground tilt       2.96°   →  1.24°
        box height error +2.83%   →  -0.19%

    Also, compute_ransac_transform does not fix a random seed; with only 5 observations the result varies by about
    0.3% from run to run. Here it is run N times with fixed seeds and the set with the smallest median residual is kept, so the same
    input always yields the same scene.

    To get back to upstream behavior (scene04 was built this way), pass --min-tag-px 0.
    """
    from figs.render.capture_generation import extract_positions
    import figs.utilities.capture_helper as ch

    tfm_data = json.loads(P["sfm_tfm"].read_text())
    sparse = o3d.io.read_point_cloud(P["sfm_spc"].as_posix())

    camera_config = cfg["camera"]
    if camera_config is None:
        raise SystemExit("The camera block is null; this project should use self-calibrated intrinsics")

    if TAG_RULE == "present":
        Psfm, Parc, px = _tag_observations_present(P, cfg, camera_config)
    else:
        Psfm, Parc = extract_positions(P["sfm"], cfg["extractor"], camera_config)
        Psfm, Parc = np.asarray(Psfm), np.asarray(Parc)

    if MIN_TAG_PX > 0:
        if TAG_RULE != "present":
            px = _tag_pixel_sizes(P, cfg)
        if len(px) != Psfm.shape[1]:
            raise SystemExit(f"[scale] tag size array length {len(px)} does not match observation count {Psfm.shape[1]}")
        keep = px >= MIN_TAG_PX
        if keep.sum() < MIN_TAG_OBS:                      # not enough: fall back to the largest few
            keep = np.zeros(len(px), bool)
            keep[np.argsort(-px)[:MIN_TAG_OBS]] = True
            print(f"[scale] ⚠ tags ≥{MIN_TAG_PX}px: only {(px >= MIN_TAG_PX).sum()} of them; "
                  f"taking the largest {MIN_TAG_OBS} instead (smallest {px[keep].min():.0f}px)")
        print(f"[scale] tag size filter ≥{MIN_TAG_PX}px: kept {keep.sum()}/{len(px)} observations, "
              f"side lengths {sorted(px[keep].astype(int).tolist())}")
        Psfm, Parc = Psfm[:, keep], Parc[:, keep]
    else:
        print("[scale] no tag size filtering (--min-tag-px 0, upstream behavior)")

    best = None                                            # fixed seeds, keep the solution with the smallest residual
    for seed in range(RANSAC_RESTARTS):
        np.random.seed(seed)
        c_, R_, t_ = ch.compute_ransac_transform(Psfm, Parc)
        res = np.median(np.linalg.norm(Parc - (c_ * R_ @ Psfm + t_[:, None]), axis=0))
        if best is None or res < best[0]:
            best = (res, c_, R_, t_)
    resid, cs, Rs, ts = best
    print(f"[scale] {RANSAC_RESTARTS} fixed-seed reruns, kept the one with the smallest median residual: "
          f"residual {resid * 100:.2f} cm")
    print(f"[scale] Sim(3) solved scale cs = {cs:.6f}")
    print(f"[scale] rotation Rs=\n{Rs}")
    print(f"[scale] translation ts = {ts.ravel()}")

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

    if LEVEL_R > 0:
        print(f"[level] leveling with the ground within {LEVEL_R:g} m (rotation only; scale and origin unchanged)")
        Rl, tilt = _ground_level_rotation(pts, LEVEL_R)
        for frame in tfm_data["frames"]:
            T = np.array(frame["transform_matrix"])
            T[:3, :3], T[:3, 3] = Rl @ T[:3, :3], Rl @ T[:3, 3]
            frame["transform_matrix"] = T.tolist()
        pts = pts @ Rl.T
        sparse.points = o3d.utility.Vector3dVector(pts)
        print(f"[level] rotated by {tilt:.2f}°; re-checking after leveling:")
        _, after = _ground_level_rotation(pts, LEVEL_R)
        print(f"[level] remaining tilt after leveling {after:.2f}°")

    P["tfm"].write_text(json.dumps(tfm_data, indent=4))
    o3d.io.write_point_cloud(P["spc"].as_posix(), sparse)

    bb = sparse.get_axis_aligned_bounding_box()
    ext_mm = bb.get_extent()
    print(f"[scale] wrote {P['tfm'].name} and {P['spc'].name}")
    print(f"[scale] sparse point cloud bounding box (meters): "
          f"{ext_mm[0]:.2f} x {ext_mm[1]:.2f} x {ext_mm[2]:.2f}")
    print(f"[scale] ⚠ this is the number Chapter 7 verifies; check now that it is plausible for the real room size")


def stage_train(scene, cfg, P):
    """ns-train, output streamed directly. The three scale-preserving flags are exactly the same as upstream."""
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
    global SELECT_MODE, MIN_TAG_PX, TAG_RULE, LEVEL_R  # must be declared before they are read as defaults below
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--select", choices=["uniform", "sharp"], default="uniform",
                    help="uniform=original authors' farthest point sampling (default); sharp=sharpest frame in each bin")
    ap.add_argument("--min-tag-px", type=float, default=MIN_TAG_PX,
                    help="scale solve only trusts observations with tag side >= this many pixels (default 100, i.e. within about 2.4 m); "
                         "0 = no filter, back to upstream behavior")
    ap.add_argument("--tag-rule", choices=["exact", "present"], default=TAG_RULE,
                    help="which images the scale solve uses: exact=exactly one tag detected (upstream, default); "
                         "present=accept whenever marker_id is present, ignoring false detections of other ids caused by grass texture")
    ap.add_argument("--level-ground", type=float, default=LEVEL_R,
                    help="after the scale solve, level using the ground within this radius (meters), fixing tilt from a tag that is not lying flat; "
                         "0 = no leveling (default, upstream behavior)")
    ap.add_argument("--stage", required=True,
                    choices=["frames", "sfm", "check", "scale", "train", "all"])
    args = ap.parse_args()
    SELECT_MODE = args.select
    MIN_TAG_PX = args.min_tag_px
    TAG_RULE = args.tag_rule
    LEVEL_R = args.level_ground

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
