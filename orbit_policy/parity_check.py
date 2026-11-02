#!/usr/bin/env python3
"""
Check that orbit_runtime.OrbitPolicy reproduces the training environment: replay bundle_v4/reference.pt and
compare every value it computes, step by step, with what the training environment recorded.

reference.pt was recorded with the real DroneVLAOrbitEnv when the bundle was exported (two 40-step runs:
clockwise and counter-clockwise). The same images and states are fed to the inference code here, and each
step compares: observation, visual feature, VLM feature, raw actor output, filtered action, VAE latent,
history, and the processed action (= the body rates and thrust sent to the flight controller).
With the height integrator (v4) it also compares f_I and the total thrust sent to the physics.
Step timings are measured too.

PASS: largest difference < 1e-3 for the command sent to the flight controller, for the raw actor output and
(v4) for the total thrust. Different GPUs and PyTorch / transformers versions typically give 1e-5 to 1e-4.

This needs only torch and transformers -- no grad_nav and no 3DGS scene.

Usage: python parity_check.py [--bundle bundle_v4] [--device cuda|cpu]
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from orbit_runtime import OrbitPolicy, SceneState  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default=os.path.join(HERE, "bundle_v4"), help="folder with weights.pt and reference.pt")
ap.add_argument("--device", default=None)
ap.add_argument("--tol", type=float, default=1e-3)
ap.add_argument("--tf32", action="store_true", help="use TF32 matmuls (diagnostics; reference.pt was recorded in full FP32)")
a = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = a.tf32
torch.backends.cudnn.allow_tf32 = a.tf32        # PyTorch uses TF32 for convolutions by default; switch it off explicitly (as for reference.pt)
ref = torch.load(os.path.join(a.bundle, "reference.pt"), map_location="cpu", weights_only=True)
# v3 and later: the bundle meta records thrust_exponent / t_hover_prog; OrbitPolicy refuses other values
meta = torch.load(os.path.join(a.bundle, "weights.pt"), map_location="cpu", weights_only=True)["meta"]
t0 = time.time()
pol = OrbitPolicy(a.bundle, instruction=0, device=a.device, canon_quat=False,
                  T_hover=meta.get("t_hover_prog", 0.65), thrust_exponent=meta.get("thrust_exponent", 1.67))
dev = pol.device
print(f"device {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}), torch {torch.__version__}, "
      f"loaded in {time.time() - t0:.1f} s")
_ver = {"heading": "v2 or later", "world": "v1"}.get(pol.obs_frame, "?")
print(f"bundle {os.path.abspath(a.bundle)}: observation layout {pol.obs_frame} ({_ver})")
print(f"thrust conversion T_hover {pol.T_hover}, thrust_exponent {pol.p}; height integrator {pol.hint if pol.hint is not None else 'none'}; "
      f"episode start {pol.start_mode}")

# ---- CLIP on its own (the same 224x224 image)
with torch.no_grad():
    px = (ref["image224"].to(dev) - pol.clip_mean) / pol.clip_std
    d_px = float((px.cpu() - ref["pixel_values_processor"]).abs().max())
    e = pol.clip(pixel_values=px).image_embeds.cpu()
    d_clip = float((e - ref["clip_image_embeds"]).abs().max())
print(f"CLIP: preprocessing diff {d_px:.2e}, image embedding diff {d_clip:.2e} (embedding magnitude about {float(ref['clip_image_embeds'].abs().max()):.2f})")


def st(s):
    return SceneState(z=s["z"], quat=tuple(s["quat"]), vel=tuple(s["vel"]))


worst = {}
def upd(k, x, y):
    worst[k] = max(worst.get(k, 0.0), float((x.detach().float().cpu() - y.float()).abs().max()))


times, clip_times = [], []
for ep in ref["episodes"]:
    pol.instruction = ep["task"]
    img = lambda i: ep["images"][i].to(dev)  # noqa: E731
    pol.reset(st(ep["states"][0]), img(0))
    upd("observation", pol.obs[0], ep["obs"][0]); upd("VAE observation", pol.vae_obs[0], ep["vae_obs"][0])
    upd("VLM feature", pol.vlm[0], ep["vlm"][0]); upd("visual feature", pol.visual_info[0], ep["visual"][0])
    for t in range(len(ep["mu"])):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        cmd = pol.act()
        pol.observe(st(ep["states"][t + 1]), img(t + 1))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
        (clip_times if pol.progress % 10 == 0 else times).append(dt)
        upd("raw actor output", pol.last_mu, ep["mu"][t]); upd("filtered action", pol.lpf[0], ep["act"][t])
        upd("VAE latent", pol.last_z, ep["z"][t]); upd("history", pol.hist[0], ep["hist"][t])
        upd("command to FC", pol.actions, ep["proc"][t])
        if "thrust_total" in ep:            # v4 bundle: integrator correction and total thrust to the physics (incl. f_I)
            upd("integrator f_I", torch.tensor(pol.f_I), ep["fI"][t])
            upd("total thrust incl. f_I", torch.tensor(cmd.sim_thrust_frac), ep["thrust_total"][t])
        upd("observation", pol.obs[0], ep["obs"][t + 1]); upd("VAE observation", pol.vae_obs[0], ep["vae_obs"][t + 1])
        upd("VLM feature", pol.vlm[0], ep["vlm"][t + 1]); upd("visual feature", pol.visual_info[0], ep["visual"][t + 1])
        if t in (0, len(ep["mu"]) - 1):
            print(f"  task {ep['task']} step {t + 1:2d}: body rate FLU = ({cmd.body_rate_flu[0]:+.3f}, {cmd.body_rate_flu[1]:+.3f}, "
                  f"{cmd.body_rate_flu[2]:+.3f}) rad/s, sim thrust {cmd.sim_thrust_frac:.3f} -> PX4 throttle {cmd.px4_thrust:.3f}")

print("largest absolute difference (both runs, every step):")
for k, v in worst.items():
    print(f"  {k:24s} {v:.2e}")
ok = worst["command to FC"] < a.tol and worst["raw actor output"] < a.tol and worst.get("total thrust incl. f_I", 0.0) < a.tol
ms = lambda xs: 1000 * sorted(xs)[len(xs) // 2]  # noqa: E731
print(f"time per step (median): normal step {ms(times):.1f} ms, step with CLIP {ms(clip_times):.1f} ms (control period 50 ms)")
print("PASS" if ok else "FAIL", f"(tolerance {a.tol:g})")
sys.exit(0 if ok else 1)
