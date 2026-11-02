"""
Inference for the orbit-the-box policy (GRaD-Nav++, scene sunny_baseball_field). This is the code that runs
on the drone's Jetson; run_in_sim.py runs the same code inside the simulator.

Depends only on torch and transformers (the CLIP image encoder); no grad_nav, nerfstudio or torchvision.
Every step mirrors the training environment line by line (grad_nav with the simulator patch:
calculateObservations in envs/drone_vla_orbit.py, step / process_GS_data in envs/drone_vla_long_task.py,
evaluate_policy in grad_nav's algorithms/gradnav_vla_moe.py). The only differences: no observation noise,
the VAE uses its mean (no sampling), and v1's quaternion is put in the sign convention used in training.

The observation layout is set by the bundle's weights.pt meta["obs_frame"], never by the caller, so that
weights and observation layout cannot be mismatched (a mismatch raises no error; the drone just flies wrong):
    "world"   (v1; also bundles without the key): obs[2:6] scene-frame quaternion with w <= 0;
              obs[14:17] scene-frame velocity. Needs the correct ENU -> scene yaw alignment
              (yaw_offset of scene_state_from_enu)
    "heading" (v2 and later): obs[2:6] the quaternion with yaw removed (w >= 0); obs[14:17] velocity in
              the heading frame (forward, left, up). Both are independent of world yaw, so the yaw
              alignment does not affect the observation
All other fields are the same in both: z, vz, action, previous action, visual feature (32), VAE latent (24).

v4 (bundle meta has height_integrator and start_mode), exactly as in the training environment:
    Height integrator: the mixed thrust fraction gets f_I added. Each step
        f_I <- clamp(f_I + k_I * dt * clamp(H - z, +-e_clip), +-f_max), z = height in the latest observation,
        f_I = 0 at hand-over; thrust sent = clamp(f + f_I, 0, 0.5). The action in the observation
        (self.actions) does not include f_I.
    deploy_warm start: start thrust command = meta start_cmd; the VAE history holds 5 copies of the
        hand-over observation; the low-pass filter starts from the hover action (otherwise all zero, and
        the first step uses the raw action).
    If meta has thrust_exponent / t_hover_prog (v3 and later) and they differ from the constructor
    arguments, construction fails (start with the same values).
    Bundles without these keys (v1, v2) behave exactly as before.

Call order (20 Hz, one step = 0.05 s):
    pol = OrbitPolicy(bundle_dir, instruction=0)          # 0 = clockwise, 1 = counter-clockwise
    pol.reset(state, image)                               # the moment the policy takes over
    loop:
        cmd = pol.act()                                   # this step's command (see PX4Command)
        ... send cmd, wait 0.05 s ...
        pol.observe(state, image)                         # the new state and image after sending

state = SceneState, in the scene (ArUco) frame, for the body origin (the flight controller):
    z     height above ground [m] (scene ground = 0)
    quat  attitude quaternion (x, y, z, w), body FLU (forward, left, up) to scene
    vel   linear velocity in the world frame [m/s] (scene frame, z up)
image = RGB camera, float32 in [0, 1], shape (144, 256, 3). On the real drone the Gemini 2's 640x360 frame
        is first reduced with prepare_camera_image() (training renders 256x144 directly at 0.4x resolution).

async_vlm=True (used by the flight node): CLIP image encoding (about 65 ms on the Jetson) runs in a
background thread on its own CUDA stream; the control loop does not wait for it. An image is sent every
10 steps and its result is used from the next step on (2-3 steps late; training uses it on the same step).
reset() still computes it synchronously, so the first observation matches training.
Default False: run_in_sim.py and parity_check.py compare step by step.

Output command (PX4Command) -> MAVROS /mavros/setpoint_raw/attitude (mavros_msgs/AttitudeTarget):
    type_mask = 128 (IGNORE_ATTITUDE: body rates + thrust only)
    body_rate = body_rate_flu [rad/s], body FLU. MAVROS converts it to PX4's FRD itself
                (setpoint_raw.cpp attitude_cb: transform_frame_baselink_aircraft)
    thrust    = px4_thrust (0-1). MAVROS thrust_scaling = 1.0
"""
import math
import os
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gradnav_models.actor_vla_moe import TaskAwareMoEActor
from gradnav_models.vae import VAE
from gradnav_models.squeeze_net_nodl import VisualPerceptionNet
# Coordinate helpers live in orbit_geometry.py (no torch); the same names are re-exported here
from orbit_geometry import (SceneState, quat_mul, yaw_of,  # noqa: F401
                            scene_state_from_enu, yaw_offset_from_start,
                            tilt_quat, heading_frame_vel)

OBS_FRAMES = ("world", "heading")    # see "observation layout" above
START_MODES = ("upstream", "deploy", "deploy_warm")    # see "v4" above

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)     # clip-vit-base-patch16 preprocessor_config.json
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_HF_ID = "openai/clip-vit-base-patch16"         # used when the bundle has no clip-vit-base-patch16/ folder


@dataclass
class PX4Command:
    body_rate_flu: tuple     # rad/s, body FLU (the convention of MAVROS AttitudeTarget.body_rate)
    px4_thrust: float        # 0-1, PX4's normalised thrust (throttle)
    sim_thrust_frac: float   # the simulator's thrust command: fraction of max_thrust (0-0.5); includes f_I with the height integrator (= what training fed the physics)
    raw_action: tuple        # the 4 policy outputs after tanh and the low-pass filter (-1 to 1)
    type_mask: int = 128     # AttitudeTarget.IGNORE_ATTITUDE


def canonical_quat(q):
    """q and -q are the same attitude. All v1 training starts have w <= 0, so use w <= 0 here.
    (MAVROS / PX4 often give w >= 0; without this the policy would see an input it never trained on.)"""
    return -q if float(q[3]) > 0 else q


def prepare_camera_image(rgb_uint8_hw3, device="cpu"):
    """Real camera RGB (HxWx3, uint8, RGB order) -> the (144, 256, 3) float [0, 1] image the policy expects.
    Downscaled by average pooling (area average); the 3DGS render in training is 256x144 directly."""
    t = torch.as_tensor(rgb_uint8_hw3, device=device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = F.adaptive_avg_pool2d(t, (144, 256))
    return t[0].permute(1, 2, 0).contiguous()


class AsyncVLM:
    """Background thread for CLIP image encoding (see async_vlm above).

    submit(img224, step): the main thread hands over a (1, 3, 224, 224) image (on the GPU, computed on the
        main stream). The background thread first waits for the main stream's event (image ready) on its
        own CUDA stream, then runs CLIP. A request that has not started yet is replaced by a newer one
        (queue length 1).
    poll(): returns (feature (1, D) CPU tensor, step of the image) when a new result is ready, else None.
    Results are always moved to the CPU before being handed over: .cpu() in the background thread waits
    for its own stream, so the main thread only ever sees finished data, never a half-written tensor."""

    def __init__(self, fn, device):
        self._fn = fn
        self._device = device
        self._cuda = device.type == "cuda"
        self._stream = torch.cuda.Stream(device=device) if self._cuda else None
        self._cv = threading.Condition()
        self._pending = None
        self._result = None
        self._epoch = 0          # after clear(), requests sent earlier are dropped even if they finish
        self._stop = False
        self.busy = False
        self.n_done = 0
        self.last_ms = float("nan")
        self.error = None
        self._th = threading.Thread(target=self._run, name="orbit_vlm", daemon=True)
        self._th.start()

    def submit(self, img224, step):
        ev = None
        if self._cuda:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self._device))
        with self._cv:
            self._pending = (img224, ev, int(step), self._epoch)
            self._cv.notify()

    def poll(self):
        with self._cv:
            r, self._result = self._result, None
        return r

    def clear(self):
        with self._cv:
            self._pending = None
            self._result = None
            self._epoch += 1

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                img, ev, step, epoch = self._pending
                self._pending = None
                self.busy = True
            t0 = time.perf_counter()
            try:
                if self._cuda:
                    with torch.cuda.stream(self._stream):
                        self._stream.wait_event(ev)
                        out = self._fn(img).to("cpu")
                else:
                    out = self._fn(img).to("cpu")
            except Exception as e:          # the main thread decides what to do (see OrbitPolicy.vlm_error)
                self.error = repr(e)
                out = None
            self.last_ms = (time.perf_counter() - t0) * 1e3
            with self._cv:
                self.busy = False
                if out is not None and epoch == self._epoch:
                    self._result = (out, step)
                    self.n_done += 1


class OrbitPolicy:
    def __init__(self, bundle_dir, instruction=0, device=None, vae_mode="mean", canon_quat=True,
                 T_hover=0.65, thrust_exponent=1.67, async_vlm=False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = torch.load(os.path.join(bundle_dir, "weights.pt"), map_location="cpu", weights_only=True)
        self.meta = w["meta"]
        m, d = self.meta, w["dims"]
        self.num_obs, self.num_hist, self.num_latent = d["num_obs"], d["num_history"], d["num_latent"]

        self.actor = TaskAwareMoEActor(d["num_obs"], d["vlm_dim"], d["num_actions"], w["cfg_network"],
                                       device=str(self.device))
        self.actor.load_state_dict(w["actor"])
        vc = w["vae_cfg"]
        self.vae = VAE(d["num_obs"], d["num_history"], d["num_latent"], kld_weight=vc["kl_weight"],
                       activation="elu", decoder_hidden_dims=vc["decoder_units"],
                       encoder_hidden_dims=vc["encoder_units"])
        self.vae.load_state_dict(w["vae"])
        self.visual = VisualPerceptionNet(3, d["visual_dim"])
        self.visual.load_state_dict(w["visual_net"])
        self.img_proj = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.Tanh())      # models/clip.py
        self.txt_proj = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.Tanh())
        self.vlm_fc = nn.Linear(64 + 512, d["vlm_dim"])
        self.img_proj.load_state_dict(w["img_proj"]); self.txt_proj.load_state_dict(w["txt_proj"])
        self.vlm_fc.load_state_dict(w["vlm_fc"])
        for mod in (self.actor, self.vae, self.visual, self.img_proj, self.txt_proj, self.vlm_fc):
            mod.to(self.device).eval()
        self.rms_mean = w["obs_rms_mean"].to(self.device)
        self.rms_std = torch.sqrt(w["obs_rms_var"].to(self.device) + 1e-5)
        self.text_embeds = w["text_embeds"].to(self.device)      # (2, 512), already normalised

        from transformers import CLIPVisionModelWithProjection
        clip_dir = os.path.join(bundle_dir, "clip-vit-base-patch16")
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            clip_dir if os.path.isdir(clip_dir) else CLIP_HF_ID).to(self.device).eval()
        self.clip_mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

        self.instruction = int(instruction)
        self.task_name = m["task_names"][self.instruction]
        self.vae_mode = vae_mode
        self.canon_quat = canon_quat            # v1 (world) only; v2's tilt_quat already uses w >= 0
        self.obs_frame = m.get("obs_frame", "world")       # bundles made before this key existed = v1
        if self.obs_frame not in OBS_FRAMES:
            raise ValueError(f"unknown observation layout in the bundle: obs_frame={self.obs_frame!r} (supported: {OBS_FRAMES})")
        self.T_hover, self.p = float(T_hover), float(thrust_exponent)
        # v3 and later: if the bundle meta has thrust_exponent / t_hover_prog, they must equal the constructor
        # arguments (a mismatch raises no error in flight, the drone just flies wrong). The flight node loads
        # the policy during pre-flight checks, so stopping here is safe.
        for key, have, arg in (("thrust_exponent", self.p, "thrust_exponent"), ("t_hover_prog", self.T_hover, "T_hover")):
            if key in m and abs(float(m[key]) - have) > 1e-6:
                raise ValueError(f"the bundle requires {key} = {float(m[key])} but got {have}: pass {arg}={float(m[key])}")
        # v4: height integrator and episode start (training environment: envs/assets/height_integrator.py,
        # ORBIT_START_MODE in envs/drone_vla_orbit.py). Bundles without these keys keep the old behaviour (v1-v3).
        hi = m.get("height_integrator") or {}
        self.hint = None
        if hi.get("enabled", False) is True:
            self.hint = dict(k_I=float(hi["k_I"]), f_max=float(hi["f_max"]), freeze_s=float(hi.get("freeze_s", 0.0)),
                             e_clip=None if hi.get("e_clip") is None else float(hi["e_clip"]),
                             H=float(m["orbit_height"]), dt=float(m["dt"]))
        self.start_mode = m.get("start_mode", "upstream")
        if self.start_mode not in START_MODES:
            raise ValueError(f"unknown start mode in the bundle: start_mode={self.start_mode!r} (supported: {START_MODES})")
        start_f = float(m["start_cmd"][3]) if self.start_mode != "upstream" else float(m["hover_thrust_frac"])
        self.start_action = torch.tensor([0.0, 0.0, 0.0, start_f], device=self.device)
        self.lpf_start = torch.tensor([[0.0, 0.0, 0.0, start_f / 0.25 - 1.0]], device=self.device)   # the action whose mixed thrust = start_f
        self.f_I = 0.0              # height-integrator thrust correction (thrust fraction; 0 at hand-over)
        self.z_obs = None           # height in the latest observation (for the integrator)
        self.vlm_async = AsyncVLM(self._vlm_feature, self.device) if async_vlm else None
        self.vlm_step = -1          # step of the image the current VLM feature came from

    def vlm_lag(self):
        """How many steps behind this step the current VLM feature is (0-9 in synchronous mode)."""
        return self.progress - self.vlm_step

    def vlm_error(self):
        return None if self.vlm_async is None else self.vlm_async.error

    def close(self):
        if self.vlm_async is not None:
            self.vlm_async.close()

    # ---------------------------------------------------------------- perception
    @torch.no_grad()
    def _vlm_feature(self, img224):
        px = (img224 - self.clip_mean) / self.clip_std          # CLIPProcessor on a 224 input: resize/crop are identity, only normalisation is left
        e = self.clip(pixel_values=px).image_embeds
        e = e / e.norm(dim=-1, keepdim=True)
        t = self.text_embeds[self.instruction:self.instruction + 1]
        fused = torch.cat([self.img_proj(e), self.txt_proj(t) * self.meta["text_alpha"]], dim=-1)
        return self.vlm_fc(fused)

    @torch.no_grad()
    def _build_obs(self, st, image):
        img = torch.as_tensor(image, device=self.device, dtype=torch.float32)
        img224 = F.adaptive_avg_pool2d(img.permute(2, 0, 1).unsqueeze(0), (224, 224))   # process_GS_data
        self.visual_info = self.visual(img224)
        if self.vlm_async is not None and self.progress > 0:
            r = self.vlm_async.poll()
            if r is not None:
                self.vlm, self.vlm_step = r[0].to(self.device), r[1]
            if self.progress % self.meta["vlm_every"] == 0:
                self.vlm_async.submit(img224, self.progress)
        elif self.progress % self.meta["vlm_every"] == 0:        # mask = progress_buf % 10 == 0
            self.vlm = self._vlm_feature(img224)
            self.vlm_step = self.progress
        if self.obs_frame == "heading":       # v2: attitude and velocity independent of world yaw (envs/drone_vla_orbit.py)
            q = torch.tensor(tilt_quat(st.quat), device=self.device, dtype=torch.float32)
            v = torch.tensor(heading_frame_vel(st.vel, st.quat), device=self.device, dtype=torch.float32)
        else:                                 # v1: scene-frame quaternion and velocity
            q = torch.tensor(st.quat, device=self.device, dtype=torch.float32)
            if self.canon_quat:
                q = canonical_quat(q)
            v = torch.tensor(st.vel, device=self.device, dtype=torch.float32)
        self.z_obs = float(st.z)
        head = [torch.tensor([st.z, st.vel[2]], device=self.device), q, self.actions, self.prev_actions, v,
                self.visual_info[0]]
        self.obs = torch.cat(head + [self.latent]).unsqueeze(0)
        self.vae_obs = torch.cat(head + [torch.zeros_like(self.latent)]).unsqueeze(0)

    # ---------------------------------------------------------------- control flow
    def reset(self, state, image):
        self.actions = self.start_action.clone()
        self.prev_actions = self.start_action.clone()
        self.prev_prev_actions = self.start_action.clone()
        self.latent = torch.zeros(self.num_latent, device=self.device)
        self.hist_buf = torch.zeros(self.num_hist, self.num_obs, device=self.device)
        self.hist = torch.zeros(1, self.num_hist * self.num_obs, device=self.device)
        self.lpf = None
        self.f_I = 0.0
        self.progress = 0
        if self.vlm_async is not None:
            self.vlm_async.clear()          # drop results not yet collected from before (e.g. warm-up)
        self._build_obs(state, image)
        if self.start_mode == "deploy_warm":
            # v4: as at the start of a training episode, the VAE history holds 5 copies of the hand-over observation
            # and the low-pass filter starts from the hover action (deploy and upstream: all zero, first step uses the raw action)
            self.hist_buf = self.vae_obs[0].unsqueeze(0).repeat(self.num_hist, 1).clone()
            self.hist = self.hist_buf.reshape(1, -1).clone()
            self.lpf = self.lpf_start.clone()

    @torch.no_grad()
    def act(self):
        norm = (self.obs - self.rms_mean) / self.rms_std
        mu, _, _, _ = self.actor(norm, self.vlm, deterministic=True)
        self.last_mu = mu[0].clone()
        a = torch.tanh(mu)
        self.lpf = a.clone() if self.lpf is None else 0.5 * a + 0.5 * self.lpf
        z, (z_mu, _) = self.vae.forward(self.hist)
        self.last_z = (z_mu if self.vae_mode == "mean" else z)[0].clone()
        # ---- from here on: the part of env.step before the physics
        self.latent = self.last_z
        self.hist_buf = torch.roll(self.hist_buf, shifts=-1, dims=0)
        self.hist_buf[-1] = self.vae_obs[0]
        self.hist = self.hist_buf.reshape(1, -1).clone()
        a0 = torch.clamp(self.lpf[0], -1.0, 1.0)
        br = self.meta["br_delay_factor"] * (a0[:3] * 0.5) + (1 - self.meta["br_delay_factor"]) * self.prev_actions[:3]
        br = torch.clamp(br, -0.5, 0.5)
        th = self.meta["thrust_delay_factor"] * ((a0[3:] + 1) * 0.25) + (1 - self.meta["thrust_delay_factor"]) * self.prev_actions[3:]
        new = torch.cat([br, th])
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        self.actions = new
        frac = float(th[0])
        if self.hint is not None:
            # v4: add the height integral f_I to the thrust (= envs/assets/height_integrator.py; self.actions in the observation excludes f_I)
            h = self.hint
            if self.progress * h["dt"] >= h["freeze_s"]:
                e = h["H"] - self.z_obs
                if h["e_clip"] is not None:
                    e = min(max(e, -h["e_clip"]), h["e_clip"])
                self.f_I = min(max(self.f_I + h["k_I"] * h["dt"] * e, -h["f_max"]), h["f_max"])
            frac = min(max(frac + self.f_I, 0.0), 0.5)
        self.progress += 1
        return PX4Command(body_rate_flu=tuple(float(x) for x in br), px4_thrust=self.px4_thrust(frac),
                          sim_thrust_frac=frac, raw_action=tuple(float(x) for x in self.lpf[0]))

    def observe(self, state, image):
        self._build_obs(state, image)

    # ---------------------------------------------------------------- thrust conversion
    def px4_thrust(self, frac):
        """Simulator: force = frac x max_thrust (proportional to the command), hover frac = m g / max_thrust.
        Real drone (THR_MDL_FAC = 0, MC_BAT_SCALE_EN = 0): force ~ throttle^p, hover throttle T_hover.
        So throttle = T_hover x (frac / frac_hover)^(1/p). Defaults: T_hover 0.65, p 1.67 (v4 bundles require 1.3)."""
        r = max(frac, 0.0) / self.meta["hover_thrust_frac"]
        return float(min(1.0, self.T_hover * r ** (1.0 / self.p)))
