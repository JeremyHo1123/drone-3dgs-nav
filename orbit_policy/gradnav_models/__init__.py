"""grad_nav's model code (upstream Qianzhong-Chen/grad_nav @ 95218ab), copied for inference.

actor_vla_moe.py, moe.py, model_utils.py and vae.py are identical to upstream, except that the two
`from models import ...` lines in actor_vla_moe.py are relative imports (so the whole grad_nav repo is not needed).
squeeze_net_nodl.py is new: same structure and parameter names as upstream models/squeeze_net.py,
but without torchvision and without downloading ImageNet weights (all weights come from the bundle).
These files are under the NVIDIA Source Code License (non-commercial), see ../LICENSE-grad_nav.md.
"""
