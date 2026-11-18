"""PADA bridge: paths, code-variant loading, and diag-pair bundle validation.

PADA reuses the CRGF core (`crgf_proto/supermimic/crgf.py`) on top of the OPCL
proto package variant (acceleration-level WM, OBS 93, state-relative goal) --
exactly the combination the CRGF audits used for the B0/B1/B2 checkpoints.  The
two `supermimic` package roots are merged as a namespace package: proto first
(its mdp/networks/motion/env win), crgf_proto second (provides `crgf.py`).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_TORCH = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(*candidates: str) -> str | None:
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def repo_root() -> Path:
    p = _first_existing("/workspace/supermimic", "/home/sustech/Desktop/supermimic_vibed")
    if p is None:
        raise FileNotFoundError("supermimic_vibed repo not found")
    return Path(p)


def proto_root() -> Path:
    p = _first_existing(
        "/workspace/opcl_work/supermimic_opcl_proto",
        "/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/supermimic_opcl_proto",
    )
    if p is None:
        raise FileNotFoundError("OPCL proto package not found")
    return Path(p)


def pairs_root() -> Path:
    p = _first_existing(
        "/workspace/opcl_work/opcl_autonomous_20261117/server_artifacts",
        "/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts",
    )
    if p is None:
        raise FileNotFoundError("diag pair server_artifacts not found")
    return Path(p)


def import_stack() -> None:
    """Install sys.path so `supermimic.*` resolves to the proto variant with
    `supermimic.crgf` coming from crgf_proto (namespace-package merge).

    Installs a SAFE isaaclab stub when isaaclab is unavailable (CPU hosts).
    The crgf_proto stub passes every missing attribute through a permissive
    placeholder, including dunders like `__file__`; on torch>=2.13 the first
    Adam construction imports torch._dynamo, whose `@torch.library.custom_op`
    source introspection then walks sys.modules and crashes on such a module.
    Our stub keeps the permissive attribute behavior for normal names but
    exposes a real `__file__` and raises AttributeError on dunder/private
    lookups.  `_crgf_bootstrap.install` accepts any pre-installed module
    flagged `__crgf_stub__`, so CRGF behavior is unchanged.
    """
    repo = repo_root()
    proto = proto_root()
    crgf_pkg = repo / "crgf_proto"
    for p in (str(repo / "pada_proto"), str(repo / "pada_proto" / "tools"),
              str(crgf_pkg), str(crgf_pkg / "tools"), str(proto)):
        if p not in sys.path:
            sys.path.insert(0, p)
    here = crgf_pkg / "tools"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        import isaaclab.sim  # noqa: F401
        return
    except Exception:
        pass
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "__crgf_stub__", False):
        return
    import types

    from _crgf_bootstrap import _AnyCfg, _StubModule

    stub = _StubModule("isaaclab")
    stub.__crgf_stub__ = True
    stub.__file__ = str(repo / "pada_proto" / "pada_bridge.py")  # real path for inspect
    _orig_getattr = stub.__getattr__

    def _safe_getattr(name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _orig_getattr(name)

    stub.__getattr__ = _safe_getattr  # type: ignore[method-assign]
    sys.modules["isaaclab"] = stub
    for name in ("isaaclab.sim", "isaaclab.actuators", "isaaclab.assets"):
        sub = _StubModule(name)
        sub.__file__ = stub.__file__
        sub.__getattr__ = _safe_getattr  # type: ignore[method-assign]
        sys.modules[name] = sub


def fix_paths() -> None:
    from pathlib import Path as _P

    from _crgf_bootstrap import fix_data_paths

    fix_data_paths(_P(repo_root() / "crgf_proto"))


def torch_load(path):
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH.load(path, map_location="cpu", weights_only=False)


def load_pair_bundle(run: str, iteration: int, device: str = "cpu", strict: bool = True) -> dict:
    """Load and VALIDATE one diag pair bundle (fail-closed).

    Checks (per PADA0_EXPERIMENT_PROTOCOL §1):
      same_policy_update_verified, pair_validation_reason, pi_normalizer_changed,
      strict state-dict loads of pi_old/pi_new/wm/critic into the proto networks.
    """
    import torch

    import_stack()
    fix_paths()
    from supermimic.networks import Critic, Policy, WorldModel  # proto variant

    path = pairs_root() / run / "diag_pairs" / f"pair_it_{iteration:06d}.pt"
    bundle = torch_load(path)
    problems = []
    if not bundle.get("same_policy_update_verified", False):
        problems.append("same_policy_update_verified != true")
    if bundle.get("pair_validation_reason") != "VALID_SINGLE_POLICY_UPDATE":
        problems.append(f"validation={bundle.get('pair_validation_reason')}")
    if bundle.get("is_synthetic", False):
        problems.append("bundle flags itself synthetic")
    if bundle.get("pi_normalizer_changed", False):
        problems.append("pi normalizer changed across the update")
    if int(bundle["pi_updates_after"]) - int(bundle["pi_updates_before"]) != 1:
        problems.append("pi_updates delta != 1")

    hist = int(bundle["args"].get("hist", 10))
    # proto dims (OBS 93 without the reference-relative block; goal 10x73)
    from supermimic.mdp import OBS_DIM
    from supermimic.motion import GOAL_FRAMES

    goal_dim = GOAL_FRAMES * 73
    pi_old = Policy(OBS_DIM, goal_dim, hist=hist)
    pi_new = Policy(OBS_DIM, goal_dim, hist=hist)
    try:
        wm = WorldModel(residual_hidden=0)
    except TypeError:
        wm = WorldModel()
    pi_old.load_state_dict(bundle["pi_old"], strict=True)
    pi_new.load_state_dict(bundle["pi_new"], strict=True)
    wm.load_state_dict(bundle["wm"], strict=True)
    critic_in = None
    critic = None
    if "critic" in bundle:
        critic_in = bundle["critic"]["norm.mean"].shape[0]
        critic = Critic(critic_in)
        critic.load_state_dict(bundle["critic"], strict=True)
    for m in (pi_old, pi_new, wm, critic):
        if m is not None:
            m.eval().to(device)
    if problems:
        if strict:
            raise ValueError(f"pair {run}@{iteration} failed validation: {problems}")
        print(json.dumps({"pair": f"{run}@{iteration}", "warnings": problems}))
    return {
        "run": run, "iteration": iteration, "path": str(path),
        "sha256": sha256(path),
        "pi_updates_before": int(bundle["pi_updates_before"]),
        "pi_updates_after": int(bundle["pi_updates_after"]),
        "pi_param_l2_diff": float(bundle.get("pi_param_l2_diff", float("nan"))),
        "hist": hist, "obs_dim": OBS_DIM, "goal_dim": goal_dim,
        "critic_in_dim": critic_in,
        "pi_old": pi_old, "pi_new": pi_new, "wm": wm, "critic": critic,
        "args": dict(bundle["args"]),
    }


def setup_world(motion_dir: str | None = None, device: str = "cpu", pair: dict | None = None):
    """Build kin/motion/mdp/proto shared by all PADA tools (proto variant)."""
    import_stack()
    fix_paths()
    import torch

    from supermimic.crgf import Protocol
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot import default_joint_pos_list

    dev = device
    md = motion_dir or str(repo_root() / "data" / "lafan1_g1")
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(md, kin, dev)
    default_q = torch.tensor(default_joint_pos_list(), device=dev)
    a = (pair or {}).get("args", {})
    try:
        mdp = MDP(kin, motion, default_q,
                  phase_features=bool(a.get("phase_conditioned_policy", False)),
                  clip_features=bool(a.get("clip_conditioned_policy", False)))
    except TypeError:
        mdp = MDP(kin, motion, default_q)
    from supermimic.motion import GOAL_FRAMES

    proto = Protocol(mdp, GOAL_FRAMES)
    return {"kin": kin, "motion": motion, "mdp": mdp, "proto": proto, "device": dev,
            "motion_dir": md}


def select_starts(motion, n_clips: int, frames_per_clip: int, seed: int, margin: int = 64):
    """Deterministic clip/frame start selection (CRGF margin convention)."""
    import torch

    names = motion.names
    g = torch.Generator().manual_seed(seed)
    clip_ids = torch.randperm(len(names), generator=g)[:n_clips].sort().values.tolist()
    starts = []
    for c in clip_ids:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - margin - 1)
        n = min(frames_per_clip, hi - s0)
        frames = s0 + 1 + torch.randperm(hi - s0 - 1, generator=g)[:n].sort().values
        starts.append({"clip": c, "clip_name": names[c], "frames": frames.tolist()})
    return starts
