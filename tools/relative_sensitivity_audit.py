"""Compare WM action sensitivity on identical replay starts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

def wm_from(ck, dev):
    h = int(ck["wm"]["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in ck["wm"] else 0
    m = WorldModel(residual_hidden=h).to(dev); m.load_state_dict(ck["wm"]); m.eval(); return m

def starts(buf, motion, B, hist, horizon):
    while True:
        s, h = buf.sample_starts(2 * B, hist)
        ok = s["f"] + horizon < motion.clip_end[motion.clip_id[s["f"]]]
        ix = ok.nonzero(as_tuple=False).squeeze(-1)[:B]
        if len(ix) == B:
            return ({k: v[ix] for k, v in s.items()}, {k: v[ix] for k, v in h.items()})

def sensitivity(wm, pi, mdp, motion, s, h, horizon, probe):
    B = s["f"].shape[0]; S = unflatten_state(s["s"]); xi, f = s["xi"], s["f"]
    hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    goal = motion.goal(f, pi.goal_frames); acts = []
    with torch.no_grad():
        for _ in range(horizon):
            act = pi(hist, goal).detach().requires_grad_(True); acts.append(act)
            S = wm.step(S, act, normalize_xi(xi), use_residual=wm.residual_net is not None)
            f = f + 1; hist = torch.cat((hist[:, 1:], mdp.obs(S, act, xi, f).unsqueeze(1)), 1); goal = motion.goal(f, pi.goal_frames)
    S = unflatten_state(s["s"])
    for act in acts: S = wm.step(S, act, normalize_xi(xi), use_residual=wm.residual_net is not None)
    grads = torch.autograd.grad((invariant_features(S) * probe).sum(), acts, allow_unused=True)
    return torch.stack([(g.square().mean(-1).sqrt() if g is not None else torch.zeros(B, device=probe.device)) for g in grads], 1).square().sum(1).sqrt().detach()

ap = argparse.ArgumentParser(); ap.add_argument("reference"); ap.add_argument("candidates", nargs="+"); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--horizon", type=int, default=16); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max_ratio", type=float, default=1.5); ap.add_argument("--min_ratio", type=float, default=.67)
a = ap.parse_args(); dev = a.device
ref = torch.load(a.reference, map_location=dev, weights_only=False); meta = ref.get("args", {}); hist = int(meta.get("hist", 4)); gf = int(meta.get("goal_frames", 4))
state = ref["buffer"]; T, N = state["data"]["s"].shape[:2]; buf = ReplayBuffer(T, N, dev); buf.load_state_dict(state)
kin = G1Kinematics(dev); motion = MotionLib.from_dir(meta.get("motion_dir", "data/lafan1_g1"), kin, dev); mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
pi = Policy(OBS_DIM, 73 * gf, hist=hist).to(dev); pi.load_state_dict(ref["pi"]); pi.eval(); pi.goal_frames = gf
torch.manual_seed(0); s, h = starts(buf, motion, a.batch, hist, a.horizon); probe = torch.randn(a.batch, invariant_features(unflatten_state(s["s"])).shape[-1], device=dev)
base = sensitivity(wm_from(ref, dev), pi, mdp, motion, s, h, a.horizon, probe); out = {"reference": a.reference, "reference_mean": float(base.mean()), "results": []}
for p in a.candidates:
    x = torch.load(p, map_location=dev, weights_only=False); z = sensitivity(wm_from(x, dev), pi, mdp, motion, s, h, a.horizon, probe); ratio = z / base.clamp_min(1e-6)
    out["results"].append({"checkpoint": p, "mean": float(z.mean()), "ratio_mean": float(ratio.mean()), "ratio_p10": float(torch.quantile(ratio, .1)), "ratio_p90": float(torch.quantile(ratio, .9)), "gate_pass": bool((ratio >= a.min_ratio).all() and (ratio <= a.max_ratio).all())})
print(json.dumps(out, indent=2))
