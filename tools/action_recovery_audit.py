"""Audit action-recovery auxiliary quality and its relation to WM error."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, flatten_state, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--audit_fraction", type=float, default=.25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if ck.get("wm_action_recovery") is None or "buffer" not in ck:
        ap.error("checkpoint must contain wm_action_recovery and replay buffer")
    dev = args.device; a = ck.get("args", {})
    st = ck["buffer"]; T, N = st["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, dev); buf.load_state_dict(st)
    hidden = int(ck["wm_action_recovery"]["0.weight"].shape[0])
    head = nn.Sequential(nn.Linear(142, hidden), nn.SiLU(), nn.Linear(hidden, 29)).to(dev)
    head.load_state_dict(ck["wm_action_recovery"]); head.eval()
    wm_state = ck["wm"]; rh = int(wm_state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_state else 0
    wm = WorldModel(residual_hidden=rh).to(dev); wm.load_state_dict(wm_state); wm.eval()
    kin = G1Kinematics(dev); motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, dev)
    MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    n_audit = max(1, int(round(N * args.audit_fraction))); n_audit = min(n_audit, N - 1)
    groups = {"train": torch.arange(0, N - n_audit, device=dev), "audit": torch.arange(N - n_audit, N, device=dev)}
    out = {"checkpoint": str(args.checkpoint), "groups": {}}
    for name, env_ids in groups.items():
        recs, wms = [], []
        for _ in range(args.batches):
            w = buf.sample_windows(args.batch, 1, env_ids=env_ids)
            with torch.no_grad():
                s0 = unflatten_state(w["s"][:, 0])
                pred = wm.step(s0, w["a"][:, 0], normalize_xi(w["xi"][:, 0]), use_residual=rh > 0)
                sf = flatten_state(s0); pf = flatten_state(pred)
                apred = head(torch.cat((sf, pf), -1))
                rec = ((apred - w["a"][:, 0]) / 20.0).square().mean(-1)
                std = torch.sqrt(wm.out_norm.var + wm.out_norm.eps ** 2).clamp_min(1e-4)
                err = (wm.state_error(pred, unflatten_state(w["s1"][:, 0])) / std).square().mean(-1)
                recs.append(rec); wms.append(err)
        rec, err = torch.cat(recs), torch.cat(wms)
        rc, ec = rec - rec.mean(), err - err.mean()
        corr = (rc * ec).mean() / (rc.square().mean().sqrt() * ec.square().mean().sqrt()).clamp_min(1e-8)
        out["groups"][name] = {"samples": int(rec.numel()), "action_mse": float(rec.mean()),
                                "wm_error": float(err.mean()), "corr_action_mse_wm_error": float(corr),
                                "action_mse_p90": float(torch.quantile(rec, .9)),
                                "wm_error_p90": float(torch.quantile(err, .9))}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
