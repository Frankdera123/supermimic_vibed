"""Audit same-state counterfactual action-transition discrimination."""
from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--audit_fraction", type=float, default=.25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    a = ck.get("args", {})
    if "wm_action_transition_encoder" not in ck or "wm_action_encoder" not in ck:
        ap.error("checkpoint must contain contrastive action heads")
    dev = args.device
    st = ck["buffer"]["data"]
    T, N = st["s"].shape[:2]
    buf = ReplayBuffer(T, N, dev)
    buf.load_state_dict(ck["buffer"])
    th = int(ck["wm_action_transition_encoder"]["0.weight"].shape[0])
    ah = int(ck["wm_action_encoder"]["0.weight"].shape[0])
    trans_dim = 70 if a.get("wm_action_contrastive_delta", False) else 142
    td = int(ck["wm_action_transition_encoder"]["2.weight"].shape[0])
    ad = int(ck["wm_action_encoder"]["2.weight"].shape[0])
    trans = nn.Sequential(nn.Linear(trans_dim, th), nn.SiLU(), nn.Linear(th, td)).to(dev)
    act = nn.Sequential(nn.Linear(29, ah), nn.SiLU(), nn.Linear(ah, ad)).to(dev)
    trans.load_state_dict(ck["wm_action_transition_encoder"])
    act.load_state_dict(ck["wm_action_encoder"])
    trans.eval(); act.eval()
    wm_state = ck["wm"]
    rh = int(wm_state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_state else 0
    wm = WorldModel(residual_hidden=rh).to(dev)
    wm.load_state_dict(wm_state); wm.eval()
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, dev)
    MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    n_audit = max(1, min(int(round(N * args.audit_fraction)), N - 1))
    groups = {"train": torch.arange(0, N - n_audit, device=dev),
              "audit": torch.arange(N - n_audit, N, device=dev)}
    temp = float(a.get("wm_action_contrastive_temperature", .1))
    normalize_actions = bool(a.get("wm_action_contrastive_normalize_actions", False))
    out = {"checkpoint": str(args.checkpoint), "groups": {}}
    for name, env_ids in groups.items():
        correct, margins, real_correct, real_margins = [], [], [], []
        pos_sims, neg_sims, wm_errs = [], [], []
        for _ in range(args.batches):
            w = buf.sample_windows(args.batch, 1, env_ids=env_ids)
            with torch.no_grad():
                s0 = unflatten_state(w["s"][:, 0])
                xi = normalize_xi(w["xi"][:, 0])
                action = w["a"][:, 0]
                neg_action = action[torch.randperm(action.shape[0], device=dev)]
                pos_state = wm.step(s0, action, xi, use_residual=rh > 0)
                neg_state = wm.step(s0, neg_action, xi, use_residual=rh > 0)
                cur = flatten_state(s0)
                if a.get("wm_action_contrastive_delta", False):
                    std = torch.maximum(wm.out_norm.std(), torch.tensor(1e-6, device=dev))
                    pos_feat = wm.state_error(pos_state, s0) / std
                    neg_feat = wm.state_error(neg_state, s0) / std
                else:
                    pos_feat = torch.cat((cur, flatten_state(pos_state)), -1)
                    neg_feat = torch.cat((cur, flatten_state(neg_state)), -1)
                ae = action / 20.0 if normalize_actions else action
                nae = neg_action / 20.0 if normalize_actions else neg_action
                zpos = torch.nn.functional.normalize(trans(pos_feat), dim=-1)
                zneg = torch.nn.functional.normalize(trans(neg_feat), dim=-1)
                real_state = unflatten_state(w["s1"][:, 0])
                if a.get("wm_action_contrastive_delta", False):
                    real_feat = wm.state_error(real_state, s0) / std
                else:
                    real_feat = torch.cat((cur, flatten_state(real_state)), -1)
                zreal = torch.nn.functional.normalize(trans(real_feat), dim=-1)
                ztrue = torch.nn.functional.normalize(act(ae), dim=-1)
                zfalse = torch.nn.functional.normalize(act(nae), dim=-1)
                pos_true = (zpos * ztrue).sum(-1)
                pos_false = (zpos * zfalse).sum(-1)
                neg_true = (zneg * ztrue).sum(-1)
                neg_false = (zneg * zfalse).sum(-1)
                correct.append(torch.cat((pos_true > pos_false, neg_false > neg_true)))
                margins.append(torch.cat((pos_true - pos_false, neg_false - neg_true)))
                real_true = (zreal * ztrue).sum(-1)
                real_false = (zreal * zfalse).sum(-1)
                real_correct.append(real_true > real_false)
                real_margins.append(real_true - real_false)
                pos_sims.append(torch.cat((pos_true, neg_false)))
                neg_sims.append(torch.cat((pos_false, neg_true)))
                wm_errs.append(wm.state_error(pos_state, unflatten_state(w["s1"][:, 0])).square().mean(-1))
        c = torch.cat(correct); m = torch.cat(margins)
        rc = torch.cat(real_correct); rm = torch.cat(real_margins)
        ps = torch.cat(pos_sims); ns = torch.cat(neg_sims)
        e = torch.cat(wm_errs)
        # Margins contain positive and negative cases; factual WM error is
        # defined only for the positive transition, so correlate that half.
        mg = m[:e.numel()]
        mc, ec = mg - mg.mean(), e - e.mean()
        corr = (mc * ec).mean() / (mc.square().mean().sqrt() * ec.square().mean().sqrt()).clamp_min(1e-8)
        out["groups"][name] = {
            "samples": int(c.numel()), "accuracy": float(c.float().mean()),
            "margin_mean": float(m.mean()), "margin_p10": float(torch.quantile(m, .1)),
            "real_transition_accuracy": float(rc.float().mean()),
            "real_transition_margin_mean": float(rm.mean()),
            "positive_similarity": float(ps.mean()), "negative_similarity": float(ns.mean()),
            "margin_wm_error_corr": float(corr),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
