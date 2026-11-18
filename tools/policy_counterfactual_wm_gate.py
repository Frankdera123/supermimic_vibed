"""Compare two policies by counterfactual rollouts in one frozen world model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

ap = argparse.ArgumentParser()
ap.add_argument("policy_checkpoints", nargs=2, type=Path)
ap.add_argument("--wm_checkpoint", type=Path, nargs="+", required=True)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--batches", type=int, default=8)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--trace_state", type=Path, default=None,
                 help="fixed_policy_eval trace containing exact initial_state/xi/f")
ap.add_argument("--scores_out", type=Path, default=None,
                 help="optional torch file with per-start WM scores")
args = ap.parse_args()
dev = args.device
wm_ck = torch.load(args.wm_checkpoint[0], map_location=dev, weights_only=False)
base = wm_ck.get("args", {})
buf_state = wm_ck.get("buffer")
if buf_state is None:
    raise SystemExit("WM checkpoint must contain a replay buffer")
T, N = buf_state["data"]["s"].shape[:2]
buf = ReplayBuffer(T, N, dev); buf.load_state_dict(buf_state)
kin = G1Kinematics(dev); motion = MotionLib.from_dir(base.get("motion_dir", "data/lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
goal_frames, hist = int(base.get("goal_frames", 10)), int(base.get("hist", 10))
wm_models=[]
for wm_path in args.wm_checkpoint:
    ck_wm = torch.load(wm_path, map_location=dev, weights_only=False)
    wm_state = ck_wm["wm"]
    hidden = int(wm_state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_state else 0
    model = WorldModel(residual_hidden=hidden).to(dev); model.load_state_dict(wm_state); model.eval()
    wm_models.append(model)
policies = []
for path in args.policy_checkpoints:
    ck = torch.load(path, map_location=dev, weights_only=False)
    p = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev); p.load_state_dict(ck["pi"]); p.eval(); policies.append(p)
totals, term_probs, winner = [], [], []
with torch.no_grad():
    for _ in range(args.batches):
        if args.trace_state is not None:
            tr = torch.load(args.trace_state, map_location=dev, weights_only=False)
            if _ > 0:
                break
            S0 = tuple(x.to(dev) for x in tr["initial_state"])
            xi, f = tr["initial_xi"].to(dev), tr["initial_f"].to(dev)
            a_prev = torch.zeros(xi.shape[0], 29, device=dev)
            batch_size = xi.shape[0]
        else:
            s, h = buf.sample_starts(args.batch, hist)
            S0 = unflatten_state(s["s"]); xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
            batch_size = args.batch
        xi_n = normalize_xi(xi)
        if args.trace_state is not None:
            obs0 = torch.zeros(batch_size, hist, OBS_DIM, device=dev)
        else:
            obs0 = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        wm_returns=[]; wm_terms=[]
        for wm in wm_models:
            batch_returns=[]; batch_terms=[]
            for p in policies:
                S = tuple(x.clone() for x in S0); obs_hist = obs0.clone(); ff=f.clone(); aprev=a_prev.clone()
                ret=torch.zeros(args.batch,device=dev); term=torch.zeros(args.batch,device=dev)
                for _t in range(args.horizon):
                    act=p(obs_hist, motion.goal(ff,goal_frames)).clamp(-20,20)
                    S=wm.step(S,act,xi_n)
                    r=mdp.reward(S,ff,act,aprev); d=mdp.soft_termination(S,ff)
                    ret += r; term += d
                    obs=mdp.obs(S,act,xi,ff); obs_hist=torch.cat((obs_hist[:,1:],obs.unsqueeze(1)),1)
                    ff=ff+1; aprev=act
                batch_returns.append(ret/args.horizon); batch_terms.append(term/args.horizon)
            wm_returns.append(torch.stack(batch_returns)); wm_terms.append(torch.stack(batch_terms))
        batch_returns=torch.stack(wm_returns); batch_terms=torch.stack(wm_terms)
        # Keep the WM axis for calibration, while winner uses pessimistic min.
        pessimistic=batch_returns.min(0).values
        totals.append(batch_returns); term_probs.append(batch_terms)
        winner.append((pessimistic[1] > pessimistic[0]).float())
R=torch.cat(totals,2); D=torch.cat(term_probs,2); W=torch.cat(winner)
mean_r = R.mean(0); std_r = R.std(0, unbiased=False)
calibration = []
for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
    score = mean_r - lam * std_r
    margin = score[1] - score[0]
    calibration.append({"lambda": lam,
                        "policy1_wins_fraction": float((margin > 0).float().mean()),
                        "mean_margin_policy1_minus_policy0": float(margin.mean())})
out={"wm_checkpoints":[str(x) for x in args.wm_checkpoint],"policies":[str(x) for x in args.policy_checkpoints],"samples":int(W.numel()),
     "predicted_return_mean_by_wm": [[float(x) for x in row.mean(1)] for row in R],
     "pessimistic_return_mean":[float(x) for x in R.min(0).values.mean(1)],
     "predicted_term_mean_by_wm": [[float(x) for x in row.mean(1)] for row in D],
     "policy1_wins_fraction_pessimistic":float(W.mean()),
     "mean_pessimistic_margin_policy1_minus_policy0":float((R.min(0).values[1]-R.min(0).values[0]).mean()),
     "mean_std_score_calibration": calibration}
if args.scores_out is not None:
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean_return": mean_r.cpu(), "std_return": std_r.cpu(),
                "pessimistic_return": R.min(0).values.cpu()}, args.scores_out)
if args.trace_state is not None:
    actual = torch.load(args.trace_state, map_location=dev, weights_only=False)["rewards"].float().sum(1)
    actual_margin = actual[1] - actual[0]
    predicted_margin = (mean_r[1] - mean_r[0])
    pred_center = predicted_margin - predicted_margin.mean()
    act_center = actual_margin - actual_margin.mean()
    out["actual_return_mean"] = [float(x) for x in actual.mean(1)]
    out["actual_policy1_win_fraction"] = float((actual_margin > 0).float().mean())
    out["predicted_vs_actual_margin_corr"] = float((pred_center * act_center).mean() / (pred_center.square().mean().sqrt() * act_center.square().mean().sqrt()).clamp_min(1e-8))
    out["predicted_winner_accuracy"] = float(((predicted_margin > 0) == (actual_margin > 0)).float().mean())
print(json.dumps(out,indent=2))
