"""Collect policy-occupied factual transitions from Isaac for critic fitting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, flatten_state, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.strategic_critic import StrategicErrorCritic


ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--motion_dir", type=str, default=None)
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--action_candidates", type=int, default=1,
                help="counterfactual actions per sampled state (1 keeps rollout mode)")
ap.add_argument("--candidate_noise", type=float, default=.5,
                help="normalized action noise std for counterfactual candidates")
ap.add_argument("--critic_checkpoint", type=Path, default=None,
                help="optional StrategicErrorCritic checkpoint; keep only top candidate per state")
ap.add_argument("--critic_action_penalty", type=float, default=0.0,
                help="penalty on squared candidate deviation from base action")
ap.add_argument("--critic_reward_weight", type=float, default=0.0,
                help="weight for WM-predicted reward delta versus base action")
ap.add_argument("--critic_reward_horizon", type=int, choices=(1, 2), default=1,
                help="predicted reward horizon used by the reward-aware selector")
ap.add_argument("--seed", type=int, default=0,
                help="torch seed for reproducible frame and candidate sampling")
add_launcher_args(ap)
args = ap.parse_args(); dev = args.device
torch.manual_seed(args.seed)
if (args.num_envs < 1 or args.steps < 1 or args.action_candidates < 1
        or args.candidate_noise < 0.0 or args.critic_action_penalty < 0.0
        or args.critic_reward_weight < 0.0):
    ap.error("num_envs/steps/action_candidates must be positive and candidate_noise nonnegative")
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
base = ck.get("args", {})
motion_dir = args.motion_dir or base.get("motion_dir")
if not motion_dir:
    ap.error("--motion_dir is required when checkpoint args have no motion_dir")
goal_frames, hist = int(base.get("goal_frames", 10)), int(base.get("hist", 10))
kin = G1Kinematics(dev); motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
policy = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev)
policy.load_state_dict(ck["pi"]); policy.eval()
critic = None
critic_mean = critic_std = None
wm = None
if args.critic_reward_weight > 0.0:
    st = ck["wm"]; key = "residual_net.0.weight"
    rh = int(st[key].shape[0]) if key in st else 0
    wm = WorldModel(residual_hidden=rh).to(dev); wm.load_state_dict(st); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
if args.critic_checkpoint is not None:
    cck = torch.load(args.critic_checkpoint, map_location=dev, weights_only=False)
    critic = StrategicErrorCritic(hidden=int(cck.get("hidden", 256))).to(dev)
    critic.load_state_dict(cck["model"]); critic.eval()
    critic_mean = cck["mean"].to(dev); critic_std = cck["std"].to(dev)
    critic_target_mean = torch.as_tensor(cck.get("target_mean", 0.0), device=dev)
    critic_target_std = torch.as_tensor(cck.get("target_std", 1.0), device=dev).clamp_min(1e-3)
    critic_target_min = torch.as_tensor(cck.get("target_min", -float("inf")), device=dev)
    critic_target_max = torch.as_tensor(cck.get("target_max", float("inf")), device=dev)
groups = args.action_candidates
if args.num_envs % groups:
    ap.error("num_envs must be divisible by action_candidates")
n_states = args.num_envs // groups
sim_cfg = make_sim(dev, SimpleNamespace(num_envs=args.num_envs))
with launch_simulation(sim_cfg, args):
    counterfactual = groups > 1
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp,
                      gamma=float(base.get("gamma", .99)), eval_mode=counterfactual,
                      reset_noise=not counterfactual)
    obs_hist = torch.zeros(args.num_envs, hist, OBS_DIM, device=dev)
    rows = {k: [] for k in ("s", "a", "xi", "s1", "f", "done")}
    probe_scores = []
    real_reward_deltas = []
    with torch.no_grad():
        for _ in range(args.steps):
            if counterfactual:
                frames, _ = env.sample_frames(n_states)
                frames = frames.repeat_interleave(groups)
                env.reset(torch.arange(args.num_envs, device=dev), frames)
                env.refresh()
                obs_hist.zero_()
            S = env.state()
            xi_before = env.xi.clone()
            a_prev_before = env.a_prev.clone()
            f_before = env.f.clone()
            obs = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(obs_hist),
                torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1))
            goal = motion.goal(env.f, goal_frames)
            base_action = policy(obs_hist, goal).clamp(-20.0, 20.0)
            if counterfactual:
                base_action = base_action.view(n_states, groups, -1)
                noise = torch.randn_like(base_action) * (args.candidate_noise * 20.0)
                noise[:, 0].zero_()
                action = (base_action + noise).clamp(-20.0, 20.0).reshape(args.num_envs, -1)
            else:
                action = base_action
            out = env.step(action)
            keep = None
            if critic is not None:
                state_flat = flatten_state(S).view(n_states, groups, -1)[:, 0]
                xi_flat = env.xi.view(n_states, groups, -1)[:, 0]
                cand_feat = critic.features(
                    state_flat[:, None].expand(-1, groups, -1).reshape(-1, state_flat.shape[-1]),
                    action.view(n_states, groups, -1).reshape(-1, action.shape[-1]),
                    xi_flat[:, None].expand(-1, groups, -1).reshape(-1, xi_flat.shape[-1]))
                score = (critic((cand_feat - critic_mean) / critic_std)
                         * critic_target_std + critic_target_mean).clamp(
                             critic_target_min, critic_target_max).view(n_states, groups)
                if args.critic_action_penalty > 0.0:
                    score = score - args.critic_action_penalty * (
                        (action.view(n_states, groups, -1) - base_action) / 20.0
                    ).square().mean(-1)
                if args.critic_reward_weight > 0.0:
                    with torch.no_grad():
                        base_action_flat = base_action.reshape(args.num_envs, -1)
                        xi_norm = normalize_xi(xi_before)
                        pred = wm.step(S, action, xi_norm)
                        base_pred = wm.step(S, base_action_flat, xi_norm)
                        pred_r = mdp.reward(pred, f_before + 1, action, a_prev_before)
                        base_r = mdp.reward(base_pred, f_before + 1, base_action_flat, a_prev_before)
                        if args.critic_reward_horizon == 2:
                            pred_o = mdp.obs(pred, action, xi_before, f_before + 1)
                            base_o = mdp.obs(base_pred, base_action_flat, xi_before, f_before + 1)
                            pred_hist = torch.cat((obs_hist[:, 1:], pred_o[:, None]), dim=1)
                            base_hist = torch.cat((obs_hist[:, 1:], base_o[:, None]), dim=1)
                            next_goal = motion.goal(f_before + 1, goal_frames)
                            pred_a2 = policy(pred_hist, next_goal).clamp(-20.0, 20.0)
                            base_a2 = policy(base_hist, next_goal).clamp(-20.0, 20.0)
                            pred2 = wm.step(pred, pred_a2, xi_norm)
                            base2 = wm.step(base_pred, base_a2, xi_norm)
                            pred_r = pred_r + float(base.get("gamma", .99)) * mdp.reward(
                                pred2, f_before + 2, pred_a2, action)
                            base_r = base_r + float(base.get("gamma", .99)) * mdp.reward(
                                base2, f_before + 2, base_a2, base_action_flat)
                        reward_delta = ((pred_r - base_r) / 0.1).view(n_states, groups)
                    score = score + args.critic_reward_weight * reward_delta
                best = score.argmax(-1)
                keep = torch.arange(n_states, device=dev) * groups + best
                probe_scores.append(score[torch.arange(n_states, device=dev), best].detach().cpu())
                real_r = out["r"].view(n_states, groups)
                real_reward_deltas.append((real_r[torch.arange(n_states, device=dev), best]
                                           - real_r[:, 0]).detach().cpu())
            else:
                probe_scores.append(torch.ones(out["s"].shape[0]))
            for key in rows:
                value = out[key] if keep is None else out[key][keep]
                rows[key].append(value.detach().cpu())
    payload = {k: torch.cat(v, 0) for k, v in rows.items()}
    payload["probe_score"] = torch.cat(probe_scores, 0)
    metadata = {"checkpoint": str(args.checkpoint),
                "num_envs": args.num_envs, "steps": args.steps,
                "motion_dir": motion_dir,
                "critic_reward_weight": args.critic_reward_weight,
                "critic_reward_horizon": args.critic_reward_horizon,
                "seed": args.seed}
    if real_reward_deltas:
        reward_delta = torch.cat(real_reward_deltas)
        metadata.update({"selected_real_reward_delta_mean": float(reward_delta.mean()),
                         "selected_real_reward_delta_p50": float(torch.quantile(reward_delta, .5)),
                         "selected_real_reward_delta_positive_fraction": float((reward_delta > 0).float().mean())})
    payload["metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {len(payload['s'])} policy-occupied transitions to {args.output}")
