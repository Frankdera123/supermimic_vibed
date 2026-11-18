"""Generate reusable simulator counterfactual transition pairs for WM calibration."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, flatten_state
from supermimic.rotations import mat_to_quat
from supermimic.mdp import OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--samples", type=int, default=256)
ap.add_argument("--directions", type=int, default=4)
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--action_scale", type=float, default=2.0)
ap.add_argument("--policy_checkpoint", type=Path, default=None,
                help="optional checkpoint; use its policy actions as counterfactual centers")
ap.add_argument("--policy_burnin_steps", type=int, default=0,
                help="roll policy forward this many steps before collecting counterfactual state")
ap.add_argument("--delta", type=float, nargs="+", default=[.1, .5, 1.0])
ap.add_argument("--counterfactual_horizon", type=int, default=1,
                help="repeat each plus/minus action sequence for this many simulator steps")
ap.add_argument("--eval_motion_dir", type=str, default="data/lafan1_g1_test_s4")
add_launcher_args(ap)
args = ap.parse_args()
if args.counterfactual_horizon < 1:
    ap.error("--counterfactual_horizon must be positive")
dev = args.device; B = args.samples; K = args.directions
gen = torch.Generator(device=dev).manual_seed(args.seed)
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.eval_motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
valid = torch.where(motion.clip_len >= 8)[0]
clip = valid[torch.randint(valid.numel(), (B,), generator=gen, device=dev)]
off = torch.randint(1, 4, (B,), generator=gen, device=dev)
frames = motion.clip_start[clip] + off
xi = torch.zeros(B, 33, device=dev); xi[:, 0] = 1.0
if args.policy_checkpoint is None:
    action = torch.randn(B, 29, generator=gen, device=dev) * args.action_scale
else:
    ck = torch.load(args.policy_checkpoint, map_location=dev, weights_only=False)
    ck_args = ck.get("args", {})
    goal_frames = int(ck_args.get("goal_frames", 10)); hist = int(ck_args.get("hist", 10))
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev)
    pi.load_state_dict(ck["pi"]); pi.eval()
    out_policy = str(args.policy_checkpoint)
directions = torch.randn(B, K, 29, generator=gen, device=dev)
directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
total = (1 + 2*K) * B
sim_cfg = make_sim(dev, SimpleNamespace(num_envs=total))
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, total, motion, mdp,
                      gamma=float(ck_args.get("gamma", .99)) if args.policy_checkpoint is not None else .99,
                      eval_mode=True, reset_noise=False)
    if args.policy_checkpoint is not None:
        # Use the same observation context as strategic_error_audit: zero
        # history with the current simulator observation in the last slot.
        env.reset(torch.arange(B, device=dev), frames)
        env.refresh()
        histories = torch.zeros(B, hist, OBS_DIM, device=dev)
        # Roll out the current policy before taking the counterfactual slice.
        for _ in range(args.policy_burnin_steps + 1):
            state0 = tuple(x[:B] for x in env.state())
            obs = mdp.obs(state0, env.a_prev[:B], env.xi[:B], env.f[:B])
            histories[:, :-1] = histories[:, 1:].clone()
            histories[:, -1] = obs
            goals = motion.goal(env.f[:B], goal_frames)
            with torch.no_grad():
                action = pi(histories, goals).clamp(-20, 20)
            if _ < args.policy_burnin_steps:
                env.step(torch.cat((action, torch.zeros(total - B, 29, device=dev)), 0))
    pairs = []
    for delta in args.delta:
        if args.policy_checkpoint is None or args.policy_burnin_steps == 0:
            env.reset(torch.arange(total, device=dev), frames.repeat(1 + 2*K))
        else:
            # Replicate the on-policy simulator state into plus/minus slots.
            S = env.state()
            rep = torch.arange(total, device=dev)
            Sr = tuple(x[:B].repeat(1 + 2*K, *([1] * (x.ndim - 1))) for x in S)
            quat = mat_to_quat(Sr[1])
            env.robot.write_root_link_pose_to_sim_index(
                root_pose=torch.cat((Sr[0] + env.scene.env_origins[rep], quat[:, 1:4], quat[:, 0:1]), -1), env_ids=rep)
            env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((Sr[2], Sr[3]), -1), env_ids=rep)
            env.robot.write_joint_position_to_sim_index(position=Sr[4], env_ids=rep)
            env.robot.write_joint_velocity_to_sim_index(velocity=Sr[5], env_ids=rep)
            env.f.copy_(env.f[:B].repeat(1 + 2*K)); env.ep_step.zero_()
            env.a_prev.zero_()
            env.robot.set_joint_position_target_index(target=env.default_q + env.xi[:, 1:30], env_ids=rep)
        env.xi.copy_(xi.repeat(1 + 2*K, 1)); env.refresh()
        acts = [action]
        for k in range(K):
            acts.extend((action + delta * directions[:, k], action - delta * directions[:, k]))
        s0 = env.state()
        for _ in range(args.counterfactual_horizon):
            env.step(torch.cat(acts, 0))
        s1 = env.state()
        pairs.append({"delta": float(delta), "s0": flatten_state(tuple(x[:B] for x in s0)).cpu(),
                      "plus": torch.stack([flatten_state(tuple(x[(1+2*k)*B:(2+2*k)*B] for x in s1)).cpu() for k in range(K)], 1),
                      "minus": torch.stack([flatten_state(tuple(x[(2+2*k)*B:(3+2*k)*B] for x in s1)).cpu() for k in range(K)], 1)})
    out = {"frames": frames.cpu(), "xi": xi.cpu(), "action": action.cpu(),
           "directions": directions.cpu(), "deltas": list(args.delta),
           "eval_motion_dir": args.eval_motion_dir, "seed": args.seed,
           "counterfactual_horizon": args.counterfactual_horizon,
           "pairs": pairs}
    if args.policy_checkpoint is not None:
        out["policy_checkpoint"] = out_policy
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, args.output)
print(f"saved {args.output} samples={B} directions={K} deltas={args.delta}")
