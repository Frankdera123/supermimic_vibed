"""Evaluate saved policies in the real simulator from identical fixed motion starts."""
from __future__ import annotations
import argparse, json, os, sys
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import add_launcher_args, launch_simulation
import torch
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.mdp import normalize_xi
from supermimic.robot import default_joint_pos_list

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", type=Path)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--perturb", action="store_true",
                    help="apply the same random friction and joint-offset context to all policies")
parser.add_argument("--eval_motion_dir", type=str, default=None,
                    help="optional motion directory for evaluation (overrides checkpoint training directory)")
parser.add_argument("--clip_name", type=str, default=None,
                    help="evaluate starts from one exact motion clip name")
parser.add_argument("--starts_file", type=Path, default=None,
                    help="torch file with fixed clip/off tensors for paired evaluation")
parser.add_argument("--write_starts", type=Path, default=None,
                    help="save the sampled clip/off tensors and exit before launching Isaac")
parser.add_argument("--json_out", type=Path, default=None,
                    help="optional path for the clean evaluation JSON")
parser.add_argument("--trace_out", type=Path, default=None,
                    help="optional torch file with per-step features/rewards for ungated policies")
parser.add_argument("--oracle_reference_action", action="store_true",
                    help="replace the learned policy with the one-step reference joint target (diagnostic)")
parser.add_argument("--gate_mode", choices=("none", "max_error", "error_trend"), default="none",
                    help="with exactly two checkpoints, select by current error or one-step height/tilt error trend")
parser.add_argument("--blend_alpha", type=float, default=-1.0,
                    help="with exactly two checkpoints, use (1-alpha)*height + alpha*tilt actions")
parser.add_argument("--blend_error_gain", type=float, default=0.0,
                    help="smoothly adjust blend alpha by tilt-vs-height normalized error; 0 keeps fixed alpha")
parser.add_argument("--blend_action_scale", type=float, default=1.0,
                    help="multiply the blended action by this scale; 1 keeps the original action")
parser.add_argument("--risk_mpc_horizon", type=int, default=0,
                    help="if >0, use the checkpoint WM for a bounded height/tilt risk action correction")
parser.add_argument("--risk_mpc_step", type=float, default=0.0,
                    help="maximum action-space correction norm for risk MPC (0 disables)")
parser.add_argument("--risk_mpc_tilt_scale", type=float, default=1.0,
                    help="relative tilt-risk weight in the risk MPC objective")
add_launcher_args(parser)
args = parser.parse_args()

dev = args.device
gen = torch.Generator(device=args.device).manual_seed(args.seed)
cks = [torch.load(p, map_location=dev, weights_only=False) for p in args.checkpoints]
base_args = cks[0].get("args", {})
goal_frames = int(base_args.get("goal_frames", 10)); hist = int(base_args.get("hist", 10))
phase_conditioned = bool(base_args.get("phase_conditioned_policy", False))
clip_conditioned = bool(base_args.get("clip_conditioned_policy", False))
contact_conditioned = bool(base_args.get("contact_conditioned_policy", False))
current_contact = bool(base_args.get("current_contact_policy", False))
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.eval_motion_dir or base_args.get("motion_dir", "data/lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
          phase_features=phase_conditioned, clip_features=clip_conditioned,
          contact_safe_reference=bool(base_args.get("contact_safe_reward", False)),
          contact_masked_reward=bool(base_args.get("contact_masked_reward", False)),
          contact_features=contact_conditioned, current_contact_features=current_contact)
policies = []
for ck in cks:
    policy_obs_dim = OBS_DIM + (2 if phase_conditioned else 0)
    policy_obs_dim += len(motion.names) if clip_conditioned else 0
    policy_obs_dim += int(motion.body_p.shape[1]) if contact_conditioned else 0
    policy_obs_dim += int(motion.body_p.shape[1]) if current_contact else 0
    p = Policy(policy_obs_dim, 73 * goal_frames, hist=hist).to(dev)
    p.load_state_dict(ck["pi"]); p.eval(); policies.append(p)
wm_eval = None
if args.risk_mpc_horizon > 0 or args.risk_mpc_step > 0.0:
    st = cks[0].get("wm")
    if st is None:
        raise RuntimeError("risk MPC requires a checkpoint containing wm")
    key = "residual_net.0.weight"
    rh = int(st[key].shape[0]) if key in st else 0
    wm_eval = WorldModel(residual_hidden=rh).to(dev)
    wm_eval.load_state_dict(st); wm_eval.eval()
    for p in wm_eval.parameters(): p.requires_grad_(False)
    if args.risk_mpc_horizon <= 0 or args.risk_mpc_step <= 0.0:
        raise RuntimeError("risk MPC requires both --risk_mpc_horizon and --risk_mpc_step")
gated = args.gate_mode != "none"
blended = args.blend_alpha >= 0.0
if blended and not 0.0 <= args.blend_alpha <= 1.0:
    raise RuntimeError("--blend_alpha must be in [0, 1]")
if args.blend_error_gain < 0.0:
    raise RuntimeError("--blend_error_gain must be non-negative")
if args.blend_action_scale <= 0.0:
    raise RuntimeError("--blend_action_scale must be positive")
if blended and len(policies) != 2:
    raise RuntimeError("--blend_alpha requires exactly two checkpoints: height-focused then tilt-focused")
if gated and len(policies) != 2:
    raise RuntimeError("--gate_mode requires exactly two checkpoints: height-focused then tilt-focused")

valid = torch.where(motion.clip_len >= args.steps + 6)[0]
if args.clip_name is not None:
    valid = valid[torch.tensor([motion.names[int(i)] == args.clip_name for i in valid], device=dev)]
    if valid.numel() == 0:
        raise RuntimeError(f"clip {args.clip_name!r} is unavailable or too short")
if valid.numel() == 0:
    raise RuntimeError("no motion clip is long enough for requested evaluation steps")
if args.starts_file is not None:
    starts = torch.load(args.starts_file, map_location=dev, weights_only=False)
    clip = starts["clip"].to(device=dev, dtype=torch.long)
    off = starts["off"].to(device=dev, dtype=torch.long)
    if clip.ndim != 1 or off.shape != clip.shape or clip.numel() == 0:
        raise RuntimeError("starts_file must contain non-empty 1-D clip/off tensors")
    if (clip < 0).any() or (clip >= motion.clip_len.numel()).any():
        raise RuntimeError("starts_file contains an invalid motion clip index")
    max_off = (motion.clip_len[clip] - args.steps - 1).clamp_min(1)
    if (off < 1).any() or (off > max_off).any():
        raise RuntimeError("starts_file contains an offset that is too short for --steps")
    args.batch = int(clip.numel())
else:
    clip = valid[torch.randint(valid.numel(), (args.batch,), generator=gen, device=dev)]
    max_off = (motion.clip_len[clip] - args.steps - 1).clamp_min(1)
    off = (torch.rand(args.batch, generator=gen, device=dev) * max_off.float()).long() + 1
if args.write_starts is not None:
    args.write_starts.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clip": clip.detach().cpu(), "off": off.detach().cpu()}, args.write_starts)
    print(f"wrote {args.write_starts} ({clip.numel()} starts)")
    raise SystemExit(0)
frames = motion.clip_start[clip] + off
all_frames = frames if (gated or blended) else frames.repeat(len(policies))
total = args.batch if (gated or blended) else args.batch * len(policies)
sim_args = SimpleNamespace(num_envs=total)
sim_cfg = make_sim(dev, sim_args)

with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, total, motion, mdp, gamma=float(base_args.get("gamma", .99)), eval_mode=True, reset_noise=False,
                      fall_grace_ref_height=float(base_args.get("fall_grace_ref_height", 0.0)),
                      fall_grace_ref_tilt=float(base_args.get("fall_grace_ref_tilt", 0.0)))
    env.reset(torch.arange(total, device=dev), all_frames)
    if args.perturb:
        pgen = torch.Generator(device=dev).manual_seed(args.seed + 100000)
        base_xi = torch.zeros(args.batch, 33, device=dev)
        base_xi[:, 0] = torch.empty(args.batch, device=dev).uniform_(0.3, 1.2, generator=pgen)
        base_xi[:, 1:30] = torch.empty(args.batch, 29, device=dev).uniform_(-0.01, 0.01, generator=pgen)
        base_xi[:, 30:33] = torch.stack((
            torch.empty(args.batch, device=dev).uniform_(-0.025, 0.025, generator=pgen),
            torch.empty(args.batch, device=dev).uniform_(-0.05, 0.05, generator=pgen),
            torch.empty(args.batch, device=dev).uniform_(-0.05, 0.05, generator=pgen)), -1)
        env.xi.copy_(base_xi if (gated or blended) else base_xi.repeat(len(policies), 1))
        default_com = env.robot.data.body_com_pose_b.torch[:, env.torso_id, :3].clone()
        env.robot.set_coms_index(
            coms=(default_com + env.xi[:, 30:33]).unsqueeze(1),
            body_ids=[env.torso_id],
        )
        env._apply_friction()
    env.refresh()
    obs_hist = torch.zeros(total, hist, policy_obs_dim, device=dev)
    trace_initial_state = tuple(x[:args.batch].clone() for x in env.state()) if args.trace_out is not None and not gated else None
    trace_initial_xi = env.xi[:args.batch].clone() if trace_initial_state is not None else None
    trace_initial_f = env.f[:args.batch].clone() if trace_initial_state is not None else None
    alive = torch.ones(total, dtype=torch.bool, device=dev)
    failed = torch.zeros(total, dtype=torch.bool, device=dev)
    height_fail = torch.zeros(total, dtype=torch.bool, device=dev)
    tilt_fail = torch.zeros(total, dtype=torch.bool, device=dev)
    ret = torch.zeros(total, device=dev)
    first_fail = torch.full((total,), args.steps, dtype=torch.long, device=dev)
    fail_height_err = torch.zeros(total, device=dev)
    fail_tilt_err = torch.zeros(total, device=dev)
    fail_speed = torch.zeros(total, device=dev)
    fail_angular_speed = torch.zeros(total, device=dev)
    fail_action_norm = torch.zeros(total, device=dev)
    fail_frame = torch.zeros(total, device=dev)
    prev_height_err = torch.zeros(total, device=dev)
    prev_tilt_err = torch.zeros(total, device=dev)
    trace_features = (torch.zeros(len(policies), args.steps, args.batch, 5, device=dev)
                      if args.trace_out is not None and not gated else None)
    trace_rewards = (torch.zeros(len(policies), args.steps, args.batch, device=dev)
                     if args.trace_out is not None and not gated else None)
    with torch.no_grad():
        for t in range(args.steps):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o[:, None].expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o[:, None]), 1))
            if gated or blended:
                goal = motion.goal(env.f, goal_frames)
                a_height = policies[0](obs_hist, goal).clamp(-20, 20)
                a_tilt = policies[1](obs_hist, goal).clamp(-20, 20)
                height_err = (motion.root_p[env.f, 2] - S[0][:, 2]).abs() / max(env.mdp.cfg.term_height, 1e-6)
                tilt_err = (motion.root_R[env.f, 2, 2] - S[1][:, 2, 2]).abs() / max(env.mdp.cfg.term_tilt, 1e-6)
                if blended:
                    alpha = args.blend_alpha
                    if args.blend_error_gain > 0.0:
                        # Keep the user-specified base mixture while allowing
                        # the current relative risk to make a smooth correction.
                        base_logit = torch.logit(torch.as_tensor(
                            args.blend_alpha, device=dev).clamp(1e-4, 1.0 - 1e-4))
                        alpha = torch.sigmoid(base_logit + args.blend_error_gain * (tilt_err - height_err))
                    if torch.is_tensor(alpha):
                        alpha = alpha.unsqueeze(-1)
                    act = args.blend_action_scale * ((1.0 - alpha) * a_height + alpha * a_tilt)
                elif args.gate_mode == "error_trend" and t > 0:
                    use_height = ((height_err - prev_height_err) >=
                                  (tilt_err - prev_tilt_err)).unsqueeze(-1)
                    act = torch.where(use_height, a_height, a_tilt)
                else:
                    use_height = (height_err >= tilt_err).unsqueeze(-1)
                    act = torch.where(use_height, a_height, a_tilt)
            else:
                actions=[]
                for j, p in enumerate(policies):
                    sl=slice(j*args.batch,(j+1)*args.batch)
                    actions.append(p(obs_hist[sl], motion.goal(env.f[sl], goal_frames)).clamp(-20,20))
                act=torch.cat(actions,0)
                if trace_features is not None:
                    for j in range(len(policies)):
                        sl = slice(j * args.batch, (j + 1) * args.batch)
                        h_err = (motion.root_p[env.f[sl], 2] - S[0][sl, 2]).abs() / max(env.mdp.cfg.term_height, 1e-6)
                        t_err = (motion.root_R[env.f[sl], 2, 2] - S[1][sl, 2, 2]).abs() / max(env.mdp.cfg.term_tilt, 1e-6)
                        speed = S[2][sl].square().sum(-1).sqrt()
                        ang = S[3][sl].square().sum(-1).sqrt()
                        trace_features[j, t, :, :] = torch.stack((h_err, t_err, speed, ang, actions[j].square().sum(-1).sqrt()), -1)
            if args.oracle_reference_action:
                target_f = (env.f + 1).clamp(max=motion.root_p.shape[0] - 1)
                act = ((motion.q[target_f] - env.default_q - env.xi[:, 1:30]) /
                       env.alpha).clamp(-20.0, 20.0)
            if wm_eval is not None:
                # Keep the learned policy fixed.  Only a bounded, one-shot
                # correction is differentiated through the frozen WM toward
                # the future height/tilt tracking risk.
                with torch.enable_grad():
                    S0 = tuple(x.detach() for x in S)
                    a_var = act.detach().requires_grad_(True)
                    Smp = S0
                    xi_n = normalize_xi(env.xi.detach())
                    for _ in range(args.risk_mpc_horizon):
                        Smp = wm_eval.step(Smp, a_var, xi_n)
                    tf = (env.f + args.risk_mpc_horizon).clamp(max=motion.root_p.shape[0] - 1)
                    hz = ((Smp[0][:, 2] - motion.root_p[tf, 2]) / max(env.mdp.cfg.term_height, 1e-6)).square()
                    tz = ((Smp[1][:, 2, 2] - motion.root_R[tf, 2, 2]) / max(env.mdp.cfg.term_tilt, 1e-6)).square()
                    risk = (hz + args.risk_mpc_tilt_scale * tz).sum()
                    grad = torch.autograd.grad(risk, a_var, allow_unused=False)[0]
                    gnorm = grad.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    act = a_var - args.risk_mpc_step * grad / gnorm
                    act = act.clamp(-20.0, 20.0).detach()
            act=torch.where(alive[:,None],act,torch.zeros_like(act))
            out=env.step(act)
            if gated:
                prev_height_err = height_err.detach()
                prev_tilt_err = tilt_err.detach()
            ret += out["r"] * alive.float()
            if trace_rewards is not None:
                for j in range(len(policies)):
                    sl = slice(j * args.batch, (j + 1) * args.batch)
                    trace_rewards[j, t] = out["r"][sl]
            new_fail = out["term"] & alive
            S1 = env.state()
            f1 = env.f
            height_term = (motion.root_p[f1, 2] - S1[0][:, 2]).abs() > env.mdp.cfg.term_height
            tilt_term = (motion.root_R[f1, 2, 2] - S1[1][:, 2, 2]).abs() > env.mdp.cfg.term_tilt
            height_fail |= new_fail & height_term
            tilt_fail |= new_fail & tilt_term
            if new_fail.any():
                root_height_err = (motion.root_p[f1, 2] - S1[0][:, 2]).abs()
                root_tilt_err = (motion.root_R[f1, 2, 2] - S1[1][:, 2, 2]).abs()
                speed = S1[2].square().sum(-1).sqrt()
                angular_speed = S1[3].square().sum(-1).sqrt()
                action_norm = act.square().sum(-1).sqrt()
                fail_height_err[new_fail] = root_height_err[new_fail]
                fail_tilt_err[new_fail] = root_tilt_err[new_fail]
                fail_speed[new_fail] = speed[new_fail]
                fail_angular_speed[new_fail] = angular_speed[new_fail]
                fail_action_norm[new_fail] = action_norm[new_fail]
                fail_frame[new_fail] = f1[new_fail].float()
            first_fail[new_fail & (first_fail == args.steps)] = t + 1
            failed |= new_fail
            alive &= ~out["done"]
    out=[]
    result_names = ([f"gate:{args.checkpoints[0]}|{args.checkpoints[1]}"] if gated else
                    [f"blend:{args.blend_alpha:.3f}:{args.checkpoints[0]}|{args.checkpoints[1]}"] if blended else
                    list(args.checkpoints))
    for j,p in enumerate(result_names):
        sl=slice(0,args.batch) if (gated or blended) else slice(j*args.batch,(j+1)*args.batch)
        fail_sl = failed[sl]
        ret_norm_sl = ret[sl] / (args.steps * 0.1)
        out.append({"checkpoint":str(p),"return_norm":float(ret_norm_sl.mean()),"survival_at_end":float(alive[sl].float().mean()),"failure_rate":float(fail_sl.float().mean()),"height_failure_rate":float(height_fail[sl].float().mean()),"tilt_failure_rate":float(tilt_fail[sl].float().mean()),"first_failure_step_mean":float(first_fail[sl].float().mean()),
                   "failure_height_err_mean":float(fail_height_err[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "failure_tilt_err_mean":float(fail_tilt_err[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "failure_speed_mean":float(fail_speed[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "failure_angular_speed_mean":float(fail_angular_speed[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "failure_action_norm_mean":float(fail_action_norm[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "failure_frame_mean":float(fail_frame[sl][fail_sl].mean()) if fail_sl.any() else 0.0,
                   "per_start":[{"clip":int(clip[i].item()),"offset":int(off[i].item()),
                                 "return_norm":float(ret_norm_sl[i].item()),
                                 "survived":bool(alive[sl][i].item()),
                                 "first_failure_step":int(first_fail[sl][i].item()),
                                 "height_failure":bool(height_fail[sl][i].item()),
                                 "tilt_failure":bool(tilt_fail[sl][i].item())}
                                for i in range(args.batch)]})
    result = {"steps": args.steps, "batch": args.batch, "seed": args.seed,
              "starts": int(valid.numel()), "results": out}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    if args.trace_out is not None:
        if gated:
            raise RuntimeError("--trace_out requires ungated policy evaluation")
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": trace_features.cpu(), "rewards": trace_rewards.cpu(),
                    "clip": clip.cpu(), "off": off.cpu(), "steps": args.steps,
                    "initial_state": tuple(x.cpu() for x in trace_initial_state),
                    "initial_xi": trace_initial_xi.cpu(), "initial_f": trace_initial_f.cpu()}, args.trace_out)
