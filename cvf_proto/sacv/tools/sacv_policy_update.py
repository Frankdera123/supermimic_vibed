"""CVF-2a Phase E: single policy update with swapped critics (GPU; gated by Phase C/D).

Faithfully replicates the original training policy-update path
(supermimic_opcl_proto/supermimic/train.py::policy_update), with the pre-registered
deviations of the cold-optimizer experiment (protocol §E1):
  * pi starts from pi_old@2100; Adam(lr=2e-4, betas=(0.9,0.999)) freshly initialized
    (original optimizer state is not stored in ckpt_002100.pt) -- shared by all groups;
  * imagination starts = Train-split real starts (TrackingSim reset to (clip, frame),
    nominal xi, repeat-10 o_hist), B=24 vs the original 4096-buffer sampling -- SNR
    limitation recorded, identical across groups (paired design);
  * obs noise via mdp.add_noise with a shared torch.manual_seed reset per group, so the
    imagined trajectory (pi is un-changed within the single update) is IDENTICAL for all
    groups; the only difference is the bootstrap critic (or none for P-noV);
  * opt_step semantics copied: zero_grad -> backward -> clip_grad_norm_(1.0) ->
    clip_grad_value_(0.1) -> step if finite.

Groups: P-init (no step) / P0 (C-init=B0@2100_post) / P1 / P2 (fine-tuned) / P-noV.
Evaluation (E2): deterministic closed-loop sim rollouts (T_eval=96, soft survival)
from the Val+Test starts, paired per-start Delta J vs P-init, plus fall rate /
ep_len / termination distribution and update-norm diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_SACV), str(_CVF), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto"), str(_REPO / "cvf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from supermimic.mdp import flatten_state, normalize_xi  # noqa: E402

GAMMA = 0.99
HZ = 32
V_MAX = 10.0
LR = 2e-4
SEED = 2702
T_EVAL = 96
_P0_REF: dict = {}   # in-memory P0 update reference for cosine diagnostics


def load_critic(path_or_pair):
    import pada_bridge as bridge

    bridge.import_stack()
    from supermimic.networks import Critic

    if isinstance(path_or_pair, str):
        ck = torch.load(path_or_pair, map_location="cpu", weights_only=False)
        sd = ck["critic"]
        name = ck.get("group", Path(path_or_pair).stem)
    else:
        sd = path_or_pair            # raw state dict of C-init (critic_post)
        name = "C-init"
    c = Critic(sd["norm.mean"].shape[0])
    c.load_state_dict(sd, strict=True)
    c.eval()
    return name, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", type=Path, required=True, help="C1 fine-tuned ckpt")
    ap.add_argument("--p2", type=Path, default=None, help="C2 fine-tuned ckpt")
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/metrics/policy_update.json")
    ap.add_argument("--eval_minutes", type=float, default=25.0)
    ap.add_argument("--skip_eval", action="store_true")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    dev = getattr(args, "device", None) or "cuda:0"
    t0 = time.time()

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair2100)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    pi_old = pair2100["pi_old"]
    wm = pair2100["wm"]

    critics = {}
    n_p0, c_p0 = load_critic(torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                                        map_location="cpu", weights_only=False)["critic_post"])
    critics["P0"] = (n_p0, c_p0.to(dev))
    n_p1, c_p1 = load_critic(str(args.p1))
    critics["P1"] = (n_p1, c_p1.to(dev))
    if args.p2:
        n_p2, c_p2 = load_critic(str(args.p2))
        critics["P2"] = (n_p2, c_p2.to(dev))

    # ---------------- imagination starts from Train-split real starts ----------------
    man = json.loads((_REPO / "results/cvf/sacv/manifests/CVF1A_SPLIT_MANIFEST.json").read_text())
    train_starts = [(cd, fr) for cd in man["splits"]["train"]["per_clip"].values()
                    for fr in cd["frames"]]
    eval_starts = [(cd, fr) for split in ("val", "test")
                   for cd in man["splits"][split]["per_clip"].values() for fr in cd["frames"]]

    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    import isaaclab.sim as sim_utils

    def det_cfg(n):
        return sim_utils.SimulationCfg(
            dt=0.02, device=dev,
            physics=NewtonCfg(
                deterministic_mode="run_to_run",
                solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal",
                                           integrator="implicitfast", use_mujoco_contacts=True,
                                           disable_sensors=True),
                num_substeps=4,
            ),
        )

    log = {"schema": "sacv_policy_update_v1", "started": time.strftime("%F %T"),
           "groups": list(critics.keys()) + ["P-noV", "P-init"],
           "deviations": ["cold Adam state (ckpt lacks optimizer)", "B=24 vs 4096 (SNR)",
                          "repeat-10 o_hist starts (contract)", "nominal xi only",
                          "batched B<=4 (MJWarp deterministic-mode int32 workspace limit)"],
           "imagination": {}, "updates": {}, "eval": {}}

    CHUNK = 4                      # MJWarp run_to_run workspace limit (empirical: 8 overflows)
    B_im = len(train_starts)
    with launch_simulation(det_cfg(CHUNK), args):
        from supermimic.env import TrackingSim

        env = TrackingSim(det_cfg(CHUNK), CHUNK, motion, mdp, gamma=GAMMA,
                          eval_mode=True, reset_noise=False)
        xi_chunk = torch.zeros(CHUNK, 33, device=dev); xi_chunk[:, 0] = 1.0
        xi_n_chunk = normalize_xi(xi_chunk)

        def group_update(tag, critic=None, use_critic=True, do_step=True):
            """One faithful policy update; imagination batched in CHUNK-sized resets.

            torch.manual_seed is reset once per group so the obs-noise stream is
            identical across groups (pi is unchanged within a single update, so the
            imagined trajectory itself is identical; only the bootstrap differs)."""
            pi = copy.deepcopy(pi_old)
            opt = torch.optim.Adam(pi.parameters(), lr=LR, betas=(0.9, 0.999))
            torch.manual_seed(SEED)
            for p in wm.parameters():
                p.requires_grad_(False)
            for p in pi.parameters():
                p.requires_grad_(True)
            if critic is not None:
                for p in critic.parameters():
                    p.requires_grad_(False)
            J_parts, boot_parts, end_parts = [], [], []
            for ci in range(0, B_im, CHUNK):
                fr = torch.tensor([f for _, f in train_starts[ci:ci + CHUNK]], device=dev)
                env.reset(torch.arange(CHUNK, device=dev), fr)
                env.xi[:] = xi_chunk
                env.refresh()
                S = env.state()
                a_prev = torch.zeros(CHUNK, 29, device=dev)
                o0 = proto.obs(S, a_prev, xi_chunk, fr)
                o_hist = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
                g, gw = motion.goal(fr, S[0], S[1])
                Ss, As, Aps, Fs = [], [], [], []
                for t in range(HZ):
                    Aps.append(a_prev)
                    a = pi(o_hist, g)
                    S1 = tuple(proto.wm_step(wm, S, a, xi_n_chunk))
                    f1 = fr + 1
                    o1 = mdp.add_noise(mdp.obs(S1, a, xi_chunk))
                    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
                    g, gw = motion.goal(f1, S1[0], S1[1])
                    S, a_prev, fr = S1, a, f1
                    Ss.append(S); As.append(a); Fs.append(fr)
                flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
                S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
                f_all = flat(Fs)
                r = mdp.reward(S_all, f_all, flat(As), flat(Aps))
                d = mdp.soft_termination(S_all, f_all)
                r, d = r.view(HZ, CHUNK), d.view(HZ, CHUNK)
                alive = torch.cumprod(1 - d, 0)
                c_t = torch.cat((torch.ones(1, CHUNK, device=dev), alive[:-1]), 0)
                disc = GAMMA ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
                J_parts.append((disc * c_t * r).sum(0))
                end_parts.append({"c_end": alive[-1], "S_end": S, "f_end": fr, "gw_end": gw})
            J = torch.cat(J_parts)                      # (B_im,) with grad across chunks
            j_prefix = J.detach().clone()
            boot = torch.zeros_like(J)
            if use_critic and critic is not None:
                boots = []
                for ep in end_parts:
                    gw_end = proto.wm_goal(ep["f_end"], ep["S_end"])
                    boots.append((GAMMA ** HZ) * ep["c_end"]
                                 * critic(proto.critic_input(ep["S_end"], ep["f_end"], xi_n_chunk, gw_end)).clamp(max=V_MAX))
                boot = torch.cat(boots)
                J = J + boot
            loss = -J.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_raw = float(torch.sqrt(sum((p.grad ** 2).sum() for p in pi.parameters()
                                          if p.grad is not None)))
            gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(pi.parameters(), 0.1)
            upd = {}
            if do_step and bool(torch.isfinite(gn)):
                opt.step()
                with torch.no_grad():
                    ref_pi = list(pi_old.parameters())
                    l2 = float(sum((a2 - a1).norm() ** 2 for a2, a1 in
                                   zip(pi.parameters(), ref_pi)) ** 0.5)
                upd = {"pi_param_l2_diff": l2}
            else:
                upd = {"pi_param_l2_diff": 0.0, "stepped": False}
            log["updates"][tag] = {
                "j_prefix_mean": float(j_prefix.mean()), "j_prefix_std": float(j_prefix.std()),
                "bootstrap_mean": float(boot.detach().mean()) if use_critic and critic is not None else 0.0,
                "loss": float(loss.detach()), "grad_norm_pre_clip": gn_raw,
                "grad_norm_post_normclip": float(gn), "stepped": do_step and bool(torch.isfinite(gn)),
                **upd}
            if tag == "P0":
                _P0_REF["params"] = [p.detach().cpu() for p in pi.parameters()]
            elif _P0_REF.get("params") is not None and upd.get("pi_param_l2_diff", 0) > 0:
                with torch.no_grad():
                    upd_vec = [(a2 - a1).reshape(-1) for a2, a1 in
                               zip(pi.parameters(), pi_old.parameters())]
                    ref_vec = [(b2 - a1).reshape(-1) for b2, a1 in
                               zip((t.to(dev) for t in _P0_REF["params"]), pi_old.parameters())]
                    denom = max(sum(u.norm() * v.norm() for u, v in zip(upd_vec, ref_vec)), 1e-12)
                    cos = float(sum((u * v).sum() for u, v in zip(upd_vec, ref_vec)) / denom)
                log["updates"][tag]["cos_update_vs_P0"] = cos
            return pi

        for tag, (name, critic) in critics.items():
            log["imagination"][tag] = {"critic": name}
        group_update("P-init", critic=None, use_critic=False, do_step=False)

        if args.skip_eval:
            args.out.write_text(json.dumps(log, indent=2))
            print(json.dumps({"updates": {k: v["pi_param_l2_diff"] for k, v in log["updates"].items()}}))
            return 0

        # ---------------- E2: deterministic closed-loop sim evaluation ----------------
        B_ev = len(eval_starts)

        def roll_J(pi_module) -> dict:
            """Batched (CHUNK-sized resets) deterministic closed-loop evaluation."""
            J_softs, J_hards, steps_l, fell_l = [], [], [], []
            with torch.no_grad():
                for ci in range(0, B_ev, CHUNK):
                    n_c = min(CHUNK, B_ev - ci)
                    xi_one_e = torch.zeros(n_c, 33, device=dev); xi_one_e[:, 0] = 1.0
                    f0_e = torch.tensor([fr for _, fr in eval_starts[ci:ci + n_c]], device=dev)
                    env.reset(torch.arange(n_c, device=dev), f0_e)
                    env.xi[:] = xi_one_e
                    env.refresh()
                    ap = torch.zeros(n_c, 29, device=dev)
                    o0e = proto.obs(env.state(), ap, xi_one_e, f0_e)
                    oh = o0e.unsqueeze(1).expand(-1, 10, -1).contiguous()
                    c_soft = torch.ones(n_c, device=dev)
                    c_hard = torch.ones(n_c, device=dev)
                    J_soft = torch.zeros(n_c, device=dev)
                    J_hard = torch.zeros(n_c, device=dev)
                    steps = torch.zeros(n_c, device=dev)
                    fell = torch.zeros(n_c, device=dev)
                    for k in range(T_EVAL):
                        ag = proto.actor_goal(env.f, env.state())
                        a = torch.clamp(pi_module(oh, ag), -20.0, 20.0)
                        st = env.step(a)
                        S1 = env.state()
                        ds = mdp.soft_termination(S1, env.f)
                        dh = st["term"].float()
                        J_soft = J_soft + (GAMMA ** k) * c_soft * st["r"]
                        J_hard = J_hard + (GAMMA ** k) * c_hard * st["r"]
                        steps += c_soft
                        fell += dh * (1 - fell)
                        c_soft = c_soft * (1.0 - ds)
                        c_hard = c_hard * (1.0 - dh)
                        oh = torch.cat((oh[:, 1:], proto.obs(S1, a, xi_one_e, env.f).unsqueeze(1)), 1)
                    J_softs.append(J_soft); J_hards.append(J_hard)
                    steps_l.append(steps); fell_l.append(fell)
            return {"J_soft": torch.cat(J_softs).cpu(), "J_hard": torch.cat(J_hards).cpu(),
                    "mean_ep_len": torch.cat(steps_l).cpu(), "fell": torch.cat(fell_l).cpu()}

        base = roll_J(pi_old)
        log["eval"]["P-init(base=pi_old)"] = {
            "J_soft_mean": float(base["J_soft"].mean()), "J_soft_per_start": base["J_soft"].tolist(),
            "fall_rate": float(base["fell"].mean()), "mean_ep_len": float(base["mean_ep_len"].mean())}
        # run the actual updates (deterministic RNG keeps imagination identical across groups)
        pi_by = {}
        for tag, (name, critic) in critics.items():
            pi_by[tag] = group_update(tag, critic=critic, use_critic=True, do_step=True)
        pi_by["P-noV"] = group_update("P-noV", critic=None, use_critic=False, do_step=True)
        for tag, pi_g in pi_by.items():
            r = roll_J(pi_g)
            dJ = r["J_soft"] - base["J_soft"]
            log["eval"][tag] = {
                "J_soft_mean": float(r["J_soft"].mean()),
                "deltaJ_mean": float(dJ.mean()),
                "deltaJ_per_start": dJ.tolist(),
                "n_improved": int((dJ > 0).sum()), "n": int(len(dJ)),
                "fall_rate": float(r["fell"].mean()), "mean_ep_len": float(r["mean_ep_len"].mean()),
                "clips": [cd["category"] for cd, _ in eval_starts]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=2))
    print(json.dumps({"out": str(args.out),
                      "updates": {k: v.get("pi_param_l2_diff") for k, v in log["updates"].items()
                                  if isinstance(v, dict) and "pi_param_l2_diff" in v},
                      "eval": {k: v.get("deltaJ_mean") for k, v in log["eval"].items()
                               if isinstance(v, dict) and "deltaJ_mean" in v}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
