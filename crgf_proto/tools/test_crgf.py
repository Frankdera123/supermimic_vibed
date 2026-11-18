"""CPU synthetic tests for the CRGF core (no simulator, no GPU required).

Covers: reward-split closure, autograd-vs-FD agreement on the WM graph,
decomposition closure of the three path derivatives, metric correctness,
reliability-gate logic, direction utilities, and base-rollout determinism.

Run:  python tools/test_crgf.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.crgf import (base_action_rollout, delta_dir_seqs,
                             fidelity_stats, open_loop_paths,
                             path_directional_derivatives, random_dir_seqs,
                             reliability_gate, soft_term, spearman)
from supermimic.dimensions import STATE_DIM
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def random_states(B: int, motion: MotionLib, gen: torch.Generator, device="cpu",
                  min_len: int = 4):
    """Valid random states anchored at random reference frames."""
    valid = torch.where(motion.clip_len >= min_len)[0]
    clip = valid[torch.randint(valid.numel(), (B,), generator=gen)]
    f0 = motion.clip_start[clip]
    p, R, v, w, q, qd = motion.ref_state(f0)
    return (p.clone(), R.clone(), v.clone(), w.clone(), q.clone(), qd.clone()), f0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default="data/lafan1_g1")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--double", action="store_true", default=True,
                    help="float64 everywhere (FD test precision); kin/motion buffers are cast")
    args = ap.parse_args()
    dev = "cpu"
    dtype = torch.float64 if args.double else torch.float32
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(7)

    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    if dtype is torch.float64:
        # cast every float buffer so the whole graph (FK included) runs in f64;
        # int tensors (clip indices, bins) are untouched
        for name in ("axes", "off_p", "off_R", "q_lo", "q_hi"):
            setattr(kin, name, getattr(kin, name).double())
        for name in dir(motion):
            if name.startswith("_"):
                continue
            t = getattr(motion, name)
            if torch.is_tensor(t) and t.is_floating_point():
                setattr(motion, name, t.double())
        mdp.default_q = mdp.default_q.double()
        mdp.obs_noise = mdp.obs_noise.double()
        mdp.obs_scale = mdp.obs_scale.double()
        mdp.soft_lo = mdp.soft_lo.double()
        mdp.soft_hi = mdp.soft_hi.double()

    hidden = (16, 16)  # tame synthetic WM: bounded states, no exp/sigmoid saturation
    wm = WorldModel(hidden=hidden).to(dev)
    # the WM output layer is zero-initialized by design; randomize it so the
    # synthetic model is action-sensitive (a zero layer gives D == 0 exactly)
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, 0.001)
        wm.net[-1].bias.normal_(0, 0.001)
    goal_dim = 73 * 4
    pi = Policy(99, goal_dim, hist=4).to(dev)
    S_probe, f_probe = random_states(2, motion, gen)
    xi_probe = torch.zeros(2, 33)
    xi_probe[:, 0] = 1.0
    cin_dim = mdp.critic_features(S_probe, f_probe, normalize_xi(xi_probe)).shape[-1]
    cr = Critic(cin_dim + goal_dim).to(dev)
    if dtype is torch.float64:
        wm, pi, cr = wm.double(), pi.double(), cr.double()

    # ---------------------------------------------------------- 1. reward split
    S, f0 = random_states(args.B, motion, gen)
    a = torch.randn(args.B, 29, generator=gen) * 0.5
    a_prev = torch.randn(args.B, 29, generator=gen) * 0.5
    r_state, r_direct = __import__("supermimic.crgf", fromlist=["reward_split"]).reward_split(mdp, S, f0, a, a_prev)
    r_total = mdp.reward(S, f0, a, a_prev)
    err = float((r_state + r_direct - r_total).abs().max())
    check("reward_split closure == mdp.reward", err < 1e-6, f"max|diff|={err:.2e}")

    # --------------------------------------- 2. autograd vs FD on the WM graph
    B, H = args.B, args.H
    xi = torch.zeros(B, 33)
    xi[:, 0] = 1.0
    xi_n = normalize_xi(xi).to(dtype)
    Hf = 3  # short horizon keeps the synthetic (untrained) WM in a smooth regime
    f_valid = torch.where(motion.clip_len >= Hf + 2)[0]
    clip = f_valid[torch.randint(f_valid.numel(), (B,), generator=gen)]
    f0 = motion.clip_start[clip] + 1
    S0 = tuple(x.to(dtype) for x in motion.ref_state(f0))
    # perturb the start orientation away from perfect alignment with the
    # reference: rot_angle_sq clamps acos near cos=1 (rotations.py) and that
    # kink breaks FD/autograd agreement for ANY model, trained or not
    from supermimic.rotations import so3_exp
    tilt = so3_exp(0.4 * torch.randn(B, 3, generator=gen)).to(dtype)
    S0 = (S0[0], tilt @ S0[1], S0[2], S0[3], S0[4], S0[5])
    a_seq = (torch.randn(B, Hf, 29, generator=gen) * 0.3).to(dtype)
    a_prev0 = torch.zeros(B, 29, dtype=dtype)
    a_seq = a_seq.clone().requires_grad_(True)

    paths = open_loop_paths(wm, cr, mdp, S0, f0, a_seq, a_prev0, xi_n,
                            gamma=0.99, goal_frames=4)
    grads = path_directional_derivatives(paths, a_seq)
    d_seq = random_dir_seqs(B, Hf, 29, seed=11, mode="per_step").to(dtype)
    D_wm = (grads["total"] * d_seq.reshape(B, -1)).sum(-1)

    eps = 1e-5 if dtype is torch.float64 else 1e-2

    def J_of(A):
        with torch.no_grad():
            p2 = open_loop_paths(wm, cr, mdp, S0, f0, A.clone().requires_grad_(False),
                                 a_prev0, xi_n, gamma=0.99, goal_frames=4)
            return p2["J_total"]

    fd_J_raw = {}

    def fd_J(i: int, sign: float) -> float:
        # perturb ONLY sample i (a naive broadcast would perturb every sample)
        delta = torch.zeros_like(a_seq.detach())
        delta[i] = sign * eps * d_seq[i]
        jv = J_of(a_seq.detach() + delta)
        fd_J_raw[(i, sign)] = jv
        return jv

    J_plus = torch.tensor([fd_J(i, +1.0) for i in range(B)], dtype=dtype)
    J_minus = torch.tensor([fd_J(i, -1.0) for i in range(B)], dtype=dtype)
    D_fd = (J_plus - J_minus) / (2 * eps)
    scale = torch.maximum(D_wm.abs(), D_fd.abs())
    atol = 1e-8 if dtype is torch.float64 else 1e-4
    rtol = 1e-5 if dtype is torch.float64 else 5e-2
    resid = (D_wm - D_fd).abs()
    ok_fd = bool((resid <= atol + rtol * scale).all())
    worst = int(resid.argmax())
    check("autograd D_total == central FD (WM graph)", ok_fd,
          f"max|diff|={float(resid.max()):.2e} @sample{worst} "
          f"(D_wm={float(D_wm[worst]):.3e}, D_fd={float(D_fd[worst]):.3e}, "
          f"J+={fd_J_raw[(worst, +1.0)]:.12e}, J-={fd_J_raw[(worst, -1.0)]:.12e}); "
          f"|D| range [{float(D_wm.abs().min()):.2e},{float(D_wm.abs().max()):.2e}] "
          f"(eps={eps}, {'f64' if dtype is torch.float64 else 'f32'})")

    # -------------------------------------------- 3. decomposition closure
    # (a) EXACT identity: D_total == D_state + D_direct + D_critic + D_cross,
    #     where D_cross = grad of the detached cross-term graph
    lhs = grads["total"]
    rhs = grads["state"] + grads["direct"] + grads["critic"] + grads["cross"]
    ident = float((lhs - rhs).abs().max())
    check("identity D_total == state+direct+critic+cross", ident < 1e-8,
          f"max|diff|={ident:.2e}")
    # (b) with w_action_rate=0 and no critic the residual must vanish exactly
    mdp.cfg.w_action_rate = 0.0
    a_seq2 = a_seq.detach().clone().requires_grad_(True)
    paths2 = open_loop_paths(wm, None, mdp, S0, f0, a_seq2, a_prev0, xi_n,
                             gamma=0.99, goal_frames=4)
    grads2 = path_directional_derivatives(paths2, a_seq2)
    zero_cross = float(grads2["cross_residual"].abs().max())
    check("cross residual == 0 when r_direct=0 and critic=None",
          zero_cross < 1e-6, f"max|cross|={zero_cross:.2e}")
    mdp.cfg.w_action_rate = -0.1

    # (c) the cross residual is nonzero exactly when direct reward / critic
    #     cross the (state-dependent) termination path; report its scale
    print(f"     info: max|cross|={float(grads['cross'].abs().max()):.3e}, "
          f"max|D_total|={float(grads['total'].abs().max()):.3e}")

    # ------------------------------------------------------ 4. metric checks
    rho = spearman(torch.tensor([1., 2., 3.]), torch.tensor([3., 2., 1.]))
    check("spearman perfect anti-correlation", abs(rho + 1.0) < 1e-9, f"rho={rho:.3f}")
    rho2 = spearman(torch.tensor([1., 2., 3.]), torch.tensor([1., 3., 2.]))
    check("spearman partial correlation", abs(rho2 - 0.5) < 1e-9, f"rho={rho2:.3f}")
    w = torch.tensor([1., -1., 2., 0.])
    s = torch.tensor([1., 1., -2., 0.])
    st = fidelity_stats(w, s, None)
    # (0,0) sample is undecidable; among the 3 decidable pairs only one agrees
    check("sign agreement counting", abs(st["sign_agreement"] - 1.0 / 3.0) < 1e-6,
          f"agreement={st['sign_agreement']:.4f}")
    st2 = fidelity_stats(w, s, torch.tensor([True, True, False, True]))
    check("gate filters samples", st2["n_gated"] == 3 and st2["n"] == 4,
          f"n={st2['n']} n_gated={st2['n_gated']}")

    # ------------------------------------------------------ 5. gate logic
    ok, why = reliability_gate(torch.tensor([1., 1., 1.]), torch.tensor([.01, .011, .0095]))
    check("gate passes clean sample", ok and why == "ok", why)
    ok, why = reliability_gate(torch.tensor([1., 1.0073, 1.]), torch.tensor([.01, .011]))
    check("gate fails on baseline jump", (not ok) and why.startswith("baseline_range"), why)
    ok, why = reliability_gate(torch.tensor([1., 1.]), torch.tensor([.147, -.183, .037]))
    check("gate fails on cross-scale sign flip", (not ok) and why.startswith("d_range"), why)
    ok, why = reliability_gate(torch.tensor([1., 1.]), torch.tensor([1e-6, 2e-6]))
    check("gate fails below deadband", (not ok) and why.startswith("below_deadband"), why)

    # ------------------------------------------------ 6. direction utilities
    d = random_dir_seqs(5, 4, 29, seed=3, mode="per_step")
    norms = d.reshape(5, -1).norm(dim=-1)
    check("per_step random dirs are unit norm",
          bool(((norms - 1).abs() < 1e-5).all()), f"norms~{float(norms.mean()):.4f}")
    d2 = random_dir_seqs(5, 4, 29, seed=3, mode="repeated")
    same = bool((d2 - d2[:, :1]).abs().max() < 1e-9)
    check("repeated dirs constant across steps", same)
    dn, nrm = delta_dir_seqs(torch.ones(3, 4, 29), torch.ones(3, 4, 29))
    check("delta dirs of identical policies vanish",
          bool((dn.abs() < 1e-6).all() and (nrm < 1e-6).all()),
          f"norm={float(nrm.max()):.2e}")

    # ----------------------------------------- 7. base rollout determinism
    S3, f3 = random_states(4, motion, gen, min_len=12)
    xi3 = torch.zeros(4, 33)
    xi3[:, 0] = 1.0
    obs0 = mdp.obs(S3, torch.zeros(4, 29, dtype=dtype), xi3, f3)
    # the policy consumes an observation HISTORY (B, hist, obs); a fresh start
    # repeats the start observation across the history window
    obs_hist0 = obs0.unsqueeze(1).expand(-1, 4, -1).contiguous()
    a1 = base_action_rollout(wm, pi, mdp, S3, f3, xi3, normalize_xi(xi3), obs_hist0, 4, 8)
    a2 = base_action_rollout(wm, pi, mdp, S3, f3, xi3, normalize_xi(xi3), obs_hist0, 4, 8)
    check("base action rollout deterministic",
          bool(torch.equal(a1, a2)), f"max|diff|={float((a1 - a2).abs().max()):.2e}")
    check("base rollout emits (B,H,29) actions", tuple(a1.shape) == (4, 8, 29),
          f"shape={tuple(a1.shape)}")

    # --------------------------------- 8. J_total consistency with reward
    with torch.no_grad():
        p3 = open_loop_paths(wm, cr, mdp, S0, f0, a_seq.detach(), a_prev0, xi_n,
                             gamma=0.99, goal_frames=4)
        # straight recomputation of the same objective from first principles
        S_t = S0
        a_p = a_prev0
        c_prev = torch.ones(B, dtype=dtype)
        J_manual = 0.0
        Hs = a_seq.shape[1]
        for t in range(Hs):
            at = a_seq.detach()[:, t]
            S_t = wm.step(S_t, at, xi_n, use_residual=False)
            f1 = f0 + t + 1
            c_prev = c_prev * (1 - soft_term(mdp, S_t, f1))
            J_manual += 0.99 ** t * float((c_prev * mdp.reward(S_t, f1, at, a_p)).sum())
            a_p = at
        goal_H = mdp.motion.goal(f0 + Hs, 4)
        vin = torch.cat((mdp.critic_features(S_t, f0 + Hs, xi_n), goal_H), -1)
        J_manual += 0.99 ** Hs * float((c_prev * cr(vin)).sum())
    check("J_total matches manual recomputation",
          abs(J_manual - p3["J_total"]) / max(abs(J_manual), 1e-8) < 1e-5,
          f"manual={J_manual:.6f} paths={p3['J_total']:.6f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
