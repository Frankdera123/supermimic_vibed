"""CPU unit tests for CVF-0 core library (no GPU, no simulator, no checkpoints needed)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CVF_ROOT = _HERE.parent
_REPO = _CVF_ROOT.parent
sys.path.insert(0, str(_CVF_ROOT))
sys.path.insert(0, str(_REPO / "pada_proto"))

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402
from cvf_core import (  # noqa: E402
    POSE_ZERO_PENETRATION_IDX,
    VEL_IDX,
    clip_clustered_bootstrap,
    critic_tangent_grad,
    foot_ball_clearances,
    joint_limit_slack,
    loco_summary,
    perturb_state,
    rollout_wm_continuation,
    screen_safe_direction,
)
from supermimic.mdp import normalize_xi  # noqa: E402
from supermimic.networks import Critic, Policy, WorldModel  # noqa: E402


def _make_synthetic_stack(dev="cpu"):
    world = bridge.setup_world(device=dev)
    motion, mdp, proto, kin = world["motion"], world["mdp"], world["proto"], world["kin"]
    torch.manual_seed(123)
    wm = WorldModel().to(dev).eval()
    # Non-zero small weights so WM has non-trivial dynamics
    for p in wm.parameters():
        p.data.normal_(0.0, 0.02)
    pi = Policy(93, 730, hist=10).to(dev).eval()
    for p in pi.parameters():
        p.data.normal_(0.0, 0.02)
    f_probe = motion.clip_start[:2]
    S_probe = motion.ref_state(f_probe)
    xi_n = normalize_xi(torch.tensor([[1.0] + [0.0] * 32] * 2, device=dev))
    vin = proto.critic_input(S_probe, f_probe, xi_n, proto.wm_goal(f_probe, S_probe))
    critic = Critic(vin.shape[-1]).to(dev).eval()
    for p in critic.parameters():
        p.data.normal_(0.0, 0.05)
    return motion, mdp, proto, kin, wm, pi, critic


def test_perturb_identity_and_critic_grad_vs_fd():
    motion, mdp, proto, kin, wm, pi, critic = _make_synthetic_stack()
    f0 = motion.clip_start[:3] + 25
    S0 = motion.ref_state(f0)
    xi_n = normalize_xi(torch.tensor([[1.0] + [0.0] * 32] * 3))

    # 1. Zero delta identity
    S_same = perturb_state(S0, torch.zeros(3, 70))
    for a, b in zip(S0, S_same):
        assert torch.allclose(a, b, atol=1e-6), "perturb_state(S, 0) != S"

    # 2. Critic autograd tangent gradient vs central FD (at float64 precision)
    critic64 = critic.double()
    S0_64 = tuple(x.double() for x in S0)
    xi_n_64 = xi_n.double()
    kin64 = kin
    kin64.axes = kin64.axes.double(); kin64.off_p = kin64.off_p.double(); kin64.off_R = kin64.off_R.double()
    motion.root_p = motion.root_p.double(); motion.root_R = motion.root_R.double()
    motion.root_v = motion.root_v.double(); motion.root_w = motion.root_w.double()
    motion.root_v_local = motion.root_v_local.double(); motion.root_w_local = motion.root_w_local.double()
    motion.q = motion.q.double(); motion.qd = motion.qd.double()
    motion.body_p = motion.body_p.double(); motion.body_R = motion.body_R.double()
    motion.body_v = motion.body_v.double(); motion.body_w = motion.body_w.double()

    res = critic_tangent_grad(critic64, mdp, motion, proto, S0_64, f0, xi_n_64, clamp_vmax=None)
    g_auto = res["grad_raw"]
    g = torch.Generator().manual_seed(7)
    u = torch.randn(3, 70, generator=g, dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    eps = 1e-5
    with torch.no_grad():
        Sp = perturb_state(S0_64, eps * u)
        Sm = perturb_state(S0_64, -eps * u)
        vp = critic64(proto.critic_input(Sp, f0, xi_n_64, proto.wm_goal(f0, Sp)))
        vm = critic64(proto.critic_input(Sm, f0, xi_n_64, proto.wm_goal(f0, Sm)))
    d_fd = (vp - vm) / (2 * eps)
    d_auto = (g_auto * u).sum(-1)
    rel = ((d_fd - d_auto).abs() / d_auto.abs().clamp_min(1e-6)).max().item()
    assert rel < 1e-4, f"Critic tangent grad vs FD mismatch: rel={rel}"

    # Restore float32 for downstream tests
    critic.float()
    kin64.axes = kin64.axes.float(); kin64.off_p = kin64.off_p.float(); kin64.off_R = kin64.off_R.float()
    motion.root_p = motion.root_p.float(); motion.root_R = motion.root_R.float()
    motion.root_v = motion.root_v.float(); motion.root_w = motion.root_w.float()
    motion.root_v_local = motion.root_v_local.float(); motion.root_w_local = motion.root_w_local.float()
    motion.q = motion.q.float(); motion.qd = motion.qd.float()
    motion.body_p = motion.body_p.float(); motion.body_R = motion.body_R.float()
    motion.body_v = motion.body_v.float(); motion.body_w = motion.body_w.float()


def test_survival_indexing_matches_training():
    motion, mdp, proto, kin, wm, pi, critic = _make_synthetic_stack()
    f0 = motion.clip_start[:2] + 30
    S0 = motion.ref_state(f0)
    xi = torch.zeros(2, 33); xi[:, 0] = 1.0
    xi_n = normalize_xi(xi)
    ap0 = torch.zeros(2, 29)
    o0 = proto.obs(S0, ap0, xi, f0)
    oh9 = o0.unsqueeze(1).expand(-1, 9, -1).contiguous()
    T = 6
    out = rollout_wm_continuation(
        wm, pi, mdp, proto, S0, f0, oh9, ap0, xi, xi_n, T=T,
        gamma=0.99, hist_mode="last_only", need_grad=True, horizons_out=(T,)
    )
    # Independent manual calculation matching train.py:2937-2941
    with torch.no_grad():
        S = tuple(x.clone() for x in S0)
        oh = torch.cat((oh9, o0.unsqueeze(1)), 1)
        ap = ap0.clone()
        f = f0.clone()
        rs, ds = [], []
        for _ in range(T):
            a = torch.clamp(pi(oh, proto.actor_goal(f, S)), -20.0, 20.0)
            S = proto.wm_step(wm, S, a, xi_n)
            f = f + 1
            rs.append(mdp.reward(S, f, a, ap))
            ds.append(mdp.soft_termination(S, f))
            ap = a
            oh = torch.cat((oh[:, 1:], proto.obs(S, a, xi, f).unsqueeze(1)), 1)
        r = torch.stack(rs)
        d = torch.stack(ds)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, 2), alive[:-1]), 0)
        disc = 0.99 ** torch.arange(T, dtype=torch.float32).unsqueeze(-1)
        J_ref = (disc * c_t * r).sum(0)
    diff = (out["J_soft"][T] - J_ref).abs().max().item()
    assert diff < 1e-6, f"Survival indexing mismatch vs train.py: {diff}"

    # Check WM continuation autograd vs central FD (eps=5e-3 in float32)
    g = torch.Generator().manual_seed(9)
    u = torch.randn(2, 70, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    eps = 5e-3
    Jp = rollout_wm_continuation(wm, pi, mdp, proto, S0, f0, oh9, ap0, xi, xi_n, T=T,
                                 delta=eps * u, need_grad=False, horizons_out=(T,))["J_soft"][T]
    Jm = rollout_wm_continuation(wm, pi, mdp, proto, S0, f0, oh9, ap0, xi, xi_n, T=T,
                                 delta=-eps * u, need_grad=False, horizons_out=(T,))["J_soft"][T]
    d_fd = (Jp - Jm) / (2 * eps)
    d_auto = (out["grads_soft"][T] * u).sum(-1)
    rel = ((d_fd - d_auto).abs() / d_auto.abs().clamp_min(1e-4)).max().item()
    assert rel < 1e-2, f"WM continuation autograd vs FD mismatch: rel={rel}"


def test_foot_clearance_and_safe_screening():
    motion, mdp, proto, kin, _, _, _ = _make_synthetic_stack()
    f0 = motion.clip_start[:4] + 20
    S0 = motion.ref_state(f0)
    c0 = foot_ball_clearances(kin, S0)
    assert c0.shape == (4, 8)
    # Raising pelvis z by +0.20 m should increase all 8 ball clearances by exactly +0.20 m
    S_up = (S0[0] + torch.tensor([0.0, 0.0, 0.20]), *S0[1:])
    c_up = foot_ball_clearances(kin, S_up)
    assert torch.allclose(c_up - c0, torch.full_like(c0, 0.20), atol=1e-5)

    # Zero-penetration pose direction must preserve exact foot clearances (up to yaw/xy invariance)
    g = torch.Generator().manual_seed(11)
    d_raw = torch.randn(4, 70, generator=g)
    u_pose, meta_p = screen_safe_direction(kin, S0, d_raw, eps_max=0.05, subspace="pose_safe",
                                           min_foot_clearance=0.010, min_joint_slack=0.010)
    assert torch.allclose(u_pose.norm(dim=-1), torch.ones(4), atol=1e-5)
    for i in range(4):
        assert meta_p["min_slack_pm"][i] >= 0.010 - 1e-5, f"Joint limit violated: {meta_p['min_slack_pm'][i]}"
        if meta_p["mode"][i] == "pose_zero_pen":
            # Verify no leg joint or p_z or roll/pitch component leaked
            non_zero = (u_pose[i].abs() > 1e-8).nonzero(as_tuple=False).squeeze(-1).tolist()
            assert set(non_zero).issubset(set(POSE_ZERO_PENETRATION_IDX))

    u_vel, _ = screen_safe_direction(kin, S0, d_raw, eps_max=0.05, subspace="vel")
    non_zero_v = (u_vel.abs() > 1e-8).any(0).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert set(non_zero_v).issubset(set(VEL_IDX))


def test_bootstrap_and_loco():
    vals = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    clips = ["a", "a", "b", "b", "c", "c"]
    bs = clip_clustered_bootstrap(vals, clips, n_boot=500, seed=0)
    assert abs(bs["mean"] - 4.0 / 6.0) < 1e-6
    assert bs["n_clips"] == 3
    loco = loco_summary(vals, clips)
    assert abs(loco["min_mean"] - 0.5) < 1e-6
    assert abs(loco["max_mean"] - 1.0) < 1e-6


if __name__ == "__main__":
    tests = [
        test_perturb_identity_and_critic_grad_vs_fd,
        test_survival_indexing_matches_training,
        test_foot_clearance_and_safe_screening,
        test_bootstrap_and_loco,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)}/{len(tests)} CVF-0 CPU TESTS PASSED")
