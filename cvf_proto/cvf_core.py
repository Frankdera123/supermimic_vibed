"""CVF-0 core library: 70-D tangent-space perturbation, Critic gradient, FK clearance screening,
WM continuation rollout (strictly aligned with training survival indexing c_t * r_t), and
clip-clustered bootstrap / LOCO statistics.
"""
from __future__ import annotations

import torch

FOOT_BALL_LOCAL_OFFSETS = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)
FOOT_BALL_RADIUS = 0.005

VEL_IDX = list(range(6, 12)) + list(range(41, 70))
POSE_IDX = list(range(0, 6)) + list(range(12, 41))
# Pose DOFs that cannot alter foot sole z-height: dp_xy (0:2), dth_yaw (5:6), waist+arms dq[12:29] (24:41)
POSE_ZERO_PENETRATION_IDX = [0, 1, 5] + list(range(24, 41))


def perturb_state(S, delta: torch.Tensor):
    """Apply 70-D local tangent perturbation S (+) delta matching WorldModel.step coordinates."""
    from supermimic.rotations import so3_exp, yaw_mat

    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
    H_y = yaw_mat(R)
    p1 = p + (H_y @ dp.unsqueeze(-1)).squeeze(-1)
    R1 = R @ so3_exp(dth)
    v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
    w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
    return (p1, R1, v1, w1, q + dq, qd + dqd)


def critic_tangent_grad(critic, mdp, motion, proto, S, f, xi_n,
                        clamp_vmax: float | None = 10.0) -> dict:
    """Exact autograd gradient of V_psi(S (+) delta) w.r.t. delta at delta=0."""
    B = S[0].shape[0]
    dev = S[0].device
    S_det = tuple(x.detach() for x in S)
    delta = torch.zeros(B, 70, device=dev, dtype=S[0].dtype, requires_grad=True)
    S_p = perturb_state(S_det, delta)
    gw = proto.wm_goal(f, S_p)
    vin = proto.critic_input(S_p, f, xi_n, gw)
    v_raw = critic(vin)
    v_eff = v_raw.clamp(max=clamp_vmax) if clamp_vmax is not None else v_raw
    grad_raw = torch.autograd.grad(v_raw.sum(), delta, retain_graph=True)[0].detach()
    grad_eff = torch.autograd.grad(v_eff.sum(), delta, retain_graph=False)[0].detach()
    clamp_active = (v_raw >= clamp_vmax) if clamp_vmax is not None else torch.zeros(B, dtype=torch.bool, device=dev)
    return {
        "v_raw": v_raw.detach(),
        "v_eff": v_eff.detach(),
        "grad_raw": grad_raw,
        "grad_eff": grad_eff,
        "clamp_active": clamp_active,
    }


def foot_ball_clearances(kin, S) -> torch.Tensor:
    """Compute world-z clearance above ground (z_world - radius) for all 8 MJCF foot spheres."""
    p, R, v, w, q, qd = S
    bp, bR, _, _ = kin.forward(p, R, v, w, q, qd, all_bodies=True)
    l_idx = kin.body_names.index("left_ankle_roll_link")
    r_idx = kin.body_names.index("right_ankle_roll_link")
    offs = torch.tensor(FOOT_BALL_LOCAL_OFFSETS, device=p.device, dtype=p.dtype)  # (4, 3)
    clears = []
    for idx in (l_idx, r_idx):
        ankle_p = bp[:, idx]    # (B, 3)
        ankle_R = bR[:, idx]    # (B, 3, 3)
        # world_pos = ankle_p + ankle_R @ offs^T
        world_p = ankle_p.unsqueeze(1) + torch.einsum("bij,kj->bki", ankle_R, offs)
        clears.append(world_p[..., 2] - FOOT_BALL_RADIUS)
    return torch.cat(clears, dim=-1)  # (B, 8)


def joint_limit_slack(kin, q: torch.Tensor) -> torch.Tensor:
    """Per-joint distance to nearest hard joint limit [q_lo, q_hi]."""
    return torch.minimum(q - kin.q_lo.to(q.device), kin.q_hi.to(q.device) - q)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def screen_safe_direction(kin, S, d_raw: torch.Tensor, eps_max: float, subspace: str,
                          min_foot_clearance: float = 0.010,
                          min_joint_slack: float = 0.010) -> tuple[torch.Tensor, dict]:
    """Project raw 70-D direction onto a physically safe subspace and verify zero-penetration
    and joint-limit bounds across [-eps_max, +eps_max].
    """
    B = d_raw.shape[0]
    dev = d_raw.device
    q = S[4]
    slack = joint_limit_slack(kin, q)  # (B, 29)

    if subspace == "vel":
        mask = torch.zeros(B, 70, device=dev, dtype=d_raw.dtype)
        mask[:, VEL_IDX] = 1.0
        u = _normalize_rows(d_raw * mask)
        return u, {"mode": ["vel"] * B, "min_clear_pm": [float("nan")] * B, "min_slack_pm": [float("nan")] * B}

    if subspace != "pose_safe":
        raise ValueError(f"Unknown subspace: {subspace}")

    u_out = torch.zeros_like(d_raw)
    modes = []
    min_clears = []
    min_slacks = []
    for i in range(B):
        Si = tuple(x[i:i + 1] for x in S)
        di = d_raw[i:i + 1].clone()
        # Try full pose first only if base state is clearly airborne, else zero-penetration subset
        base_clear = float(foot_ball_clearances(kin, Si).min())
        chosen_mode = "pose_zero_pen"
        candidate_idx = POSE_ZERO_PENETRATION_IDX
        if base_clear > min_foot_clearance + 0.025:
            # Test full pose with joint-limit screening
            m_full = torch.zeros(1, 70, device=dev, dtype=d_raw.dtype)
            m_full[0, POSE_IDX] = 1.0
            ui = _normalize_rows(di * m_full)
            # Zero out any joint component that would violate slack at +/- eps_max
            dq = ui[0, 12:41]
            bad_j = (slack[i] - eps_max * dq.abs()) < min_joint_slack
            if bad_j.any():
                m_full[0, 12:41][bad_j] = 0.0
                if m_full.sum() > 0:
                    ui = _normalize_rows(di * m_full)
            c_plus = float(foot_ball_clearances(kin, perturb_state(Si, eps_max * ui)).min())
            c_minus = float(foot_ball_clearances(kin, perturb_state(Si, -eps_max * ui)).min())
            if min(c_plus, c_minus) >= min_foot_clearance:
                chosen_mode = "pose_airborne_full"
                candidate_idx = POSE_IDX

        m_use = torch.zeros(1, 70, device=dev, dtype=d_raw.dtype)
        m_use[0, candidate_idx] = 1.0
        ui = _normalize_rows(di * m_use)
        # Enforce joint limit safety iteratively (renormalizing changes remaining components)
        for _ in range(3):
            dq = ui[0, 12:41]
            bad_j = (slack[i] - eps_max * dq.abs()) < min_joint_slack
            if not bad_j.any():
                break
            m_use[0, 12:41][bad_j] = 0.0
            if m_use.sum() == 0:
                m_use[0, [0, 1, 5]] = 1.0
                ui = _normalize_rows(di * m_use)
                break
            ui = _normalize_rows(di * m_use)

        c_p = float(foot_ball_clearances(kin, perturb_state(Si, eps_max * ui)).min())
        c_m = float(foot_ball_clearances(kin, perturb_state(Si, -eps_max * ui)).min())
        s_p = float(joint_limit_slack(kin, perturb_state(Si, eps_max * ui)[4]).min())
        s_m = float(joint_limit_slack(kin, perturb_state(Si, -eps_max * ui)[4]).min())
        u_out[i] = ui[0]
        modes.append(chosen_mode)
        min_clears.append(min(c_p, c_m))
        min_slacks.append(min(s_p, s_m))

    return u_out, {"mode": modes, "min_clear_pm": min_clears, "min_slack_pm": min_slacks}


def rollout_wm_continuation(wm, pi, mdp, proto, S_base, f0, o_hist_prev9, a_prev0,
                            xi, xi_n, T: int, gamma: float = 0.99,
                            tz: float = 0.02, tt: float = 0.04,
                            hist_mode: str = "last_only",
                            delta: torch.Tensor | None = None,
                            need_grad: bool = False,
                            horizons_out: tuple[int, ...] = (16, 32, 64, 128, 200)) -> dict:
    """Closed-loop WM continuation rollout from S_base (+) delta.

    Strictly matches training survival indexing:
        c_0 = 1.0,  c_t = prod_{k=0}^{t-1} (1 - d_k),  J = sum_{t=0}^{T-1} gamma^t * c_t * r_t
    """
    B = S_base[0].shape[0]
    dev = S_base[0].device
    S_det = tuple(x.detach() for x in S_base)
    if delta is None:
        delta = torch.zeros(B, 70, device=dev, dtype=S_base[0].dtype, requires_grad=need_grad)
    elif need_grad and not delta.requires_grad:
        delta = delta.detach().requires_grad_(True)

    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        S0 = perturb_state(S_det, delta)
        o_cur = proto.obs(S0, a_prev0.detach(), xi, f0)
        if hist_mode == "last_only":
            o_hist = torch.cat((o_hist_prev9.detach(), o_cur.unsqueeze(1)), dim=1)
        elif hist_mode == "repeat10":
            o_hist = o_cur.unsqueeze(1).expand(-1, o_hist_prev9.shape[1] + 1, -1).contiguous()
        else:
            raise ValueError(f"Unknown hist_mode: {hist_mode}")

        S = S0
        ap = a_prev0.detach()
        f = f0.clone()
        c_soft = torch.ones(B, device=dev, dtype=S_base[0].dtype)
        c_hard = torch.ones(B, device=dev, dtype=S_base[0].dtype)
        J_soft = torch.zeros(B, device=dev, dtype=S_base[0].dtype)
        J_hard = torch.zeros(B, device=dev, dtype=S_base[0].dtype)
        by_h_soft = {}
        by_h_hard = {}
        alive_by_h = {}
        for t in range(T):
            g = proto.actor_goal(f, S)
            a = torch.clamp(pi(o_hist, g), -20.0, 20.0)
            S = proto.wm_step(wm, S, a, xi_n)
            f = f + 1
            r = mdp.reward(S, f, a, ap)
            d_s = mdp.soft_termination(S, f, tz=tz, tt=tt)
            d_h = mdp.termination(S, f).to(r.dtype)
            # Training-aligned survival: c_t multiplies r_t BEFORE multiplying by (1 - d_t)
            J_soft = J_soft + (gamma ** t) * c_soft * r
            J_hard = J_hard + (gamma ** t) * c_hard * r
            c_soft = c_soft * (1.0 - d_s)
            c_hard = c_hard * (1.0 - d_h)
            ap = a
            o1 = proto.obs(S, a, xi, f)
            o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), dim=1)
            step = t + 1
            if step in horizons_out:
                by_h_soft[step] = J_soft
                by_h_hard[step] = J_hard.detach()
                alive_by_h[step] = (c_soft.detach(), c_hard.detach())

        grads_by_h = {}
        if need_grad:
            for h in sorted(by_h_soft):
                retain = (h != max(by_h_soft))
                g_h = torch.autograd.grad(by_h_soft[h].sum(), delta, retain_graph=retain)[0]
                grads_by_h[h] = g_h.detach()

    return {
        "J_soft": {h: v.detach() for h, v in by_h_soft.items()},
        "J_hard": by_h_hard,
        "alive": alive_by_h,
        "grads_soft": grads_by_h,
        "S_end": tuple(x.detach() for x in S),
    }


def clip_clustered_bootstrap(values: torch.Tensor, clip_ids: list[str],
                             n_boot: int = 10000, seed: int = 2700) -> dict:
    """Clip-clustered bootstrap CI95 (resampling clips with replacement)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    uniq = sorted(set(clip_ids))
    by_clip = {c: torch.tensor([i for i, cid in enumerate(clip_ids) if cid == c], dtype=torch.long)
               for c in uniq}
    g = torch.Generator().manual_seed(seed)
    boots = torch.empty(n_boot, dtype=torch.float64)
    vals = values.double().cpu()
    for b in range(n_boot):
        picks = torch.randint(0, len(uniq), (len(uniq),), generator=g).tolist()
        idx = torch.cat([by_clip[uniq[p]] for p in picks])
        boots[b] = vals[idx].mean()
    return {
        "n": int(n),
        "n_clips": len(uniq),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "ci95": [float(torch.quantile(boots, 0.025)), float(torch.quantile(boots, 0.975))],
    }


def loco_summary(values: torch.Tensor, clip_ids: list[str]) -> dict:
    """Leave-one-clip-out mean range across clips."""
    uniq = sorted(set(clip_ids))
    out = {}
    vals = values.double().cpu()
    for drop in uniq:
        keep = [i for i, c in enumerate(clip_ids) if c != drop]
        out[f"drop_{drop}"] = float(vals[keep].mean()) if keep else float("nan")
    return {
        "per_drop": out,
        "min_mean": min(out.values()) if out else float("nan"),
        "max_mean": max(out.values()) if out else float("nan"),
    }
