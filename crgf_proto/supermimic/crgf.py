"""Control-Relevant Gradient Fidelity (CRGF) core.

Directional-derivative decomposition of the imagined policy objective into
three gradient paths, shared direction sets, fidelity metrics, and fail-closed
reliability gating.  WM-side evaluations are exact autograd directional
derivatives; simulator-side evaluations pair this module with central finite
differences of open-loop action sequences (tools/crgf_sim_fd_audit.py).

Objective and paths.  For a prescribed open-loop action sequence
``A = (a_0..a_{H-1})`` rolled through the world model from a fixed start, the
imagined objective matches ``policy_update`` (train.py) with ``no_critic=False``:

    J(A) = sum_t gamma^t c_t (r_state_t + r_direct_t) + gamma^H c_{H-1} V(s_H)

with ``c_t`` the soft-termination survival, ``r_state`` the state-mediated
reward (body/root tracking + joint-limit) and ``r_direct`` the direct
action-reward term (action-rate).  Three path derivatives w.r.t. ``A``:

  D_state  grad of sum_t gamma^t c_t r_state_t
           (dynamics Jacobian x analytic reward; includes the d(c_t)/dA
           contribution times r_state)
  D_direct grad of sum_t gamma^t c_t.detach() r_direct_t
           (model-free: no WM Jacobian, c detached)
  D_critic grad of gamma^H c_{H-1}.detach() V(s_H)
           (critic function x WM state Jacobian at the horizon end)

The full objective derivative D_total differs from D_state + D_direct +
D_critic only by the detached cross terms (d(c_t)/dA * r_direct_t and
gamma^H d(c_{H-1})/dA * V), reported as ``cross_residual``.
"""

from __future__ import annotations

import inspect

import torch

from .mdp import NUM_J
from .motion import CONTROL_DT


# ------------------------------------------------------------- protocol adapter
class Protocol:
    """Unifies the two SuperMimic code variants this audit must support.

    Main repo (supermimic_vibed): frame-index goal ``motion.goal(f, K)``,
    ``mdp.obs(S, a, xi, f)``, ``mdp.critic_features(S, f, xi_n)`` concatenated
    with the goal, ``wm.step(..., use_residual=...)``.
    OPCL proto: state-relative goal ``motion.goal(f, p, R) -> (actor, wm_goal)``
    (GOAL_FRAMES x 73 actor + 3 per frame for the critic), ``mdp.obs(S, a, xi)``,
    ``mdp.critic_features(S, f, xi_n, wm_goal)``, ``wm.step(S, a, xi_n)``.
    """

    def __init__(self, mdp, goal_frames: int, v_max: float | None = None):
        self.mdp = mdp
        self.motion = mdp.motion
        self.K = goal_frames
        self.v_max = v_max
        self.obs_has_f = len(inspect.signature(mdp.obs).parameters) >= 4
        goal_params = inspect.signature(self.motion.goal).parameters
        self.goal_takes_state = len([p for p in goal_params if p != "self"]) >= 3
        cf_params = [p for p in inspect.signature(mdp.critic_features).parameters if p != "self"]
        self.cf_takes_goal = len(cf_params) >= 4

    def wm_step(self, wm, S, a, xi_n):
        """WM step with optional use_residual kwarg (proto lacks it)."""
        params = [p for p in inspect.signature(wm.step).parameters if p != "self"]
        if "use_residual" in params:
            return wm.step(S, a, xi_n, use_residual=False)
        return wm.step(S, a, xi_n)

    def obs(self, S, a_prev, xi, f):
        if self.obs_has_f:
            return self.mdp.obs(S, a_prev, xi, f)
        return self.mdp.obs(S, a_prev, xi)

    def actor_goal(self, f, S=None):
        """Policy goal at frame f (state-relative in the proto)."""
        if not self.goal_takes_state:
            return self.motion.goal(f, self.K)
        return self.motion.goal(f, S[0], S[1])[0]

    def wm_goal(self, f, S):
        """Critic goal at frame f (state-relative in the proto)."""
        if not self.goal_takes_state:
            return self.motion.goal(f, self.K)
        return self.motion.goal(f, S[0], S[1])[1]

    def critic_input(self, S, f, xi_n, gw):
        if not self.cf_takes_goal:
            return torch.cat((self.mdp.critic_features(S, f, xi_n), gw), -1)
        v = self.mdp.critic_features(S, f, xi_n, gw)
        return v.clamp(max=self.v_max) if self.v_max is not None else v


# ---------------------------------------------------------------- reward split
def reward_split(mdp, S, f, a, a_prev):
    """Exact decomposition of ``mdp.reward`` into state-mediated and direct parts.

    Uses the ``terms=True`` diagnostics of ``mdp.reward`` so the split is
    consistent with the training reward by construction:
      r_total   = track*dt + w_action_rate * rate * dt + w_joint_limit * lim * dt
      r_state   = track*dt + w_joint_limit * lim * dt          (only via S)
      r_direct  = w_action_rate * rate * dt                    (only via a)
    """
    r, terms = mdp.reward(S, f, a, a_prev, terms=True)
    r_state = terms["track"] + mdp.cfg.w_joint_limit * terms["joint_limit"] * CONTROL_DT
    r_direct = mdp.cfg.w_action_rate * terms["action_rate"] * CONTROL_DT
    return r_state, r_direct


def soft_term(mdp, S, f, tz: float = 0.02, tt: float = 0.04):
    """Differentiable termination probability used by imagined rollouts."""
    return mdp.soft_termination(S, f, tz=tz, tt=tt)


def ori_cosine_stats(mdp, S, f, margin: float = 1e-5):
    """Per-body relative-rotation cosines WITHOUT the acos clamp (diagnostic).

    The training reward uses ``rot_angle_sq`` (rotations.py), which clamps
    ``acos`` at ``cos = 1 - 1e-6``; whenever the tracked bodies are nearly
    aligned with the reference the reward landscape has a kink and central
    finite differences of the true response are unreliable there.  This helper
    reproduces only the relative-rotation computation of ``mdp.reward`` to flag
    such samples; it never enters the objective itself.

    Returns (cos: (B, n_body), clamp_hi: (B,) bool, clamp_lo: (B,) bool) where
    ``clamp_*`` marks any body outside ``[-1+margin, 1-margin]``.
    """
    from .rotations import yaw_mat

    p, R, v, w, q, qd = S
    _, bR, _, _ = mdp.kin.forward(p, R, v, w, q, qd)
    m = mdp.motion
    rp, rR = m.root_p[f], m.root_R[f]
    dR = yaw_mat(R @ rR.transpose(-1, -2))
    ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
    tr = (ref_bR * bR).sum((-1, -2))  # trace(ref_bR^T bR) per body
    cos = (tr - 1.0) / 2.0
    hi = (cos > 1.0 - margin).any(-1)
    lo = (cos < -1.0 + margin).any(-1)
    return cos, hi, lo


# ------------------------------------------------------------------- rollout
def open_loop_paths(wm, critic, mdp, S0, f0, a_seq, a_prev0, xi_n,
                    gamma: float = 0.99, tz: float = 0.02, tt: float = 0.04,
                    critic_weight: float = 1.0, goal_frames: int = 4,
                    record_states: bool = False, v_max: float | None = None,
                    action_gain: float = 1.0):
    """Roll the WM open-loop on prescribed actions and build all path graphs.

    ``a_seq``: (B, H, A) leaf tensor with ``requires_grad``.  The critic input
    at the horizon end matches ``policy_update``: ``_critic_input(S, f, xi_n,
    g)`` with the goal refreshed at the final imagined frame
    (``mdp.motion.goal(f0 + H, goal_frames)``).  Returns graph tensors and
    diagnostics; derivatives are obtained via ``torch.autograd.grad`` with
    ``retain_graph=True`` on the shared graph.
    """
    B, H, A = a_seq.shape
    proto = Protocol(mdp, goal_frames, v_max=v_max)
    S = S0
    a_prev = a_prev0
    c_prev = torch.ones(B, device=a_seq.device)
    L_state = a_seq.new_zeros(())
    L_direct_attached = a_seq.new_zeros(())
    L_direct_iso = a_seq.new_zeros(())
    J_state_vec = a_seq.new_zeros(B)
    J_direct_vec = a_seq.new_zeros(B)
    disc = [gamma ** t for t in range(H)]
    r_state_steps, r_direct_steps, d_steps, c_steps, hard_steps = [], [], [], [], []
    state_steps = [] if record_states else None
    S_H = None
    f_H = None
    for t in range(H):
        a_t = a_seq[:, t]
        # ``action_gain`` mirrors train.py's ``wm_action_effect_gain``: the WM
        # input is the SCALED action while rewards keep the ORIGINAL action
        # (the exact semantic split that the historical gain experiment used).
        a_wm = torch.clamp(a_t * action_gain, -20.0, 20.0)
        S = proto.wm_step(wm, S, a_wm, xi_n)
        if record_states:
            state_steps.append(tuple(x.detach() for x in S))
        f1 = f0 + t + 1
        r_state, r_direct = reward_split(mdp, S, f1, a_t, a_prev)
        d = soft_term(mdp, S, f1, tz=tz, tt=tt)
        c = c_prev * (1.0 - d)
        L_state = L_state + disc[t] * (c * r_state).sum()
        L_direct_attached = L_direct_attached + disc[t] * (c * r_direct).sum()
        L_direct_iso = L_direct_iso + disc[t] * (c.detach() * r_direct).sum()
        J_state_vec = J_state_vec + disc[t] * (c * r_state).detach()
        J_direct_vec = J_direct_vec + disc[t] * (c * r_direct).detach()
        r_state_steps.append(r_state.detach())
        r_direct_steps.append(r_direct.detach())
        d_steps.append(d.detach())
        c_steps.append(c.detach())
        hard_steps.append(mdp.termination(S, f1).detach())
        a_prev = a_t
        c_prev = c
        S_H, f_H = S, f1
    L_critic_iso = a_seq.new_zeros(())
    L_critic_attached = a_seq.new_zeros(())
    L_cross_iso = a_seq.new_zeros(())
    J_critic_vec = a_seq.new_zeros(B)
    V = None
    if critic is not None:
        gw = proto.wm_goal(f_H, S_H)
        vin = proto.critic_input(S_H, f_H, xi_n, gw)
        V = critic(vin)
        L_critic_iso = (gamma ** H) * critic_weight * (c_prev.detach() * V).sum()
        J_critic_vec = (gamma ** H) * critic_weight * (c_prev * V).detach()
        L_critic_attached = (gamma ** H) * critic_weight * (c_prev * V).sum()
        L_cross_iso = L_cross_iso + (gamma ** H) * critic_weight * (c_prev * V.detach()).sum()
    # L_cross_iso carries exactly the detached cross terms that separate
    # D_total from D_state + D_direct + D_critic:
    #   grad(L_cross_iso) = sum_t gamma^t r_direct_t * d(c_t)/dA
    #                     + gamma^H V * d(c_H)/dA   (values detached, c attached)
    L_cross_iso = L_cross_iso + (L_direct_attached - L_direct_iso)
    L_total = L_state + L_direct_attached + L_critic_attached
    with torch.no_grad():
        J_return = float((L_state + L_direct_attached).item())
        J_total = float(L_total.item())
    return {
        "graphs": (L_state, L_direct_iso, L_critic_iso, L_total, L_cross_iso),
        "J_return": J_return,
        "J_total": J_total,
        "J_return_vec": J_state_vec + J_direct_vec,
        "J_total_vec": J_state_vec + J_direct_vec + J_critic_vec,
        "r_state": torch.stack(r_state_steps),      # (H, B)
        "r_direct": torch.stack(r_direct_steps),
        "soft_d": torch.stack(d_steps),
        "alive_c": torch.stack(c_steps),
        "hard_done": torch.stack(hard_steps),
        "states": state_steps,
        "S_H": S_H,
        "V": V.detach() if V is not None else None,
    }


def path_directional_derivatives(paths, a_seq) -> dict:
    """Exact directional derivatives of each path w.r.t. the action sequence.

    Returns flattened gradient vectors (B, H*A) for state / direct / critic /
    total paths plus the exact cross residual of the decomposition
    (``grad(L_cross_iso)``).
    """
    L_state, L_direct_iso, L_critic_iso, L_total, L_cross_iso = paths["graphs"]
    B, H, A = a_seq.shape
    outs = {}
    for name, L in (("state", L_state), ("direct", L_direct_iso),
                    ("critic", L_critic_iso), ("total", L_total),
                    ("cross", L_cross_iso)):
        if not L.requires_grad:
            outs[name] = torch.zeros(B, H * A, device=a_seq.device)
            continue
        g = torch.autograd.grad(L, a_seq, retain_graph=True, allow_unused=False)[0]
        outs[name] = g.reshape(B, H * A)
    outs["cross_residual"] = outs["cross"]
    return outs


# ----------------------------------------------------------------- directions
def normalize_dir_seqs(d: torch.Tensor) -> torch.Tensor:
    """Unit-normalize each sample over the flattened (H*A) action sequence."""
    flat = d.reshape(d.shape[0], -1)
    return d / flat.norm(dim=-1).clamp_min(1e-8).view(-1, 1, 1)


def random_dir_seqs(B: int, H: int, A: int, seed: int, mode: str = "per_step",
                    device="cpu") -> torch.Tensor:
    """Random unit action-sequence directions.

    ``per_step``: iid unit directions per step; ``repeated``: one unit
    direction held across all steps (coherent push, closest analogue of the
    one-step audits).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "per_step":
        d = torch.randn(B, H, A, generator=g)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    elif mode == "repeated":
        d0 = torch.randn(B, A, generator=g)
        d0 = d0 / d0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        d = d0.unsqueeze(1).expand(B, H, A).clone()
    else:
        raise ValueError(f"unknown random direction mode: {mode}")
    return normalize_dir_seqs(d.to(device))


def delta_dir_seqs(a_new: torch.Tensor, a_old: torch.Tensor):
    """Unit-normalized policy-update action deltas da = pi_new(o) - pi_old(o).

    Returns (unit directions (B, H, A), original delta norms (B,)).
    """
    d = a_new - a_old
    return normalize_dir_seqs(d), d.reshape(d.shape[0], -1).norm(dim=-1)


def ascent_dirs(grad_total: torch.Tensor, B: int, H: int, A: int) -> torch.Tensor:
    """Unit-normalized WM steepest-ascent directions of the full objective."""
    return normalize_dir_seqs(grad_total.reshape(B, H, A))


# -------------------------------------------------------------------- metrics
def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties get the mean rank) of a 1-D tensor."""
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, device=x.device, dtype=torch.float64)
    sx = x[order].double()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and bool(sx[j + 1] == sx[i]):
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.square().sum() * ry.square().sum()).sqrt().clamp_min(1e-12)
    return float((rx * ry).sum() / denom)


def fidelity_stats(d_wm: torch.Tensor, d_sim: torch.Tensor, gate: torch.Tensor):
    """Fidelity metrics on gated (decidable) samples only; fail-closed.

    d_wm / d_sim: (N,) directional derivative pairs; gate: (N,) bool tensor or
    None (no gating).
    """
    stats = {"n": int(d_wm.numel())}
    if gate is None:
        gate = torch.ones(d_wm.numel(), dtype=torch.bool)
    stats["n_gated"] = int(gate.sum())
    if gate.sum() == 0:
        stats.update({"sign_agreement": None, "mae": None, "rmse": None,
                      "median_ratio": None, "spearman": None})
        return stats
    w, s = d_wm[gate], d_sim[gate]
    decidable = (w.abs() + s.abs()) > 0
    stats["sign_agreement"] = (
        float(((torch.sign(w) == torch.sign(s)) & decidable).sum() / decidable.sum())
        if decidable.any() else None)
    stats["mae"] = float((w - s).abs().mean())
    stats["rmse"] = float((w - s).square().mean().sqrt())
    stats["median_ratio"] = float((w.abs() / s.abs().clamp_min(1e-8)).median())
    stats["spearman"] = spearman(w, s)
    return stats


# ---------------------------------------------------------- reliability gate
def reliability_gate(base_returns: torch.Tensor, d_values: torch.Tensor,
                     base_tol: float = 1e-5, d_floor: float = 5e-4,
                     d_rel: float = 0.20):
    """Fail-closed screening for finite-difference directional derivatives.

    ``base_returns``: (R,) repeats of the unperturbed return for one sample.
    ``d_values``: (M,) pooled D estimates across eps values, repetitions and
    execution orders.  A sample passes only if
      (a) baseline repeat range <= ``base_tol``;
      (b) pooled D range <= max(``d_floor``, ``d_rel`` * |mean(D)|);
      (c) |mean(D)| > ``d_floor`` (above the numerical deadband).
    Returns (pass: bool, reason: str).  Thresholds follow the exploratory
    protocol of the 2026-11-18 OPCL reproducibility fix; they are screening
    rules, not statistical proofs of differentiability.
    """
    if base_returns.numel() < 2:
        return False, "insufficient_baseline_repeats"
    if d_values.numel() < 2:
        return False, "insufficient_d_repeats"
    base_range = float(base_returns.max() - base_returns.min())
    if base_range > base_tol:
        return False, f"baseline_range={base_range:.3e}"
    d_mean = float(d_values.mean())
    d_range = float(d_values.max() - d_values.min())
    tol = max(d_floor, d_rel * abs(d_mean))
    if d_range > tol:
        return False, f"d_range={d_range:.3e}>tol={tol:.3e}"
    if abs(d_mean) <= d_floor:
        return False, f"below_deadband={abs(d_mean):.3e}"
    return True, "ok"


# ------------------------------------------------------------ policy rollout
@torch.no_grad()
def base_action_rollout(wm, pi, mdp, S0, f0, xi, xi_n, obs_hist0, goal_frames: int,
                        H: int, action_gain: float = 1.0):
    """Closed-loop policy rollout in the WM; returns the recorded action sequence.

    Matches the imagination loop of ``policy_update`` (noise-free observations,
    a_prev = 0 at the start, ``wm_action_effect_gain`` default 1.0).
    ``obs_hist0`` must be a (B, hist, obs_dim) observation history; for fresh
    motion-frame starts, repeat the start observation across the window.
    Works with both code variants via :class:`Protocol` (frame-index goal or
    state-relative actor goal).
    """
    proto = Protocol(mdp, goal_frames)
    S = S0
    a_prev = torch.zeros(S0[0].shape[0], NUM_J, device=S0[0].device,
                         dtype=obs_hist0.dtype)
    o_hist = obs_hist0
    f = f0
    acts = []
    for _ in range(H):
        g = proto.actor_goal(f, S)
        a = pi(o_hist, g)
        acts.append(a)
        a_model = torch.clamp(a * action_gain, -20.0, 20.0)
        S = proto.wm_step(wm, S, a_model, xi_n)
        f1 = f + 1
        o1 = proto.obs(S, a, xi, f1)
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        f = f1
        a_prev = a
    return torch.stack(acts, 1)  # (B, H, A)
