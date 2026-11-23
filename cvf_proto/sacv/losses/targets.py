"""SACV value-target math: survival indexing, multi-horizon discounted returns,
matched frozen-reference targets, and pair differences.

All functions are pure and float64-accurate; they implement exactly the frozen
contract in docs/cvf/sacv/CVF1A_DATA_CONTRACT.md §4:

  c_0 = 1,  c_t = prod_{k=0}^{t-1} (1 - d_k)          (training-aligned survival)
  G_T(x) = sum_{k=0}^{T-1} gamma^k * c_k * r_k        (independent target)
  Y_T(x) = G_T(x) + gamma^T * c_T * V_ref(x_T)        (matched training target)
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


def survival_soft(d: torch.Tensor) -> torch.Tensor:
    """c_t = prod_{k<t}(1 - d_k), c_0 = 1.  d: (B, T) termination in {0,1}.

    Note the training-aligned ordering: r_t is multiplied by c_t where c_t does
    NOT yet include step t's own termination (r is earned at the step, then the
    episode can end)."""
    B, T = d.shape
    ones = d.new_ones(B, dtype=DTYPE) if d.dtype != DTYPE else d.new_ones(B)
    dd = (1.0 - d.double())
    c = [ones]
    for t in range(1, T):
        c.append(c[-1] * dd[:, t - 1])
    return torch.stack(c, dim=1)  # (B, T) float64


def discounted_return(r: torch.Tensor, d: torch.Tensor, gamma: float,
                      horizons: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """G_T = sum_{k<T} gamma^k c_k r_k for each T in `horizons` (float64).

    r, d: (B, T) per-step reward and termination.  Returns {T: (B,) float64}.
    One long rollout yields every shorter horizon (no re-collection)."""
    assert r.shape == d.shape
    B, T = r.shape
    c = survival_soft(d)                       # (B, T) float64
    disc = torch.tensor([gamma ** k for k in range(T)], dtype=DTYPE, device=r.device)
    contrib = c * r.double() * disc.view(1, -1)  # (B, T): gamma^k c_k r_k
    cum = torch.cumsum(contrib, dim=1)
    out = {}
    for h in horizons:
        assert 1 <= h <= T, f"horizon {h} exceeds rollout length {T}"
        out[h] = cum[:, h - 1].clone()
    return out


def survival_after(d: torch.Tensor, T: int) -> torch.Tensor:
    """c_T = prod_{k<T} (1 - d_k): survival AFTER T steps, the tail weight.

    This is the survival *entering* step T, i.e. one factor beyond c_{T-1}; it is
    zero whenever a termination occurs anywhere in steps 0..T-1 (including T-1)."""
    assert 1 <= T <= d.shape[1], f"T={T} outside rollout length {d.shape[1]}"
    return torch.prod(1.0 - d.double()[:, :T], dim=1)


def survival_at(d: torch.Tensor, T: int) -> torch.Tensor:
    """Alias of survival_after (c_T).  v1 returned c_{T-1}, which kept a spurious
    tail when termination happened at the very last step."""
    return survival_after(d, T)


def matched_target(G_T: torch.Tensor, c_T: torch.Tensor, gamma: float,
                   T: int, v_ref_xT: torch.Tensor) -> torch.Tensor:
    """Y_T = G_T + gamma^T c_T V_ref(x_T).  All inputs (B,), returns (B,) float64.

    V_ref must be the shared frozen reference critic evaluated on the TRUE
    simulator state x_T; never assign real-return labels to WM states."""
    return G_T.double() + (gamma ** T) * c_T.double() * v_ref_xT.double()


def pair_difference(y_plus: torch.Tensor, y_minus: torch.Tensor) -> torch.Tensor:
    """Delta Y for a plus/minus branch pair of the SAME start/direction/epsilon."""
    return (y_plus.double() - y_minus.double())


def finite_difference_response(dy: torch.Tensor, eps: float) -> torch.Tensor:
    """D = Delta Y / (2 eps) for symmetric +-eps action prefixes."""
    return dy.double() / (2.0 * eps)
