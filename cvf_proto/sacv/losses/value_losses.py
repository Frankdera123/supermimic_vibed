"""SACV Critic fine-tuning losses (C0 / C1 / C2), frozen per
docs/cvf/sacv/CVF1A_EXPERIMENT_PROTOCOL.md §1.

Mathematical contract (pre-registered):
  C0: L      = E[(V(x) - Y_WM(x))^2]                       (WM scalar supervision)
  C1: L_val  = 1/2 E[(V(x+) - Y+)^2 + (V(x-) - Y-)^2]       (sim scalar supervision)
  C2: L      = L_val + alpha * L_pair
      L_pair = E[((V(x+) - V(x-)) - (Y+ - Y^-))^2]

L_pair creates NO information beyond the two scalar labels (Delta Y is a linear
functional of Y+, Y-); C2 only re-weights the local difference error in the
optimization functional.  Every loss here reduces MEAN over the batch but the
per-sample terms are invariant to batch composition (no cross-batch statistics).
"""
from __future__ import annotations

import torch


def wm_scalar_loss(v: torch.Tensor, y_wm: torch.Tensor) -> torch.Tensor:
    """C0 objective (scalar, mean over batch)."""
    return ((v.double() - y_wm.double()) ** 2).mean()


def value_loss(v_plus: torch.Tensor, v_minus: torch.Tensor,
               y_plus: torch.Tensor, y_minus: torch.Tensor) -> torch.Tensor:
    """C1 objective: 1/2 E[(V+ - Y+)^2 + (V- - Y-)^2]."""
    term = 0.5 * ((v_plus.double() - y_plus.double()) ** 2
                  + (v_minus.double() - y_minus.double()) ** 2)
    return term.mean()


def pair_loss(v_plus: torch.Tensor, v_minus: torch.Tensor,
              y_plus: torch.Tensor, y_minus: torch.Tensor) -> torch.Tensor:
    """C2 pair term: E[((V+ - V-) - (Y+ - Y-))^2]."""
    dv = (v_plus.double() - v_minus.double()) - (y_plus.double() - y_minus.double())
    return (dv ** 2).mean()


def combined_loss(v_plus: torch.Tensor, v_minus: torch.Tensor,
                  y_plus: torch.Tensor, y_minus: torch.Tensor,
                  alpha: float) -> dict[str, torch.Tensor]:
    """C2 = C1 + alpha * L_pair.  Returns components for logging (same graph)."""
    l_val = value_loss(v_plus, v_minus, y_plus, y_minus)
    l_pair = pair_loss(v_plus, v_minus, y_plus, y_minus)
    return {"loss": l_val + alpha * l_pair, "value": l_val, "pair": l_pair}


def per_sample_value_terms(v_plus: torch.Tensor, v_minus: torch.Tensor,
                           y_plus: torch.Tensor, y_minus: torch.Tensor) -> torch.Tensor:
    """Per-sample C1 terms (B,): exposes batch-composition invariance for tests."""
    return 0.5 * ((v_plus.double() - y_plus.double()) ** 2
                  + (v_minus.double() - y_minus.double()) ** 2)


def pair_scale_normalizer(delta_y_train: torch.Tensor) -> float:
    """Frozen sigma(Delta Y) on the Train split (pre-registered alpha rescale,
    computed once before training and never updated afterwards)."""
    return float(delta_y_train.double().std().clamp_min(1e-8))
