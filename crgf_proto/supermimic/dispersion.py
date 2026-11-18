"""Model-disagreement gating utilities for optional policy/world-model coupling.

The target predictions are deliberately detached from the policy objective: they
only decide whether the robust imagined objective is trusted for each start.
This keeps the world-model/policy gradient isolation used by SuperMimic intact.
"""

from __future__ import annotations

import torch
from torch import nn


def relative_disagreement(online_return: torch.Tensor,
                          target_return: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """Return per-start relative disagreement between two model returns.

    Both inputs may have arbitrary leading dimensions but must be broadcastable.
    The returned tensor is detached so it cannot create a gradient path through
    the target/EMA model or the gate decision.
    """
    online_return, target_return = torch.broadcast_tensors(online_return, target_return)
    center = 0.5 * (online_return + target_return)
    return ((online_return - target_return).abs() /
            center.abs().clamp_min(eps)).detach()


def gate_mask(online_return: torch.Tensor,
              target_return: torch.Tensor,
              threshold: float,
              eps: float = 1e-6) -> torch.Tensor:
    """Boolean per-start mask selecting robust imagined updates."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return relative_disagreement(online_return, target_return, eps) <= threshold


def gate_weight(online_return: torch.Tensor,
                target_return: torch.Tensor,
                threshold: float,
                temperature: float,
                eps: float = 1e-6) -> torch.Tensor:
    """Return a detached soft trust weight in ``[0, 1]`` per start.

    ``temperature=0`` is intentionally equivalent to the hard gate. For a
    positive temperature, disagreement below ``threshold`` receives weight
    near one and disagreement above it decays smoothly.
    """
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    disagreement = relative_disagreement(online_return, target_return, eps)
    if temperature == 0.0:
        return (disagreement <= threshold).float()
    return torch.sigmoid((threshold - disagreement) / temperature)


def quantile_gate_weight(online_return: torch.Tensor,
                         target_return: torch.Tensor,
                         trusted_fraction: float,
                         temperature: float = 0.0,
                         eps: float = 1e-6) -> torch.Tensor:
    """Trust a fixed fraction of the least-disagreeing starts in a batch."""
    if not 0.0 <= trusted_fraction <= 1.0:
        raise ValueError("trusted_fraction must be in [0, 1]")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    disagreement = relative_disagreement(online_return, target_return, eps)
    if disagreement.numel() == 0:
        return disagreement.float()
    cutoff = torch.quantile(disagreement, trusted_fraction)
    if temperature == 0.0:
        return (disagreement <= cutoff).float()
    return torch.sigmoid((cutoff - disagreement) / temperature)


def gated_objective(robust_cost: torch.Tensor,
                    nominal_cost: torch.Tensor,
                    online_return: torch.Tensor,
                    target_return: torch.Tensor,
                    threshold: float,
                    eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Select robust or nominal cost per start and return (cost, mask).

    ``robust_cost`` remains differentiable; the mask is detached and therefore
    cannot backpropagate through either model used to estimate disagreement.
    """
    robust_cost, nominal_cost = torch.broadcast_tensors(robust_cost, nominal_cost)
    mask = gate_mask(online_return, target_return, threshold, eps)
    return torch.where(mask, robust_cost, nominal_cost), mask


@torch.no_grad()
def ema_update_(target: nn.Module, source: nn.Module, beta: float) -> None:
    """In-place EMA update for parameters and floating-point buffers."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    target_params = dict(target.named_parameters())
    source_params = dict(source.named_parameters())
    if target_params.keys() != source_params.keys():
        raise ValueError("source and target parameter structures differ")
    for name, target_param in target_params.items():
        target_param.mul_(beta).add_(source_params[name], alpha=1.0 - beta)
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("source and target buffer structures differ")
    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.is_floating_point():
            target_buffer.mul_(beta).add_(source_buffer, alpha=1.0 - beta)
        else:
            target_buffer.copy_(source_buffer)
