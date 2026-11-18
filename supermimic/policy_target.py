"""Policy-sensitive target WM utilities.

The target model is deliberately separate from the replay-fitted nominal WM.
This module contains state-management, audit gating, and an optional
zero-initialized policy-sensitive adapter; it does not alter the training
recipe until the caller explicitly opts in.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .rotations import so3_exp, yaw_mat


class PolicyTargetWorldModel(nn.Module):
    """A frozen-copy/EMA target wrapper with an explicit replay-fidelity gate."""

    def __init__(self, nominal: nn.Module, ema_beta: float = 0.99,
                 adapter_hidden: int = 0, adapter_max_delta: float = 0.0):
        super().__init__()
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        # A WorldModel carries architecture choices (hidden width, residual
        # head, normalizers) that are not recoverable from a zero-argument
        # constructor.  Deep-copy the supplied nominal model so the target is
        # structurally identical, including non-parameter buffers.
        self.model = copy.deepcopy(nominal)
        self.model.to(next(nominal.parameters()).device)
        self.ema_beta = float(ema_beta)
        self.adapter_hidden = int(adapter_hidden)
        self.adapter_max_delta = float(adapter_max_delta)
        if self.adapter_max_delta < 0.0:
            raise ValueError("adapter_max_delta must be nonnegative")
        if self.adapter_hidden < 0:
            raise ValueError("adapter_hidden must be nonnegative")
        self.adapter = None
        if self.adapter_hidden > 0:
            out_dim = int(self.model.out_norm.mean.numel())
            self.adapter = nn.Sequential(
                nn.Linear(self.model.in_dim, self.adapter_hidden),
                nn.SiLU(),
                nn.Linear(self.adapter_hidden, out_dim),
            ).to(next(nominal.parameters()).device)
            # A zero output head makes the task model exactly nominal at init.
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        self.enabled = False
        self.register_buffer("audit_error", torch.tensor(float("inf")))
        self.register_buffer("replay_kl_proxy", torch.tensor(float("inf")))
        self.freeze()

    @torch.no_grad()
    def sync_from(self, nominal: nn.Module):
        """Hard-sync all target parameters and buffers from nominal WM."""
        self.model.load_state_dict(nominal.state_dict())

    @torch.no_grad()
    def ema_from(self, nominal: nn.Module):
        """EMA-sync floating parameters while copying non-floating buffers."""
        beta = self.ema_beta
        for target, source in zip(self.model.parameters(), nominal.parameters()):
            target.mul_(beta).add_(source, alpha=1.0 - beta)
        for target, source in zip(self.model.buffers(), nominal.buffers()):
            if target.is_floating_point():
                target.mul_(beta).add_(source, alpha=1.0 - beta)
            else:
                target.copy_(source)

    def adapter_parameters(self):
        """Return trainable task-adapter parameters, if an adapter is enabled."""
        return tuple(self.adapter.parameters()) if self.adapter is not None else ()

    def adapter_norm(self):
        """Return the current Euclidean adapter parameter norm."""
        params = self.adapter_parameters()
        if not params:
            return torch.zeros((), device=self.audit_error.device)
        return torch.sqrt(torch.stack([p.square().sum() for p in params]).sum())

    @torch.no_grad()
    def snapshot_adapter(self):
        """Return a device-local rollback token for the task adapter."""
        if self.adapter is None:
            return None
        return {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}

    @torch.no_grad()
    def restore_adapter(self, token):
        """Restore a token produced by :meth:`snapshot_adapter`."""
        if self.adapter is None:
            if token is not None:
                raise ValueError("cannot restore an adapter token without an adapter")
            return
        if token is None:
            raise ValueError("adapter rollback token is missing")
        expected = set(self.adapter.state_dict())
        if set(token) != expected:
            raise ValueError("adapter rollback token has incompatible parameters")
        self.adapter.load_state_dict(token)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad_(True)
            self.adapter.train()

    @staticmethod
    def _apply_delta(S, delta):
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
        return (p + (H @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth),
                v + (R @ dv.unsqueeze(-1)).squeeze(-1),
                w + (R @ dw.unsqueeze(-1)).squeeze(-1), q + dq, qd + dqd)

    def step(self, S, a, xi_n, use_residual=False):
        """Predict one transition, optionally adding the task adapter correction."""
        base = self.model.step(S, a, xi_n, use_residual=use_residual)
        if self.adapter is None:
            return base
        x = self.model.in_norm(self.model.inputs(S, a, xi_n))
        correction = self.adapter(x)
        if self.adapter_max_delta > 0.0:
            rms = correction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            correction = correction * (self.adapter_max_delta / rms).clamp_max(1.0)
        correction = correction * self.model.out_norm.std()
        delta = self.model.targets(S, base) + correction
        return self._apply_delta(S, delta)

    @torch.no_grad()
    def audit_gate(self, error: torch.Tensor, max_error: float) -> bool:
        """Enable target use only when an independent replay audit passes."""
        value = error.detach().float().mean()
        self.audit_error.copy_(value)
        self.enabled = bool(value <= max_error)
        return self.enabled

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
