"""Policy-occupancy transition odds used as a detached replay signal.

The classifier is intentionally frozen after offline fitting.  This module
only owns checkpoint validation, normalization, temperature calibration, and
bounded odds conversion; it never contributes gradients to the world model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransitionOdds(nn.Module):
    """Load a transition classifier and produce bounded normalized weights."""

    def __init__(self, checkpoint, device=None):
        super().__init__()
        if not isinstance(checkpoint, dict):
            raise ValueError("transition checkpoint must be a mapping")
        input_dim = int(checkpoint.get("input_dim", -1))
        hidden = int(checkpoint.get("hidden", 64))
        if input_dim <= 0 or hidden <= 0:
            raise ValueError("transition checkpoint has invalid input_dim or hidden")
        if "model" not in checkpoint or "mean" not in checkpoint or "std" not in checkpoint:
            raise ValueError("transition checkpoint is missing model, mean, or std")
        self.model = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1))
        self.model.load_state_dict(checkpoint["model"])
        self.mean = checkpoint["mean"].detach().float()
        self.std = checkpoint["std"].detach().float().clamp_min(1e-4)
        if self.mean.numel() != input_dim or self.std.numel() != input_dim:
            raise ValueError("transition normalization dimension does not match input_dim")
        self.temperature = float(checkpoint.get("temperature", 1.0))
        if self.temperature <= 0.0:
            raise ValueError("transition checkpoint temperature must be positive")
        self.to(device=device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def input_dim(self):
        return int(self.mean.numel())

    @torch.no_grad()
    def weights(self, features, scale=1.0, minimum=0.25, maximum=4.0):
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"transition classifier expects {self.input_dim} features, "
                             f"got {features.shape[-1]}")
        if scale < 0.0:
            raise ValueError("scale must be nonnegative")
        logits = self.model((features.float() - self.mean) / self.std).squeeze(-1)
        probability = (logits / self.temperature).sigmoid().clamp(1e-3, 1.0 - 1e-3)
        odds = probability / (1.0 - probability)
        odds = odds / odds.mean().clamp_min(1e-6)
        return (1.0 + scale * (odds - 1.0)).clamp(minimum, maximum)
