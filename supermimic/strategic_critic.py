"""Small critic for predicting factual world-model error in state/action space."""
from __future__ import annotations

import torch
import torch.nn as nn


class StrategicErrorCritic(nn.Module):
    """Predict ``log1p`` normalized WM error from a factual transition query.

    The critic is deliberately independent of the deployable policy and WM
    parameters.  It is trained on detached simulator-grounded targets and can
    therefore be queried on bounded action candidates before a real probe.
    """

    def __init__(self, state_dim: int = 71, action_dim: int = 29,
                 xi_dim: int = 33, hidden: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.xi_dim = int(xi_dim)
        self.input_dim = self.state_dim + self.action_dim + self.xi_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def features(self, state: torch.Tensor, action: torch.Tensor,
                 xi: torch.Tensor, mean: torch.Tensor | None = None,
                 std: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat((state, action / 20.0, xi), dim=-1)
        if mean is not None and std is not None:
            x = (x - mean) / std.clamp_min(1e-4)
        return x
