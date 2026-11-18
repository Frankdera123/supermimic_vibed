"""Isolated policy/critic branch for probe-gated concurrent updates."""
from __future__ import annotations

from copy import deepcopy

import torch.nn as nn
import torch


class ShadowPolicyBranch:
    """Keep a policy/critic copy that can be promoted or discarded atomically."""

    def __init__(self, policy: nn.Module, critic: nn.Module):
        self.live_policy = policy
        self.live_critic = critic
        self.shadow_policy = deepcopy(policy)
        self.shadow_critic = deepcopy(critic)
        self.reset_from_live()

    def reset_from_live(self) -> None:
        self.shadow_policy.load_state_dict(self.live_policy.state_dict())
        self.shadow_critic.load_state_dict(self.live_critic.state_dict())

    def promote(self) -> None:
        """Atomically copy the validated shadow state into the live branch."""
        policy_state = deepcopy(self.shadow_policy.state_dict())
        critic_state = deepcopy(self.shadow_critic.state_dict())
        self.live_policy.load_state_dict(policy_state)
        self.live_critic.load_state_dict(critic_state)

    def discard(self) -> None:
        """Discard shadow changes and restart from the current live state."""
        self.reset_from_live()

    def state_dict(self) -> dict:
        """Return a CPU-independent state payload for checkpointing."""
        return {
            "policy": deepcopy(self.shadow_policy.state_dict()),
            "critic": deepcopy(self.shadow_critic.state_dict()),
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow_policy.load_state_dict(state["policy"])
        self.shadow_critic.load_state_dict(state["critic"])

    def make_optimizers(self, lr: float, **kwargs):
        """Build independent Adam optimizers for shadow policy and critic."""
        return (
            torch.optim.Adam(self.shadow_policy.parameters(), lr=lr, **kwargs),
            torch.optim.Adam(self.shadow_critic.parameters(), lr=lr, **kwargs),
        )


@torch.no_grad()
def cap_parameter_step(module: nn.Module, before: list[torch.Tensor], max_norm: float) -> float:
    """Project one optimizer step into an L2 parameter-space trust region."""
    if max_norm <= 0.0:
        return 0.0
    params = list(module.parameters())
    delta_sq = torch.stack([(p.detach() - b).square().sum() for p, b in zip(params, before)]).sum()
    delta_norm = float(delta_sq.sqrt().cpu())
    scale = min(1.0, max_norm / max(delta_norm, 1e-12))
    if scale < 1.0:
        for p, b in zip(params, before):
            p.copy_(b + scale * (p - b))
    return delta_norm * scale
