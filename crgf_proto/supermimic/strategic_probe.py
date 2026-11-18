"""Isolated storage for factual transitions collected by strategic probes."""
from __future__ import annotations

import torch


class StrategicProbeBuffer:
    """Small ring buffer kept separate from the ordinary replay buffer.

    ``transition`` values must have a leading batch dimension.  Scores and
    source ids are metadata and never enter the nominal WM loss implicitly.
    """

    def __init__(self, capacity: int, device: str):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.data: dict[str, torch.Tensor] = {}
        self.score = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.source = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0

    def _allocate(self, transition: dict[str, torch.Tensor]) -> None:
        if self.data:
            return
        for key, value in transition.items():
            if value.ndim < 1:
                raise ValueError(f"transition field {key!r} needs a batch dimension")
            self.data[key] = torch.zeros(
                (self.capacity,) + tuple(value.shape[1:]),
                dtype=value.dtype,
                device=self.device,
            )

    def add(self, transition: dict[str, torch.Tensor], score: torch.Tensor,
            source: torch.Tensor | int = 0) -> None:
        if not transition:
            raise ValueError("transition cannot be empty")
        batch = next(iter(transition.values())).shape[0]
        if batch < 1 or any(value.shape[0] != batch for value in transition.values()):
            raise ValueError("all transition fields must share a non-empty batch dimension")
        score = torch.as_tensor(score, device=self.device, dtype=torch.float32).flatten()
        if score.numel() != batch:
            raise ValueError("score must have one value per transition")
        source = torch.as_tensor(source, device=self.device, dtype=torch.long)
        source = source.expand(batch).flatten() if source.numel() == 1 else source.flatten()
        if source.numel() != batch:
            raise ValueError("source must be scalar or have one value per transition")
        self._allocate(transition)
        if set(transition) != set(self.data):
            raise ValueError("transition fields changed after first add")
        for i in range(batch):
            row = self.ptr
            for key, value in transition.items():
                self.data[key][row].copy_(value[i].to(self.device))
            self.score[row] = score[i]
            self.source[row] = source[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, prioritized: bool = False,
               alpha: float = 1.0) -> dict[str, torch.Tensor]:
        if self.size < 1:
            raise ValueError("cannot sample from an empty probe buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative")
        if prioritized:
            weights = self.score[:self.size].detach().clamp_min(1e-6).pow(alpha)
            indices = torch.multinomial(weights / weights.sum(), batch_size, replacement=True)
        else:
            indices = torch.randint(self.size, (batch_size,), device=self.device)
        out = {key: value[indices] for key, value in self.data.items()}
        out["probe_score"] = self.score[indices]
        out["probe_source"] = self.source[indices]
        return out

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "data": {key: value.clone() for key, value in self.data.items()},
            "score": self.score.clone(),
            "source": self.source.clone(),
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("probe buffer capacity mismatch")
        self.data = {key: value.to(self.device).clone() for key, value in state["data"].items()}
        self.score.copy_(state["score"].to(self.device))
        self.source.copy_(state["source"].to(self.device))
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
