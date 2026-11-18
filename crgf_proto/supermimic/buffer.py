"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .dimensions import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    # Cause labels are optional for backwards-compatible checkpoints.  They
    # identify which termination boundary the next state crossed.
    "term_height": ((), torch.bool),
    "term_tilt": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str):
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        # Optional world-model sampling priorities. Uniform sampling remains
        # the default; priorities are stored per time row, not per transition.
        self.priority = torch.ones(T, dtype=torch.float32, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        # New data starts at the current maximum priority so it is not silently
        # starved when prioritized sampling is enabled.
        self.priority[self.ptr] = self.priority.max().clamp_min(1.0)
        for k in FIELDS:
            value = tr.get(k)
            if value is None:
                # Older callers/checkpoints have no cause labels; preserve the
                # old behavior and leave the optional field false.
                self.data[k][self.ptr].zero_()
            else:
                self.data[k][self.ptr] = value
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def state_dict(self):
        """Serialize replay contents so staged/curriculum resumes keep data coverage."""
        return {"data": {k: v.clone() for k, v in self.data.items()},
                "priority": self.priority.clone(), "ptr": self.ptr, "size": self.size}

    def load_state_dict(self, state):
        for k in self.data:
            if k in state["data"]:
                self.data[k].copy_(state["data"][k])
        self.priority.copy_(state.get("priority", self.priority))
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def update_priorities(self, age: torch.Tensor, value: torch.Tensor):
        """Update priorities using age-ordered start indices (0 = oldest)."""
        age, value = torch.broadcast_tensors(age.long(), value.float())
        if ((age < 0) | (age >= self.size)).any():
            raise ValueError("priority age is outside the buffered range")
        # Bound extreme early-training errors so one bad window cannot starve
        # all other replay rows; relative ordering is preserved below the cap.
        self.priority[self._row(age)] = value.detach().clamp(1e-6, 1e6)

    def sample_windows(self, B: int, k: int, prioritized: bool = False,
                       priority_alpha: float = 1.0, return_indices: bool = False,
                       env_ids: torch.Tensor | None = None,
                       recent_fraction: float = 0.0):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        if k < 1 or k > self.size:
            raise ValueError(f"Cannot sample {k}-step windows from {self.size} buffered steps")
        n_starts = self.size - k + 1
        if not 0.0 <= recent_fraction <= 1.0:
            raise ValueError("recent_fraction must be in [0, 1]")
        recent_start = max(0, n_starts - max(1, int(n_starts * recent_fraction))) \
            if recent_fraction > 0.0 else 0
        if prioritized:
            if priority_alpha < 0.0:
                raise ValueError("priority_alpha must be non-negative")
            ages0 = torch.arange(n_starts, device=self.device)
            rows0 = self._row(ages0)
            weights = self.priority[rows0].clamp_min(1e-6).pow(priority_alpha)
            if recent_fraction > 0.0:
                weights[:recent_start] = 0.0
            t0 = torch.multinomial(weights / weights.sum(), B, replacement=True)
        else:
            t0 = torch.randint(recent_start if recent_fraction > 0.0 else 0,
                               n_starts, (B,), device=self.device)
        if env_ids is None:
            env = torch.randint(0, self.N, (B,), device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
            if env_ids.numel() < 1 or (env_ids < 0).any() or (env_ids >= self.N).any():
                raise ValueError("env_ids must contain valid, non-empty environment indices")
            env = env_ids[torch.randint(0, env_ids.numel(), (B,), device=self.device)]
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "f", "r", "done", "term_height", "term_tilt", "ep_step")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        if return_indices:
            out["window_age"] = t0
            out["window_env"] = env
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "s1", "a", "a_prev", "xi", "f", "r", "term_height", "term_tilt")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
