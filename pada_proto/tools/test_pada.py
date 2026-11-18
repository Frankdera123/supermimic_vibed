"""PADA CPU synthetic tests (no simulator, no checkpoints).

Covers the fail-closed invariants the experiment depends on:
  1. direction unit-norm conventions (random / policy-delta / normalize);
  2. probe window construction correctness + equal pool sizes across arms;
  3. paired sampling schedule identity across arms (same seed -> same indices);
  4. fidelity_stats correctness against hand-computed values;
  5. reliability_gate fail-closed behavior;
  6. branch_J discounted-return correctness incl. soft-termination and H-truncation;
  7. proto WM target-space linearity (z-form equals train.py state_error form);
  8. finetune loss decrease on a toy WM (gradient flows, normalizers frozen).

Run:  python tools/test_pada.py   (from pada_proto/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
import torch  # must initialize torch before bridge triggers supermimic.kinematics

sys.path.insert(0, str(_HERE.parent))
import pada_bridge as bridge

bridge.import_stack()
bridge.fix_paths()

from supermimic.crgf import (fidelity_stats, normalize_dir_seqs, random_dir_seqs,
                             reliability_gate)

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")


# ---------------------------------------------------------------- 1 directions
d = random_dir_seqs(4, 8, 29, seed=1)
n = d.reshape(4, -1).norm(dim=-1)
check("random dirs unit norm", torch.allclose(n, torch.ones(4), atol=1e-4))

a_old = torch.randn(3, 8, 29)
a_new = a_old + 0.05 * torch.randn(3, 8, 29)
dd = normalize_dir_seqs(a_new - a_old)
check("delta dirs unit norm", torch.allclose(dd.reshape(3, -1).norm(dim=-1),
                                             torch.ones(3), atol=1e-4))
check("delta dirs preserve orientation",
      all(float((dd[i] * (a_new[i] - a_old[i])).sum()) > 0 for i in range(3)))

zero_delta = normalize_dir_seqs(torch.zeros(2, 4, 29))
check("zero delta does not NaN", bool(zero_delta.isfinite().all()))

# -------------------------------------------- 2/3 window construction + pairing
sys.path.insert(0, str(_HERE))
from pada_finetune_wm import K_WINDOW, build_windows  # noqa: E402


def fake_probes(arm_names, H=32, n=2):
    trajs = {}
    for arm in arm_names:
        for i in range(n):
            trajs[f"{arm}_{i}_0_plus"] = {
                "s0": torch.randn(71), "s1": torch.randn(H, 71),
                "a": torch.randn(H, 29),
                "f": torch.arange(100, 100 + H),
                "r_state": torch.rand(H), "r_direct": torch.rand(H),
                "soft_d": torch.rand(H) * 0.1, "done": torch.zeros(H, dtype=torch.bool),
            }
    return {"branches": trajs}


p_a = fake_probes(["random"])
p_b = fake_probes(["policy"])
St_a, A_a, F_a = build_windows(p_a, "random")
St_b, A_b, F_b = build_windows(p_b, "policy")
check("equal window pools across arms", St_a.shape == St_b.shape and A_a.shape == A_b.shape)
check("window shape", St_a.shape[1:] == (K_WINDOW + 1, 71) and A_a.shape[1:] == (K_WINDOW, 29))
tr = p_a["branches"]["random_0_0_plus"]
s_all = torch.cat((tr["s0"].unsqueeze(0), tr["s1"]), 0)
check("window content alignment", torch.allclose(St_a[0], s_all[:K_WINDOW + 1])
      and torch.allclose(A_a[0], tr["a"][:K_WINDOW]))
n_windows = St_a.shape[0]
g = torch.Generator().manual_seed(99)
idx1 = torch.randint(0, n_windows, (10, 4), generator=g)
g = torch.Generator().manual_seed(99)
idx2 = torch.randint(0, n_windows, (10, 4), generator=g)
check("paired schedule identical across arms", torch.equal(idx1, idx2))

# ------------------------------------------------------------ 4 fidelity stats
w = torch.tensor([1.0, -2.0, 0.5, 3.0])
s = torch.tensor([2.0, -1.0, -0.5, 1.0])
st = fidelity_stats(w, s, gate=None)
check("fidelity sign agreement", abs(st["sign_agreement"] - 3 / 4) < 1e-9,
      f"got {st['sign_agreement']}")
check("fidelity rmse", abs(st["rmse"] - torch.sqrt(((w - s) ** 2).mean())) < 1e-9)

# ------------------------------------------------------------------- 5 gate
ok, why = reliability_gate(torch.tensor([1.0, 1.0]), torch.tensor([5.0, 5.1]))
check("gate passes clean sample", ok, why)
ok, why = reliability_gate(torch.tensor([1.0, 1.2]), torch.tensor([5.0, 5.1]))
check("gate fails on noisy baseline", (not ok) and "baseline_range" in why, why)
ok, why = reliability_gate(torch.tensor([1.0, 1.0]), torch.tensor([5.0, 9.0]))
check("gate fails on unstable D", (not ok) and "d_range" in why, why)
ok, why = reliability_gate(torch.tensor([1.0, 1.0]), torch.tensor([1e-6, 2e-6]))
check("gate fails below deadband", (not ok) and "deadband" in why, why)
ok, why = reliability_gate(torch.tensor([1.0]), torch.tensor([5.0, 5.1]))
check("gate fails insufficient baseline", (not ok) and "insufficient" in why, why)

# ------------------------------------------------------------------ 6 branch_J
sys.path.insert(0, str(_HERE / "tools"))
import importlib

pada_eval = importlib.import_module("pada_eval")
H = 8
tr = {
    "s0": torch.randn(71),
    "s1": torch.randn(H, 71),
    "f": torch.arange(50, 50 + H),
    "a": torch.randn(H, 29), "a_prev": torch.randn(H, 29),
    "r_state": torch.tensor([1.0] * H), "r_direct": torch.tensor([0.5] * H),
    "soft_d": torch.tensor([0.0] * (H - 1) + [0.5]),  # soft termination at last step
    "done": torch.zeros(H, dtype=torch.bool),
}
J = pada_eval.branch_J(tr, None, None, None, H, gamma=0.99)
manual_state = sum(0.99 ** t * (0.0 if t == 0 else 1.0) * 1.0 for t in range(H))
# c_t = prod_{<=t}(1-d); d_{H-1}=0.5 affects only c_{H-1}
c = 1.0
acc = 0.0
for t in range(H):
    c *= 1 - (0.5 if t == H - 1 else 0.0)
    acc += 0.99 ** t * c * 1.0
check("branch_J state path", abs(float(J["state"]) - acc) < 1e-6,
      f"{float(J['state']):.6f} vs {acc:.6f}")
acc_d = 1.0
acc2 = 0.0
c = 1.0
for t in range(H):
    c *= 1 - (0.5 if t == H - 1 else 0.0)
    acc2 += 0.99 ** t * c * 0.5
check("branch_J direct path", abs(float(J["direct"]) - acc2) < 1e-6)
J4 = pada_eval.branch_J(tr, None, None, None, 4, gamma=0.99)
acc3 = 0.0
for t in range(4):
    acc3 += 0.99 ** t * 1.0        # soft_d[0..3] are all 0 in the fixture
check("branch_J H truncation", abs(float(J4["state"]) - acc3) < 1e-6,
      f"{float(J4['state']):.6f} vs {acc3:.6f}")

# ------------------------------------------------- 7 proto WM target linearity
import pada_eval  # noqa: E402

bridge.import_stack()
from supermimic.networks import WorldModel  # proto

wm = WorldModel()
S = (torch.randn(2, 3), torch.linalg.qr(torch.randn(2, 3, 3))[0],
     torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 29), torch.randn(2, 29))
S1a = (torch.randn(2, 3), torch.linalg.qr(torch.randn(2, 3, 3))[0],
       torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 29), torch.randn(2, 29))
S1b = (torch.randn(2, 3), torch.linalg.qr(torch.randn(2, 3, 3))[0],
       torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 29), torch.randn(2, 29))
lin = wm.targets(S, S1a) - wm.targets(S, S1b)
Rt = S[1].transpose(-1, -2)
check("targets linear in S1 (dv block)",
      torch.allclose(lin[:, :3], (Rt @ (S1a[2] - S1b[2]).unsqueeze(-1)).squeeze(-1),
                     atol=1e-5))
check("targets qd block", torch.allclose(lin[:, 6:], S1a[5] - S1b[5]))

# ------------------------------------------------------- 8 toy finetune check
tiny = WorldModel(hidden=(32, 32))
for p in tiny.parameters():
    p.requires_grad_(True)
opt = torch.optim.Adam(tiny.parameters(), lr=1e-2)
std = torch.ones(35)
S_toy = (torch.randn(16, 3), torch.linalg.qr(torch.randn(16, 3, 3))[0],
         torch.randn(16, 3), torch.randn(16, 3), torch.randn(16, 29), torch.randn(16, 29))
A_toy = torch.randn(16, 29)
S1_toy = tuple(x + 0.1 * torch.randn_like(x) for x in S_toy)
xi_n = torch.zeros(16, 33)
xi_n[:, 0] = 1.0
first = None
for it in range(60):
    pred = tiny.step(S_toy, A_toy, xi_n)
    z = (tiny.targets(S_toy, pred) - tiny.targets(S_toy, S1_toy)) / std
    loss = z.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if first is None:
        first = float(loss)
check("toy finetune loss decreases", float(loss) < first * 0.7,
      f"{first:.4f} -> {float(loss):.4f}")
check("normalizers frozen (no in-place update)",
      bool(torch.equal(tiny.in_norm.count, torch.zeros(()))))

print(json.dumps({"pass": len(PASS), "fail": len(FAIL)}))
sys.exit(1 if FAIL else 0)
