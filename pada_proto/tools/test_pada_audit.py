"""PADA-0a audit tests (CPU only).

Test 1  linear function: autograd directional derivative == central FD
Test 2  quadratic function: center-autograd / plus-point-autograd / central-FD
        exact relations
Test 3  path decomposition on the real stack: matched-builder identity
        (D_total = D_state+D_direct+D_critic, attached survival) and equality
        of the matched total return with crgf's L_total; crgf's own iso
        identity D_total = D_state+D_direct_iso+D_critic_iso+D_cross
Test 4  survival gradient: surrogate proves attached-vs-detached difference is
        the survival term (analytically matched); real stack: exact identity
        g_attached - g_iso = grad(sum disc * c * r_direct.detach())
Test 5  sample identity: recomputed gate labels reproduce the original
        eval_rows.csv gated sample set exactly
Test 6  original-result protection: SHA256 of protected files unchanged across
        an audit-tool run; `git diff --check` clean

Run:  python tools/test_pada_audit.py   (from pada_proto/)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
import torch  # must init before bridge (see test_pada.py note)

sys.path.insert(0, str(_HERE.parent))
import pada_bridge as bridge

bridge.import_stack()
bridge.fix_paths()

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {info}")


REPO = bridge.repo_root()

# ------------------------------------------------------------ Test 1: linear
torch.manual_seed(0)
w = torch.randn(12)
A = torch.randn(1, 4, 3, requires_grad=True)
d = torch.randn(1, 4, 3)
d = d / d.reshape(-1).norm()


def f_lin(a):
    return (a.reshape(-1) * w).sum()


J = f_lin(A)
g = torch.autograd.grad(J, A)[0]
D_ag = float((g * d).sum())
eps = 0.1
D_fd = float((f_lin(A + eps * d) - f_lin(A - eps * d)) / (2 * eps))
check("T1 linear: autograd == central FD", abs(D_ag - D_fd) < 1e-6,
      f"{D_ag:.8f} vs {D_fd:.8f}")

# ---------------------------------------------------------- Test 2: quadratic
Q = torch.randn(12, 12)
Q = Q @ Q.T / 12


def f_quad(a):
    x = a.reshape(-1)
    return 0.5 * x @ Q @ x


A2 = torch.randn(1, 4, 3, requires_grad=True)
d2 = torch.randn(1, 4, 3)
d2 = d2 / d2.reshape(-1).norm()
g_c = torch.autograd.grad(f_quad(A2), A2)[0]
D_center = float((g_c * d2).sum())
A_plus = (A2 + eps * d2).detach().requires_grad_(True)
D_plus = float((torch.autograd.grad(f_quad(A_plus), A_plus)[0] * d2).sum())
D_fd2 = float((f_quad(A2 + eps * d2) - f_quad(A2 - eps * d2)) / (2 * eps))
dQd = float((d2.reshape(-1) @ Q @ d2.reshape(-1)))
check("T2 quadratic: central FD == center autograd (exact for quadratic)",
      abs(D_fd2 - D_center) < 1e-5, f"{D_fd2:.6f} vs {D_center:.6f}")
check("T2 quadratic: plus-point autograd = center + eps*dQd",
      abs(D_plus - (D_center + eps * dQd)) < 1e-5,
      f"{D_plus:.6f} vs {D_center + eps * dQd:.6f}")

# ------------------------------------- real-stack setup shared by Tests 3 & 4
pair = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
world = bridge.setup_world(device="cpu", pair=pair)
motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
critic = pair["critic"]
wm = pair["wm"]
from supermimic.mdp import normalize_xi  # noqa: E402

xi_n = normalize_xi(torch.tensor([[1.0] + [0.0] * 32]))
f0 = torch.tensor([int(motion.clip_start[0]) + 10])
S0 = motion.ref_state(f0)
H_t = 6
a_seq = (0.5 * torch.randn(1, H_t, 29)).clone().requires_grad_(True)

from pada_eval_matched_fd import (matched_path_returns,  # noqa: E402
                                  numeric_path_returns)
from supermimic.crgf import (open_loop_paths, path_directional_derivatives,  # noqa: E402
                             reward_split, soft_term)

# ------------------------------------------- Test 3: path decomposition (real)
rets = matched_path_returns(wm, critic, mdp, proto, S0, f0, a_seq, xi_n)
D = {}
for p in ("state", "direct", "critic", "total"):
    g = torch.autograd.grad(rets[p], a_seq, retain_graph=True, allow_unused=False)[0]
    D[p] = float((g * a_seq).sum())  # a_seq is the direction here (self-dot)
resid = D["total"] - (D["state"] + D["direct"] + D["critic"])
scale = max(abs(D["total"]), 1e-9)
check("T3 matched identity D_total = D_state+D_direct+D_critic (attached)",
      abs(resid) < 1e-4 * scale, f"resid={resid:.3e}, scale={scale:.3e}")

paths_crgf = open_loop_paths(wm, critic, mdp, S0, f0, a_seq,
                             torch.zeros(1, 29), xi_n, goal_frames=proto.K)
g_total_crgf = path_directional_derivatives(paths_crgf, a_seq)["total"][0]
g_total_matched = torch.autograd.grad(rets["total"], a_seq, retain_graph=True,
                                      allow_unused=False)[0].reshape(-1)
dev_norm = max(float(g_total_crgf.norm()), 1e-9)
rel = float((g_total_crgf - g_total_matched).norm()) / dev_norm
check("T3 matched J_total grad == crgf L_total grad", rel < 1e-6, f"rel={rel:.3e}")

gd = path_directional_derivatives(paths_crgf, a_seq)
iso_sum = gd["state"] + gd["direct"] + gd["critic"] + gd["cross"]
resid_iso = float((gd["total"] - iso_sum).norm()) / dev_norm
# float32 accumulation across 5 backward passes is visible at the ~1e-6
# level; redo the identity in float64 for an exact check
wm_d = WorldModelD = None
try:
    wm64 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")["wm"].double()
    critic64 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")["critic"].double()
    S0_64 = tuple(x.double() for x in S0)
    a64 = a_seq.detach().double().requires_grad_(True)
    pc = open_loop_paths(wm64, critic64, mdp, S0_64, f0, a64,
                         torch.zeros(1, 29, dtype=torch.float64), xi_n.double(),
                         goal_frames=proto.K)
    gd64 = path_directional_derivatives(pc, a64)
    resid64 = float((gd64["total"] - (gd64["state"] + gd64["direct"]
                                      + gd64["critic"] + gd64["cross"])).norm())
    scale64 = max(float(gd64["total"].norm()), 1e-30)
    check("T3 crgf iso identity (float64, exact)",
          resid64 < 1e-12 * max(scale64, 1.0), f"resid={resid64:.3e}, scale={scale64:.3e}")
    check("T3 crgf iso identity (float32, tolerance)",
          resid_iso < 1e-4, f"rel resid={resid_iso:.3e} (float32 accumulation)")
except Exception as e:  # pragma: no cover
    check("T3 crgf iso identity (float32, tolerance)", resid_iso < 1e-4,
          f"rel resid={resid_iso:.3e} (float64 check unavailable: {e})")

# ------------------------------------------- Test 4: survival gradient
# surrogate: soft termination depending directly on the action
beta = torch.tensor([1.0, -2.0, 0.5])
vv = torch.tensor([1.0, 0.5, -1.0])
a_s = torch.tensor([0.3, -0.2, 0.6], requires_grad=True)
soft = torch.sigmoid(a_s @ beta)                    # d(A)
c = 1.0 - soft
r_direct = (a_s ** 2).sum()
gam = 0.9
L_att = gam * c * r_direct                          # survival ATTACHED
L_iso = gam * c.detach() * r_direct                 # survival DETACHED
D_att = float(torch.autograd.grad(L_att, a_s, retain_graph=True)[0] @ beta)
D_iso = float(torch.autograd.grad(L_iso, a_s)[0] @ beta)
surv_direct = gam * float(r_direct.detach() * (-(soft * (1 - soft)) * (beta @ beta)))
check("T4 surrogate: attached - iso == survival term (direct path)",
      abs((D_att - D_iso) - surv_direct) < 1e-6 and abs(D_att - D_iso) > 1e-9,
      f"diff={D_att - D_iso:.6f}, analytic={surv_direct:.6f}")
V = (a_s @ vv) ** 2
L_c_att = gam * c * V
L_c_iso = gam * c.detach() * V
Dc_att = float(torch.autograd.grad(L_c_att, a_s, retain_graph=True)[0] @ beta)
Dc_iso = float(torch.autograd.grad(L_c_iso, a_s)[0] @ beta)
check("T4 surrogate: critic path attached != iso (survival gradient present)",
      abs(Dc_att - Dc_iso) > 1e-9, f"diff={Dc_att - Dc_iso:.6f}")

# real stack: g_attached - g_iso must equal grad(sum disc * c * r_direct.detach())
g_direct_attached = torch.autograd.grad(rets["direct"], a_seq, retain_graph=True,
                                        allow_unused=False)[0]
g_direct_iso = gd["direct"].reshape(1, H_t, 29)
surv = torch.zeros(1, H_t, 29)
cs = [torch.ones(1)]
with torch.no_grad():
    pass
# rebuild the survival term with r_direct detached (graph through c only)
a_tmp = a_seq.clone().requires_grad_(True)
S = S0
f = f0
c_prev = torch.ones(1)
L_surv = torch.zeros(())
a_prev = torch.zeros(1, 29)
for t in range(H_t):
    a_t = a_tmp[:, t]
    S = proto.wm_step(wm, S, torch.clamp(a_t, -20.0, 20.0), xi_n)
    f1 = f + 1
    _, r_d = reward_split(mdp, S, f1, a_t, a_prev)
    d = soft_term(mdp, S, f1)
    c_prev = c_prev * (1.0 - d)
    L_surv = L_surv + (0.99 ** t) * (c_prev * r_d.detach()).sum()
    a_prev = a_t
    f = f1
g_surv = torch.autograd.grad(L_surv, a_tmp)[0]
diff = g_direct_attached - g_direct_iso.reshape(1, H_t, 29)
check("T4 real stack: g_attached - g_iso == grad(survival-only term)",
      bool(torch.allclose(diff, g_surv, atol=1e-6)),
      f"max diff={float((diff - g_surv).abs().max()):.3e}, "
      f"||surv||={float(g_surv.norm()):.3e}")

# ------------------------------------------- Test 5: sample identity
sys.path.insert(0, str(_HERE))
import csv as _csv

meta = json.load(open(REPO / "results/pada/dirs/directions_meta.json"))
ev = torch.load(REPO / "results/pada/collect/eval_data.pt", map_location="cpu",
                weights_only=False)
eval_sets = ("eval_random", "eval_pol_seen", "eval_pol_B", "eval_pol_C")
set_dirs = {n: torch.load(REPO / f"results/pada/dirs/{n}.pt", map_location="cpu",
                          weights_only=False)["dirs"] for n in eval_sets}
k_per_set = {n: int(x.shape[1]) for n, x in set_dirs.items()}
horizons = sorted({8, ev["meta"]["H"]})
from pada_eval_matched_fd import compute_sim_ground_truth  # noqa: E402

sim_D, gates = compute_sim_ground_truth(ev, meta, REPO / "results/pada/dirs",
                                        world, proto, critic, horizons,
                                        eval_sets, k_per_set)
orig = list(_csv.DictReader(open(REPO / "results/pada/eval/eval_rows.csv")))
orig_gated = {}
for r in orig:
    orig_gated.setdefault((r["set"], int(r["H"])), set()).add((int(r["si"]), int(r["ji"])))
mism = []
for (set_name, h), ids in orig_gated.items():
    new_ids = {(si, ji) for (s, si, ji), lab in gates.items()
               if s == set_name and lab[h][0]}
    if new_ids != ids:
        mism.append((set_name, h, len(ids), len(new_ids)))
check("T5 recomputed gates reproduce original gated sample set",
      not mism, f"mismatches={mism}")

# ------------------------------------------- Test 6: protection
def h_of(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


protected = ["results/pada/eval/eval_rows.csv",
             "results/pada/eval/eval_summary.json",
             "results/pada/eval/level1.json"]
before = {r: h_of(REPO / r) for r in protected}
import argparse as _ap  # noqa: E402

sys.argv = ["pada_eval_matched_fd.py",
            "--dirs_dir", str(REPO / "results/pada/dirs"),
            "--collect_dir", str(REPO / "results/pada/collect"),
            "--ckpt_dir", str(REPO / "results/pada/ckpts"),
            "--original_eval", str(REPO / "results/pada/eval"),
            "--out_dir", str(REPO / "results/pada/offline_audit_smoke"),
            "--models", "M0", "--max_samples", "2", "--bootstrap", "200"]
import importlib

mod = importlib.import_module("pada_eval_matched_fd")
importlib.reload(mod)
mod.main()
after = {r: h_of(REPO / r) for r in protected}
check("T6 protected files unchanged after audit run", before == after)
gd = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True,
                    cwd=REPO)
check("T6 git diff --check clean", gd.returncode == 0 and not gd.stdout.strip(),
      gd.stdout[:200])

print(json.dumps({"pass": len(PASS), "fail": len(FAIL)}))
sys.exit(1 if FAIL else 0)
