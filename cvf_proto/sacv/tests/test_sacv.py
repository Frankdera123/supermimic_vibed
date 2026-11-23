"""SACV Phase A CPU unit tests (Goal §10 test families 1-15).

Run:  python -m pytest cvf_proto/sacv/tests/test_sacv.py -q   (or python test_sacv.py)
All tests are CPU-only; the real-critic autograd/FD check (family 7) loads the
frozen pair bundle once and is skipped when the asset is unavailable.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_REPO = _SACV.parents[1]
for p in (str(_SACV), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto"), str(_REPO / "cvf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sacv.losses import targets as T  # noqa: E402
from sacv.losses import value_losses as L  # noqa: E402
from sacv.evaluation import metrics as M  # noqa: E402

GAMMA = 0.99
SA = Path("/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts")
MANIFEST = _REPO / "results/cvf/sacv/manifests/mother_asset_manifest.json"
SPLIT = _REPO / "results/cvf/sacv/manifests/CVF1A_SPLIT_MANIFEST.json"


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# ---------------------------------------------------------------- family 1
def test_discounted_return_math():
    Bn, Tn = 3, 20
    r = torch.ones(Bn, Tn)
    d = torch.zeros(Bn, Tn)
    out = T.discounted_return(r, d, GAMMA, horizons=(1, 5, 20))
    for h in (1, 5, 20):
        expected = sum(GAMMA ** k for k in range(h))
        for b in range(Bn):
            approx(float(out[h][b]), expected, 1e-9)
    # closed form for infinite horizon sanity at T=20
    approx(float(out[20][0]), (1 - GAMMA ** 20) / (1 - GAMMA), 1e-9)
    # varying reward
    r2 = torch.arange(1, Tn + 1, dtype=torch.float64).unsqueeze(0).repeat(Bn, 1)
    out2 = T.discounted_return(r2, d, GAMMA, horizons=(Tn,))
    expected = sum((k + 1) * GAMMA ** k for k in range(Tn))
    approx(float(out2[Tn][0]), expected, 1e-9)


# ---------------------------------------------------------------- family 2
def test_soft_survival_indexing():
    d = torch.zeros(2, 8)
    d[0, 3] = 1.0   # soft termination at step index 3
    c = T.survival_soft(d)
    expected_row0 = [1, 1, 1, 1, 0, 0, 0, 0]  # c_t excludes own step's d
    expected_row1 = [1] * 8
    for t in range(8):
        approx(float(c[0, t]), expected_row0[t])
        approx(float(c[1, t]), expected_row1[t])
    # brute-force product identity
    d2 = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.float64)
    c2 = T.survival_soft(d2)
    for t in range(8):
        brute = 1.0
        for k in range(t):
            brute *= (1.0 - float(d2[0, k]))
        approx(float(c2[0, t]), brute)


# ---------------------------------------------------------------- family 3
def test_hard_termination_truncation_and_tail():
    Bn, Tn = 1, 10
    r = torch.ones(Bn, Tn)
    d = torch.zeros(Bn, Tn)
    d[0, 4] = 1.0  # terminated (hard) at step 4 -> r_4 earned, nothing after
    out = T.discounted_return(r, d, GAMMA, horizons=(Tn,))
    expected = sum(GAMMA ** k for k in range(5))
    approx(float(out[10][0]), expected)
    c_T = T.survival_at(d, 10)
    approx(float(c_T), 0.0)
    # matched target with c_T=0 must drop the tail entirely
    y = T.matched_target(out[Tn], c_T, GAMMA, Tn, torch.tensor([7.0]))
    approx(float(y[0]), expected)


# ---------------------------------------------------------------- family 4
def test_plus_minus_pair_identity():
    B = 4
    y_p = torch.randn(B) + 3.0
    y_m = torch.randn(B) - 3.0
    dy = T.pair_difference(y_p, y_m)
    dy_swapped = T.pair_difference(y_m, y_p)
    assert torch.allclose(dy, -dy_swapped)
    eps = 0.015
    fd = T.finite_difference_response(dy, eps)
    fd_swapped = T.finite_difference_response(dy_swapped, eps)
    assert torch.allclose(fd, -fd_swapped)
    # deterministic zero when labels identical
    assert torch.allclose(T.pair_difference(y_p, y_p), torch.zeros(B, dtype=torch.float64))


# ---------------------------------------------------------------- family 5
def test_pair_loss_analytic_and_grads():
    torch.manual_seed(0)
    B = 8
    v_p = torch.randn(B, requires_grad=True)
    v_m = torch.randn(B, requires_grad=True)
    y_p = torch.randn(B)
    y_m = torch.randn(B)
    lp = L.pair_loss(v_p, v_m, y_p, y_m)
    # perfect fit -> zero
    lp0 = L.pair_loss(y_p, y_m, y_p, y_m)
    approx(float(lp0), 0.0, 1e-12)
    # analytic gradient: d/dv_p = 2((v_p - v_m) - dy)/B ; d/dv_m = -2((v_p-v_m)-dy)/B
    lp.backward()
    dv = (v_p.detach() - v_m.detach()) - (y_p - y_m)
    g_p = 2 * dv / B
    g_m = -2 * dv / B
    assert torch.allclose(v_p.grad, g_p, atol=1e-10)
    assert torch.allclose(v_m.grad, g_m, atol=1e-10)


# ---------------------------------------------------------------- family 6
def test_batch_size_consistency_b1_vs_bn():
    torch.manual_seed(1)
    B = 16
    v_p, v_m = torch.randn(B), torch.randn(B)
    y_p, y_m = torch.randn(B), torch.randn(B)
    full = L.value_loss(v_p, v_m, y_p, y_m)
    per = [L.value_loss(v_p[i:i + 1], v_m[i:i + 1], y_p[i:i + 1], y_m[i:i + 1])
           for i in range(B)]
    approx(float(full), sum(float(x) for x in per) / B, 1e-12)
    pf = L.pair_loss(v_p, v_m, y_p, y_m)
    pp = [L.pair_loss(v_p[i:i + 1], v_m[i:i + 1], y_p[i:i + 1], y_m[i:i + 1]) for i in range(B)]
    approx(float(pf), sum(float(x) for x in pp) / B, 1e-12)


# ---------------------------------------------------------------- family 7
def test_critic_autograd_vs_finite_difference():
    # (a) analytic small MLP.  FD identity is checked in float64: float32
    # roundoff amplification (~|V|*eps/h) exceeds truncation error at small h;
    # production training precision (bf16 AMP) is a separate domain.
    torch.manual_seed(2)
    net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, 1)).double()
    x = torch.randn(1, 16, requires_grad=True).double()
    u = torch.randn(1, 16).double()
    u = u / u.norm()
    v = net(x)
    g = torch.autograd.grad(v.sum(), x)[0]
    auto = float((g * u).sum())
    h = 1e-4
    fd = float((net((x + h * u).detach()) - net((x - h * u).detach())) / (2 * h))
    approx(auto, fd, 1e-9 * max(1.0, abs(auto)))

    # (b) frozen real critic B0@2100_post on random 938-D inputs (skip if absent)
    if not (SA / "online_B0/diag_pairs/pair_it_002100.pt").exists():
        print("skip real-critic FD check (bundle not found)")
        return
    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.networks import Critic

    bundle = torch.load(SA / "online_B0/diag_pairs/pair_it_002100.pt",
                        map_location="cpu", weights_only=False)
    critic = Critic(bundle["critic_post"]["norm.mean"].shape[0])
    critic.load_state_dict(bundle["critic_post"], strict=True)
    critic.eval().double()
    x = torch.randn(1, 938).double().requires_grad_(True) * 0.5
    u = torch.randn(1, 938).double()
    u = u / u.norm()
    v_raw = critic(x)
    g = torch.autograd.grad(v_raw.sum(), x)[0]
    auto = float((g * u).sum())
    h = 1e-4
    fd = float((critic((x + h * u).detach()) - critic((x - h * u).detach())) / (2 * h))
    assert abs(auto - fd) < 1e-5 * max(1.0, abs(auto)), f"auto={auto} fd={fd}"


# ---------------------------------------------------------------- family 8
def test_no_cross_batch_reduction():
    torch.manual_seed(3)
    B = 12
    v_p, v_m = torch.randn(B), torch.randn(B)
    y_p, y_m = torch.randn(B), torch.randn(B)
    terms_full = L.per_sample_value_terms(v_p, v_m, y_p, y_m)
    # shuffle companions: per-sample terms must be identical
    perm = torch.randperm(B)
    terms_perm = L.per_sample_value_terms(v_p[perm], v_m[perm], y_p[perm], y_m[perm])
    assert torch.allclose(terms_full, terms_perm[torch.argsort(perm)])
    # batch-mean invariant to batch composition
    full = L.value_loss(v_p, v_m, y_p, y_m)
    half = L.value_loss(v_p[:6], v_m[:6], y_p[:6], y_m[:6])
    approx(float(full), float((terms_full[:6].mean() + terms_full[6:].mean()) / 2), 1e-12)
    assert float(half) == float(terms_full[:6].mean())


# ---------------------------------------------------------------- family 9
def test_c1_c2_identity_and_stream():
    torch.manual_seed(4)
    B = 10
    v_p, v_m = torch.randn(B), torch.randn(B)
    y_p, y_m = torch.randn(B), torch.randn(B)
    alpha = 0.3
    comp = L.combined_loss(v_p, v_m, y_p, y_m, alpha)
    l_val = L.value_loss(v_p, v_m, y_p, y_m)
    l_pair = L.pair_loss(v_p, v_m, y_p, y_m)
    approx(float(comp["loss"]), float(l_val + alpha * l_pair), 1e-12)
    approx(float(comp["value"]), float(l_val), 1e-12)
    approx(float(comp["pair"]), float(l_pair), 1e-12)
    # same sampling stream: identical permutation gives identical subset losses
    perm = torch.randperm(B)
    a1 = L.combined_loss(v_p[perm][:5], v_m[perm][:5], y_p[perm][:5], y_m[perm][:5], alpha)
    a2 = L.combined_loss(v_p[perm][:5], v_m[perm][:5], y_p[perm][:5], y_m[perm][:5], alpha)
    approx(float(a1["loss"]), float(a2["loss"]))
    # alpha=0 must reduce exactly to C1
    approx(float(L.combined_loss(v_p, v_m, y_p, y_m, 0.0)["loss"]), float(l_val), 1e-12)


# ----------------------------------------------------------- family 10/11
def _freeze_check(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    before = copy.deepcopy([p.detach().clone() for p in module.parameters()])
    # a training step that only touches the critic
    critic = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    x = torch.randn(4, 8)
    loss = critic(x).pow(2).mean()
    loss.backward()
    opt.step()
    after = list(module.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a), "frozen module changed during update"


def test_freeze_wm_parameters():
    wm = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
    _freeze_check(wm)


def test_freeze_policy_parameters():
    pi = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    _freeze_check(pi)


# ---------------------------------------------------------------- family 12
def test_norm_statistics_invariance():
    import pada_bridge as bridge

    bridge.import_stack()
    from supermimic.networks import RunningNorm

    norm = RunningNorm(938)
    norm.eval()
    x = torch.randn(7, 938) * 3 + 1.5
    m0, v0, c0 = norm.mean.clone(), norm.var.clone(), norm.count.clone()
    out1 = norm(x)
    out2 = norm(x[:1])  # different batch composition
    assert torch.equal(norm.mean, m0) and torch.equal(norm.var, v0) and torch.equal(norm.count, c0), \
        "eval-mode forward must not update RunningNorm statistics"
    assert torch.allclose(out1[0], out2[0]), "per-sample norm output must not depend on batch"
    # explicit update DOES change stats (documents the intended training-time behavior)
    norm.update(x)
    assert not torch.equal(norm.mean, m0)


# ---------------------------------------------------------------- family 13
def test_split_leakage_and_feasibility():
    assert SPLIT.exists(), "split manifest missing - run build_split_manifest.py"
    man = json.loads(SPLIT.read_text())
    sets = {s: set(man["splits"][s]["clips"]) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]), "clip overlap train/val"
    assert not (sets["train"] & sets["test"]), "clip overlap train/test"
    assert not (sets["val"] & sets["test"]), "clip overlap val/test"
    assert not (set(man["cvf0_excluded_clips"]) & (sets["train"] | sets["val"] | sets["test"])), \
        "CVF-0 clips leaked into new splits"
    for s, info in man["splits"].items():
        assert info["n_clips"] == len(info["clips"])
        for clip, cd in info["per_clip"].items():
            assert len(cd["frames"]) == 2, f"{clip} needs >=2 starts"
            assert abs(cd["frames"][0] - cd["frames"][1]) >= man["min_frame_gap"], \
                f"{clip} start frames too close"
            for f0 in cd["frames"]:
                assert cd["clip_start"] + 1 <= f0, f"{clip} f0 at clip head"
                assert f0 + man["h_prefix"] + man["t_max"] <= cd["clip_end"] - 1, \
                    f"{clip} horizon overruns clip_end"
            assert len(cd["csv_sha256"]) == 64
    sizes = {s: man["splits"][s]["n_clips"] for s in sets}
    assert sizes["train"] >= 12 and sizes["val"] >= 4 and sizes["test"] >= 6, \
        f"pilot minimums violated: {sizes}"


# ---------------------------------------------------------------- family 14
def test_policy_optimizer_state_consistency():
    torch.manual_seed(5)
    pi = torch.nn.Sequential(torch.nn.Linear(938, 256), torch.nn.GELU(),
                             torch.nn.Linear(256, 29))
    # cold-optimizer shared init: two groups build independent optimizers from the
    # same cloned initial state -> identical first update under identical grads
    state0 = copy.deepcopy(pi.state_dict())
    grads = [torch.randn_like(p) for p in pi.parameters()]
    outs = []
    for _ in range(2):
        pi.load_state_dict(state0)
        opt = torch.optim.Adam(pi.parameters(), lr=2e-4)
        for p, g in zip(pi.parameters(), grads):
            p.grad = g.clone()
        opt.step()
        outs.append(copy.deepcopy(pi.state_dict()))
    for k in outs[0]:
        assert torch.equal(outs[0][k], outs[1][k]), "shared-init optimizer updates diverged"
    # and differ from init (a real update happened)
    assert any(not torch.equal(outs[0][k], state0[k]) for k in state0)


# ---------------------------------------------------------------- family 15
def test_checkpoint_and_data_hash_invariance():
    assert MANIFEST.exists(), "mother manifest missing - run verify_mother_assets.py"
    man = json.loads(MANIFEST.read_text())
    assert man["verdict"] == "VERIFIED", "mother assets were never fully verified"
    for name, rec in man["files"].items():
        p = Path(rec["path"])
        assert p.exists(), f"missing {p}"
        h = _sha256_file(p)
        assert h == rec["sha256"], f"mother file changed since verification: {name}"
    for name, rec in man["oracle_data"].items():
        p = Path(rec["path"])
        assert p.exists(), f"missing {p}"
        h = _sha256_file(p)
        assert h == rec["sha256"], f"CVF-0 oracle data changed: {name}"
    if SPLIT.exists():
        sm = json.loads(SPLIT.read_text())
        for s in ("train", "val", "test"):
            for clip, cd in sm["splits"][s]["per_clip"].items():
                csv = _REPO / "data/lafan1_g1" / f"{clip}.csv"
                assert _sha256_file(csv) == cd["csv_sha256"], f"motion csv changed: {clip}"


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------- metric smoke checks
def test_metrics_sanity():
    ref = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5])
    pred = torch.tensor([1.1, -0.9, -2.0, 2.0, 0.4])
    approx(M.sign_agreement(pred, ref), 3 / 5)
    approx(M.directional_rmse(pred, ref), math.sqrt((0.1 ** 2 + 0.1 ** 2 + 4 ** 2 + 4 ** 2 + 0.1 ** 2) / 5), 1e-9)
    # ratios: 1.1, 0.9, 1.0, 1.0, 0.8 -> median 1.0
    approx(M.magnitude_ratio(pred, ref), 1.0, 1e-9)
    assert M.spearman(ref, ref) == 1.0
    st = M.exact_paired_sign_test(torch.tensor([1.0, -1.0, 2.0, 3.0]))
    assert st["n"] == 4 and st["n_plus"] == 3
    assert abs(st["p"] - 0.625) < 1e-9
    # oracle-2 prefix weighting convention
    d = M.gamma_survival_weighted_prefix_fd(2.0, 1.0, 1.0, 1.0, GAMMA, 32, 0.015)
    approx(d, (GAMMA ** 32) * (2.0 - 1.0) / (2 * 0.015))
    # asymmetric survival enters per-branch
    d2 = M.gamma_survival_weighted_prefix_fd(2.0, 1.0, 1.0, 0.0, GAMMA, 32, 0.015)
    approx(d2, (GAMMA ** 32) * 2.0 / (2 * 0.015))


# ------------------------------------------- v2 contract regressions (ERRATUM)
def test_survival_after_last_step_termination():
    d = torch.zeros(1, 8)
    d[0, 7] = 1.0                       # termination at the very last step of the window
    approx(float(T.survival_after(d, 8)[0]), 0.0)    # tail weight must vanish
    approx(float(T.survival_after(d, 7)[0]), 1.0)    # c_7 does not include step 7
    d2 = torch.tensor([[0.1, 0.2, 0.0, 0.5]], dtype=torch.float64)
    approx(float(T.survival_after(d2, 3)[0]), 0.9 * 0.8 * 1.0)
    approx(float(T.survival_after(d2, 4)[0]), 0.9 * 0.8 * 0.5)
    torch.manual_seed(7)
    dd = torch.rand(3, 12) * 0.3
    cs = T.survival_soft(dd)                      # c_0 .. c_11
    for t in range(1, 12):                        # survival_after(d, t) is exactly c_t
        assert torch.allclose(cs[:, t], T.survival_after(dd, t)), f"indexing mismatch at t={t}"


def _v2_pairs(split: str = "val", horizon: int = 96):
    path = _REPO / f"results/cvf/sacv/datasets/v2/{split}_pairs.pt"
    if not path.exists():
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d["schema"] == "sacv_pairs_v2", d["schema"]
    return [p for p in d["pairs"] if p["horizon"] == horizon]


def test_pairs_v2_state_label_alignment():
    """The critic state must be x_H (prefix end) and every label must be reproducible
    from the raw step records.  Guards against the v1 misalignment (ERRATUM sec. 2)."""
    pairs = _v2_pairs()
    if pairs is None:
        print("skip (no v2 dataset)")
        return
    for p in pairs:
        for side in ("plus", "minus"):
            st = p[f"state_{side}"]
            assert st["snap_group"] == "snaps_prefix" and st["snap_key"] == "prefix_32", st
        assert abs(p["Delta_G"] - (p["G_plus"] - p["G_minus"])) < 1e-12
        assert abs(p["Delta_Y"] - (p["Y_plus"] - p["Y_minus"])) < 1e-12
        assert abs(p["c_plus"] - p["c_plus"]) == 0 and 0.0 <= p["c_plus"] <= 1.0 + 1e-9

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    v_ref_sd = torch.load(SA / "online_B0/diag_pairs/pair_it_002100.pt",
                          map_location="cpu", weights_only=False)["critic_post"]
    v_ref = Critic(938)
    v_ref.load_state_dict(v_ref_sd, strict=True)
    v_ref.eval()
    proto = bridge.setup_world(device="cpu")["proto"]
    xi_n = normalize_xi(torch.zeros(1, 33) + torch.tensor([[1.0] + [0.0] * 32]))
    raw_cache: dict = {}
    step = max(1, len(pairs) // 4)
    for p in pairs[::step]:
        for side in ("plus", "minus"):
            st = p[f"state_{side}"]
            if st["batch"] not in raw_cache:
                raw_cache[st["batch"]] = torch.load(
                    next((_REPO / "results/cvf/sacv/datasets/raw").rglob(st["batch"])),
                    map_location="cpu", weights_only=False)
            b = raw_cache[st["batch"]]
            br = b["branches"][st["branch"]]
            slot = st["slot"]
            # (a) x_H is the prefix-end snapshot and is NOT the continuation-end snapshot
            S_H = br["snaps_prefix"]["prefix_32"]["S_flat"][slot]
            S_T = br["snaps_cont"]["cont_96"]["S_flat"][slot]
            assert not torch.allclose(S_H, S_T), "prefix-end and continuation-end states coincide?"
            # (b) labels re-derived from the raw step records
            pl = br["prefix_len"]
            assert pl == 32
            r = br["full"]["r"][slot:slot + 1, pl:]
            dsoft = br["full"]["d_soft"][slot:slot + 1, pl:]
            G = T.discounted_return(r, dsoft, GAMMA, (96,))[96]
            c = T.survival_after(dsoft, 96)
            snap_T = br["snaps_cont"]["cont_96"]
            S = unflatten_state(snap_T["S_flat"][slot:slot + 1])
            f = snap_T["f"][slot:slot + 1]
            with torch.no_grad():
                v = float(v_ref(proto.critic_input(S, f, xi_n, proto.wm_goal(f, S)))[0])
            assert abs(float(G) - p[f"G_{side}"]) < 1e-9
            assert abs(float(c) - p[f"c_{side}"]) < 1e-12
            assert abs(v - p[f"V_ref_{side}"]) < 1e-4
            assert abs(float(G) + GAMMA ** 96 * float(c) * v - p[f"Y_{side}"]) < 1e-4
            # (c) the re-derived return equals the collector's online accumulation
            assert abs(float(G) - float(br["cont"]["V_soft"][96][slot])) < 1e-4


def test_chord_delta_roundtrip():
    """chord_delta(Sm, Sp) is the exact tangent taking Sm onto Sp in perturb_state coords."""
    from cvf_core import perturb_state
    from sacv.evaluation.sacv_dump_preds import chord_delta

    torch.manual_seed(11)

    def rand_state(n):
        from supermimic.rotations import so3_exp

        return (torch.randn(n, 3), so3_exp(torch.randn(n, 3) * 1.5), torch.randn(n, 3) * 2,
                torch.randn(n, 3) * 3, torch.randn(n, 29) * 0.5, torch.randn(n, 29) * 4)

    Sm, Sp = rand_state(16), rand_state(16)
    back = perturb_state(Sm, chord_delta(Sm, Sp))
    for a, b_ in zip(back, Sp):
        assert torch.allclose(a, b_, atol=1e-4), float((a - b_).abs().max())
    assert torch.allclose(chord_delta(Sm, Sm), torch.zeros(16, 70), atol=1e-6)


def test_stats_primitives():
    from sacv.evaluation import sacv_stats as S

    x = np.array([0.3, -0.1, -2.0, 1.0, 0.0])
    y = np.array([1.0, -0.5, -1.0, 2.0, 0.0])
    approx(S._spearman(x, y), 1.0)
    assert S._sign(x, y) == 1.0
    # ties are rank-averaged: ranks x = [1.5, 1.5, 3], y = [1, 2, 3] -> 1.5 / sqrt(1.5 * 2)
    approx(S._spearman(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])), 1.5 / np.sqrt(3.0), 1e-9)
    approx(S._binom_two_sided(8, 8), 2 / 256, 1e-12)
    approx(S._binom_two_sided(4, 8), 1.0, 1e-12)
    # layer construction: single-sided termination and cross-epsilon inconsistency are excluded
    meta = [
        {"clip": "a", "frame": 1, "family": "delta_2100", "eps": 0.015, "Delta_G": 0.0030, "c_plus": 1.0, "c_minus": 1.0, "term_plus": 0.0, "term_minus": 0.0},
        {"clip": "a", "frame": 1, "family": "delta_2100", "eps": 0.03, "Delta_G": 0.0060, "c_plus": 1.0, "c_minus": 1.0, "term_plus": 0.0, "term_minus": 0.0},
        {"clip": "a", "frame": 2, "family": "delta_2300", "eps": 0.015, "Delta_G": 0.0030, "c_plus": 1.0, "c_minus": 1.0, "term_plus": 0.0, "term_minus": 0.0},
        {"clip": "b", "frame": 3, "family": "delta_2300", "eps": 0.03, "Delta_G": -0.0090, "c_plus": 1.0, "c_minus": 1.0, "term_plus": 0.0, "term_minus": 0.0},
        {"clip": "b", "frame": 4, "family": "ascent", "eps": 0.015, "Delta_G": 0.02, "c_plus": 1.0, "c_minus": 0.1, "term_plus": 0.0, "term_minus": 1.0},
        {"clip": "c", "frame": 5, "family": "random", "eps": 0.015, "Delta_G": 0.02, "c_plus": 1.0, "c_minus": 0.1, "term_plus": 0.0, "term_minus": 0.0},
    ]
    L = S.build_layers(meta)
    assert L["all"] == [0, 1, 2, 3, 4, 5]
    assert L["nobif"] == [0, 1, 2, 3]
    assert L["gated"] == [0, 1], L["gated"]        # D(eps)=0.1 for both eps of pair (0,1)
    clips = ["a", "a", "a", "b"]
    idx = [0, 1, 2, 3]
    ci1 = S.clip_boot(lambda s: float(np.mean(s)), idx, clips, n_boot=200, seed=3)
    ci2 = S.clip_boot(lambda s: float(np.mean(s)), idx, clips, n_boot=200, seed=3)
    assert ci1 == ci2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(fns)} passed")
    sys.exit(0 if n_pass == len(fns) else 1)
