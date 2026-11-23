"""PADA-0a offline matched-FD evaluator (CPU, no simulator, no training).

Audits and corrects two measurement issues of the original PADA-0 evaluation
(pada_eval.py) WITHOUT touching original outputs:

Issue 1 (evaluation point): the original D_wm was the autograd directional
derivative at the PERTURBED point A+eps*d, while D_sim is a CENTRAL finite
difference around A.  This tool computes three separated metrics:
    A. D_wm_legacy_plusgrad -- original definition (reproduced for continuity)
    B. D_wm_center          -- grad_A J_wm(A)^T d at the UNPERTURBED A
                               (the matched primary gradient metric)
    C. D_wm_fd              -- [J_wm(A+eps d) - J_wm(A-eps d)] / (2 eps)
                               with the SAME eps as the simulator side

Issue 2 (path decomposition): the original compared WM's survival-DETACHED
path gradients (D_direct_iso / D_critic_iso from crgf.open_loop_paths) against
simulator per-path central FDs, whose returns INCLUDE survival-factor
gradients (c_t depends on the action through the trajectory).  This tool
rebuilds WM path returns with EXACTLY the simulator branch_J definitions
(attached survival for every path) via matched_path_returns(), and verifies
the decomposition identity
    D_total = D_state + D_direct + D_critic      (matched, attached form)
as well as crgf's own identity
    D_total = D_state + D_direct_iso + D_critic_iso + D_cross.

Sample identity: gates are recomputed deterministically from the recorded
simulator data (model-independent) and ASSERTED equal to the gated sample set
of the original eval_rows.csv; all statistics use those identical gated
samples.  Original files under results/pada/eval/ are never written.

Outputs (results/pada/offline_audit/):
    matched_eval_rows.csv      per (model, set, si, ji, clip, H, eps) rows
    matched_eval_summary.json  fidelity stats + paired/clustered bootstrap
    path_decomposition.json    identity residuals + per-path M2 vs M1
    gradient_consistency.json  legacy vs center vs FD consistency/nonlinearity
    data_integrity.json        SHA256/provenance/local-only + protection hashes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402

from pada_eval import branch_J  # noqa: E402  (identical sim-side computation)
from supermimic.crgf import (Protocol, fidelity_stats, open_loop_paths,  # noqa: E402
                             path_directional_derivatives, reliability_gate,
                             reward_split, soft_term, spearman)
from supermimic.mdp import flatten_state, normalize_xi, unflatten_state  # noqa: E402

POLICY_SETS = ("eval_pol_seen", "eval_pol_B", "eval_pol_C")
PATHS = ("state", "direct", "critic", "total")
M0KEY = "M0"
GAMMA, TZ, TT = 0.99, 0.02, 0.04


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matched_path_returns(wm, critic, mdp, proto, S0, f0, a_seq, xi_n,
                         gamma=GAMMA, tz=TZ, tt=TT, critic_weight=1.0,
                         per_sample=True):
    """WM path returns with EXACTLY the simulator branch_J definitions.

    J_state  = sum_t gamma^t c_t r_state_t          (c attached)
    J_direct = sum_t gamma^t c_t r_direct_t         (c attached)
    J_critic = gamma^H c_{H-1} V(s_H)               (c attached)
    J_total  = J_state + J_direct + J_critic

    c_t = prod_{k<=t} (1 - soft_term(S_k, f_k)) computed on the WM rollout;
    r_state/r_direct via crgf.reward_split (identical to the recorded sim
    rewards); V via the diag-pair critic on the final WM state with the
    state-relative goal, mirroring branch_J's real-state evaluation.
    """
    B, H, A = a_seq.shape
    c_prev = torch.ones(B, device=a_seq.device)
    J_state = a_seq.new_zeros(())
    J_direct = a_seq.new_zeros(())
    a_prev = torch.zeros(B, A, device=a_seq.device)
    S = S0
    f = f0
    for t in range(H):
        a_t = a_seq[:, t]
        a_wm = torch.clamp(a_t, -20.0, 20.0)      # action_gain=1.0 (as crgf)
        S = proto.wm_step(wm, S, a_wm, xi_n)
        f1 = f + 1
        r_state, r_direct = reward_split(mdp, S, f1, a_t, a_prev)
        d = soft_term(mdp, S, f1, tz=tz, tt=tt)
        c_prev = c_prev * (1.0 - d)
        # r_state / r_direct are per-sample scalars (B,); keep the batch axis
        term_s = (c_prev * r_state) if per_sample else (c_prev * r_state).sum()
        term_d = (c_prev * r_direct) if per_sample else (c_prev * r_direct).sum()
        J_state = J_state + (gamma ** t) * term_s
        J_direct = J_direct + (gamma ** t) * term_d
        a_prev = a_t
        f = f1
    J_critic = a_seq.new_zeros(())
    V = None
    if critic is not None:
        gw = proto.wm_goal(f, S)
        vin = proto.critic_input(S, f, xi_n, gw)
        V = critic(vin)
        # V is per-sample (B,); keep the batch axis (a .sum(-1) here silently
        # collapses the batch and mixes rows in the gradient)
        J_critic = (gamma ** H) * critic_weight * ((c_prev * V) if per_sample
                                                   else (c_prev * V).sum())
    return {"state": J_state, "direct": J_direct, "critic": J_critic,
            "total": J_state + J_direct + J_critic, "V": V, "c_last": c_prev}


def numeric_path_returns(wm, critic, mdp, proto, S0, f0, a_seq, xi_n):
    """Graph-free rollout matching matched_path_returns numerically.

    Returns per-step reward/survival lists plus the FULL-horizon path returns
    (critic at s_H).  H=8 truncated returns are assembled by the caller from
    the per-step prefix sums plus numeric_h8_critic."""
    with torch.no_grad():
        B, H, A = a_seq.shape
        c_prev = torch.ones(B, device=a_seq.device)
        js, jd = [], []
        S = S0
        f = f0
        a_prev = torch.zeros(B, A, device=a_seq.device)
        for t in range(H):
            a_t = a_seq[:, t]
            S = proto.wm_step(wm, S, torch.clamp(a_t, -20.0, 20.0), xi_n)
            f1 = f + 1
            r_state, r_direct = reward_split(mdp, S, f1, a_t, a_prev)
            d = soft_term(mdp, S, f1, tz=TZ, tt=TT)
            c_prev = c_prev * (1.0 - d)
            js.append((c_prev * r_state).cpu().clone())
            jd.append((c_prev * r_direct).cpu().clone())
            a_prev = a_t
            f = f1
        J_critic = torch.zeros(js[0].shape[0])
        if critic is not None:
            gw = proto.wm_goal(f, S)
            vin = proto.critic_input(S, f, xi_n, gw)
            J_critic = (GAMMA ** H) * (c_prev.cpu() * critic(vin).cpu())
        J_state = sum((GAMMA ** t) * js[t] for t in range(H))
        J_direct = sum((GAMMA ** t) * jd[t] for t in range(H))
        return {"per_step_state": js, "per_step_direct": jd,
                "state": J_state, "direct": J_direct, "critic": J_critic,
                "total": J_state + J_direct + J_critic}


def numeric_h8_critic(wm, critic, mdp, proto, S0, f0, a_seq, xi_n, step=8):
    """gamma^8 * c_7 * V(s_8) per sample for the H=8 truncated critic term."""
    with torch.no_grad():
        B, H, A = a_seq.shape
        c_prev = torch.ones(B, device=a_seq.device)
        S = S0
        f = f0
        a_prev = torch.zeros(B, A, device=a_seq.device)
        for t in range(step):
            S = proto.wm_step(wm, S, torch.clamp(a_seq[:, t], -20.0, 20.0), xi_n)
            f1 = f + 1
            d = soft_term(mdp, S, f1, tz=TZ, tt=TT)
            c_prev = c_prev * (1.0 - d)
            a_prev = a_seq[:, t]
            f = f1
        gw = proto.wm_goal(f, S)
        vin = proto.critic_input(S, f, xi_n, gw)
        return (GAMMA ** step) * (c_prev.cpu() * critic(vin).cpu())


def compute_sim_ground_truth(ev, meta, dirs_dir, world, proto, critic,
                             horizons, eval_sets, k_per_set):
    """Deterministic model-independent simulator FD ground truth + gate labels.

    Same computation as the original pada_eval.py gate (branch_J rewards,
    reliability_gate on pooled-eps state-path D with base-repeat range check),
    refactored so the audit tests and the tool share one implementation."""
    gamma = GAMMA
    eps_list = ev["meta"]["eps_eval"]
    S_e = len(meta["eval_starts"])
    base_J = {}
    for si in range(S_e):
        for rep in range(ev["meta"]["base_reps"]):
            tr = ev["branches"][f"base_{si}_{rep}"]
            for h in horizons:
                base_J[(si, rep, h)] = branch_J(tr, critic, proto, world, h, gamma)
    sim_D = {}
    gates = {}
    for set_name in eval_sets:
        dirs = torch.load(Path(dirs_dir) / f"{set_name}.pt", map_location="cpu",
                          weights_only=False)["dirs"]
        k = int(dirs.shape[1])
        for si in range(S_e):
            for ji in range(k):
                per_eps = {e: {} for e in eps_list}
                for eps in eps_list:
                    for br in ("plus", "minus"):
                        tr = ev["branches"][f"{set_name}_{si}_{ji}_{eps}_{br}"]
                        per_eps[eps][br] = {h: branch_J(tr, critic, proto, world, h, gamma)
                                            for h in horizons}
                sim_D[(set_name, si, ji)] = per_eps
                labels = {}
                for h in horizons:
                    base_vals = [float(base_J[(si, rep, h)]["total"]) for rep in
                                 range(ev["meta"]["base_reps"])]
                    d_state = torch.tensor(
                        [float(per_eps[e]["plus"][h]["state"] - per_eps[e]["minus"][h]["state"])
                         / (2 * e) for e in eps_list])
                    ok, why = reliability_gate(torch.tensor(base_vals), d_state)
                    labels[h] = (bool(ok), why)
                gates[(set_name, si, ji)] = labels
    return sim_D, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs_dir", type=Path, default=Path("results/pada/dirs"))
    ap.add_argument("--collect_dir", type=Path, default=Path("results/pada/collect"))
    ap.add_argument("--ckpt_dir", type=Path, default=Path("results/pada/ckpts"))
    ap.add_argument("--original_eval", type=Path, default=Path("results/pada/eval"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/pada/offline_audit"))
    ap.add_argument("--repo_root", type=Path, default=Path("."))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default="M0,M1_500,M1_1000,M1_1500,M2_500,M2_1000,M2_1500")
    ap.add_argument("--primary_step", type=int, default=1500)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=2600)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--skip_identity_assert", action="store_true")
    args = ap.parse_args()
    dev = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- data integrity (phase-1 record) ------------------------
    prov = json.load(open(args.dirs_dir.parent / "provenance.json"))
    integrity = {"head": None, "artifacts": {}, "local_only_ignored": [],
                 "protected_before": {}}
    try:
        import subprocess

        integrity["head"] = subprocess.run(["git", "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    for rel, rec in prov["artifacts_sha256"].items():
        p = args.dirs_dir.parent / rel
        if p.exists():
            integrity["artifacts"][rel] = {"status": "exists",
                                           "sha256_match": sha256(p) == rec}
        else:
            integrity["artifacts"][rel] = {"status": "missing", "sha256_recorded": rec}
    import subprocess as sp

    tracked = set(sp.run(["git", "ls-files", "results/pada"], capture_output=True,
                         text=True).stdout.split())
    ck_hashes = {}
    for p in sorted(Path("results/pada").rglob("*")):
        if p.is_file() and str(p) not in tracked:
            integrity["local_only_ignored"].append(str(p))
            if p.suffix == ".pt" and "ckpts" in str(p) and "smoke" not in str(p):
                ck_hashes[p.name] = sha256(p)
    integrity["ckpt_sha256_supplement"] = ck_hashes
    for rel in ("eval/eval_rows.csv", "eval/eval_summary.json", "eval/level1.json"):
        p = args.dirs_dir.parent / rel
        integrity["protected_before"][rel] = sha256(p)

    # ---------------- load assets ------------------------
    meta = json.load(open(args.dirs_dir / "directions_meta.json"))
    ev = torch.load(args.collect_dir / "eval_data.pt", map_location="cpu",
                    weights_only=False)
    assert ev["meta"]["gamma"] == GAMMA
    H_full = ev["meta"]["H"]
    horizons = sorted({8, H_full})
    eps_list = ev["meta"]["eps_eval"]
    S_e = len(meta["eval_starts"])
    eval_sets = ("eval_random",) + POLICY_SETS
    set_dirs = {n: torch.load(args.dirs_dir / f"{n}.pt", map_location="cpu",
                              weights_only=False)["dirs"] for n in eval_sets}
    k_per_set = {n: int(d.shape[1]) for n, d in set_dirs.items()}

    pair = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    critic = pair["critic"].to(dev)
    xi_n_dev = normalize_xi(torch.tensor([[1.0] + [0.0] * 32], device=dev))

    def eval_frame(si):
        return meta["eval_starts"][si]["frame"]

    def clip_of(si):
        return meta["eval_starts"][si]["clip_name"]

    # ---------------- simulator ground truth + gates (model-independent) ---
    print("[1/4] sim-side FD ground truth + gates ...")
    sim_D, gates = compute_sim_ground_truth(
        ev, meta, args.dirs_dir, world, proto, critic, horizons, eval_sets, k_per_set)

    # ---------------- sample identity vs original eval ----------------------
    print("[2/4] sample-identity assertion vs original eval_rows.csv ...")
    orig_rows = list(csv.DictReader(open(args.original_eval / "eval_rows.csv")))
    orig_gated = {}     # (set, h) -> {(si, ji)}
    for r in orig_rows:
        orig_gated.setdefault((r["set"], int(r["H"])), set()).add((int(r["si"]), int(r["ji"])))
    mismatches = []
    for (set_name, h), ids in orig_gated.items():
        new_ids = {(si, ji) for (s, si, ji), lab in gates.items() if s == set_name and lab[h][0]}
        if new_ids != ids:
            mismatches.append({"set": set_name, "H": h,
                               "only_original": sorted(ids - new_ids),
                               "only_recomputed": sorted(new_ids - ids)})
    if mismatches and not args.skip_identity_assert:
        raise SystemExit(f"SAMPLE IDENTITY FAILURE: {json.dumps(mismatches)}")
    identity_ok = not mismatches

    # ---------------- model registry ------------------------
    model_specs = {}
    for m in args.models.split(","):
        if m == "M0":
            model_specs[m] = ("pair", None)
        else:
            tag, st = m.split("_")
            model_specs[m] = ({"M1": "random", "M2": "policy"}[tag], int(st))

    def load_wm(mkey):
        kind, st = model_specs[mkey]
        if kind == "pair":
            return pair["wm"]
        ck = torch.load(args.ckpt_dir / f"M_{kind}_{st:05d}.pt", map_location=dev,
                        weights_only=False)
        wm = bridge.load_pair_bundle("online_B0", 2100, device=dev)["wm"]
        wm.load_state_dict(ck["wm"])
        return wm.to(dev).eval()

    # ---------------- per-sample audit ------------------------
    print("[3/4] WM-side matched metrics (legacy / center / FD) ...")
    rows = []
    consistency = []
    decomp_resid = []
    n_samples = 0
    for mkey in model_specs:
        wm = load_wm(mkey)
        for set_name in eval_sets:
            dirs_all = set_dirs[set_name].to(dev)
            K = k_per_set[set_name]
            samples = [(si, ji) for si in range(S_e) for ji in range(K)
                       if gates[(set_name, si, ji)][horizons[0]][0]
                       or gates[(set_name, si, ji)][horizons[1]][0]]
            if args.max_samples:
                samples = samples[:max(0, args.max_samples - n_samples)]
            if not samples:
                continue
            Bn = len(samples)
            n_samples += Bn
            base_b = torch.stack([ev["base_actions"][si].to(dev) for si, _ in samples], 0)
            dirv_b = torch.stack([dirs_all[si, ji] for si, ji in samples], 0)
            S0_b = unflatten_state(torch.stack(
                [ev["branches"][f"{set_name}_{si}_{ji}_{eps_list[0]}_plus"]["s0"]
                 for si, ji in samples], 0).to(dev))
            f0_b = torch.tensor([eval_frame(si) for si, _ in samples], device=dev)
            dv_b = dirv_b.reshape(Bn, -1)
            z29 = torch.zeros(Bn, 29, device=dev)
            xi_b = xi_n_dev.expand(Bn, -1)

            # metric B: matched center autograd (H full + H8)
            a_c = base_b.clone().requires_grad_(True)
            rets = matched_path_returns(wm, critic, mdp, proto, S0_b, f0_b,
                                        a_c, xi_b)
            center_grads = {}
            for p in PATHS:
                g = torch.autograd.grad(rets[p].sum(), a_c, retain_graph=True,
                                        allow_unused=False)[0]
                center_grads[p] = (g.reshape(Bn, -1) * dv_b).sum(-1)
            center_iso = {}
            paths_c = open_loop_paths(wm, critic, mdp, S0_b, f0_b, a_c, z29,
                                      xi_b, gamma=GAMMA, goal_frames=proto.K)
            grads_c = path_directional_derivatives(paths_c, a_c)
            for p in ("state", "direct", "critic", "total", "cross"):
                center_iso[p] = (grads_c[p].reshape(Bn, -1) * dv_b).sum(-1)
            for b, (si, ji) in enumerate(samples):
                decomp_resid.append({
                    "model": mkey, "set": set_name, "si": si, "ji": ji,
                    "resid_total_minus_sum": float(center_grads["total"][b]
                        - center_grads["state"][b] - center_grads["direct"][b]
                        - center_grads["critic"][b]),
                    "resid_crgf_identity": float(center_iso["total"][b]
                        - center_iso["state"][b] - center_iso["direct"][b]
                        - center_iso["critic"][b] - center_iso["cross"][b]),
                    "survival_direct": float(center_grads["direct"][b] - center_iso["direct"][b]),
                    "survival_critic": float(center_grads["critic"][b] - center_iso["critic"][b]),
                })
            a_c8 = base_b[:, :8].clone().contiguous().requires_grad_(True)
            rets8 = matched_path_returns(wm, critic, mdp, proto, S0_b, f0_b,
                                         a_c8, xi_b)
            center8 = {}
            for p in PATHS:
                g8 = torch.autograd.grad(rets8[p].sum(), a_c8, retain_graph=True,
                                         allow_unused=False)[0]
                center8[p] = (g8.reshape(Bn, -1) * dirv_b[:, :8].reshape(Bn, -1)).sum(-1)

            # metric A: legacy plus-point autograd (original definition)
            legacy = {}
            for eps in eps_list:
                for h in horizons:
                    a_seq = (base_b + eps * dirv_b)[:, :h].contiguous().requires_grad_(True)
                    paths = open_loop_paths(wm, critic, mdp, S0_b, f0_b, a_seq, z29,
                                            xi_b, gamma=GAMMA,
                                            goal_frames=proto.K)
                    grads = path_directional_derivatives(paths, a_seq)
                    for p in PATHS:
                        legacy[(eps, h, p)] = (grads[p].reshape(Bn, -1)
                                               * dirv_b[:, :h].reshape(Bn, -1)).sum(-1)

            # metric C: matched WM central FD (numeric, same eps as sim)
            fd_vals = {}
            for eps in eps_list:
                num = {}
                for br, sgn in (("plus", 1.0), ("minus", -1.0)):
                    a_br = (base_b + sgn * eps * dirv_b)[:, :H_full].contiguous()
                    num[br] = numeric_path_returns(wm, critic, mdp, proto, S0_b, f0_b,
                                                   a_br, xi_b)
                for h in horizons:
                    for p in PATHS:
                        if h == H_full:
                            fd_vals[(eps, h, p)] = (num["plus"][p] - num["minus"][p]) / (2 * eps)
                        else:
                            st_p = sum((GAMMA ** t) * num["plus"]["per_step_state"][t]
                                       for t in range(h))
                            st_m = sum((GAMMA ** t) * num["minus"]["per_step_state"][t]
                                       for t in range(h))
                            di_p = sum((GAMMA ** t) * num["plus"]["per_step_direct"][t]
                                       for t in range(h))
                            di_m = sum((GAMMA ** t) * num["minus"]["per_step_direct"][t]
                                       for t in range(h))
                            cr_p = numeric_h8_critic(wm, critic, mdp, proto, S0_b, f0_b,
                                                     (base_b + eps * dirv_b)[:, :8].contiguous(),
                                                     xi_b)
                            cr_m = numeric_h8_critic(wm, critic, mdp, proto, S0_b, f0_b,
                                                     (base_b - eps * dirv_b)[:, :8].contiguous(),
                                                     xi_b)
                            per = {"state": (st_p - st_m) / (2 * eps),
                                   "direct": (di_p - di_m) / (2 * eps),
                                   "critic": (cr_p - cr_m) / (2 * eps)}
                            per["total"] = per["state"] + per["direct"] + per["critic"]
                            for pp in PATHS:
                                fd_vals[(eps, h, pp)] = per[pp]

            # assemble rows
            for b, (si, ji) in enumerate(samples):
                labels = gates[(set_name, si, ji)]
                for eps in eps_list:
                    for h in horizons:
                        row = {"model": mkey, "set": set_name, "si": si, "ji": ji,
                               "clip": clip_of(si), "H": h, "eps": eps,
                               "gate": int(labels[h][0]), "gate_reason": labels[h][1]}
                        for p in ("state", "direct", "critic", "total", "cross"):
                            if p != "cross":
                                row[f"D_sim_{p}"] = float(
                                    (sim_D[(set_name, si, ji)][eps]["plus"][h][p]
                                     - sim_D[(set_name, si, ji)][eps]["minus"][h][p]) / (2 * eps))
                                row[f"D_wm_legacy_{p}"] = float(legacy[(eps, h, p)][b])
                                row[f"D_wm_center_{p}"] = float((center_grads[p] if h == H_full
                                                                 else center8[p])[b])
                                row[f"D_wm_fd_{p}"] = float(fd_vals[(eps, h, p)][b])
                            row[f"D_wm_center_iso_{p}"] = (float(center_iso[p][b])
                                                           if h == H_full else "")
                        rows.append(row)
                        if labels[h][0]:
                            for p in PATHS:
                                consistency.append({
                                    "model": mkey, "set": set_name, "si": si, "ji": ji,
                                    "H": h, "eps": eps, "path": p,
                                    "legacy": float(legacy[(eps, h, p)][b]),
                                    "center": float((center_grads[p] if h == H_full
                                                     else center8[p])[b]),
                                    "fd": float(fd_vals[(eps, h, p)][b]),
                                    "sim": row[f"D_sim_{p}"]})
        del wm
        print(f"    model {mkey} done ({round(time.time()-t0,1)}s)")

    # ---------------- write matched rows ------------------------
    fields = ["model", "set", "si", "ji", "clip", "H", "eps", "gate", "gate_reason"]
    for p in ("state", "direct", "critic", "total", "cross"):
        fields += [f"D_sim_{p}", f"D_wm_legacy_{p}", f"D_wm_center_{p}", f"D_wm_fd_{p}",
                   f"D_wm_center_iso_{p}"]
    with open(args.out_dir / "matched_eval_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # ---------------- statistics on ORIGINAL gated samples ------------------------
    print("[4/4] statistics ...")
    primary = args.primary_step
    M1, M2 = f"M1_{primary}", f"M2_{primary}"
    gseed = torch.Generator().manual_seed(args.bootstrap_seed)

    def gated_rows(metric_prefix, model, set_name, h, path):
        out = [r for r in rows if r["model"] == model and r["set"] == set_name
               and r["H"] == h and r["gate"] == 1]
        # pool eps by mean (as the original did) then report
        agg = {}
        for r in out:
            key = (r["si"], r["ji"])
            agg.setdefault(key, {})[r["eps"]] = r
        pooled = []
        for key, per_eps in sorted(agg.items()):
            wm_v = sum(per_eps[e][f"{metric_prefix}_{path}"] for e in eps_list) / len(eps_list)
            sim_v = sum(per_eps[e][f"D_sim_{path}"] for e in eps_list) / len(eps_list)
            pooled.append({"si": key[0], "ji": key[1],
                           "clip": per_eps[eps_list[0]]["clip"],
                           "wm": wm_v, "sim": sim_v})
        return pooled

    def stats_for(metric_prefix, model, set_name, h, path, pool=True):
        if pool:
            samples = gated_rows(metric_prefix, model, set_name, h, path)
            w = torch.tensor([s["wm"] for s in samples])
            s = torch.tensor([s["sim"] for s in samples])
        else:
            samples = [r for r in rows if r["model"] == model and r["set"] == set_name
                       and r["H"] == h and r["gate"] == 1]
            w = torch.tensor([r[f"{metric_prefix}_{path}"] for r in samples])
            s = torch.tensor([r[f"D_sim_{path}"] for r in samples])
        st = fidelity_stats(w, s, gate=None)
        st["sign_binary"] = [float(torch.sign(w[i]) == torch.sign(s[i]))
                             for i in range(len(w))]
        return st, samples

    summary = {"identity_ok": identity_ok,
               "identity_mismatches": mismatches,
               "eps_list": eps_list, "horizons": horizons,
               "models": list(model_specs),
               "fidelity": {}, "per_eps_fidelity": {},
               "paired": {}, "per_clip": {}, "leave_one_clip_out": {},
               "snapshot_curve": {}}
    for mkey in model_specs:
        for set_name in eval_sets:
            for h in horizons:
                for path in PATHS:
                    st, _ = stats_for("D_wm_legacy", mkey, set_name, h, path)
                    summary["fidelity"][f"legacy|{mkey}|{set_name}|H{h}|{path}"] = st
                    st, _ = stats_for("D_wm_center", mkey, set_name, h, path)
                    summary["fidelity"][f"center|{mkey}|{set_name}|H{h}|{path}"] = st
                    st, _ = stats_for("D_wm_fd", mkey, set_name, h, path)
                    summary["fidelity"][f"fd|{mkey}|{set_name}|H{h}|{path}"] = st
                for path in PATHS:
                    for e in eps_list:
                        sel = [r for r in rows if r["model"] == mkey and r["set"] == set_name
                               and r["H"] == h and r["eps"] == e and r["gate"] == 1]
                        if not sel:
                            continue
                        w = torch.tensor([r[f"D_wm_center_{path}"] for r in sel])
                        sg = torch.tensor([r[f"D_sim_{path}"] for r in sel])
                        summary["per_eps_fidelity"][f"center|{mkey}|{set_name}|H{h}|{path}|eps{e}"] = \
                            fidelity_stats(w, sg, gate=None)

    # per-clip counts + metrics (center metric, primary models)
    for set_name in eval_sets:
        for h in horizons:
            for m in (M0KEY := "M0", M1, M2):
                samples = gated_rows("D_wm_center", m, set_name, h, "total")
                by_clip = {}
                for s in samples:
                    by_clip.setdefault(s["clip"], []).append(s)
                summary["per_clip"][f"{set_name}|H{h}|{m}"] = {
                    clip: {"n": len(v),
                           "sign_agree": sum(x["wm"] * x["sim"] > 0 for x in v) / len(v)}
                    for clip, v in by_clip.items()}

    # paired bootstrap: per-sample and clip-clustered (center metric, total path)
    def paired(set_name, h, path, m_a=M2, m_b=M1, metric="D_wm_center"):
        va, ka = _sign_vec(metric, m_a, set_name, h, path)
        vb, kb = _sign_vec(metric, m_b, set_name, h, path)
        if ka != kb or len(va) == 0:
            return {"n": 0}
        clips = [next(s["clip"] for s in gated_rows("D_wm_center", m_a, set_name, h, path)
                      if (s["si"], s["ji"]) == k) for k in ka]
        d = va - vb
        n = len(d)
        boots = []
        for _ in range(args.bootstrap):
            idx = torch.randint(0, n, (n,), generator=gseed)
            boots.append(float(d[idx].mean()))
        bt = torch.tensor(boots)
        lo, hi = torch.quantile(bt, 0.025).item(), torch.quantile(bt, 0.975).item()
        # clip-clustered bootstrap: resample clips with replacement
        uniq_clips = sorted(set(clips))
        boots_c = []
        for _ in range(args.bootstrap):
            pick = [uniq_clips[int(torch.randint(0, len(uniq_clips), (1,), generator=gseed))]
                    for _ in uniq_clips]
            vals = torch.cat([d[[i for i, c in enumerate(clips) if c == c_pick]] for c_pick in pick])
            boots_c.append(float(vals.mean()))
        btc = torch.tensor(boots_c)
        return {"n": n, "mean_diff": float(d.mean()),
                "ci95": [lo, hi],
                "clip_cluster_ci95": [torch.quantile(btc, 0.025).item(),
                                       torch.quantile(btc, 0.975).item()],
                "frac_m2": float(va.mean()), "frac_m1": float(vb.mean()),
                "n_clips": len(uniq_clips)}

    def _sign_vec(metric, mkey, set_name, h, path):
        samples = gated_rows(metric, mkey, set_name, h, path)
        samples = sorted(samples, key=lambda s: (s["si"], s["ji"]))
        w = torch.tensor([s["wm"] for s in samples])
        sg = torch.tensor([s["sim"] for s in samples])
        return (torch.sign(w) == torch.sign(sg)).float(), [(s["si"], s["ji"]) for s in samples]

    for set_name in eval_sets:
        for h in horizons:
            for metric in ("D_wm_center", "D_wm_legacy", "D_wm_fd"):
                for path in ("total", "state", "direct", "critic"):
                    summary["paired"][f"{metric}|{set_name}|H{h}|{path}"] = \
                        paired(set_name, h, path, metric=metric)

    # leave-one-clip-out (center metric, total path, primary models)
    for set_name in eval_sets:
        for h in horizons:
            entry = {}
            for m in (M0KEY, M1, M2):
                samples = gated_rows("D_wm_center", m, set_name, h, "total")
                clips = sorted({s["clip"] for s in samples})
                entry[m] = {}
                for drop in clips:
                    sub = [s for s in samples if s["clip"] != drop]
                    if not sub:
                        entry[m][f"drop_{drop}"] = None
                        continue
                    w = torch.tensor([s["wm"] for s in sub])
                    sg = torch.tensor([s["sim"] for s in sub])
                    entry[m][f"drop_{drop}"] = {
                        "n": len(sub),
                        "sign_agree": float(((torch.sign(w) == torch.sign(sg)).float().mean()))}
            summary["leave_one_clip_out"][f"{set_name}|H{h}"] = entry

    # snapshot curve (center metric, total path, policy sets)
    for set_name in POLICY_SETS:
        for h in horizons:
            curve = {}
            for mkey in model_specs:
                st, _ = stats_for("D_wm_center", mkey, set_name, h, "total")
                curve[mkey] = {"sign_agreement": st["sign_agreement"],
                               "n_gated": st["n_gated"]}
            summary["snapshot_curve"][f"{set_name}|H{h}"] = curve

    with open(args.out_dir / "matched_eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---------------- path decomposition ------------------------
    max_resid = max(abs(r["resid_total_minus_sum"]) for r in decomp_resid) if decomp_resid else None
    # survival-gradient magnitude: legacy direct (iso) vs matched direct (attached, center)
    surv = {}
    for set_name in POLICY_SETS:
        for h in horizons:
            for m in (M0KEY, M1, M2):
                a_samples = gated_rows("D_wm_legacy", m, set_name, h, "direct")
                b_samples = gated_rows("D_wm_center", m, set_name, h, "direct")
                if a_samples and b_samples:
                    av = [s["wm"] for s in a_samples]
                    bv = [s["wm"] for s in b_samples]
                    surv[f"{set_name}|H{h}|{m}"] = {
                        "n": len(av),
                        "mean_abs_iso": sum(abs(x) for x in av) / len(av),
                        "mean_abs_attached_center": sum(abs(x) for x in bv) / len(bv)}
    with open(args.out_dir / "path_decomposition.json", "w") as fh:
        json.dump({"identity_max_abs_residual_center": max_resid,
                   "identity_note": "D_total_center - (D_state+D_direct+D_critic)_center, "
                                    "matched attached definitions; exact by construction, "
                                    "residual should be ~1e-6 relative",
                   "survival_iso_vs_attached": surv,
                   "per_sample_residuals": decomp_resid}, fh, indent=1)

    # ---------------- gradient consistency ------------------------
    cons = {}
    for p in PATHS:
        for h in horizons:
            lc, cf = [], []
            for c in consistency:
                if c["path"] != p or c["H"] != h:
                    continue
                lc.append(abs(c["legacy"] - c["center"]))
                cf.append(abs(c["fd"] - c["center"]))
            if lc:
                scale = max(max(abs(c["center"]) for c in consistency
                                if c["path"] == p and c["H"] == h), 1e-12)
                cons[f"{p}|H{h}"] = {
                    "n": len(lc),
                    "mean_abs_legacy_minus_center": sum(lc) / len(lc),
                    "mean_abs_fd_minus_center": sum(cf) / len(cf),
                    "rel_to_typical_scale": (sum(lc) / len(lc)) / scale}
    # legacy reproduction check vs original eval_rows.csv
    orig_map = {}
    for r in orig_rows:
        orig_map[(r["model"], r["set"], int(r["si"]), int(r["ji"]), int(r["H"]),
                  float(r["eps"]))] = r
    repo = []
    for r in rows:
        key = (r["model"], r["set"], r["si"], r["ji"], r["H"], float(r["eps"]))
        o = orig_map.get(key)
        if o is None:
            continue
        for p in PATHS:
            d_new, d_old = r[f"D_wm_legacy_{p}"], float(o[f"D_wm_{p}"])
            denom = max(abs(d_old), 1e-8)
            repo.append(abs(d_new - d_old) / denom)
    cons["legacy_reproduction_vs_original"] = {
        "n_compared": len(repo),
        "mean_rel_dev": sum(repo) / len(repo) if repo else None,
        "max_rel_dev": max(repo) if repo else None,
        "note": "original computed on cuda:0; this audit on cpu -> expected "
                "small float deviation; large deviations would indicate a "
                "structural change"}
    with open(args.out_dir / "gradient_consistency.json", "w") as fh:
        json.dump(cons, fh, indent=1)
    with open(args.out_dir / "per_sample_consistency.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "set", "si", "ji", "H", "eps",
                                           "path", "legacy", "center", "fd", "sim"])
        w.writeheader()
        for c in consistency:
            w.writerow(c)

    # ---------------- protection ------------------------
    integrity["protected_after"] = {rel: sha256(args.dirs_dir.parent / rel)
                                    for rel in integrity["protected_before"]}
    integrity["protected_unchanged"] = all(
        integrity["protected_before"][k] == integrity["protected_after"][k]
        for k in integrity["protected_before"])
    with open(args.out_dir / "data_integrity.json", "w") as fh:
        json.dump(integrity, fh, indent=1)

    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows),
                      "samples": n_samples, "identity_ok": identity_ok,
                      "protected_unchanged": integrity["protected_unchanged"],
                      "elapsed_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
