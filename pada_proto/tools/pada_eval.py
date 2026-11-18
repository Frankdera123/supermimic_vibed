"""PADA-0 evaluation: Level 1-4 metrics + pre-registered verdict inputs.

Runs entirely offline on the recorded simulator trajectories (pada_sim_collect
outputs) plus WM-side autograd rollouts.  Produces:
  results/pada/eval_rows.csv        per (model, set, start, dir, H, path) D values
  results/pada/eval_summary.json    gated fidelity stats, Level-1/2 tables,
                                    paired bootstrap CIs, verdict-check inputs
  results/pada/verdict_inputs.json  the exact PASS/INCONCLUSIVE/NO-GO checks

Metrics follow PADA0_EXPERIMENT_PROTOCOL §6; the gate is model-independent
(computed once from sim data only) so every model is scored on identical
gated samples.
"""
from __future__ import annotations

import argparse
import csv
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

from supermimic.crgf import (Protocol, fidelity_stats, open_loop_paths,  # noqa: E402
                             path_directional_derivatives, reliability_gate,
                             spearman)
from supermimic.mdp import flatten_state, normalize_xi, unflatten_state  # noqa: E402

POLICY_SETS = ("eval_pol_seen", "eval_pol_B", "eval_pol_C")
PRIMARY_PATH = "total"


def branch_J(tr, critic, proto, world, H, gamma):
    """Discounted soft-termination returns of a recorded branch (dict of tensors)."""
    dev = tr["s1"].device
    c_prev = torch.ones((), device=dev)
    J_state = torch.zeros((), device=dev)
    J_direct = torch.zeros((), device=dev)
    c_last = torch.ones((), device=dev)
    S_last_flat, f_last = None, None
    for t in range(H):
        r_s = tr["r_state"][t].clone().detach()
        r_d = tr["r_direct"][t].clone().detach()
        d = tr["soft_d"][t].clone().detach()
        c_prev = c_prev * (1 - d)
        J_state = J_state + (gamma ** t) * c_prev * r_s
        J_direct = J_direct + (gamma ** t) * c_prev * r_d
        c_last = c_prev
        S_last_flat, f_last = tr["s1"][t], int(tr["f"][t])
    J_critic = None
    if critic is not None:
        S_last = unflatten_state(S_last_flat.unsqueeze(0).to(world["device"]))
        xi_n = normalize_xi(torch.tensor([[1.0] + [0.0] * 32], device=world["device"]))
        gw = proto.wm_goal(torch.tensor([f_last], device=world["device"]), S_last)
        vin = proto.critic_input(S_last, torch.tensor([f_last], device=world["device"]),
                                 xi_n, gw)
        with torch.no_grad():
            J_critic = (gamma ** H) * c_last * critic(vin)[0].cpu()
    return {"state": J_state.cpu(), "direct": J_direct.cpu(), "critic": J_critic,
            "total": J_state + J_direct + (J_critic if J_critic is not None else 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs_dir", type=Path, required=True)
    ap.add_argument("--collect_dir", type=Path, required=True)
    ap.add_argument("--ckpt_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--snapshot_steps", default="500,1000,1500")
    ap.add_argument("--primary_step", type=int, default=1500)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--max_minutes", type=float, default=25.0)
    ap.add_argument("--train_run", default="online_B0")
    ap.add_argument("--train_iter", type=int, default=2100)
    ap.add_argument("--model_a", default=None, help="default M2_{primary_step}")
    ap.add_argument("--model_b", default=None, help="default M1_{primary_step}")
    args = ap.parse_args()
    dev = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()

    meta = json.load(open(args.dirs_dir / "directions_meta.json"))
    gamma = 0.99
    probes = torch.load(args.collect_dir / "probes.pt", map_location="cpu", weights_only=False)
    ev = torch.load(args.collect_dir / "eval_data.pt", map_location="cpu", weights_only=False)
    assert ev["meta"]["gamma"] == gamma
    H_full = ev["meta"]["H"]
    horizons = sorted({8, H_full})
    PRIMARY_H = H_full
    MODEL_A = args.model_a or f"M2_{args.primary_step}"
    MODEL_B = args.model_b or f"M1_{args.primary_step}"
    eps_list = ev["meta"]["eps_eval"]

    pair = bridge.load_pair_bundle(args.train_run, args.train_iter, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    critic = pair["critic"].to(dev)

    S_e = len(meta["eval_starts"])

    def eval_frame(si):
        return meta["eval_starts"][si]["frame"]

    # ---------------- sim-side ground truth (model-independent) ----------------
    print("computing sim-side FD ground truth ...")
    eval_sets = ("eval_random",) + POLICY_SETS
    set_dirs = {name: torch.load(args.dirs_dir / f"{name}.pt", map_location="cpu",
                                 weights_only=False)["dirs"]
                for name in eval_sets}
    k_per_set = {name: int(d.shape[1]) for name, d in set_dirs.items()}
    # base returns per start (2 reps) for the gate
    base_J = {}
    for si in range(S_e):
        for rep in range(ev["meta"]["base_reps"]):
            tr = ev["branches"][f"base_{si}_{rep}"]
            for h in tuple(horizons):
                base_J[(si, rep, h)] = branch_J(tr, critic, proto, world, h, gamma)
    sim_D = {}      # (set, si, ji) -> {h: {path: [D_eps0, D_eps1]}}, gate label
    gates = {}
    for set_name in eval_sets:
        for si in range(S_e):
            for ji in range(k_per_set[set_name]):
                per_eps = {e: {} for e in eps_list}
                for eps in eps_list:
                    for br in ("plus", "minus"):
                        tr = ev["branches"][f"{set_name}_{si}_{ji}_{eps}_{br}"]
                        per_eps[eps][br] = {h: branch_J(tr, critic, proto, world, h, gamma)
                                            for h in tuple(horizons)}
                sim_D[(set_name, si, ji)] = per_eps
                # gate: base repeat range + pooled D range (state path) + deadband
                ok_labels = {}
                for h in tuple(horizons):
                    base_vals = [base_J[(si, rep, h)]["total"] for rep in
                                 range(ev["meta"]["base_reps"])]
                    d_state = torch.tensor(
                        [float(per_eps[e]["plus"][h]["state"] - per_eps[e]["minus"][h]["state"])
                         / (2 * e) for e in eps_list])
                    ok, why = reliability_gate(torch.tensor(base_vals), d_state)
                    ok_labels[h] = (ok, why)
                gates[(set_name, si, ji)] = ok_labels

    # ---------------- models ----------------
    snap_steps = [int(s) for s in args.snapshot_steps.split(",")]
    # primary models first so a wall-clock fuse can only truncate the
    # curve-sensitivity snapshots, never the verdict-bearing runs
    models = {"M0": ("pair", None),
              f"M1_{args.primary_step}": ("random", args.primary_step),
              f"M2_{args.primary_step}": ("policy", args.primary_step)}
    for arm, tag in (("random", "M1"), ("policy", "M2")):
        for st in snap_steps:
            if st != args.primary_step:
                models[f"{tag}_{st}"] = (arm, st)

    def load_wm(key):
        kind, st = models[key]
        if kind == "pair":
            return pair["wm"]
        ck = torch.load(args.ckpt_dir / f"M_{kind}_{st:05d}.pt", map_location=dev,
                        weights_only=False)
        wm = bridge.load_pair_bundle(args.train_run, args.train_iter, device=dev)["wm"]
        wm.load_state_dict(ck["wm"])
        return wm.to(dev).eval()

    xi_n_dev = normalize_xi(torch.tensor([[1.0] + [0.0] * 32], device=dev))

    # ---------------- WM-side directional derivatives + state responses --------
    rows = []          # Level-3 per-sample rows
    lvl2 = []          # Level-2 per-sample aggregates
    for mkey in models:
        wm = load_wm(mkey)
        for set_name in eval_sets:
            dirs_all = set_dirs[set_name].to(dev)
            for si in range(S_e):
                for ji in range(k_per_set[set_name]):
                    if not gates[(set_name, si, ji)][PRIMARY_H][0] and \
                       not gates[(set_name, si, ji)][8][0]:
                        continue
                    base = ev["base_actions"][si].to(dev)
                    dirv = dirs_all[si, ji]
                    S0 = unflatten_state(ev["branches"][f"{set_name}_{si}_{ji}_{eps_list[0]}_plus"]["s0"]
                                         .unsqueeze(0).to(dev))
                    f0 = torch.tensor([eval_frame(si)], device=dev)
                    for h in tuple(horizons):
                        gate_ok = gates[(set_name, si, ji)][h][0]
                        if not gate_ok:
                            continue
                        for eps in eps_list:
                            a_plus = (base + eps * dirv)[:h].contiguous().unsqueeze(0)
                            a_minus = (base - eps * dirv)[:h].contiguous().unsqueeze(0)
                            outs = {}
                            for br, seq in (("plus", a_plus), ("minus", a_minus)):
                                a_seq = seq.clone().requires_grad_(True)
                                paths = open_loop_paths(wm, critic, mdp, S0, f0, a_seq,
                                                        torch.zeros(1, 29, device=dev),
                                                        xi_n_dev, gamma=gamma,
                                                        goal_frames=proto.K)
                                grads = path_directional_derivatives(paths, a_seq)
                                outs[br] = (grads, paths)
                            dv = dirv[:h].reshape(-1)
                            row = {"model": mkey, "set": set_name, "si": si, "ji": ji,
                                   "H": h, "eps": eps}
                            for p in ("state", "direct", "critic", "total"):
                                d_wm = float((outs["plus"][0][p][0] * dv).sum())
                                d_sim = float(
                                    (sim_D[(set_name, si, ji)][eps]["plus"][h][p]
                                     - sim_D[(set_name, si, ji)][eps]["minus"][h][p]) / (2 * eps))
                                row[f"D_wm_{p}"] = d_wm
                                row[f"D_sim_{p}"] = d_sim
                            rows.append(row)
                        if h != PRIMARY_H:
                            continue
                        # ---- Level 2: state FD at the larger eps (pre-registered) ----
                        eps = eps_list[-1]
                        with torch.no_grad():
                            state_resp = {}
                            for br, seq in (("plus", (base + eps * dirv).unsqueeze(0)),
                                            ("minus", (base - eps * dirv).unsqueeze(0))):
                                paths = open_loop_paths(wm, critic, mdp, S0, f0,
                                                        seq.clone(), torch.zeros(1, 29, device=dev),
                                                        xi_n_dev, gamma=gamma,
                                                        goal_frames=proto.K,
                                                        record_states=True)
                                flat = [flatten_state(tuple(x[0] for x in S))
                                        for S in paths["states"]]
                                state_resp[br] = torch.stack(flat, 0).cpu()  # (H, 71)
                            tr_p = ev["branches"][f"{set_name}_{si}_{ji}_{eps}_plus"]["s1"]
                            tr_m = ev["branches"][f"{set_name}_{si}_{ji}_{eps}_minus"]["s1"]
                            d_sim_s = tr_p - tr_m                       # (H, 71)
                            d_wm_s = state_resp["plus"] - state_resp["minus"]
                            for (tagH, sl, sl_wm) in (("8", slice(0, 8), slice(0, 8)),
                                                      ("32", slice(0, H_full), slice(0, H_full))):
                                ds, dw = d_sim_s[sl], d_wm_s[sl_wm]
                                per_step_cos = torch.stack(
                                    [torch.nn.functional.cosine_similarity(dw[t], ds[t], dim=0)
                                     for t in range(ds.shape[0])])
                                traj_cos = torch.nn.functional.cosine_similarity(
                                    dw.reshape(-1), ds.reshape(-1), dim=0)
                                mag_ratio = (dw.norm(dim=-1) / ds.norm(dim=-1).clamp_min(1e-8))
                                rel_err = ((dw - ds).norm(dim=-1) / ds.norm(dim=-1).clamp_min(1e-8))
                                dec = (dw.abs() + ds.abs()) > 0
                                sign_agree = float((((torch.sign(dw) == torch.sign(ds)) & dec)
                                                    .float().sum()) / dec.sum().clamp_min(1))
                                lvl2.append({"model": mkey, "set": set_name, "si": si,
                                             "ji": ji, "H": tagH,
                                             "mean_step_cos": float(per_step_cos.mean()),
                                             "traj_cos": float(traj_cos),
                                             "median_mag_ratio": float(mag_ratio.median()),
                                             "mean_rel_err": float(rel_err.mean()),
                                             "sign_agree_71": sign_agree,
                                             "sign_agree_height": float(
                                                 (torch.sign(dw[:, 2]) == torch.sign(ds[:, 2])).float().mean()),
                                             "rmse": float((dw - ds).square().mean().sqrt())})
        del wm
        if (time.time() - t0) > args.max_minutes * 60:
            print(f"WARNING: time budget reached at model {mkey}; partial results")
            break

    # ---------------- Level 1 ----------------
    print("Level-1 prediction metrics ...")
    lvl1 = {}
    _ng = torch.no_grad()
    _ng.__enter__()
    for mkey in models:
        wm = load_wm(mkey)
        std = wm.out_norm.std().detach()
        rows1 = {"eval_base": [], "probe_random": [], "probe_policy": []}
        # teacher-forced one-step on eval base transitions
        for si in range(S_e):
            for rep in range(ev["meta"]["base_reps"]):
                tr = ev["branches"][f"base_{si}_{rep}"]
                S_all = torch.cat((tr["s0"].unsqueeze(0), tr["s1"]), 0).to(dev)
                Ss = unflatten_state(S_all)
                A = tr["a"].to(dev)
                Hh = S_all.shape[0] - 1
                for j in range(Hh):
                    S_j = tuple(x[j:j + 1] for x in Ss)
                    pred = wm.step(S_j, A[j:j + 1], xi_n_dev)
                    z = (wm.targets(S_j, pred) - wm.targets(S_j, tuple(x[j + 1:j + 2] for x in Ss))) / std
                    rows1["eval_base"].append(float(z.square().mean().sqrt()))
        # free-running H=8/32 along base actions from s0
        free = {"8": [], "32": []}
        for si in range(S_e):
            rep = 0
            tr = ev["branches"][f"base_{si}_{rep}"]
            S = unflatten_state(tr["s0"].unsqueeze(0).to(dev))
            A = tr["a"].to(dev)
            S_true = torch.cat((tr["s0"].unsqueeze(0), tr["s1"]), 0).to(dev)
            Ss_true = unflatten_state(S_true)
            for t in range(H_full):
                S = wm.step(S, A[t:t + 1], xi_n_dev)
                z = (wm.targets(unflatten_state(S_true[t:t + 1]), S)
                     - wm.targets(unflatten_state(S_true[t:t + 1]),
                                  unflatten_state(S_true[t + 1:t + 2]))) / std
                err = float(z.square().mean().sqrt())
                if t < 8:
                    free["8"].append(err)
                free["32"].append(err)
        # probe transitions (descriptive)
        for arm in ("random", "policy"):
            acc = []
            for k, tr in probes["branches"].items():
                if not k.startswith(arm + "_"):
                    continue
                S_all = torch.cat((tr["s0"].unsqueeze(0), tr["s1"]), 0).to(dev)
                Ss = unflatten_state(S_all)
                A = tr["a"].to(dev)
                Hh = S_all.shape[0] - 1
                for j in range(0, Hh, 4):          # subsample for speed
                    S_j = tuple(x[j:j + 1] for x in Ss)
                    pred = wm.step(S_j, A[j:j + 1], xi_n_dev)
                    z = (wm.targets(S_j, pred) - wm.targets(S_j, tuple(x[j + 1:j + 2] for x in Ss))) / std
                    acc.append(float(z.square().mean().sqrt()))
            rows1[f"probe_{arm}"] = acc
        import math as _m

        lvl1[mkey] = {"one_step_eval_base_rmse": sum(rows1["eval_base"]) / len(rows1["eval_base"]),
                      "one_step_probe_random_rmse": (sum(rows1["probe_random"]) / len(rows1["probe_random"])),
                      "one_step_probe_policy_rmse": (sum(rows1["probe_policy"]) / len(rows1["probe_policy"])),
                      "free_H8_rmse": sum(free["8"]) / len(free["8"]),
                      "free_H32_rmse": sum(free["32"]) / len(free["32"])}
        del wm
    _ng.__exit__(None, None, None)

    # ---------------- aggregate + paired bootstrap ----------------
    print("aggregating ...")
    df_rows = {}
    for r in rows:
        key = (r["model"], r["set"], r["si"], r["ji"], r["H"], r["eps"])
        df_rows.setdefault(key, {}).update(r)
    # pooled per (model, set, H): mean D across eps then fidelity stats
    summary = {"fidelity": {}, "level1": lvl1, "level2": {}, "verdict_checks": {}}
    for (mkey, set_name, si, ji, h) in {(r["model"], r["set"], r["si"], r["ji"], r["H"])
                                        for r in rows}:
        eps_rows = [r for r in df_rows.values()
                    if (r["model"], r["set"], r["si"], r["ji"], r["H"]) == (mkey, set_name, si, ji, h)]
        merged = {"model": mkey, "set": set_name, "si": si, "ji": ji, "H": h}
        for p in ("state", "direct", "critic", "total"):
            d_wm = sum(r[f"D_wm_{p}"] for r in eps_rows) / len(eps_rows)
            d_sim = sum(r[f"D_sim_{p}"] for r in eps_rows) / len(eps_rows)
            merged[f"D_wm_{p}"] = d_wm
            merged[f"D_sim_{p}"] = d_sim
        summary.setdefault("_merged", []).append(merged)
    merged_all = summary["_merged"]
    del summary["_merged"]

    def stat_group(mkey, set_name, h, path):
        sel = [m for m in merged_all if m["model"] == mkey and m["set"] == set_name
               and m["H"] == h]
        if not sel:
            return {"n_gated": 0}
        w = torch.tensor([m[f"D_wm_{path}"] for m in sel])
        s = torch.tensor([m[f"D_sim_{path}"] for m in sel])
        st = fidelity_stats(w, s, gate=None)
        st["sign_binary"] = [float(torch.sign(w[i]) == torch.sign(s[i])) for i in range(len(w))]
        st["median_ratio"] = float((w.abs() / s.abs().clamp_min(1e-8)).median())
        st["spearman"] = spearman(w, s)
        return st

    for mkey in models:
        for set_name in eval_sets:
            for h in tuple(horizons):
                for path in ("state", "direct", "critic", "total"):
                    summary["fidelity"][f"{mkey}|{set_name}|H{h}|{path}"] = \
                        stat_group(mkey, set_name, h, path)
    for r in lvl2:
        summary["level2"].setdefault(f"{r['model']}|{r['set']}|H{r['H']}", []).append(r)
    summary["level2"] = {k: {kk: sum(d[kk] for d in v) / len(v)
                             for kk in ("mean_step_cos", "traj_cos", "median_mag_ratio",
                                        "mean_rel_err", "sign_agree_71", "sign_agree_height",
                                        "rmse")} | {"n": len(v)}
                         for k, v in summary["level2"].items()}

    # paired bootstrap: primary metric sign binary per sample, M2_1500 vs M1_1500
    def sign_vec(mkey, set_name, h, path):
        sel = sorted([m for m in merged_all if m["model"] == mkey and m["set"] == set_name
                      and m["H"] == h], key=lambda m: (m["si"], m["ji"]))
        w = torch.tensor([m[f"D_wm_{path}"] for m in sel])
        sg = torch.tensor([m[f"D_sim_{path}"] for m in sel])
        return (torch.sign(w) == torch.sign(sg)).float(), [(m["si"], m["ji"]) for m in sel]

    gseed = torch.Generator().manual_seed(2600)

    def paired_ci(set_name, h, path, m_a=None, m_b=None):
        m_a = m_a or MODEL_A
        m_b = m_b or MODEL_B
        va, ka = sign_vec(m_a, set_name, h, path)
        vb, kb = sign_vec(m_b, set_name, h, path)
        assert ka == kb
        if len(va) == 0:
            return {"n": 0}
        d = va - vb
        boots = []
        n = len(d)
        for _ in range(args.bootstrap):
            idx = torch.randint(0, n, (n,), generator=gseed)
            boots.append(float(d[idx].mean()))
        boots = torch.tensor(boots)
        lo, hi = torch.quantile(boots, 0.025).item(), torch.quantile(boots, 0.975).item()
        # two-sided sign test on non-zero paired diffs
        nz = d[d != 0]
        from math import comb

        k = int((nz > 0).sum())
        n_nz = int(nz.numel())
        p_two = min(1.0, sum(comb(n_nz, i) for i in range(0, min(k, n_nz - k) + 1))
                    / 2 ** n_nz * 2) if n_nz > 0 else 1.0
        return {"n": n, "mean_diff": float(d.mean()), "ci95": [lo, hi],
                "sign_test_p": p_two, "positive_frac_m2": float(va.mean()),
                "positive_frac_m1": float(vb.mean())}

    for set_name in eval_sets:
        for h in tuple(horizons):
            summary["verdict_checks"][f"{set_name}|H{h}|total"] = \
                paired_ci(set_name, h, "total")
            summary["verdict_checks"][f"{set_name}|H{h}|state"] = \
                paired_ci(set_name, h, "state")

    # ---------------- persist ----------------
    fields = ["model", "set", "si", "ji", "H", "eps"]
    for p in ("state", "direct", "critic", "total"):
        fields += [f"D_wm_{p}", f"D_sim_{p}"]
    with open(args.out_dir / "eval_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    with open(args.out_dir / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(args.out_dir / "level1.json", "w") as fh:
        json.dump(lvl1, fh, indent=1)
    print(json.dumps({"out": str(args.out_dir), "rows": len(rows),
                      "gated_total": len({(m['set'], m['si'], m['ji'], m['H'])
                                          for m in merged_all}),
                      "elapsed_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
