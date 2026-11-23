"""CVF-1a Phase C: Critic fine-tuning under frozen supervision groups (GPU; APPROVED=true).

v2 (2026-11-23) -- corrected contract: the critic is evaluated at the prefix-end
state x_H (datasets/v2, schema sacv_pairs_v2), where the labels are defined and
where the policy update bootstraps.  v1 trained/evaluated at x_{H+T} and is INVALID
(docs/cvf/sacv/CVF1B_ERRATUM.md).  v1 refuses to load here (fail-closed).

Groups (protocol sec. 1), all from the SAME frozen C-init (B0@2100_post):
  C1  L_value = 1/2 E[(V+ - Y+)^2 + (V- - Y-)^2]           real-simulator scalar targets
  C2  L = L_value + alpha * L_pair / sigma^2(Delta Y_train)  + explicit pair term
  C0  L_value with Y replaced by Y_WM (WM closed-loop continuation from the SAME x_H,
      same frozen V_ref tail contract)
Frozen everywhere: WM, pi_old, Critic RunningNorm (never updated), architecture, lr,
optimizer config, sampling stream.  Matched steps / batch / sample exposure / init.

The pair scale is the registered definition (1 / sigma^2 of the TRUE Delta Y on the
Train split, frozen once).  The init measurement the protocol asks for -- value loss,
pair loss and their critic-parameter gradient norms -- is computed before training and
stored in every checkpoint (``init_scales``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_SACV), str(_CVF), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

sys.path.insert(0, str(_SACV))
from sacv.losses import value_losses as L  # noqa: E402
from sacv.evaluation import metrics as M  # noqa: E402

GAMMA = 0.99
PRIMARY_H = 96            # frozen by data-contract sec. 4.4 rule
CKPT_STEPS = (250, 500, 1000)
PAIRS_SCHEMA = "sacv_pairs_v2"
C_INIT_SHA256 = "65118115d917958430279afc5ffc37a8666c52104bcea3f10d4de1ae8bd73f54"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(path: Path, horizon: int) -> list[dict]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("schema") != PAIRS_SCHEMA:
        raise SystemExit(f"{path.name}: schema {d.get('schema')!r} != {PAIRS_SCHEMA!r} "
                         "(v1 datasets are INVALID: critic state was x_{H+T}, see ERRATUM)")
    return [p for p in d["pairs"] if p["horizon"] == horizon]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("C1", "C2", "C0"), required=True)
    ap.add_argument("--alpha", type=float, default=0.3, help="C2 pair weight (protocol sec. 1)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=2700)
    ap.add_argument("--horizon", type=int, default=PRIMARY_H)
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets/v2")
    ap.add_argument("--raw_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/raw")
    ap.add_argument("--out_dir", type=Path, default=_REPO / "results/cvf/sacv/checkpoints/v2")
    ap.add_argument("--gen_wm_targets_only", action="store_true")
    ap.add_argument("--pair_scale_mode", choices=("sigma2", "gradmatch"), default="gradmatch")
    ap.add_argument("--tag", default=None,
                    help="variant name stored in checkpoints and used in file names "
                         "(e.g. C2gm03); defaults to the group name")
    args = ap.parse_args()
    args.name = args.tag or args.group
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair2100)
    proto, wm, pi_old = world["proto"], pair2100["wm"], pair2100["pi_old"]
    v_ref_sd = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                          map_location="cpu", weights_only=False)["critic_post"]   # C-init
    v_ref = Critic(v_ref_sd["norm.mean"].shape[0])
    v_ref.load_state_dict(v_ref_sd, strict=True)
    v_ref.eval().to(dev)
    xi_one = torch.zeros(1, 33, device=dev); xi_one[:, 0] = 1.0
    xi_n = normalize_xi(xi_one)     # WM / critic_features convention (CVF-0 phase0)

    train = load_pairs(args.datasets / "train_pairs.pt", args.horizon)
    val = load_pairs(args.datasets / "val_pairs.pt", args.horizon)
    if not train or not val:
        print(json.dumps({"error": "empty dataset - run sacv_build_dataset.py first"}))
        return 1
    split_sha = sha256_file(args.datasets / "train_pairs.pt")

    raw_cache: dict[str, dict] = {}

    def branch_of(pref: dict) -> tuple[dict, int]:
        if pref["batch"] not in raw_cache:
            raw_cache[pref["batch"]] = torch.load(
                next(args.raw_root.rglob(pref["batch"])), map_location="cpu", weights_only=False)
        b = raw_cache[pref["batch"]]
        br = b["branches"][pref["branch"]] if pref["branch"] != "base0" else b["base_res"][0]
        return br, pref["slot"]

    def critic_row(pref: dict) -> torch.Tensor:
        """Critic input row (1, 938) at the prefix-end state x_H (fail-closed)."""
        if pref.get("snap_group") != "snaps_prefix" or pref.get("snap_key") != "prefix_32":
            raise SystemExit(f"pair state is not x_H: {pref}")
        br, slot = branch_of(pref)
        snap = br["snaps_prefix"][pref["snap_key"]]
        S = unflatten_state(snap["S_flat"][slot:slot + 1].to(dev))   # proto lives on dev
        f = snap["f"][slot:slot + 1].to(dev)
        return proto.critic_input(S, f, xi_n, proto.wm_goal(f, S))

    with torch.no_grad():
        X_tr_p = torch.cat([critic_row(p["state_plus"]) for p in train], 0)
        X_tr_m = torch.cat([critic_row(p["state_minus"]) for p in train], 0)
        X_va_p = torch.cat([critic_row(p["state_plus"]) for p in val], 0)
        X_va_m = torch.cat([critic_row(p["state_minus"]) for p in val], 0)
    f64 = lambda xs: torch.tensor(xs, device=dev, dtype=torch.float64)  # noqa: E731
    y_tr_p, y_tr_m = f64([p["Y_plus"] for p in train]), f64([p["Y_minus"] for p in train])
    dG_tr, dY_tr = f64([p["Delta_G"] for p in train]), f64([p["Delta_Y"] for p in train])
    dG_va, dY_va = f64([p["Delta_G"] for p in val]), f64([p["Delta_Y"] for p in val])
    y_va = torch.cat([f64([p["Y_plus"] for p in val]), f64([p["Y_minus"] for p in val])])

    # ---- init measurement (protocol sec. 1 / C4): loss magnitudes and gradient norms
    def init_scales() -> dict:
        c = Critic(v_ref_sd["norm.mean"].shape[0]).to(dev)
        c.load_state_dict(v_ref_sd, strict=True)
        v = c(torch.cat([X_tr_p, X_tr_m], 0))
        vp, vm = v[:len(train)], v[len(train):]
        l_val = L.value_loss(vp, vm, y_tr_p, y_tr_m)
        l_pair = L.pair_loss(vp, vm, y_tr_p, y_tr_m)
        params = list(c.parameters())
        g_v = torch.autograd.grad(l_val, params, retain_graph=True)
        g_p = torch.autograd.grad(l_pair, params)
        n = lambda g: float(torch.sqrt(sum((x.double() ** 2).sum() for x in g)))  # noqa: E731
        cos = float(sum((a.double() * b.double()).sum() for a, b in zip(g_v, g_p))
                    / max(n(g_v) * n(g_p), 1e-12))
        d = vp - vm   # predicted pair response Delta V = V(x_H+) - V(x_H-)
        return {"train_value_loss": float(l_val), "train_pair_loss": float(l_pair),
                "train_pair_over_value": float(l_pair / l_val),
                "grad_norm_value": n(g_v), "grad_norm_pair": n(g_p),
                "grad_norm_pair_over_value": n(g_p) / max(n(g_v), 1e-12),
                "grad_cos_value_pair": cos,
                "init_pair_vs_dG": M.direction_metrics(d, dG_tr),
                "init_pair_vs_dY": M.direction_metrics(d, dY_tr)}

    scales = init_scales()
    print(json.dumps({"init_scales": scales}), flush=True)

    # ---- pair-loss scale, frozen once from the Train init measurement above (ERRATUM sec. 5)
    #   sigma2   : 1 / sigma^2(Delta Y_train)   -- the v1 protocol's literal definition
    #   gradmatch: g_value^0 / g_pair^0         -- alpha = pair share of the value gradient norm
    sigma_dy = float(dY_tr.std().clamp_min(1e-8))
    if args.pair_scale_mode == "gradmatch":
        pair_scale = scales["grad_norm_value"] / max(scales["grad_norm_pair"], 1e-12)
    else:
        pair_scale = 1.0 / (sigma_dy ** 2)
    print(json.dumps({"sigma_dy": sigma_dy, "pair_scale": pair_scale,
                      "pair_scale_mode": args.pair_scale_mode, "alpha": args.alpha,
                      "alpha_eff": args.alpha * pair_scale}), flush=True)

    # ---------------- C0 targets (WM continuation from the SAME x_H) ----------------
    if args.group == "C0" or args.gen_wm_targets_only:
        wm_targets_path = args.datasets / f"train_wm_targets_h{args.horizon}.pt"
        if args.gen_wm_targets_only or not wm_targets_path.exists():
            gen = _gen_wm_targets(train, proto, wm, pi_old, v_ref, xi_one, dev, args.horizon,
                                  args.raw_root)
            torch.save({"schema": "sacv_wm_targets_v2", "horizon": args.horizon,
                        "start_state": "snaps_prefix/prefix_32 (x_H)", "targets": gen},
                       wm_targets_path)
            print(json.dumps({"wm_targets": len(gen), "out": str(wm_targets_path)}))
            if args.gen_wm_targets_only:
                return 0
        wm_t = torch.load(wm_targets_path, map_location="cpu", weights_only=False)["targets"]
        assert len(wm_t) == len(train), f"wm targets {len(wm_t)} != train pairs {len(train)}"
        y_tr_p = f64([rec["Y_wm_plus"] for rec in wm_t])
        y_tr_m = f64([rec["Y_wm_minus"] for rec in wm_t])

    # ---------------- training -------------------------------------------------------
    torch.manual_seed(args.seed)
    critic = Critic(v_ref_sd["norm.mean"].shape[0])
    critic.load_state_dict(v_ref_sd, strict=True)
    critic = critic.to(dev)
    for p in critic.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(args.seed)

    def val_probe() -> dict:
        critic.eval()
        with torch.no_grad():
            vp, vm = critic(X_va_p), critic(X_va_m)
        critic.train()
        res = {"val_value_rmse_vsY": float(torch.sqrt(0.5 * ((vp.double() - y_va[:len(val)]) ** 2
                                                             + (vm.double() - y_va[len(val):]) ** 2).mean()))}
        for tag, ref in (("dY", dY_va), ("dG", dG_va)):
            m = M.direction_metrics(vp - vm, ref)
            res[f"val_pair_{tag}"] = {"sign": m["sign"], "spearman": m["spearman"],
                                      "rmse": m["rmse"], "ratio": m["mag_ratio_median"]}
        return res

    log_rows: list[dict] = []
    t0 = time.time()
    cursor = 0
    order = torch.randperm(len(train), generator=g)
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order = torch.randperm(len(train), generator=g)
            cursor = 0
        idx = order[cursor: cursor + args.batch].to(dev)
        cursor += args.batch
        n = len(idx)
        v = critic(torch.cat([X_tr_p[idx], X_tr_m[idx]], 0))
        v_ps, v_ms = v[:n], v[n:]
        y_ps, y_ms = y_tr_p[idx], y_tr_m[idx]
        if args.group == "C0":
            loss = L.wm_scalar_loss(torch.cat([v_ps, v_ms], 0), torch.cat([y_ps, y_ms], 0))
            comp = {"value": loss.detach(), "pair": torch.zeros((), device=dev)}
        elif args.group == "C1":
            loss = L.value_loss(v_ps, v_ms, y_ps, y_ms)
            comp = {"value": loss.detach(), "pair": torch.zeros((), device=dev)}
        else:  # C2
            c = L.combined_loss(v_ps, v_ms, y_ps, y_ms, alpha=args.alpha * pair_scale)
            loss, comp = c["loss"], {k: c[k].detach() for k in ("value", "pair")}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in critic.parameters()
                                  if p.grad is not None)))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)
        opt.step()
        log_rows.append({"step": step, "loss": float(loss), "value": float(comp["value"]),
                         "pair": float(comp["pair"]), "grad_norm": gn})
        if step % 100 == 0 or step in CKPT_STEPS:
            vp = val_probe()
            log_rows[-1].update(vp)
            print(json.dumps({"step": step, "loss": float(loss), "grad_norm": gn, **vp,
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)
        if step in CKPT_STEPS:
            _save_ckpt(critic, opt, args, step, split_sha, sigma_dy, pair_scale, scales, log_rows,
                       val_probe(), args.out_dir)

    final = val_probe()
    out_meta = {"schema": "sacv_critic_meta_v2", "name": args.name, "group": args.group,
                "alpha": args.alpha, "pair_scale": pair_scale,
                "pair_scale_mode": args.pair_scale_mode,
                "alpha_eff": args.alpha * pair_scale, "sigma_dy": sigma_dy, "steps": args.steps,
                "contract": "critic at x_H; Y_T = G_T + gamma^T c_T V_ref(x_{H+T})",
                "init_scales": scales, "final_val": final, "train_sha256": split_sha}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}_h{args.horizon}_s{args.seed}_meta.json").write_text(
        json.dumps(out_meta, indent=2))
    print(json.dumps({"done": args.name, "final_val": final}))
    return 0


def _save_ckpt(critic, opt, args, step, split_sha, sigma_dy, pair_scale, scales, log_rows, val,
               out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}_h{args.horizon}_s{args.seed}_{step:04d}.pt"
    torch.save({
        "schema": "sacv_critic_ckpt_v2", "name": args.name, "group": args.group,
        "alpha": args.alpha, "pair_scale": pair_scale, "pair_scale_mode": args.pair_scale_mode,
        "alpha_eff": args.alpha * pair_scale,
        "steps": step, "horizon": args.horizon, "seed": args.seed, "lr": args.lr,
        "contract": "critic at x_H; Y_T = G_T + gamma^T c_T V_ref(x_{H+T})",
        "critic": critic.state_dict(), "optimizer": opt.state_dict(),
        "normalizer": "RunningNorm buffers frozen (C-init values; never updated)",
        "parent": {"bundle": "pair_it_002100.pt", "critic_post_sha256": C_INIT_SHA256},
        "train_split_sha256": split_sha, "sigma_dy": sigma_dy,
        "loss_defs": {"C1": "0.5*E[(V+ - Y+)^2 + (V- - Y-)^2]",
                      "C2": "C1 + alpha*pair_scale*E[((V+ - V-) - (Y+ - Y-))^2] (pair_scale_mode: sigma2 -> 1/sigma^2(DeltaY_train); gradmatch -> g_value^0/g_pair^0)",
                      "C0": "E[(V - Y_WM)^2] over both branches (Y_WM: WM continuation from x_H)"}[args.group],
        "init_scales": scales, "val_probe": val, "log_tail": log_rows[-50:],
    }, path)
    path.with_suffix(".sha256").write_text(sha256_file(path) + "\n")
    print(json.dumps({"ckpt": str(path)}), flush=True)


def _gen_wm_targets(train, proto, wm, pi_old, v_ref, xi_one, dev, horizon: int,
                    raw_root: Path) -> list[dict]:
    """Y_WM for C0: WM closed-loop continuation from the SAME real prefix-end states x_H
    (with their recorded o_hist / a_prev), same gamma^T c_T V_ref(sh_T^WM) tail contract."""
    from supermimic.mdp import normalize_xi, unflatten_state

    raw_cache: dict[str, dict] = {}
    xi_n = normalize_xi(xi_one)

    def batch_of(name: str) -> dict:
        if name not in raw_cache:
            raw_cache[name] = torch.load(next(raw_root.rglob(name)), map_location="cpu",
                                         weights_only=False)
        return raw_cache[name]

    out = []
    for p in train:
        for side in ("plus", "minus"):
            pref = p[f"state_{side}"]
            if pref.get("snap_group") != "snaps_prefix" or pref.get("snap_key") != "prefix_32":
                raise SystemExit(f"pair state is not x_H: {pref}")
            b = batch_of(pref["batch"])
            br = b["branches"][pref["branch"]] if pref["branch"] != "base0" else b["base_res"][0]
            snap = br["snaps_prefix"][pref["snap_key"]]
            slot = pref["slot"]
            S = tuple(x[slot:slot + 1].to(dev) for x in unflatten_state(snap["S_flat"]))
            f = snap["f"][slot:slot + 1].to(dev)
            ap = snap["a_prev"][slot:slot + 1].to(dev)
            oh = snap["o_hist"][slot:slot + 1].to(dev)
            c_soft = torch.ones(1, device=dev)
            G = torch.zeros(1, device=dev)
            with torch.no_grad():
                for k in range(horizon):
                    a = torch.clamp(pi_old(oh, proto.actor_goal(f, S)), -20.0, 20.0)
                    S = tuple(proto.wm_step(wm, S, a, xi_n))
                    f = f + 1
                    G = G + (GAMMA ** k) * c_soft * proto.mdp.reward(S, f, a, ap)
                    c_soft = c_soft * (1.0 - proto.mdp.soft_termination(S, f))
                    ap = a
                    oh = torch.cat((oh[:, 1:], proto.obs(S, a, xi_one, f).unsqueeze(1)), 1)
                v_tail = v_ref(proto.critic_input(S, f, xi_n, proto.wm_goal(f, S)))
            out.append(float(G + (GAMMA ** horizon) * c_soft * v_tail))
    assert len(out) == 2 * len(train)
    return [{"Y_wm_plus": out[2 * i], "Y_wm_minus": out[2 * i + 1]} for i in range(len(train))]


if __name__ == "__main__":
    sys.exit(main())
