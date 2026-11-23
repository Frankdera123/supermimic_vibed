"""CVF-1a Phase C: Critic fine-tuning under frozen supervision groups (GPU; APPROVED-gated).

Groups (protocol §1): C1 = sim scalar, C2 = sim scalar + alpha*pair, C0 = WM scalar.
All groups start from the SAME frozen C-init (B0@2100_post), share the frozen
RunningNorm (eval mode, never updated), lr=2e-4 Adam (original config), identical
batch schedule and RNG stream, and differ ONLY in the supervision signal.

C0 targets: WM closed-loop continuation from the SAME real branch-end states with
the same V_ref tail contract (precomputed via --gen_wm_targets into
train_wm_targets.pt).  WM and Policy are never trained here (single-module optimizer).

Checkpoints carry parent hash / split hash / config / seed / steps / loss defs /
validation probes, per protocol §6.
"""
from __future__ import annotations

import argparse
import copy
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
from sacv.losses import targets as vt  # noqa: E402
from sacv.evaluation import metrics as M  # noqa: E402

GAMMA = 0.99
PRIMARY_H = 96            # frozen by data-contract §4.4 rule (confirm on Val before training)
CKPT_STEPS = (250, 500, 1000, 2000)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(path: Path, horizon: int) -> list[dict]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return [p for p in d["pairs"] if p["horizon"] == horizon]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=("C1", "C2", "C0"), required=True)
    ap.add_argument("--alpha", type=float, default=0.3, help="C2 pair weight (protocol §1)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=2700)
    ap.add_argument("--horizon", type=int, default=PRIMARY_H)
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--out_dir", type=Path, default=_REPO / "results/cvf/sacv/checkpoints")
    ap.add_argument("--pair_sigma", type=float, default=None,
                    help="frozen sigma(Delta Y) normalizer; if unset, computed from Train once")
    ap.add_argument("--gen_wm_targets_only", action="store_true")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair2100)
    proto, wm, pi_old = world["proto"], pair2100["wm"], pair2100["pi_old"]
    from supermimic.networks import Critic

    raw2100 = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                         map_location="cpu", weights_only=False)
    v_ref_sd = raw2100["critic_post"]           # C-init (bridge's "critic" == pre)
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
    train_path = args.datasets / "train_pairs.pt"
    split_sha = sha256_file(train_path)

    # raw-batch cache for extended-state fetching during training
    snaps: dict[str, dict] = {}

    def eval_vref(S, f, a_prev):
        with torch.no_grad():
            gw = proto.wm_goal(f, S)
            return v_ref(proto.critic_input(S, f, xi_n, gw))

    # ---------------- C0 target generation --------------------------------
    if args.group == "C0" or args.gen_wm_targets_only:
        wm_targets_path = args.datasets / f"train_wm_targets_h{args.horizon}.pt"
        if args.gen_wm_targets_only or not wm_targets_path.exists():
            gen = _gen_wm_targets(train, proto, wm, pi_old, v_ref, xi_one, dev, args.horizon)
            torch.save({"schema": "sacv_wm_targets_v1", "horizon": args.horizon,
                        "targets": gen}, wm_targets_path)
            print(json.dumps({"wm_targets": len(gen), "out": str(wm_targets_path)}))
            if args.gen_wm_targets_only:
                return 0
    wm_t = None
    if args.group == "C0":
        wm_t = torch.load(wm_targets_path, map_location="cpu", weights_only=False)["targets"]
        assert len(wm_t) == len(train), f"wm targets {len(wm_t)} != train pairs {len(train)}"
        for p, rec in zip(train, wm_t):   # merge by pair order (generator preserved order)
            p["Y_wm_plus"] = rec["Y_wm_plus"]
            p["Y_wm_minus"] = rec["Y_wm_minus"]

    # ---------------- training --------------------------------------------
    torch.manual_seed(args.seed)
    critic = Critic(v_ref_sd["norm.mean"].shape[0])
    critic.load_state_dict(v_ref_sd, strict=True)
    critic = critic.to(dev)
    critic.train()                      # modules in train mode; RunningNorm NEVER updated
    for p in critic.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr)  # original optimizer config
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train), generator=g)

    # frozen pair-scale normalizer (protocol §1): computed once from Train, never updated
    sigma_dy = args.pair_sigma or float(torch.tensor([p["Delta_Y"] for p in train]).double().std().clamp_min(1e-8))
    pair_scale = 1.0 / (sigma_dy ** 2)

    def state_tensors(pkey: str, pref: dict):
        """Fetch one branch-end extended state as tensors on dev."""
        if pref["batch"] not in snaps:
            snaps[pref["batch"]] = torch.load(_batch_path(pref["batch"]), map_location="cpu",
                                              weights_only=False)
        batch = snaps[pref["batch"]]
        br = batch["branches"].get(pref["branch"]) if pref["branch"] != "base0" \
            else batch["base_res"][0]
        snap = br["snaps_cont"][pref["snap_key"]]
        slot = pref["slot"]
        S = unflatten_state(snap["S_flat"][slot:slot + 1].to(dev))
        f_t = snap["f"][slot:slot + 1].to(dev)
        ap_t = snap["a_prev"][slot:slot + 1].to(dev)
        return S, f_t, ap_t

    def _batch_path(name: str) -> Path:
        return next((args.datasets / "raw").rglob(name))

    def v_of(pref: dict):
        S, f_t, ap_t = state_tensors("x", pref)
        gw = proto.wm_goal(f_t, S)
        vin = proto.critic_input(S, f_t, xi_n, gw)
        return critic(vin), vin

    def val_probe() -> dict:
        critic.eval()
        errs, signs, dys, dvs = [], [], [], []
        with torch.no_grad():
            for p in val[:256]:
                vp, _ = v_of(p["snap_plus"])
                vm, _ = v_of(p["snap_minus"])
                errs.append((float(vp) - p["Y_plus"], float(vm) - p["Y_minus"]))
                dys.append(p["Y_plus"] - p["Y_minus"])
                dvs.append(float(vp) - float(vm))
        critic.train()
        e = torch.tensor(errs)
        rmse = float(torch.sqrt((e ** 2).mean()))
        dys_t, dvs_t = torch.tensor(dys), torch.tensor(dvs)
        return {"val_value_rmse": rmse, "val_pair_sign": M.sign_agreement(dvs_t, dys_t),
                "val_pair_spearman": M.spearman(dvs_t, dys_t), "n_val": len(errs)}

    log_rows = []
    t0 = time.time()
    cursor = 0
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            order = torch.randperm(len(train), generator=g)
            cursor = 0
        idx = order[cursor: cursor + args.batch]
        cursor += args.batch
        samples = [train[i] for i in idx.tolist()]
        v_ps, v_ms, y_ps, y_ms = [], [], [], []
        rows = []
        for smp in samples:
            for side in ("snap_plus", "snap_minus"):
                S_b, f_b, ap_b = state_tensors("x", smp[side])
                gw_b = proto.wm_goal(f_b, S_b)
                rows.append(proto.critic_input(S_b, f_b, xi_n, gw_b))
            if args.group == "C0":
                y_ps.append(smp["Y_wm_plus"]); y_ms.append(smp["Y_wm_minus"])
            else:
                y_ps.append(smp["Y_plus"]); y_ms.append(smp["Y_minus"])
        X = torch.cat(rows, 0)                      # (2B, 938) single forward
        v = critic(X)
        v_ps, v_ms = v[0::2], v[1::2]
        y_ps = torch.tensor(y_ps, device=dev, dtype=torch.float64)
        y_ms = torch.tensor(y_ms, device=dev, dtype=torch.float64)
        if args.group == "C0":
            loss = ((v_ps.double() - y_ps) ** 2).mean() + ((v_ms.double() - y_ms) ** 2).mean()
            comp = {"value": loss.detach(), "pair": torch.zeros((), device=dev)}
        elif args.group == "C1":
            lv = L.value_loss(v_ps, v_ms, y_ps, y_ms)
            loss, comp = lv, {"value": lv.detach(), "pair": torch.zeros((), device=dev)}
        else:  # C2
            c = L.combined_loss(v_ps, v_ms, y_ps, y_ms,
                                alpha=args.alpha * pair_scale if args.alpha else 0.0)
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
            _save_ckpt(critic, opt, args, step, split_sha, sigma_dy, log_rows,
                       val_probe(), pair2100, args.out_dir)

    final = val_probe()
    out_meta = {"group": args.group, "alpha": args.alpha, "pair_scale": pair_scale,
                "sigma_dy": sigma_dy, "steps": args.steps, "final_val": final,
                "train_sha256": split_sha}
    (args.out_dir / f"{args.group}_h{args.horizon}_meta.json").write_text(json.dumps(out_meta, indent=2))
    print(json.dumps({"done": args.group, **out_meta}))
    return 0


def _save_ckpt(critic, opt, args, step, split_sha, sigma_dy, log_rows, val, pair2100,
               out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.group}_h{args.horizon}_s{step:04d}.pt"
    torch.save({
        "schema": "sacv_critic_ckpt_v1", "group": args.group, "alpha": args.alpha,
        "steps": step, "horizon": args.horizon, "seed": args.seed, "lr": args.lr,
        "critic": critic.state_dict(), "optimizer": opt.state_dict(),
        "parent": {"bundle": "pair_it_002100.pt",
                   "critic_post_sha256": "65118115d917958430279afc5ffc37a8666c52104bcea3f10d4de1ae8bd73f54"},
        "train_split_sha256": split_sha, "sigma_dy": sigma_dy,
        "loss_defs": {"C1": "0.5*E[(V+ - Y+)^2 + (V- - Y-)^2]",
                      "C2": "C1 + alpha*pair_scale*E[((V+ - V-) - (Y+ - Y-))^2]",
                      "C0": "E[(V - Y_wm)^2]"}[args.group],
        "val_probe": val, "log_tail": log_rows[-50:],
    }, path)
    path.with_suffix(".sha256").write_text(sha256_file(path))
    print(json.dumps({"ckpt": str(path)}), flush=True)


def _gen_wm_targets(train, proto, wm, pi_old, v_ref, xi_one, dev, horizon: int) -> list[dict]:
    """Y_WM for C0: WM closed-loop continuation from the SAME real branch-end states,
    with the SAME gamma^T c_T V_ref(ŝ_T^WM) tail contract (data-contract §4.3)."""
    from supermimic.mdp import normalize_xi, unflatten_state

    raw_cache: dict[str, dict] = {}
    xi_n = normalize_xi(xi_one)

    def batch_of(name: str) -> dict:
        if name not in raw_cache:
            path = next((_REPO / "results/cvf/sacv/datasets/raw").rglob(name))
            raw_cache[name] = torch.load(path, map_location="cpu", weights_only=False)
        return raw_cache[name]

    xi_one_b = xi_one
    out = []
    for p in train:
        for side in ("plus", "minus"):
            pref = p[f"snap_{side}"]
            batch = batch_of(pref["batch"])
            br = batch["branches"].get(pref["branch"]) if pref["branch"] != "base0" \
                else batch["base_res"][0]
            snap = br["snaps_cont"][pref["snap_key"]]
            slot = pref["slot"]
            S = tuple(x[slot:slot + 1].to(dev) for x in unflatten_state(snap["S_flat"]))
            f0 = snap["f"][slot:slot + 1].to(dev)
            a_prev = snap["a_prev"][slot:slot + 1].to(dev)
            oh = snap["o_hist"][slot:slot + 1].to(dev)
            o_cur = proto.obs(S, a_prev, xi_one_b, f0)
            o_hist = torch.cat((oh[:, :-1], o_cur.unsqueeze(1)), 1)
            c_soft = torch.ones(1, device=dev)
            G = torch.zeros(1, device=dev)
            S_c = S
            f_c = f0.clone()
            ap = a_prev
            oh_c = o_hist
            with torch.no_grad():
                for k in range(horizon):
                    ag = proto.actor_goal(f_c, S_c)
                    a = torch.clamp(pi_old(oh_c, ag), -20.0, 20.0)
                    S_c = tuple(proto.wm_step(wm, S_c, a, xi_n))
                    f_c = f_c + 1
                    r = proto.mdp.reward(S_c, f_c, a, ap)
                    ds = proto.mdp.soft_termination(S_c, f_c)
                    G = G + (GAMMA ** k) * c_soft * r
                    c_soft = c_soft * (1.0 - ds)
                    ap = a
                    oh_c = torch.cat((oh_c[:, 1:], proto.obs(S_c, a, xi_one_b, f_c).unsqueeze(1)), 1)
                gw_end = proto.wm_goal(f_c, S_c)
                v_tail = v_ref(proto.critic_input(S_c, f_c, xi_n, gw_end))
            y_wm = float(G + (GAMMA ** horizon) * c_soft * v_tail)
            out.append({"pair_idx": len(out) // 2, "side": side, "Y_wm": y_wm})
    # regroup per pair
    regrouped = [{} for _ in range(len(out) // 2)]
    for rec in out:
        regrouped[rec["pair_idx"]]["Y_wm_" + rec["side"]] = rec["Y_wm"]
    for i, p in enumerate(train):
        p["Y_wm_plus"] = regrouped[i]["Y_wm_plus"]
        p["Y_wm_minus"] = regrouped[i]["Y_wm_minus"]
    return regrouped


if __name__ == "__main__":
    sys.exit(main())
