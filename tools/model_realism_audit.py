"""Audit whether a classifier can distinguish real replay transitions from WM transitions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--save_model", type=Path, default=None)
    ap.add_argument("--group_split", action="store_true",
                    help="split paired real/fake transitions by source transition, avoiding pair leakage")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "buffer" not in ck:
        raise RuntimeError("checkpoint must contain serialized replay buffer")
    data = ck["buffer"]["data"]
    ptr, size, cap = int(ck["buffer"]["ptr"]), int(ck["buffer"]["size"]), data["s"].shape[0]
    rows = (ptr - size + torch.arange(size)) % cap
    nenv = data["s"].shape[1]
    age = torch.arange(size).repeat_interleave(nenv)
    env = torch.arange(nenv).repeat(size)
    row_all = rows[age]
    s = data["s"][row_all, env].float()
    s1 = data["s1"][row_all, env].float()
    a = data["a"][row_all, env].float()
    xi = data["xi"][row_all, env].float()
    reward = data["r"][row_all, env].float() if "r" in data else None
    if len(s) > args.samples:
        idx = torch.randperm(len(s))[:args.samples]
        s, s1, a, xi = s[idx], s1[idx], a[idx], xi[idx]
        if reward is not None:
            reward = reward[idx]
    state = ck["wm"]
    key = "residual_net.0.weight"
    hidden_res = int(state[key].shape[0]) if key in state else 0
    wm = WorldModel(residual_hidden=hidden_res)
    wm.load_state_dict(state); wm.eval()
    # Use the repository's canonical state flattening to avoid quaternion layout mistakes.
    from supermimic.mdp import flatten_state
    with torch.no_grad():
        pred = wm.step(unflatten_state(s), a, normalize_xi(xi), use_residual=False)
        s1_fake = flatten_state(pred)
    real = torch.cat((s, a, s1), -1)
    fake = torch.cat((s, a, s1_fake), -1)
    x = torch.cat((real, fake)); y = torch.cat((torch.ones(len(real)), torch.zeros(len(fake))))
    n = len(x)
    if args.group_split:
        # Each real/fake pair shares exactly the same (s,a).  Splitting the
        # concatenated rows leaks the source transition across train/test and
        # makes the classifier look better than it transfers to new states.
        ng = len(real)
        groups = torch.randperm(ng)
        ntr = max(1, int(.6 * ng)); ncal = max(1, int(.2 * ng))
        gtr, gcal, gte = groups[:ntr], groups[ntr:ntr + ncal], groups[ntr + ncal:]
        tr = torch.cat((gtr, gtr + ng))
        cal = torch.cat((gcal, gcal + ng))
        te = torch.cat((gte, gte + ng))
    else:
        split = int(.7 * n)
        order = torch.randperm(n); tr, te = order[:split], order[split:]
        cal = torch.empty(0, dtype=torch.long)
    mean, std = x[tr].mean(0), x[tr].std(0).clamp_min(1e-4)
    x = (x - mean) / std
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], args.hidden), torch.nn.SiLU(),
                                torch.nn.Linear(args.hidden, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(args.epochs):
        for ids in tr[torch.randperm(len(tr))].split(args.batch):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[ids]).squeeze(-1), y[ids])
            opt.zero_grad(); loss.backward(); opt.step()
    temperature = 1.0
    if len(cal):
        with torch.no_grad():
            cal_logits = model(x[cal]).squeeze(-1)
        log_temp = torch.zeros((), requires_grad=True)
        cal_opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
        def closure():
            cal_opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cal_logits / log_temp.exp(), y[cal])
            loss.backward()
            return loss
        cal_opt.step(closure)
        temperature = float(log_temp.detach().exp().clamp(0.25, 4.0))
    with torch.no_grad():
        test_logits = model(x[te]).squeeze(-1) / temperature
        p = test_logits.sigmoid().numpy(); yy = y[te].numpy()
        ratio = test_logits.exp().clamp(max=100.0).numpy()
    real_ratio = ratio[yy > .5]
    ess = float(real_ratio.sum() ** 2 / ((real_ratio ** 2).sum() + 1e-12)) if len(real_ratio) else 0.0
    out = {"checkpoint": str(args.checkpoint), "samples_each": len(real),
           "auroc": float(roc_auc_score(yy, p)),
           "average_precision": float(average_precision_score(yy, p)),
           "brier": float(brier_score_loss(yy, p)),
           "group_split": bool(args.group_split), "temperature": temperature,
           "real_ratio_ess_fraction": ess / max(1, len(real_ratio)),
           "real_ratio_p50": float(torch.tensor(real_ratio).quantile(.5)) if len(real_ratio) else 0.0,
           "real_ratio_p95": float(torch.tensor(real_ratio).quantile(.95)) if len(real_ratio) else 0.0}
    # Held-out fake-transition diagnostics for the lower-bound audit. These
    # are correlations only: replay reward is a factual value proxy, while
    # the transition RMSE is a model-fidelity proxy. They never enter training.
    if args.group_split and reward is not None and len(te):
        te_t = te[torch.as_tensor(yy) < 0.5]
        src = (te_t - len(real)).long()
        fake_logits = test_logits[torch.as_tensor(yy) < 0.5].detach()
        fake_rmse = (s1_fake[src] - s1[src]).square().mean(-1).sqrt()
        rr = reward[src]
        def _corr(a0, b0):
            a0, b0 = a0.float(), b0.float()
            return float(((a0-a0.mean())*(b0-b0.mean())).mean()
                         / (a0.std().clamp_min(1e-8)*b0.std().clamp_min(1e-8)))
        out.update({"heldout_fake_logit_reward_corr": _corr(fake_logits, rr),
                    "heldout_fake_logit_rmse_corr": _corr(fake_logits, fake_rmse),
                    "heldout_fake_reward_mean": float(rr.mean()),
                    "heldout_fake_rmse_mean": float(fake_rmse.mean())})
    print(json.dumps(out, indent=2))
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "input_dim": x.shape[-1], "hidden": args.hidden,
                    "temperature": temperature, "group_split": bool(args.group_split),
                    "result": out}, args.save_model)


if __name__ == "__main__":
    main()
