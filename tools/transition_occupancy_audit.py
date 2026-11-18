"""Audit whether replay transitions contain a learnable current-policy footprint.

This is a diagnostic approximation to the first TOM step.  It labels the newest
buffer slice as a proxy for current-policy data and an old slice as historical
data, trains a transition classifier, and evaluates it on held-out buffer rows.
It does not change WM training or claim to implement TOM's dual objective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def load_features(path: Path, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    buf = ck.get("buffer")
    if not buf:
        raise RuntimeError(f"checkpoint {path} does not contain a serialized replay buffer")
    data, ptr, size, capacity = buf["data"], int(buf["ptr"]), int(buf["size"]), buf["data"]["s"].shape[0]
    rows = (ptr - size + torch.arange(size, device=device)) % capacity
    # One row/environment transition is a sample.  Keep the state/action/next
    # state tuple so the classifier cannot reduce to age alone.
    feats = torch.cat((
        data["s"][rows].float(), data["a"][rows].float(),
        data["xi"][rows].float(), data["s1"][rows].float(),
        data["f"][rows].float().unsqueeze(-1), data["ep_step"][rows].float().unsqueeze(-1),
    ), -1).reshape(-1, data["s"].shape[-1] * 2 + data["a"].shape[-1] + data["xi"].shape[-1] + 2)
    return ck, feats, size, data["s"].shape[1], data["f"][rows].reshape(-1), data["ep_step"][rows].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--positive_fraction", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--matched", action="store_true",
                    help="match negative rows by frame/episode-step buckets and hide absolute frame features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--save_model", type=Path, default=None,
                    help="save classifier state and feature normalization for WM weighting")
    args = ap.parse_args()
    if not 0.0 < args.positive_fraction < 0.5:
        ap.error("--positive_fraction must be in (0, .5)")
    torch.manual_seed(args.seed)
    ck, x, size, nenv, frame, ep_step = load_features(args.checkpoint)
    # Rows are age ordered; hold out the middle half to test transfer across
    # buffer time rather than reporting training separability.
    nrow = size * nenv
    x = x.reshape(size, nenv, -1)
    m = max(1, int(size * args.positive_fraction))
    pos = x[-m:].reshape(-1, x.shape[-1])
    neg = x[:m].reshape(-1, x.shape[-1])
    if args.matched:
        # Match old rows to newest rows using coarse motion/episode coverage.
        # The classifier cannot use the absolute frame or episode-step fields.
        pos_keys = torch.stack((frame[-m * nenv:] % 32, ep_step[-m * nenv:] // 8), -1)
        neg_keys = torch.stack((frame[:m * nenv] % 32, ep_step[:m * nenv] // 8), -1)
        chosen = []
        for key in pos_keys:
            d = (neg_keys - key).abs().sum(-1)
            chosen.append(int(d.argmin()))
        neg = neg[torch.tensor(chosen)]
        pos = pos[:, :-2]
        neg = neg[:, :-2]
    train_n = int(0.7 * min(len(pos), len(neg)))
    pos, neg = pos[:2 * train_n], neg[:2 * train_n]
    train = torch.cat((pos[:train_n], neg[:train_n]))
    y = torch.cat((torch.ones(train_n), torch.zeros(train_n)))
    test = torch.cat((pos[train_n:], neg[train_n:]))
    yt = torch.cat((torch.ones(len(pos) - train_n), torch.zeros(len(neg) - train_n)))
    mean, std = train.mean(0), train.std(0).clamp_min(1e-4)
    train, test = (train - mean) / std, (test - mean) / std
    model = torch.nn.Sequential(torch.nn.Linear(train.shape[-1], args.hidden), torch.nn.SiLU(),
                                torch.nn.Linear(args.hidden, 1)).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(args.epochs):
        order = torch.randperm(len(train))
        for idx in order.split(args.batch):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train[idx]).squeeze(-1), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        test_logits = model(test).squeeze(-1)
    # Calibrate the classifier on the held-out split.  This is deliberately a
    # single scalar temperature: it changes odds sharpness without fitting a
    # second feature-dependent model.
    log_temp = torch.zeros((), requires_grad=True)
    cal_opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        cal_opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            test_logits / log_temp.exp(), yt)
        loss.backward()
        return loss
    cal_opt.step(closure)
    temperature = float(log_temp.detach().exp().clamp(0.25, 4.0))
    p = (test_logits / temperature).sigmoid().numpy()
    yy = yt.numpy()
    result = {
        "checkpoint": str(args.checkpoint), "iteration": int(ck.get("iteration", -1)),
        "buffer_rows": size, "envs": nenv, "positive_rows": len(pos), "negative_rows": len(neg),
        "auroc": float(roc_auc_score(yy, p)), "average_precision": float(average_precision_score(yy, p)),
        "brier": float(brier_score_loss(yy, p)), "temperature": temperature,
        "positive_fraction": args.positive_fraction,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "input_dim": train.shape[-1], "hidden": args.hidden,
                    "temperature": temperature, "matched": args.matched,
                    "result": result}, args.save_model)


if __name__ == "__main__":
    main()
