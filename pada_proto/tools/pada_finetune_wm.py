"""Fine-tune M0 into M1 (random probes) / M2 (policy-shift probes).

Protocol §5: identical init (M0), identical window schedule (same seed, same
pool size), identical optimizer/loss; ONLY the probe data distribution
differs.  Loss mirrors train.py's normalized teacher-forced error in the proto
WM's native target space:

    z = [targets(S_j, Ŝ_{j+1}) - targets(S_j, S_{j+1})] / out_norm.std()

(targets is linear in S1 for the proto acceleration-level WM, so this equals
train.py's state_error/std form; normalizers are frozen.)
"""
from __future__ import annotations

import argparse
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

from supermimic.mdp import normalize_xi  # noqa: E402

K_WINDOW = 8


def build_windows(probes: dict, arm: str):
    """(n_windows, k+1, 71) states, (n_windows, k, 29) actions, (n_windows, k+1,) frames."""
    trajs = [v for k, v in probes["branches"].items() if k.startswith(arm + "_")]
    St, A, Fd = [], [], []
    for tr in trajs:
        s_all = torch.cat((tr["s0"].unsqueeze(0), tr["s1"]), 0)      # (H+1, 71)
        f_all = torch.cat((tr["f"][:1] - 1, tr["f"]), 0)             # (H+1,)
        Hh = s_all.shape[0] - 1
        for t in range(Hh - K_WINDOW + 1):
            St.append(s_all[t:t + K_WINDOW + 1])
            A.append(tr["a"][t:t + K_WINDOW])
            Fd.append(f_all[t:t + K_WINDOW + 1])
    return torch.stack(St), torch.stack(A), torch.stack(Fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--arms", default="random,policy")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--snapshot_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train_run", default="online_B0")
    ap.add_argument("--train_iter", type=int, default=2100)
    args = ap.parse_args()
    dev = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    probes = torch.load(args.probes, map_location="cpu", weights_only=False)
    xi_one = torch.zeros(1, 33)
    xi_one[0, 0] = 1.0
    xi_n = normalize_xi(xi_one).to(dev)

    arms = args.arms.split(",")
    data = {}
    for arm in arms:
        St, A, Fd = build_windows(probes, arm)
        data[arm] = (St, A, Fd)
        print(f"{arm}: windows={St.shape[0]}")
    sizes = {a: data[a][0].shape[0] for a in arms}
    assert len(set(sizes.values())) == 1, f"arm window pools differ: {sizes}"

    # paired index stream: same seed -> same window indices for every arm
    g = torch.Generator().manual_seed(args.seed)
    idx_stream = torch.randint(0, sizes[arms[0]], (args.steps, args.batch), generator=g)

    for arm in arms:
        t0 = time.time()
        pair = bridge.load_pair_bundle(args.train_run, args.train_iter, device=dev)
        wm = pair["wm"]                                   # fresh M0 copy per arm
        wm.to(dev)
        for p in wm.parameters():
            p.requires_grad_(True)
        # normalizers frozen: buffers, never updated here
        opt = torch.optim.Adam(wm.parameters(), lr=args.lr, betas=(0.9, 0.999))
        St, A, Fd = data[arm]
        St_d, A_d = St.to(dev), A.to(dev)
        n_snap = args.steps // args.snapshot_every
        log_rows = []
        for step in range(1, args.steps + 1):
            idx = idx_stream[step - 1].to(dev)
            s_win = St_d[idx]                              # (B, k+1, 71)
            a_win = A_d[idx]                               # (B, k, 29)
            B = s_win.shape[0]
            from supermimic.mdp import unflatten_state

            states = unflatten_state(s_win.reshape(B * (K_WINDOW + 1), 71))
            acts = a_win.reshape(B * K_WINDOW, 29)
            std = wm.out_norm.std().detach()
            loss_acc = 0.0
            for j in range(K_WINDOW):
                S_j = tuple(x[j::K_WINDOW + 1] for x in states)      # (B, ...)
                S_next_true = tuple(x[j + 1::K_WINDOW + 1] for x in states)
                pred = wm.step(S_j, acts[j::K_WINDOW], xi_n.expand(B, -1))
                inc_pred = wm.targets(S_j, pred)
                inc_true = wm.targets(S_j, S_next_true)
                z = (inc_pred - inc_true) / std
                loss_acc = loss_acc + z.square().mean()
            loss = loss_acc / K_WINDOW
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == 1:
                log_rows.append({"step": step, "loss": float(loss.detach())})
            if step % args.snapshot_every == 0 or step == args.steps:
                torch.save({"wm": {k: v.cpu() for k, v in wm.state_dict().items()},
                            "step": step, "arm": arm,
                            "init": f"{args.train_run}@{args.train_iter} pair wm",
                            "lr": args.lr, "batch": args.batch,
                            "seed": args.seed},
                           args.out_dir / f"M_{arm}_{step:05d}.pt")
        with open(args.out_dir / f"train_log_{arm}.json", "w") as fh:
            json.dump({"arm": arm, "steps": args.steps, "lr": args.lr,
                       "batch": args.batch, "rows": log_rows,
                       "elapsed_s": round(time.time() - t0, 1)}, fh, indent=1)
        print(f"{arm}: done in {round(time.time()-t0,1)} s, final loss "
              f"{log_rows[-1]['loss'] if log_rows else float('nan'):.5f}")
    print(json.dumps({"out_dir": str(args.out_dir), "sizes": sizes}))


if __name__ == "__main__":
    main()
