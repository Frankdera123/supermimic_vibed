"""Compare teacher-forced and free-running WM errors on the same replay windows."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel

def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)

def predict_free(wm, w, k):
    state = unflatten_state(w["s"][:, 0])
    out = []
    for j in range(k):
        state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]))
        out.append(state)
    return tuple(torch.stack([x[i] for x in out], 1) for i in range(6))

def predict_teacher(wm, w, k):
    out = []
    for j in range(k):
        out.append(wm.step(unflatten_state(w["s"][:, j]), w["a"][:, j], normalize_xi(w["xi"][:, j])))
    return tuple(torch.stack([x[i] for x in out], 1) for i in range(6))

def score(wm, pred, w):
    std = torch.sqrt(wm.out_norm.var + wm.out_norm.eps**2).clamp_min(1e-4)
    target = unflatten_state(w["s1"])
    e = (wm.state_error(pred, target) / std).square().mean(-1)
    mask = w["mask"]
    first = (e[:, 0] * mask[:, 0]).sum() / mask[:, 0].sum().clamp_min(1.0)
    final = (e[:, -1] * mask[:, -1]).sum() / mask[:, -1].sum().clamp_min(1.0)
    all_steps = (e * mask).sum() / mask.sum().clamp_min(1.0)
    counts = torch.stack((mask[:, 0].sum(), mask[:, -1].sum(), mask.sum()))
    return torch.cat((torch.stack((first, final, all_steps)), counts))

def evaluate(wm, buffer, env_ids, k, batch, batches):
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            w = buffer.sample_windows(batch, k, env_ids=env_ids)
            vals.append(torch.stack((score(wm, predict_teacher(wm, w, k), w),
                                     score(wm, predict_free(wm, w, k), w))))
    return torch.stack(vals).mean(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--buffer_checkpoint", type=Path, default=None,
                    help="optional checkpoint whose replay buffer supplies the audit windows")
    ap.add_argument("--audit_fraction", type=float, default=.25)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    buffer_ck = (torch.load(args.buffer_checkpoint, map_location=args.device, weights_only=False)
                 if args.buffer_checkpoint else ck)
    state = buffer_ck.get("buffer")
    if state is None:
        ap.error("checkpoint must contain replay buffer state")
    T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    n_audit = max(1, int(round(N * args.audit_fraction))); n_audit = min(n_audit, N - 1)
    ids = torch.arange(N - n_audit, N, device=args.device)
    wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval()
    val = evaluate(wm, buf, ids, args.window, args.batch, args.batches)
    print(json.dumps({"checkpoint": str(args.checkpoint), "buffer_checkpoint": str(args.buffer_checkpoint or args.checkpoint), "window": args.window,
                      "audit_envs": [int(ids[0]), int(ids[-1])],
                      "teacher_forced": {"one_step": float(val[0,0]), "final_step": float(val[0,1]), "all_steps": float(val[0,2])},
                      "free_running": {"one_step": float(val[1,0]), "final_step": float(val[1,1]), "all_steps": float(val[1,2])},
                      "valid_counts": {"one_step": float(val[0,3]), "final_step": float(val[0,4]), "all_steps": float(val[0,5])}}, indent=2))
if __name__ == "__main__": main()
