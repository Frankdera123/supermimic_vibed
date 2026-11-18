"""Summarize height/tilt termination labels stored in a SuperMimic checkpoint."""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    buf = ckpt.get("buffer")
    if not buf or "data" not in buf or "term_height" not in buf["data"]:
        raise SystemExit("checkpoint has no cause labels; collect with the instrumented trainer first")
    data = buf["data"]
    T = data["term_height"].shape[0]
    size = min(int(buf.get("size", T)), T)
    ptr = int(buf.get("ptr", size))
    rows = (ptr - size + torch.arange(size)) % T
    h = data["term_height"][rows].bool()
    t = data["term_tilt"][rows].bool()
    done = data["done"][rows].bool()
    denom = max(int(done.sum()), 1)
    print(f"buffer transitions={int(done.numel())} terminal={int(done.sum())}")
    print(f"height_terminal={int(h.sum())} ({float(h.sum())/denom:.4f} per terminal)")
    print(f"tilt_terminal={int(t.sum())} ({float(t.sum())/denom:.4f} per terminal)")
    print(f"both_terminal={int((h & t).sum())} ({float((h & t).sum())/denom:.4f} per terminal)")


if __name__ == "__main__":
    main()
