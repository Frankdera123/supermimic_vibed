"""CVF-1a Phase A: fail-closed verification of frozen mother assets (CPU-only).

Verifies, without loading CUDA:
  1. file-level SHA256 of online_B0 checkpoints / diag pair bundles vs the
     manifest recorded at save time (diag_pair_manifest.json);
  2. component-level state-dict SHA256 (pi_old / pi_new / wm / critic_pre /
     critic_post / policy normalizers) recomputed with the same deterministic
     algorithm as opcl_diag_ckpt.sha256_state_dict;
  3. ckpt_002100.pt internal structure and whether its WM state dict matches
     the pair-bundle WM (establishes the exact "WM online_B0@2100" identity);
  4. phase0 start/direction asset structure;
  5. CVF-0 oracle raw data hashes, cross-checked against the da01 inventory.

Writes results/cvf/sacv/manifests/mother_asset_manifest.json (additive; never
overwrites existing CVF-0 assets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"), str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

SA = Path("/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_bytes(t: torch.Tensor) -> bytes:
    tc = t.detach().cpu().contiguous().reshape(-1)
    return tc.view(torch.uint8).numpy().tobytes()


def sha256_state_dict(sd: dict[str, Any]) -> str:
    """Deterministic hash over sorted keys, shapes, dtypes and raw bytes
    (byte-compatible with supermimic.opcl_diag_ckpt.sha256_state_dict)."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(str(k).encode("utf-8"))
        h.update(b"\x00")
        if isinstance(v, torch.Tensor):
            h.update(f"tensor:{tuple(v.shape)}:{v.dtype}".encode("utf-8"))
            h.update(b"\x00")
            h.update(_tensor_bytes(v))
        elif isinstance(v, dict):
            h.update(b"dict\x00")
            h.update(sha256_state_dict(v).encode("utf-8"))
        else:
            h.update(repr(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def check(name: str, ok: bool, detail: Any = None) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/manifests/mother_asset_manifest.json")
    args = ap.parse_args()
    t0 = time.time()
    checks: list[dict] = []
    manifest: dict[str, Any] = {
        "schema": "sacv_mother_asset_manifest_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "CVF-1a frozen mother (WM online_B0@2100 + pi_old@2100 + critic B0@2100_post)",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available_at_verify": bool(torch.cuda.is_available()),
        "server_artifacts": str(SA),
        "files": {},
        "bundle": {},
        "ckpt_002100": {},
        "phase0": {},
        "oracle_data": {},
        "checks": checks,
    }

    def req_file(path: Path) -> bool:
        if not path.exists():
            checks.append(check(f"exists:{path.name}", False, str(path)))
            return False
        checks.append(check(f"exists:{path.name}", True, path.stat().st_size))
        return True

    # ---- 1. diag_pair_manifest.json -------------------------------------
    dpm_path = SA / "online_B0/diag_pairs/diag_pair_manifest.json"
    if not req_file(dpm_path):
        manifest["verdict"] = "BLOCKED"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2))
        return 1
    dpm = json.loads(dpm_path.read_text())
    pair2100 = next(p for p in dpm["pairs"] if p["iteration"] == 2100)
    manifest["diag_pair_manifest"] = {
        "path": str(dpm_path),
        "num_real_pairs": dpm.get("num_real_pairs"),
        "recorded_file_sha256_it2100": pair2100["file_sha256"],
        "component_sha256_it2100": pair2100["component_sha256"],
        "operation_order_it2100": pair2100["operation_order"],
    }

    # ---- 2. file-level hashes vs recorded --------------------------------
    files_to_hash = {
        "ckpt_002100": SA / "online_B0/ckpt_002100.pt",
        "ckpt_002300": SA / "online_B0/ckpt_002300.pt",
        "pair_it_002100": SA / "online_B0/diag_pairs/pair_it_002100.pt",
        "pair_it_002300": SA / "online_B0/diag_pairs/pair_it_002300.pt",
        "last": SA / "online_B0/last.pt",
    }
    recorded = {
        "pair_it_002100": pair2100["file_sha256"],
        "pair_it_002300": next(p for p in dpm["pairs"] if p["iteration"] == 2300)["file_sha256"],
    }
    for name, path in files_to_hash.items():
        if not path.exists():
            checks.append(check(f"file_hash:{name}", False, "missing"))
            continue
        got = sha256_file(path)
        manifest["files"][name] = {"path": str(path), "size": path.stat().st_size, "sha256": got}
        if name in recorded:
            checks.append(check(f"file_hash:{name}", got == recorded[name],
                                {"got": got, "recorded": recorded[name]}))

    # ---- 3. pair bundle component hashes ---------------------------------
    bundle = torch.load(files_to_hash["pair_it_002100"], map_location="cpu", weights_only=False)
    bmeta: dict[str, Any] = {
        "keys": sorted(bundle.keys()),
        "same_policy_update_verified": bool(bundle.get("same_policy_update_verified")),
        "pair_validation_reason": bundle.get("pair_validation_reason"),
        "operation_order": bundle.get("operation_order"),
        "pi_updates": [bundle.get("pi_updates_before"), bundle.get("pi_updates_after")],
        "pi_param_l2_diff": float(bundle.get("pi_param_l2_diff", float("nan"))),
        "args_core": {k: bundle["args"].get(k) for k in (
            "horizon", "gamma", "lam", "hist", "seed", "lr", "num_envs", "buffer_steps",
            "motion_dir", "critic_epochs", "critic_minibatches", "no_critic", "amp", "resume",
            "wm_window", "wm_batches", "wm_batch", "pi_batch", "opcl_enabled", "action_noise")
            if k in bundle.get("args", {})},
    }
    comp = {}
    for comp_name in ("pi_old", "pi_new", "wm", "critic_pre", "critic_post"):
        sd = bundle[comp_name] if comp_name != "critic_pre" and comp_name != "critic_post" \
            else bundle["critic_pre" if comp_name == "critic_pre" else "critic_post"]
        got = sha256_state_dict(sd)
        comp[comp_name] = got
        recorded_c = pair2100["component_sha256"].get(comp_name)
        checks.append(check(f"component:{comp_name}", got == recorded_c,
                            {"got": got, "recorded": recorded_c}))
    # policy normalizers: extract from pi_old state dict buffers
    norm_sd = {k: v for k, v in bundle["pi_old"].items() if "norm" in k}
    got_norm = sha256_state_dict(norm_sd)
    checks.append(check("component:policy_normalizer",
                        got_norm == pair2100["component_sha256"]["policy_normalizer_pre"],
                        {"got": got_norm, "recorded": pair2100["component_sha256"]["policy_normalizer_pre"]}))
    critic_sd = bundle.get("critic_post") or bundle.get("critic")
    if critic_sd is None and "critic" in bundle and isinstance(bundle["critic"], dict) \
            and "net.0.weight" not in bundle["critic"]:
        critic_sd = bundle["critic"].get("critic_post")
    bmeta["critic_in_dim"] = int(critic_sd["norm.mean"].shape[0]) if critic_sd and "norm.mean" in critic_sd else None
    bmeta["critic_hidden"] = [tuple(critic_sd[k].shape) for k in critic_sd if k.startswith("net.")]
    bmeta["recomputed_component_sha256"] = comp
    manifest["bundle"] = bmeta

    # ---- 4. ckpt_002100 internal structure -------------------------------
    ckpt = torch.load(files_to_hash["ckpt_002100"], map_location="cpu", weights_only=False)
    cmeta: dict[str, Any] = {"keys": sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}
    manifest["ckpt_002100"] = cmeta
    wm_hash_bundle = comp["wm"]
    if isinstance(ckpt, dict):
        for cand in ("wm", "wm_sd", "model_wm"):
            if cand in ckpt and isinstance(ckpt[cand], dict):
                got = sha256_state_dict(ckpt[cand])
                cmeta["wm_state_key"] = cand
                cmeta["wm_state_sha256"] = got
                checks.append(check("ckpt_wm_matches_bundle_wm", got == wm_hash_bundle,
                                    {"got": got, "bundle": wm_hash_bundle}))
                break
        else:
            # search one level deep for the WM state dict
            for k, v in ckpt.items():
                if isinstance(v, dict) and "wm" in v and isinstance(v["wm"], dict):
                    got = sha256_state_dict(v["wm"])
                    cmeta["wm_state_key"] = f"{k}.wm"
                    cmeta["wm_state_sha256"] = got
                    checks.append(check("ckpt_wm_matches_bundle_wm", got == wm_hash_bundle,
                                        {"got": got, "bundle": wm_hash_bundle}))
                    break

    # ---- 5. phase0 start/direction asset ---------------------------------
    p0_path = _REPO / "results/cvf/phase0/phase0_starts_and_dirs.pt"
    if req_file(p0_path):
        p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
        o1d = p0.get("oracle1_dirs", {})
        o2d = p0.get("oracle2_dirs", {})
        manifest["phase0"] = {
            "path": str(p0_path), "sha256": sha256_file(p0_path),
            "keys": sorted(p0.keys()),
            "n_starts": len(p0["starts"]),
            "start0_keys": sorted(p0["starts"][0].keys()) if p0["starts"] else None,
            "f0_shape": list(p0["f0"].shape) if torch.is_tensor(p0.get("f0")) else None,
            "oracle1_dirs": {k: list(v.shape) for k, v in o1d.items()},
            "oracle2_dirs": {k: list(v.shape) for k, v in o2d.items()},
            "oracle2_base_A_shape": list(p0["oracle2_base_A"].shape) if torch.is_tensor(p0.get("oracle2_base_A")) else None,
        }

    # ---- 6. oracle raw data + da01 cross-check ---------------------------
    da01_path = _REPO / "results/cvf/deep_analysis/da01_inventory_contract.json"
    da01 = json.loads(da01_path.read_text()) if da01_path.exists() else None
    inv = {}
    if da01:
        raw = json.dumps(da01)
        # da01 stores entries possibly nested; build name -> sha256 lookup
        def walk(obj):
            if isinstance(obj, dict):
                if "sha256" in obj and isinstance(obj.get("path") or obj.get("file"), str):
                    p = obj.get("path") or obj.get("file")
                    inv[Path(p).name] = obj["sha256"]
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(da01)
    for name in ("oracle1_sim.pt", "oracle2_sim.pt", "oracle1_batch_0.pt", "oracle2_batch_0.pt"):
        path = _REPO / "results/cvf/phase1b" / name
        if not path.exists():
            checks.append(check(f"oracle:{name}", False, "missing"))
            continue
        got = sha256_file(path)
        manifest["oracle_data"][name] = {"path": str(path), "size": path.stat().st_size, "sha256": got}
        if name in inv:
            checks.append(check(f"oracle_da01_match:{name}", got == inv[name],
                                {"got": got, "da01": inv[name]}))
    manifest["da01_inventory_used"] = bool(da01)

    ok_all = all(c["ok"] for c in checks)
    manifest["verdict"] = "VERIFIED" if ok_all else "BLOCKED"
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    n_fail = sum(not c["ok"] for c in checks)
    print(json.dumps({"verdict": manifest["verdict"], "checks": len(checks), "failed": n_fail,
                      "out": str(args.out), "elapsed_s": manifest["elapsed_s"]}))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
