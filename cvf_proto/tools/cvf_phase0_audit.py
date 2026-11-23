"""CVF-0 Phase 0 offline CPU-only audit:
1. Select 16 balanced starts across 8 distinct LAFAN1 motion categories (2 frames/clip).
2. Pre-screen and save Oracle 1 (state-tangent) & Oracle 2 (action-sequence) direction tensors.
3. Audit in-model error (A) across 7+ opcl_proto Critic checkpoints vs exact WM continuation gradients.

Zero GPU usage, read-only checkpoints, no training or loss edits.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CVF_ROOT = _HERE.parent
_REPO = _CVF_ROOT.parent
sys.path.insert(0, str(_CVF_ROOT))
sys.path.insert(0, str(_REPO / "pada_proto"))

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402
from cvf_core import (  # noqa: E402
    POSE_IDX,
    VEL_IDX,
    clip_clustered_bootstrap,
    critic_tangent_grad,
    loco_summary,
    perturb_state,
    rollout_wm_continuation,
    screen_safe_direction,
)
from supermimic.crgf import (  # noqa: E402
    normalize_dir_seqs,
    open_loop_paths,
    path_directional_derivatives,
    random_dir_seqs,
    spearman,
)
from supermimic.mdp import normalize_xi  # noqa: E402
from supermimic.networks import Critic  # noqa: E402


def select_balanced_8clip_starts(motion, frames_per_clip: int = 2, seed: int = 2600,
                                margin: int = 250) -> list[dict]:
    """Select 8 clips covering all 7 LAFAN1 motion families + 1 distinct subject clip."""
    names = motion.names
    families = {}
    for idx, n in enumerate(names):
        fam = n.split("1")[0].split("2")[0].split("3")[0].split("_")[0]
        families.setdefault(fam, []).append(idx)
    g = torch.Generator().manual_seed(seed)
    chosen = []
    for fam in sorted(families):
        pool = families[fam]
        pick = pool[int(torch.randint(0, len(pool), (1,), generator=g))]
        chosen.append(pick)
    # 8th clip: pick from largest family avoiding duplicates
    extra_pool = [i for i in families.get("walk", list(range(len(names)))) if i not in chosen]
    chosen.append(extra_pool[int(torch.randint(0, len(extra_pool), (1,), generator=g))])

    starts = []
    for c in chosen:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - margin - 1)
        fs = s0 + 1 + torch.randperm(hi - s0 - 1, generator=g)[:frames_per_clip].sort().values
        for f in fs.tolist():
            starts.append({"clip_id": c, "clip_name": names[c], "frame": int(f)})
    return starts


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)


def _dir_metrics(d_crit: torch.Tensor, d_ref: torch.Tensor, clips: list[str]) -> dict:
    dec = (d_crit.abs() + d_ref.abs()) > 1e-8
    sign_vec = ((torch.sign(d_crit) == torch.sign(d_ref)) & dec).float()
    bs_sign = clip_clustered_bootstrap(sign_vec, clips)
    loco = loco_summary(sign_vec, clips)
    rat = d_crit.abs() / d_ref.abs().clamp_min(1e-6)
    return {
        "n": int(d_crit.numel()),
        "sign_agree": bs_sign["mean"],
        "sign_ci95": bs_sign["ci95"],
        "loco_sign_range": [loco["min_mean"], loco["max_mean"]],
        "spearman": spearman(d_crit, d_ref),
        "med_ratio": float(rat.median()),
        "iqr_ratio": [float(torch.quantile(rat, 0.25)), float(torch.quantile(rat, 0.75))],
        "med_abs_crit": float(d_crit.abs().median()),
        "med_abs_ref": float(d_ref.abs().median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("results/cvf/phase0"))
    ap.add_argument("--seed", type=int, default=2600)
    ap.add_argument("--eps_max", type=float, default=0.05)
    ap.add_argument("--h_pre", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pair_2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    pair_2300 = bridge.load_pair_bundle("online_B0", 2300, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair_2100)
    motion, mdp, proto, kin = world["motion"], world["mdp"], world["proto"], world["kin"]
    wm = pair_2100["wm"]
    pi_old = pair_2100["pi_old"]
    pi_new = pair_2100["pi_new"]

    starts = select_balanced_8clip_starts(motion, frames_per_clip=2, seed=args.seed, margin=250)
    B = len(starts)
    clips = [s["clip_name"] for s in starts]
    f0 = torch.tensor([s["frame"] for s in starts], dtype=torch.long)
    S0 = motion.ref_state(f0)
    xi = torch.zeros(B, 33); xi[:, 0] = 1.0
    xi_n = normalize_xi(xi)

    # ---- 1. Roll WM H_pre=16 steps to get on-policy state S* and history for Oracle 1 ----
    with torch.no_grad():
        Sc = tuple(x.clone() for x in S0)
        ap_cur = torch.zeros(B, 29)
        o0 = proto.obs(S0, ap_cur, xi, f0)
        oh = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
        fc = f0.clone()
        for _ in range(args.h_pre):
            ag = proto.actor_goal(fc, Sc)
            a = torch.clamp(pi_old(oh, ag), -20.0, 20.0)
            Sc = wm.step(Sc, a, xi_n)
            fc = fc + 1
            ap_cur = a
            o1 = proto.obs(Sc, a, xi, fc)
            oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), 1)
    S_star = tuple(x.detach() for x in Sc)
    f_star = fc.clone()
    a_prev_star = ap_cur.detach()
    oh_prev9_star = oh[:, :-1].detach()

    # ---- 2. Generate Oracle 1 & Oracle 2 direction tensors ----
    c_grad_star = critic_tangent_grad(pair_2100["critic"], mdp, motion, proto, S_star, f_star, xi_n)
    gV_star = c_grad_star["grad_raw"]

    # Reachable terminal displacement dS_32 from S_star along delta_2100
    with torch.no_grad():
        # Base closed-loop rollout from S_star for H=32
        Sr = tuple(x.clone() for x in S_star)
        ohr = oh.clone()
        fr = f_star.clone()
        acts_old, acts_new = [], []
        for _ in range(32):
            ag = proto.actor_goal(fr, Sr)
            ao = torch.clamp(pi_old(ohr, ag), -20.0, 20.0)
            an = torch.clamp(pi_new(ohr, ag), -20.0, 20.0)
            acts_old.append(ao); acts_new.append(an)
            Sr = wm.step(Sr, ao, xi_n)
            fr = fr + 1
            ohr = torch.cat((ohr[:, 1:], proto.obs(Sr, ao, xi, fr).unsqueeze(1)), 1)
        A_base32 = torch.stack(acts_old, 1)
        dA_2100 = normalize_dir_seqs(torch.stack(acts_new, 1) - A_base32)
        eps_r = 0.05
        def _roll_open(A_seq):
            So = tuple(x.clone() for x in S_star)
            for t in range(A_seq.shape[1]):
                So = wm.step(So, torch.clamp(A_seq[:, t], -20.0, 20.0), xi_n)
            return So
        Sp_r = _roll_open(A_base32 + eps_r * dA_2100)
        Sm_r = _roll_open(A_base32 - eps_r * dA_2100)
        from supermimic.rotations import so3_log, yaw_mat
        def _local_diff(Sa, Sb):
            p, R, v, w, q, qd = Sb
            pa, Ra, va, wa, qa, qda = Sa
            Hy, Rt = yaw_mat(R), R.transpose(-1, -2)
            return torch.cat((
                (Hy.transpose(-1, -2) @ (pa - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Ra),
                (Rt @ (va - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wa - w).unsqueeze(-1)).squeeze(-1),
                qa - q, qda - qd,
            ), -1)
        dS_reach = _local_diff(Sp_r, Sm_r) / (2 * eps_r)

    g_rng = torch.Generator().manual_seed(args.seed + 101)
    u_vel_crit, _ = screen_safe_direction(kin, S_star, gV_star, args.eps_max, "vel")
    u_vel_reach, _ = screen_safe_direction(kin, S_star, dS_reach, args.eps_max, "vel")
    u_vel_rand0, _ = screen_safe_direction(kin, S_star, torch.randn(B, 70, generator=g_rng), args.eps_max, "vel")
    u_vel_rand1, _ = screen_safe_direction(kin, S_star, torch.randn(B, 70, generator=g_rng), args.eps_max, "vel")
    u_pose_crit, pose_meta_c = screen_safe_direction(kin, S_star, gV_star, args.eps_max, "pose_safe")
    u_pose_rand, pose_meta_r = screen_safe_direction(kin, S_star, torch.randn(B, 70, generator=g_rng), args.eps_max, "pose_safe")

    oracle1_dirs = {
        "vel_critic": u_vel_crit,
        "vel_reachable": u_vel_reach,
        "vel_random_0": u_vel_rand0,
        "vel_random_1": u_vel_rand1,
        "pose_safe_critic": u_pose_crit,
        "pose_safe_random": u_pose_rand,
    }

    # Oracle 2 action directions (B, 32, 29)
    with torch.no_grad():
        Sr2 = tuple(x.clone() for x in S0)
        oh2 = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
        fr2 = f0.clone()
        a0_old, a23_old, a23_new = [], [], []
        pi23_old, pi23_new = pair_2300["pi_old"], pair_2300["pi_new"]
        for _ in range(32):
            ag = proto.actor_goal(fr2, Sr2)
            ao = torch.clamp(pi_old(oh2, ag), -20.0, 20.0)
            a0_old.append(ao)
            a23_old.append(torch.clamp(pi23_old(oh2, ag), -20.0, 20.0))
            a23_new.append(torch.clamp(pi23_new(oh2, ag), -20.0, 20.0))
            Sr2 = wm.step(Sr2, ao, xi_n)
            fr2 = fr2 + 1
            oh2 = torch.cat((oh2[:, 1:], proto.obs(Sr2, ao, xi, fr2).unsqueeze(1)), 1)
        A0_32 = torch.stack(a0_old, 1)
        dA_2300 = normalize_dir_seqs(torch.stack(a23_new, 1) - torch.stack(a23_old, 1))
    A_req = A0_32.clone().requires_grad_(True)
    paths = open_loop_paths(wm, pair_2100["critic"], mdp, S0, f0, A_req, torch.zeros(B, 29), xi_n,
                            gamma=0.99, goal_frames=proto.K)
    dA_ascent = normalize_dir_seqs(path_directional_derivatives(paths, A_req)["total"].reshape(B, 32, 29).detach())
    dA_rand = random_dir_seqs(B, 32, 29, seed=args.seed + 202, device="cpu")
    oracle2_dirs = {
        "delta_2100": dA_2100,
        "delta_2300": dA_2300,
        "ascent": dA_ascent,
        "random": dA_rand,
    }

    torch.save({
        "starts": starts,
        "f0": f0,
        "f_star": f_star,
        "S_star": S_star,
        "a_prev_star": a_prev_star,
        "oh_prev9_star": oh_prev9_star,
        "oracle1_dirs": oracle1_dirs,
        "pose_meta_critic": pose_meta_c,
        "pose_meta_random": pose_meta_r,
        "oracle2_base_A": A0_32,
        "oracle2_dirs": oracle2_dirs,
    }, args.out_dir / "phase0_starts_and_dirs.pt")

    # ---- 3. WM exact continuation value & gradients at S_star (ground truth for Model-Internal Error A) ----
    print("Computing exact WM continuation gradients at S_star (T=32,64,128) and scalar T=200...")
    wm_grad_res = rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star, f_star, oh_prev9_star, a_prev_star,
        xi, xi_n, T=128, hist_mode="last_only", need_grad=True, horizons_out=(32, 64, 128)
    )
    wm_val_200 = rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star, f_star, oh_prev9_star, a_prev_star,
        xi, xi_n, T=200, hist_mode="last_only", need_grad=False, horizons_out=(16, 32, 64, 128, 200)
    )
    V_wm_200 = wm_val_200["J_soft"][200]
    gW_128 = wm_grad_res["grads_soft"][128]

    # ---- 4. Evaluate all 9 Critic snapshots (7 runs + pre/post @2100) ----
    sa = bridge.pairs_root()
    critic_specs = [
        ("b0@1000", sa / "b0_resume/ckpt_001000.pt", "critic"),
        ("b0@1500", sa / "b0_resume/ckpt_001500.pt", "critic"),
        ("b0@2000", sa / "b0_resume/ckpt_002000.pt", "critic"),
        ("B0@2100_pre", sa / "online_B0/diag_pairs/pair_it_002100.pt", "critic_pre"),
        ("B0@2100_post", sa / "online_B0/diag_pairs/pair_it_002100.pt", "critic_post"),
        ("B0@2300", sa / "online_B0/ckpt_002300.pt", "critic"),
        ("B1@2300", sa / "online_B1/ckpt_002300.pt", "critic"),
        ("B2@2300", sa / "online_B2/ckpt_002300.pt", "critic"),
        ("B3@2300", sa / "online_B3/ckpt_002300.pt", "critic"),
    ]

    summary = {
        "n_starts": B,
        "clips": sorted(set(clips)),
        "pose_screening": {
            "critic_modes": pose_meta_c["mode"],
            "min_clear_pm_critic": min(pose_meta_c["min_clear_pm"]),
            "min_slack_pm_critic": min(pose_meta_c["min_slack_pm"]),
            "min_clear_pm_random": min(pose_meta_r["min_clear_pm"]),
            "min_slack_pm_random": min(pose_meta_r["min_slack_pm"]),
        },
        "wm_scalar_calibration": {
            f"T{h}": {"mean_J_soft": float(wm_val_200["J_soft"][h].mean()),
                      "mean_J_hard": float(wm_val_200["J_hard"][h].mean()),
                      "alive_soft": float(wm_val_200["alive"][h][0].mean())}
            for h in (16, 32, 64, 128, 200)
        },
        "critics": {},
    }
    csv_rows = []
    per_start_rows = []

    for cname, cpath, ckey in critic_specs:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cin = ck[ckey]["norm.mean"].shape[0]
        cr = Critic(cin).eval()
        cr.load_state_dict(ck[ckey], strict=True)
        cg = critic_tangent_grad(cr, mdp, motion, proto, S_star, f_star, xi_n, clamp_vmax=10.0)
        v_raw = cg["v_raw"]
        g_raw = cg["grad_raw"]

        # Level 1 scalar calibration vs V_wm_200
        pearson = float(torch.corrcoef(torch.stack((v_raw, V_wm_200)))[0, 1])
        sp_scalar = spearman(v_raw, V_wm_200)
        mae = float((v_raw - V_wm_200).abs().mean())
        bias = float((v_raw - V_wm_200).mean())

        # Level 2 full/pose/vel cosine vs gW_128
        cos_all = _cos(g_raw, gW_128)
        cos_pose = _cos(g_raw[:, POSE_IDX], gW_128[:, POSE_IDX])
        cos_vel = _cos(g_raw[:, VEL_IDX], gW_128[:, VEL_IDX])

        dir_stats = {}
        for dname, u in oracle1_dirs.items():
            d_crit = (g_raw * u).sum(-1)
            d_wm128 = (gW_128 * u).sum(-1)
            st = _dir_metrics(d_crit, d_wm128, clips)
            dir_stats[dname] = st
            csv_rows.append({"critic": cname, "direction": dname, **st})

        summary["critics"][cname] = {
            "scalar_vs_wm200": {
                "mean_V": float(v_raw.mean()), "clamp_frac": float(cg["clamp_active"].float().mean()),
                "pearson": pearson, "spearman": sp_scalar, "mae": mae, "bias": bias,
            },
            "grad_cos_vs_wm128": {
                "med_cos_all": float(cos_all.median()),
                "med_cos_pose": float(cos_pose.median()),
                "med_cos_vel": float(cos_vel.median()),
                "med_norm_crit": float(g_raw.norm(dim=-1).median()),
                "med_norm_wm128": float(gW_128.norm(dim=-1).median()),
            },
            "directions_vs_wm128": dir_stats,
        }
        for i in range(B):
            per_start_rows.append({
                "critic": cname, "clip": clips[i], "frame": int(f0[i]),
                "V_raw": float(v_raw[i]), "V_wm200": float(V_wm_200[i]),
                "cos_all": float(cos_all[i]), "cos_pose": float(cos_pose[i]), "cos_vel": float(cos_vel[i]),
                "d_crit_vel_crit": float((g_raw[i] * oracle1_dirs["vel_critic"][i]).sum()),
                "d_wm128_vel_crit": float((gW_128[i] * oracle1_dirs["vel_critic"][i]).sum()),
            })
        print(f"  {cname:13s}: V_r={pearson:+.3f} | cos(all/pose/vel)=({cos_all.median():+.3f}/{cos_pose.median():+.3f}/{cos_vel.median():+.3f}) | vel_reach sign={dir_stats['vel_reachable']['sign_agree']:.2f} rat={dir_stats['vel_reachable']['med_ratio']:.2f}")

    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.out_dir / "phase0_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.out_dir / "phase0_per_critic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    with open(args.out_dir / "phase0_per_start.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_start_rows[0].keys()))
        w.writeheader(); w.writerows(per_start_rows)
    print(json.dumps({"out_dir": str(args.out_dir), "elapsed_s": summary["elapsed_s"]}))


if __name__ == "__main__":
    main()
