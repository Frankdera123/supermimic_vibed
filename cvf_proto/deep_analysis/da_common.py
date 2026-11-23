"""CVF Deep Analysis -- shared, independent utilities (CPU only).

Design rules
------------
* Read-only on every historical artefact (phase0/phase1b .pt, checkpoints, reports).
* Re-implements the WM continuation / critic evaluation instead of importing the
  historical `cvf_core.rollout_wm_continuation`, so that the original implementation
  can be cross-checked against an independent one (see da01_contract_audit.py).
* All outputs go to results/cvf/deep_analysis/.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hard CPU-only guard

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent.parent
sys.path.insert(0, str(REPO / "pada_proto"))
sys.path.insert(0, str(REPO / "cvf_proto"))

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available(), "Deep analysis must run CPU-only"
torch.set_num_threads(min(20, os.cpu_count() or 4))

OUT = REPO / "results" / "cvf" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)
GAMMA = 0.99
V_MAX = 10.0

SA = bridge.pairs_root()

# The 9 critics used by CVF-0 (names identical to the historical summary).
CRITIC_SPECS = [
    ("b0@1000", SA / "b0_resume/ckpt_001000.pt", "critic", 1000, "b0_resume"),
    ("b0@1500", SA / "b0_resume/ckpt_001500.pt", "critic", 1500, "b0_resume"),
    ("b0@2000", SA / "b0_resume/ckpt_002000.pt", "critic", 2000, "b0_resume"),
    ("B0@2100_pre", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_pre", 2100, "online_B0"),
    ("B0@2100_post", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_post", 2100, "online_B0"),
    ("B0@2300", SA / "online_B0/ckpt_002300.pt", "critic", 2300, "online_B0"),
    ("B1@2300", SA / "online_B1/ckpt_002300.pt", "critic", 2300, "online_B1"),
    ("B2@2300", SA / "online_B2/ckpt_002300.pt", "critic", 2300, "online_B2"),
    ("B3@2300", SA / "online_B3/ckpt_002300.pt", "critic", 2300, "online_B3"),
]
CRITIC_NAMES = [c[0] for c in CRITIC_SPECS]
PRIMARY = "B0@2100_post"

DIRS_O1 = ["vel_critic", "vel_reachable", "vel_random", "pose_safe_critic", "pose_safe_random"]
DIRS_O2 = ["delta_2100", "delta_2300", "ascent", "random"]
EPS = (0.015, 0.03)
H_O1 = (32, 64, 96, 128)

# 70-D tangent layout used by perturb_state / WorldModel.step
TANGENT_BLOCKS = {
    "root_pos": slice(0, 3), "root_ori": slice(3, 6), "root_linvel": slice(6, 9),
    "root_angvel": slice(9, 12), "joint_pos": slice(12, 41), "joint_vel": slice(41, 70),
}
# fixed physical scales (Bryson nominal errors from WorldModel loss, networks.NOMINAL_ERROR)
BRYSON = {"root_pos": 0.01, "root_ori": 0.02, "root_linvel": 0.1, "root_angvel": 0.3,
          "joint_pos": 0.1, "joint_vel": 3.14}


def bryson_vec(dtype=torch.float32):
    s = torch.zeros(70, dtype=dtype)
    for k, sl in TANGENT_BLOCKS.items():
        s[sl] = BRYSON[k]
    return s


# Critic input (938-D) feature index map -- derived from mdp.critic_features and motion.goal
def critic_feature_groups(goal_frames: int = 10) -> dict[str, list[int]]:
    g = {
        "height": [0],
        "tilt6d": list(range(1, 7)),
        "root_linvel_b": list(range(7, 10)),
        "root_angvel_b": list(range(10, 13)),
        "joint_pos": list(range(13, 42)),
        "joint_vel": list(range(42, 71)),
        "joint_pos_err": list(range(71, 100)),
        "fk_body_pos_err": list(range(100, 142)),
        "root_pos_err": list(range(142, 145)),
        "xi_n": list(range(145, 178)),
    }
    base = 178
    ref_only, rel_ori = [], []
    for k in range(goal_frames):
        o = base + 73 * k
        ref_only += list(range(o, o + 67))
        rel_ori += list(range(o + 67, o + 73))
    g["goal_ref_only"] = ref_only
    g["goal_rel_ori"] = rel_ori
    g["goal_rel_pos"] = list(range(base + 73 * goal_frames, base + 73 * goal_frames + 3 * goal_frames))
    return g


FEATURE_GROUP_ORDER = ["height", "tilt6d", "root_linvel_b", "root_angvel_b", "joint_pos", "joint_vel",
                       "joint_pos_err", "fk_body_pos_err", "root_pos_err", "xi_n", "goal_ref_only",
                       "goal_rel_ori", "goal_rel_pos"]


# ----------------------------------------------------------------------------- loading
_WORLD = None


def world():
    global _WORLD
    if _WORLD is None:
        pair = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
        w = bridge.setup_world(device="cpu", pair=pair)
        w["pair2100"] = pair
        _WORLD = w
    return _WORLD


def load_critic(path, key):
    from supermimic.networks import Critic
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck[key]
    cr = Critic(sd["norm.mean"].shape[0]).eval()
    cr.load_state_dict(sd, strict=True)
    for p in cr.parameters():
        p.requires_grad_(False)
    return cr


def load_all_critics():
    return {n: load_critic(p, k) for n, p, k, _, _ in CRITIC_SPECS}


def load_pair(run, it):
    p = bridge.load_pair_bundle(run, it, device="cpu")
    for m in (p["pi_old"], p["pi_new"], p["wm"], p["critic"]):
        if m is not None:
            for q in m.parameters():
                q.requires_grad_(False)
    return p


def load_policy_from(path, key="pi"):
    from supermimic.mdp import OBS_DIM
    from supermimic.motion import GOAL_FRAMES
    from supermimic.networks import Policy
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pi = Policy(OBS_DIM, GOAL_FRAMES * 73, hist=10).eval()
    pi.load_state_dict(ck[key], strict=True)
    for q in pi.parameters():
        q.requires_grad_(False)
    return pi


def load_wm_from(path, key="wm"):
    from supermimic.networks import WorldModel
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = WorldModel().eval()
    wm.load_state_dict(ck[key], strict=True)
    for q in wm.parameters():
        q.requires_grad_(False)
    return wm


# ----------------------------------------------------------------------------- raw data
def load_raw():
    o1 = torch.load(REPO / "results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
    o2 = torch.load(REPO / "results/cvf/phase1b/oracle2_sim.pt", map_location="cpu", weights_only=False)
    p0 = torch.load(REPO / "results/cvf/phase0/phase0_starts_and_dirs.pt", map_location="cpu", weights_only=False)
    return o1, o2, p0


def assemble_o1(o1):
    """Tidy Oracle-1 tensors. Start index s = 4*batch + env (identical to o1['starts'] order)."""
    B = o1["batches"]
    cat = lambda f: torch.cat([f(b) for b in B], 0)  # noqa: E731
    d = {
        "starts": o1["starts"],
        "clips": [s["clip_name"] for s in o1["starts"]],
        "frames": [s["frame"] for s in o1["starts"]],
        "S_star": tuple(cat(lambda b, k=k: b["S_star"][k]) for k in range(6)),
        "f_star": cat(lambda b: b["f_star"]),
        "f0": cat(lambda b: b["f0"]),
        "a_prev": cat(lambda b: b["a_prev_star"]),
        "oh9": cat(lambda b: b["oh_prev9_star"]),
        "dirs": {dn: cat(lambda b, dn=dn: b["dirs"][dn]) for dn in DIRS_O1},
        "pose_mode_c": sum([b["pose_meta_c"]["mode"] for b in B], []),
        "pose_mode_r": sum([b["pose_meta_r"]["mode"] for b in B], []),
        "base": [], "br": {},
    }
    for r in range(2):
        d["base"].append({
            key: {h: cat(lambda b, h=h, key=key: b["base_res"][r][key][h]) for h in H_O1}
            for key in ("J_soft", "J_hard", "c_soft", "c_hard")
        })
        d["base"][r]["s_end"] = cat(lambda b: b["base_res"][r]["s_end"])
    for dn in DIRS_O1:
        for e in EPS:
            for sg in ("plus", "minus"):
                k = (dn, e, sg)
                d["br"][k] = {
                    key: {h: cat(lambda b, h=h, key=key: b["branches"][k][key][h]) for h in H_O1}
                    for key in ("J_soft", "J_hard", "c_soft", "c_hard")
                }
                d["br"][k]["s_end"] = cat(lambda b: b["branches"][k]["s_end"])
    return d


def assemble_o2(o2):
    B = o2["batches"]
    cat = lambda f: torch.cat([f(b) for b in B], 0)  # noqa: E731
    d = {"starts": o2["starts"], "clips": [s["clip_name"] for s in o2["starts"]],
         "f0": cat(lambda b: b["f0"]), "base": [], "br": {}}

    def grab(res_getter):
        out = {}
        for key in ("r_state", "r_direct", "d_soft", "d_hard", "S_at_8", "f_at_8", "S_at_32", "f_at_32"):
            out[key] = cat(lambda b, key=key: res_getter(b)[key])
        for key in ("V_cont_soft", "V_cont_hard", "c_cont_soft", "c_cont_hard"):
            out[key] = {h: cat(lambda b, h=h, key=key: res_getter(b)[key][h]) for h in (32, 64, 96)}
        return out

    for r in range(2):
        d["base"].append(grab(lambda b, r=r: b["base_res"][r]))
    for dn in DIRS_O2:
        for e in EPS:
            for sg in ("plus", "minus"):
                d["br"][(dn, e, sg)] = grab(lambda b, k=(dn, e, sg): b["branches"][k])
    return d


# ----------------------------------------------------------------------------- dynamics
def perturb(S, delta):
    from supermimic.rotations import so3_exp, yaw_mat
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
    return (p + (yaw_mat(R) @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth),
            v + (R @ dv.unsqueeze(-1)).squeeze(-1), w + (R @ dw.unsqueeze(-1)).squeeze(-1), q + dq, qd + dqd)


def local_diff(Sa, Sb):
    """Tangent coordinates of Sa relative to Sb (inverse of perturb at first order)."""
    from supermimic.rotations import so3_log, yaw_mat
    p, R, v, w, q, qd = Sb
    pa, Ra, va, wa, qa, qda = Sa
    Hy, Rt = yaw_mat(R), R.transpose(-1, -2)
    return torch.cat(((Hy.transpose(-1, -2) @ (pa - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ Ra),
                      (Rt @ (va - v).unsqueeze(-1)).squeeze(-1), (Rt @ (wa - w).unsqueeze(-1)).squeeze(-1),
                      qa - q, qda - qd), -1)


def critic_value(critic, S, f, xi_n):
    w = world()
    proto = w["proto"]
    return critic(proto.critic_input(S, f, xi_n, proto.wm_goal(f, S)))


def critic_input(S, f, xi_n):
    proto = world()["proto"]
    return proto.critic_input(S, f, xi_n, proto.wm_goal(f, S))


def xi_nominal(B, dtype=torch.float32):
    from supermimic.mdp import normalize_xi
    xi = torch.zeros(B, 33, dtype=dtype)
    xi[:, 0] = 1.0
    return xi, normalize_xi(xi)


def wm_closed_loop(wm, pi, S0, f0, oh9, a_prev0, T_list, critics_boot=None, obs_noise_gen=None,
                   index_mode="c_t", act_clip=True):
    """Independent closed-loop WM continuation (training-aligned c_t * r_t indexing).

    Returns dict with J[T] (B,), c[T] (B,), S[T] state tuples, f[T] and, if critics_boot given,
    Vboot[name][T] = V(S_T) (raw, unclamped; clamp applied later).  Graph is preserved when S0
    requires grad (caller decides).
    index_mode: 'c_t' (training: c_0 = 1, J += g^t c_t r_t) or 'c_t1' (CRGF/PADA tools: c_{t+1} r_t).
    """
    w = world()
    proto, mdp = w["proto"], w["mdp"]
    B = S0[0].shape[0]
    xi, xi_n = xi_nominal(B, S0[0].dtype)
    o_cur = proto.obs(S0, a_prev0, xi, f0)
    if obs_noise_gen is not None:
        o_cur = o_cur + (2 * torch.rand(o_cur.shape, generator=obs_noise_gen) - 1) * mdp.obs_noise
    oh = torch.cat((oh9, o_cur.unsqueeze(1)), 1)
    S, f, ap = S0, f0.clone(), a_prev0
    c = torch.ones(B, dtype=S0[0].dtype)
    J = torch.zeros(B, dtype=S0[0].dtype)
    Tmax = max(T_list)
    out = {"J": {}, "c": {}, "S": {}, "f": {}, "Vboot": {}, "act_clip_frac": 0.0, "d_soft_max": 0.0}
    nclip = 0
    for t in range(Tmax):
        g = proto.actor_goal(f, S)
        a_raw = pi(oh, g)
        nclip += int((a_raw.abs() > 20.0).sum())
        a = torch.clamp(a_raw, -20.0, 20.0) if act_clip else a_raw
        S = proto.wm_step(wm, S, a, xi_n)
        f = f + 1
        r = mdp.reward(S, f, a, ap)
        d = mdp.soft_termination(S, f)
        out["d_soft_max"] = max(out["d_soft_max"], float(d.detach().max()))
        if index_mode == "c_t":
            J = J + (GAMMA ** t) * c * r
            c = c * (1.0 - d)
        else:
            c = c * (1.0 - d)
            J = J + (GAMMA ** t) * c * r
        ap = a
        o1 = proto.obs(S, a, xi, f)
        if obs_noise_gen is not None:
            o1 = o1 + (2 * torch.rand(o1.shape, generator=obs_noise_gen) - 1) * mdp.obs_noise
        oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), 1)
        if (t + 1) in T_list:
            T = t + 1
            out["J"][T], out["c"][T], out["S"][T], out["f"][T] = J, c, S, f.clone()
            if critics_boot:
                out["Vboot"].setdefault(T, {})
                for n, cr in critics_boot.items():
                    out["Vboot"][T][n] = critic_value(cr, S, f, xi_n)
    out["act_clip_frac"] = nclip / (Tmax * B * 29)
    return out


# ----------------------------------------------------------------------------- statistics
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def sign_agree(pred, ref):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    return (np.sign(pred) == np.sign(ref)).astype(float)


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=5000, seed=2701):
    """Percentile CI by resampling whole clusters (start or clip) with replacement.
    `values` may be 1-D or a tuple of aligned arrays passed to `stat(*arrays)`."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    arrays = values if isinstance(values, tuple) else (np.asarray(values, float),)
    arrays = tuple(np.asarray(a, float) for a in arrays)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    point = float(stat(*arrays))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        try:
            boots.append(float(stat(*(a[idx] for a in arrays))))
        except Exception:
            boots.append(float("nan"))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return {"point": point, "ci95": [float("nan"), float("nan")], "n_clusters": int(len(uniq))}
    return {"point": point, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "n_clusters": int(len(uniq))}


def exact_sign_test(successes, n):
    from scipy.stats import binomtest
    if n == 0:
        return float("nan")
    return float(binomtest(int(successes), int(n), 0.5).pvalue)


def leave_one_out(values_fn, clusters):
    """values_fn(mask)->float; returns min/max/per-drop over clusters."""
    clusters = np.asarray(clusters)
    res = {}
    for c in np.unique(clusters):
        res[str(c)] = float(values_fn(clusters != c))
    vals = [v for v in res.values() if np.isfinite(v)]
    return {"min": min(vals) if vals else float("nan"), "max": max(vals) if vals else float("nan"), "per_drop": res}


def holm(pvals: dict) -> dict:
    items = sorted([(k, v) for k, v in pvals.items() if np.isfinite(v)], key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def pair_metrics(pred, ref, clusters_start=None, clusters_clip=None, n_boot=3000):
    """Full fidelity block for directional-response pairs."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    n = len(pred)
    if n == 0:
        return {"n": 0}
    sa = sign_agree(pred, ref)
    err = pred - ref
    ratio = np.abs(pred) / np.maximum(np.abs(ref), 1e-8)
    out = {
        "n": int(n),
        "sign_agree": float(sa.mean()),
        "sign_p_exact": exact_sign_test(int(sa.sum()), n),
        "pearson": pearson(pred, ref),
        "spearman": spearman(pred, ref),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "med_ratio": float(np.median(ratio)),
        "ratio_iqr": [float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))],
        "ols_slope_pred_on_ref": float(np.polyfit(ref, pred, 1)[0]) if n > 2 and ref.std() > 0 else float("nan"),
        "med_abs_pred": float(np.median(np.abs(pred))),
        "med_abs_ref": float(np.median(np.abs(ref))),
    }
    if clusters_start is not None:
        out["sign_ci95_start"] = cluster_bootstrap(sa, clusters_start, n_boot=n_boot)["ci95"]
        out["n_starts"] = int(len(np.unique(clusters_start)))
    if clusters_clip is not None:
        out["sign_ci95_clip"] = cluster_bootstrap(sa, clusters_clip, n_boot=n_boot)["ci95"]
        out["n_clips"] = int(len(np.unique(clusters_clip)))
        if n >= 4:
            out["spearman_ci95_clip"] = cluster_bootstrap((pred, ref), clusters_clip, stat=lambda a, b: spearman(a, b),
                                                          n_boot=min(n_boot, 1500))["ci95"]
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)
