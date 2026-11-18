"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: pelvis position, rotation matrix, world linear/angular
velocity of the pelvis link frame, joint positions and velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dimensions import STATE_DIM, XI_DIM
from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat

OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6
PHASE_DIM = 2
NUM_J = 29

# Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
        torch.full((6,), 0.05),
    )
)
# normalization of the obs before the policy (qd and angular velocity are large)
OBS_SCALE = torch.cat(
    (torch.ones(29), torch.full((29,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(29), torch.ones(6))
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 29 + 29


class MDP:
    def __init__(self, kin: G1Kinematics, motion: MotionLib, default_q: torch.Tensor,
                 cfg: RewardCfg = RewardCfg(), phase_features: bool = False,
                 clip_features: bool = False, contact_safe_reference: bool = False,
                 contact_masked_reward: bool = False, contact_features: bool = False,
                 current_contact_features: bool = False):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        self.phase_features = bool(phase_features)
        self.clip_features = bool(clip_features)
        self.clip_feature_dim = len(motion.names) if self.clip_features else 0
        self.contact_safe_reference = bool(contact_safe_reference)
        self.contact_masked_reward = bool(contact_masked_reward)
        self.contact_features = bool(contact_features)
        self.current_contact_features = bool(current_contact_features)
        self.contact_feature_dim = int(motion.body_p.shape[1]) if self.contact_features else 0
        self.current_contact_feature_dim = int(motion.body_p.shape[1]) if self.current_contact_features else 0
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        self.obs_noise = OBS_NOISE.to(dev)
        self.obs_scale = OBS_SCALE.to(dev)
        if self.phase_features:
            self.obs_noise = torch.cat((self.obs_noise, torch.zeros(PHASE_DIM, device=dev)))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones(PHASE_DIM, device=dev)))
        if self.clip_features:
            self.obs_noise = torch.cat((self.obs_noise, torch.zeros(self.clip_feature_dim, device=dev)))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones(self.clip_feature_dim, device=dev)))
        if self.contact_features:
            self.obs_noise = torch.cat((self.obs_noise, torch.zeros(self.contact_feature_dim, device=dev)))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones(self.contact_feature_dim, device=dev)))
        if self.current_contact_features:
            self.obs_noise = torch.cat((self.obs_noise, torch.zeros(self.current_contact_feature_dim, device=dev)))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones(self.current_contact_feature_dim, device=dev)))

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1:30]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        out = torch.cat((q - self.default_q - dq0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)
        if self.phase_features:
            clip = self.motion.clip_id[f]
            phase = (f - self.motion.clip_start[clip]).float() / self.motion.clip_len[clip].sub(1).clamp_min(1).float()
            phase = 2.0 * torch.pi * phase
            out = torch.cat((out, torch.sin(phase).unsqueeze(-1), torch.cos(phase).unsqueeze(-1)), -1)
        if self.clip_features:
            clip = self.motion.clip_id[f]
            out = torch.cat((out, torch.nn.functional.one_hot(clip, self.clip_feature_dim).to(out.dtype)), -1)
        if self.contact_features:
            contact = (self.motion.body_p[f, :, 2] < 0.05).to(out.dtype)
            out = torch.cat((out, contact), -1)
        if self.current_contact_features:
            bp = self.kin.forward(*S)[0]
            out = torch.cat((out, (bp[..., 2] < 0.05).to(out.dtype)), -1)
        return out

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        if self.contact_safe_reference:
            ref_bp = torch.cat((ref_bp[..., :2], ref_bp[..., 2:3].clamp_min(0.02)), -1)
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        body_track = (
                torch.exp(-e_pos / c.body_pos_std**2)
                + torch.exp(-e_ori / c.body_ori_std**2)
                + torch.exp(-e_lv / c.body_linvel_std**2)
                + torch.exp(-e_av / c.body_angvel_std**2)
        )
        if self.contact_masked_reward:
            contact_valid = (m.body_p[f][..., 2] >= 0.0).all(-1).to(body_track.dtype)
            body_track = body_track * contact_valid
        r_track = (
                body_track
                + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)


CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), -1)
