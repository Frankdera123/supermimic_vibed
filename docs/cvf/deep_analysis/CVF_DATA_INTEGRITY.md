# CVF-0 Data Integrity & Asset Inventory

**Scope**: Independent CPU-only verification of all CVF-0 artifacts, checkpoints, SHA256 hashes, data shapes, and determinism.

## 1. Repository & Codebase Lineages

- **Main repository**: `/home/sustech/Desktop/supermimic_vibed` (branch `dev`, HEAD `6e9b4e7`).
  - Critic-enabled default: 70-D full-state WM, `OBS_DIM=99`, ELU Critic, no `V_MAX` clamp.
  - Leading training recipe: `--no_critic`, `H=64`, `--policy_bptt_truncation 4`, `--wm_teacher_forcing`.
- **OPCL proto variant (used by CVF-0)**: `/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/supermimic_opcl_proto`.
  - 35-D acceleration-level WM (`WM_OUT = 3+3+29`), `OBS_DIM=93`, semi-implicit pose integration, state-relative `wm_goal`, LayerNorm+GELU Critic `[256, 128]`, `V_MAX = 10.0` clamp, `H=32, gamma=0.99, lam=0.95`.
- **Server checkpoints**: `/home/sustech/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts`.

## 2. Accessible Critic Checkpoints (9 Total)

| Name | Run Branch | Iter | Key | `norm.count` | `opcl_enabled` | `opcl_lambda_p/o` | `b1_match_opcl` |
|---|---|---:|---|---:|---|---|---|
| `b0@1000` | `b0_resume` | 1000 | `critic` | `1.12e8` | False | 0.0 / 0.0 | False |
| `b0@1500` | `b0_resume` | 1500 | `critic` | `1.68e8` | False | 0.0 / 0.0 | False |
| `b0@2000` | `b0_resume` | 2000 | `critic` | `2.00e8` | False | 0.0 / 0.0 | False |
| `B0@2100_pre` | `online_B0` | 2100 | `critic_pre` | `2.00e8` | False | 0.0 / 0.0 | False |
| `B0@2100_post` | `online_B0` | 2100 | `critic_post` | `2.00e8` | False | 0.0 / 0.0 | False |
| `B0@2300` | `online_B0` | 2300 | `critic` | `2.00e8` | False | 0.0 / 0.0 | False |
| `B1@2300` | `online_B1` | 2300 | `critic` | `2.00e8` | False | 0.0 / 0.0 | True |
| `B2@2300` | `online_B2` | 2300 | `critic` | `2.00e8` | True | 0.01 / 0.0 | False |
| `B3@2300` | `online_B3` | 2300 | `critic` | `2.00e8` | True | 0.0 / 100.0 | False |

- All 9 Critics have input dimension `938` (`critic_in_dim=938`) and identical architecture (`[256, 128]`, LayerNorm + GELU).
- `online_B0`, `online_B1`, `online_B2`, `online_B3` all fork from `b0_resume/ckpt_002000.pt` at iteration 2000; they are **parallel branches**, not a single time series.
- Full SHA256 hashes and byte sizes for all checkpoints and `results/cvf/` artifacts are recorded in [da01_inventory_contract.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da01_inventory_contract.json).

## 3. Dataset Shapes & Determinism

- **Starts**: 16 balanced starts across 8 distinct LAFAN1 clips (2 frames/clip, 12.5% per clip):
  `dance2_subject2`, `fallAndGetUp1_subject1`, `fight1_subject5`, `fightAndSports1_subject4`, `jumps1_subject5`, `run2_subject4`, `sprint1_subject2`, `walk1_subject2`.
- **Oracle 1** (`oracle1_sim.pt`): 16 starts × 5 directions (`vel_critic`, `vel_reachable`, `vel_random`, `pose_safe_critic`, `pose_safe_random`) × 2 epsilons (`0.015`, `0.03`) × `+/-` + 2 base reps; `T_max = 128` (`horizons_out = [32, 64, 96, 128]`).
  - `base_repeat_range` across all 4 batches: `[0.0, 0.0, 0.0, 0.0]` (exact bit-for-bit determinism).
- **Oracle 2** (`oracle2_sim.pt`): 16 starts × 4 action directions (`delta_2100`, `delta_2300`, `ascent`, `random`) × 2 epsilons (`0.015`, `0.03`) × `+/-` + 2 base reps; `h_open = 32`, `t_cont = 96` (`cont_out = [32, 64, 96]`).
  - `base_repeat_range` across all 4 batches: `[0.0, 0.0, 0.0, 0.0]`.
