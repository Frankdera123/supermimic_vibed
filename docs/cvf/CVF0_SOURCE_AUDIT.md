# CVF-0_SOURCE_AUDIT（Critic Value-Gradient Fidelity Stage-0 源码与资产审计）

日期：2026-11-19。范围：纯 CPU 只读源码与本地资产审计，无训练、无 Loss 修改、无 GPU 调用。

## 1. 两套代码线与检查点边界（禁止混池）

| 资产线 | 核心源码 | 本地可用 Critic-enabled 检查点 | 关键架构与训练口径 | CVF-0 定位 |
|---|---|---|---|---|
| **主线：`opcl_proto`** | `supermimic_opcl_proto/supermimic/{networks,mdp,motion,env,train}.py` | `b0_resume` (`1000/1500/2000`) + `online_B0/B1/B2/B3` (`2100/2200/2300`，含 15 个 `VALID_SINGLE_POLICY_UPDATE` `diag_pairs`) | 35D 加速度级 WM；LN+GELU Critic `[256,128]`；`V_MAX = 10.0` clamp；`OBS_DIM=93`；状态相对 `wm_goal`；`H=32, γ=0.99, λ=0.95`；2000 轮起 `critic.norm.count = 2.00e8`（已冻结）；`env.py` 内置 `snapshot_state`/`restore_state` | **CVF-0 主研究对象**（与 OPCL/CRGF/PADA-0a 同源连续） |
| **副线：`supermimic_vibed`** | `supermimic/{networks,mdp,motion,env,train}.py` | `logs/run/last.pt` (`it=4000, H=32, no_critic=False, c_norm_cnt=2.00e8`) | 70D 全增量 WM；无 LN 的 ELU Critic `[256,128]`；无 `V_MAX` clamp；`OBS_DIM=99`；帧索引 `goal`；`env.py` 无 `restore_state` | **CVF-0b 独立复核线**（不与 `opcl_proto` 混合统计） |

注：主仓库领先并发配方（`tools/run_concurrent_ramp.sh`）采用 `--no_critic --horizon 64 --policy_bptt_truncation 4`，与 Critic-enabled 体系属于不同优化范式，CVF-0 不混用。

## 2. Simulator 状态注入与确定性约束审查

1. **`restore_state()` 可用但有两项边界条件**（`opcl_proto/supermimic/env.py:209-245`）：
   - 可显式注入 `(p, R, v, w, q, qd)` 与 `f, ep_step, a_prev, xi` 并执行 `scene.reset → write_data_to_sim → refresh`。
   - **缺口 1**：与 `reset()` 第 168 行不同，`restore_state()` **未对 `q` 做 `[q_lo, q_hi]` clamp**，状态扰动前必须由 CPU 预筛保证严格不越界。
   - **缺口 2**：CRGF Phase 0 仅验证了 `env.reset()` 在 `deterministic_mode="run_to_run" + disable_sensors=True + <=4 envs` 下逐位一致（`repeat_range = 0.0`），尚未验证 `restore_state()` 注入非参考状态后 MuJoCo-Warp 是否存在 warm-start 缓存残留 → 列为 Phase 1a GPU 烟测首项必检项。
2. **`eval_mode=True` 连续轨迹导出**（`opcl_proto/env.py:275-278`）：
   - 在 `eval_mode=True` 下遇 `term` 或 `timeout` 不自动重置，单次长程 rollout 可在离线同时计算任意截断长度 $T$ 下的 `soft_termination` 与 `hard_termination` 回报。
3. **MotionLib 片段长度**：
   - `data/lafan1_g1` 全部 40 个 clip（50 Hz，共 441,140 帧）长度范围为 `[5109, 13656]` 帧（102.2 s – 273.1 s），中位 12,065 帧。预留 `margin >= 250` 帧后 **40/40 clip 全部可用**，不会混入 clip-end 静止延拓。

## 3. 训练语义与评估口径对齐

1. **存活因子 $c_t$ 的 1 步索引修正**：
   - 训练源码（`train.py:2937-2941` 及 `opcl_proto/train.py:1136-1144`）定义 $c_0 = 1,\; c_t = \prod_{k=0}^{t-1}(1 - d_k)$，即第 $t$ 步奖励 $r_t$ 乘以 $c_t$（第 $t$ 步的终止概率 $d_t$ 折减第 $t+1$ 步及之后）。
   - CRGF/PADA 评估工具（`crgf.py:186-188`, `pada_eval.py:53-55`）写成了 `c_prev *= (1-d)` 后再乘 `r_t`（即 $c_{t+1} r_t$）。CVF-0 统一采用**与训练源码完全一致的 $c_t r_t$ 索引**。
2. **Critic 与 Policy 的精确配对**：
   - `diag_pairs` 操作顺序为 `collect → wm_update → snapshot(pi_old, critic_pre) → policy_update(pi_old→pi_new) → critic_update(在 pi_old 想象轨迹上更新 critic_pre→critic_post)`。
   - 因此 `critic_post`（即 `bundle["critic"]`）的严格匹配策略是 `pi_old`。
3. **`rot_angle_sq` 的 `acos` 扭结**（`supermimic/rotations.py:94-98`）：
   - 在完全对齐点附近存在 `clamp(..., 1-1e-6)` 扭结；Oracle 1 采用真实前缀 $H_{\text{pre}}=16$ 步到达的 on-policy 状态 $S^*$，天然避开参考零误差点。

## 4. 足底碰撞几何与关节限位边界（CPU 实测）

- **G1 足底碰撞体**（`data/g1_description/g1_29dof.xml`）：左右脚各在 `left/right_ankle_roll_link` 下挂 4 个半径 $r = 0.005\text{ m}$ 的接触球，局部坐标为 `(-0.05, ±0.025, -0.03)` 与 `(0.12, ±0.03, -0.03)`。即足底最低接触面为 $z_{\text{ball}} - 0.005$。
- **LAFAN1 参考动作足底间隙**：`min(l_ankle_z, r_ankle_z)` 在 5%/25%/50%/75%/95% 分位分别为 `0.0351 / 0.0400 / 0.0454 / 0.0551 / 0.1088 m`；双脚同时高于 `0.10 m`（离地 $> 6.5\text{ cm}$）的腾空帧仅占 `6.24%`。
- **关节硬限位余量**：参考动作中部分关节距 `[q_lo, q_hi]` 的最小余量低至 `0.0104 rad`（中位最小余量 `0.0783 rad`）。
- **结论**：支撑相下直接扰动 $(p_z, \theta_{rp}, q_{\text{leg}})$ 必然导致单侧穿地冲击与不可微接触跳变；必须将直接状态扰动限制在零穿地子空间（`Tier-V` 与 `Tier-P-safe`），而把完整支撑相耦合交由动作诱导的自然物理流形（Oracle 2）来评估。
