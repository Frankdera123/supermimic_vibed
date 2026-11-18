# GRADIENT_FIDELITY_SOURCE_AUDIT

日期：2026-11-18。审计对象：`/home/sustech/Desktop/supermimic_vibed`（主仓库）与 OPCL proto 副本（`~/Documents/Codex/2026-11-14/new-chat-2/work/supermimic_opcl_proto`）。所有结论给出代码位置，可复核。

## 1. Policy Improvement 的完整目标（主仓库）

`supermimic/train.py: policy_update`（2725 行起）：

- WM 默认冻结：`set_requires_grad(wm, joint_wm)`，`joint_wm` 默认 False（train.py:2736-2741）；Critic 冻结（2750）。冻结的是参数，梯度仍然穿过 WM（BPTT），除非 `--policy_bptt_truncation` 每 N 步 detach 状态（2833-2840）。
- 想象循环 HZ 步（默认 32，train.py:22），每步 `_imag_step`（1530）：策略 `pi(o_hist, g)` 出动作 → WM `step` → 观测 `mdp.obs(S1, a, xi, f1)` + `add_noise`（1628）→ goal `motion.goal(f1, K)`。
- 目标（2941, 3067-3068）：

  J = Σ_t γ^t c_t·r_t + γ^HZ·c_{HZ-1}·V_ψ(ŝ_HZ)

  其中 c_t = Π(1−soft_term)（2937），r 由解析 reward 给出；`--no_critic` 可去掉 bootstrap。
- 附加项（不改变三路径结构，审计时全关）：height/tilt/angular-velocity penalty、ensemble softmin、dispersion gate、realism gate、BC/AWBC、proximal。

## 2. Reward 与终止的解析结构（误差归因的关键）

`mdp.py:149-210`：

- reward = (body_track + 0.5·root_pos + 0.5·root_ori + w_action_rate·rate + w_joint_limit·lim)·CONTROL_DT；
  body_track 四个 exp 核（body pos/ori/linvel/angvel），FK 由 `kinematics.forward` 对状态可微。
- **因此 reward 路径梯度误差 100% 由动力学梯度误差诱导**——不存在学习式 reward model。
- soft_termination（203）：sigmoid 可微终止；`alive` 连乘进入 J。
- **rot_angle_sq（rotations.py:94）用 clamp 过的 acos**：cos→1（近完美对齐）处存在结构性扭结。这既影响训练梯度也影响任何 FD 类审计——方向导数在对齐邻域可能不可靠（见 EVALUATION_REPORT 的 ori_clamp 诊断）。

## 3. 三路径分解定义（本审计采用的规范）

对开环动作序列 A=(a_0..a_{H-1})，J(A) = Σ γ^t c_t(r_state_t + r_direct_t) + γ^H c_{H-1} V(ŝ_H)：

| 路径 | 定义 | 是否经过 WM Jacobian |
|---|---|---|
| D_state | grad Σ γ^t c_t·r_state_t（含 ∂c/∂A·r_state） | 是 |
| D_direct | grad Σ γ^t c_t.detach()·r_direct_t（action-rate） | **否**（模型无关） |
| D_critic | grad γ^H c.detach()·V(ŝ_H) | **是**（critic 函数 × WM 状态 Jacobian） |
| cross | ∂c/∂A·r_direct 与 γ^H·∂c/∂A·V | 混合项，单独报告 |

恒等式 D_total = D_state + D_direct + D_critic + cross 由构造成立，CPU 测试验证到机器精度（test_crgf identity 8.7e-19）。**critic 路径不是独立于动力学的：它经过 WM 状态 Jacobian**（用户要求 4 的正确表述）。

## 4. `wm_action_effect_gain` 的确切语义

- 定义：`--wm_action_effect_gain`，默认 1.0（train.py:150），正值校验（805）。
- 作用点：`_imag_step`（1561）：`a_model = clamp(a * gain)` **仅传给 WM step**；观测（1628 `mdp.obs(S1, a, ...)`）、reward（2933 用原始 As）、replay fitting 与真实采集都用原始动作。
- 结论：这是**推理期想象循环里的响应缩放**，不是模型结构校准、也不是训练机制。gain≠1 时想象闭环内部状态-观测-奖励三者语义错位（状态由 a·gain 推进，观测/reward 用 a），这是全局 gain 修复失败的机制性解释之一（见历史证据报告 §4）。

## 5. 两线代码差异（必须用 proto 代码评估 B0/B1/B2）

| 项 | supermimic_vibed | OPCL proto |
|---|---|---|
| WM 输出 | 70 维全增量（dp,dθ,dv,dω,dq,dqd，networks.py:54） | **35 维加速度级**（dv_b,dw_b,dqd，proto networks.py:53） |
| OBS_DIM | 99（含 6D 参考相对姿态） | 93（无该项） |
| goal | 帧索引 `motion.goal(f, K)`（motion.py:133） | **状态相对** `goal(f, p, R) → (actor, wm_goal)`（proto motion.py:111），GOAL_FRAMES=10、STRIDE=5 硬编码 |
| critic 输入 | `cat(critic_features(S,f,ξ_n), g)` | `critic_features(S,f,ξ_n,wm_goal)` + `clamp(max=V_MAX)`（proto train.py:1144） |
| mdp.obs | 4 参（含 f） | 3 参 |

实现响应：`supermimic/crgf.py::Protocol` 运行时检测并适配两种约定；工具经 `--proto_root` 加载 proto 包（`tools/_crgf_bootstrap.py` 装 isaaclab stub + 修复 MJCF 路径）。proto 的 `data/` 目录为空且 root 所有，未改动，运行时把 `MJCF_PATH` 指回本仓库 `data/g1_description/g1_29dof.xml`（`fix_data_paths`，附加性、可逆）。

## 6. 历史审计工具清单（复用，不重复造轮子）

- `tools/true_action_jacobian_audit.py`：单步 WM-vs-Isaac-FD 响应向量对照（sensitivity_ratio / least_squares_gain / vector_cosine / quartile bins）。
- `tools/horizon_jacobian_audit.py`：多步 WM 对动作序列敏感度（无 sim 真值）。
- `tools/multiscale_action_gram_audit.py`：多尺度响应 Gram 阵对照 + 门禁校准。
- `tools/generate_counterfactual_dataset.py`：仿真器反事实对数据（±动作扰动真值）。
- 本轮新增：`supermimic/crgf.py`（核心）、`tools/crgf_wm_directional_audit.py`、`tools/crgf_delta_action_directions.py`、`tools/crgf_sim_fd_audit.py`、`tools/test_crgf.py`。
