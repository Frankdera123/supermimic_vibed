# CVF-0_EXPERIMENT_PROTOCOL（Critic Value-Gradient Fidelity Stage-0 预注册协议）

日期：2026-11-19。本协议在任何 GPU 采集之前落盘锁定。

## 0. 核心研究问题

> 在 SuperMimic（`opcl_proto` Critic-enabled 体系）中，已学习 Critic 的状态价值梯度 $\nabla_s V_\psi(S)$ 在多大程度上忠实反映真实 Simulator 的闭环未来回报梯度 $\nabla_s V_{\text{sim}}^\pi(S)$？其误差主要源于模型内近似（Sobolev 失配）、WM→Sim 动力学现实鸿沟，还是特定子空间/方向上的幅值畸变？

## 1. 误差两段拆解与 70-D 切空间定义

在机器人的非欧状态流形 $\mathcal{M} = \mathbb{R}^3 \times SO(3) \times \mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{R}^{29} \times \mathbb{R}^{29}$ 上，定义与 `WorldModel.step` 局部坐标一致的 70-D 切空间扰动算子 $\oplus$：
$$S \oplus \delta = \big(p + H(R)\delta p,\; R\exp(\delta\theta),\; v + R\delta v,\; w + R\delta w,\; q + \delta q,\; \dot q + \delta\dot q\big)$$
Critic 沿单位方向 $u \in \mathbb{R}^{70}$ 的精确方向导数为 $D_{\text{critic}}(S, u) = \langle \nabla_\delta \widetilde{V}_\psi(S \oplus \delta)|_{\delta=0}, u\rangle$（穿透 `mdp.critic_features` 与 `motion.goal` 的 `wm_goal`）。

将总梯度误差拆分为两个正交项：
$$\underbrace{\nabla_s V_\psi(S) - \nabla_s V_{\text{sim}}^{\pi}(S)}_{\text{Total Error}} = \underbrace{\big[\nabla_s V_\psi(S) - \nabla_s V_{\text{WM}}^{\pi}(S)\big]}_{\text{(A) 模型内 Sobolev 误差（纯 CPU 可测）}} + \underbrace{\big[\nabla_s V_{\text{WM}}^{\pi}(S) - \nabla_s V_{\text{sim}}^{\pi}(S)\big]}_{\text{(B) 长程现实鸿沟（需 Sim 续推）}}$$

## 2. 真实闭环续推价值 $V_{\text{sim}}^\pi(S; T)$ 规范

- **未来策略**：`critic_post` 严格配对 `pi_old`，确定性动作 $\pi(o_{\text{hist}}, g)$，clamp 至 `[-20, 20]`。
- **观测噪声**：主协议采用 noise-free 确定性观测；可选 Phase 1c 做 Common Random Numbers (CRN) 噪声敏感性对照。
- **折扣与终止**：$\gamma = 0.99$，$c_0=1, c_k = \prod_{j=0}^{k-1}(1 - d_j)$（与训练源码一致）；主参考使用 `soft_termination`（$t_z=0.02, t_t=0.04$），辅参考同时导出 `hard_termination`；**主真值禁止在步 $T$ 拼接 learned critic bootstrap**。
- **截断收敛**：单次采集 $T_{\max}$ 步，离线导出 $T \in \{32, 64, 128, T_{\max}\}$ 梯度收敛曲线与尾项变动 $\rho_{\text{tail}}(128, T_{\max})$。
- **上下文**：nominal $\xi = [1.0, 0_{29}, 0_3]$，`eval_mode=True`，保证 $f_0 + H_{\text{pre}} + T_{\max} \le \text{clip\_end}$。

## 3. 双真值神谕架构（Dual-Oracle）

### Oracle 1：直接切空间状态扰动（测隔离状态价值梯度）
在前缀真实仿真 $H_{\text{pre}}=16$ 步到达的 on-policy 状态 $S^*$ 处注入 $S^* \oplus (\pm \epsilon u)$，保持前 9 步 `o_hist` 不变、仅更新当前槽 $o_t^\pm$（`last_only`），在真实 Simulator 中闭环续推 $T$ 步。

**物理可行性子空间分层**（严禁混池）：
- **Tier-V（零穿地速度子空间：$\delta v, \delta w, \delta\dot q$）**：100% 无穿地冲击，且已证实覆盖真实策略更新 Critic 内积的 37%–57%。
- **Tier-P-safe（零穿地位姿子空间）**：支撑相仅允许 $(\delta p_{xy}, \delta\theta_{\text{yaw}}, \delta q_{12:29})$；腾空相经 8 球 FK 验算 $\pm\epsilon$ 两侧均满足 $z_{\text{ball}} - 0.005 > 0.010\text{ m}$ 时允许全位姿；所有 $\delta q$ 须经关节硬限位余量 $> \epsilon_{\max}|\delta q_j| + 0.01\text{ rad}$ 筛查。
- **Tier-P-contact（支撑相垂向/腿部位姿）**：仅作 Phase 1a 探索性诊断，不进入主判定。

### Oracle 2：前缀动作诱导的可达流形续推（测策略可达状态梯度 & Q3）
从 $S_0$ 出发以开环动作 $A_{0:H-1} \pm \epsilon d_A$（$H \in \{8, 32\}$）在真实 Simulator 中推进 $H$ 步至自然物理末端 $(S_H^{\text{sim},\pm}, o_{\text{hist},H}^\pm)$，随后无缝切换至闭环 $\pi_{\text{old}}$ 在真实 Simulator 续推 $T$ 步：
$$D_{\text{sim,cont}}^{\text{O2}}(d_A; \epsilon, H, T) = \frac{\gamma^H c_H^{\text{sim},+} V_{\text{sim}}^\pi(S_H^{\text{sim},+}; T) - \gamma^H c_H^{\text{sim},-} V_{\text{sim}}^\pi(S_H^{\text{sim},-}; T)}{2\epsilon}$$

## 4. 起点与方向集（防 clip 集中化）

- **16 个均衡起点**：从 8 个**不同运动类别**的 LAFAN1 clip（`walk`, `run`, `sprint`, `jumps`, `dance`, `fight`, `fallAndGetUp`, 及第 8 类不同受试者）各等量抽取 2 帧（任一 clip 占比恒为 12.5%）。
- **Oracle 1 方向（6 个/起点）**：`vel_critic` (1), `vel_reachable` (1), `vel_random` (2), `pose_safe_critic` (1), `pose_safe_random` (1)。
- **Oracle 2 方向（4 个/起点）**：`delta_2100` (1), `delta_2300` (1), `ascent` (1), `random` (1)。

## 5. 四层评估指标与统计纪律

所有指标仅在通过 fail-closed 门禁（基线重复范围 $\le 10^{-5}$ + 跨 $\epsilon$ 相对差 $\le \max(5\times 10^{-4}, 0.20|D|)$ + 死区过滤 + 排除单侧跌倒分叉）的样本上计算，同时报告 per-$\epsilon$ 与 pooled-$\epsilon$、**clip 聚类 bootstrap 95% CI（$B=10000$）**与 **leave-one-clip-out (LOCO)**：
1. **Level 1**：标量校准 $V_\psi(S^*)$ vs $V_{\text{WM}}^\pi(S^*;T)$ vs $V_{\text{sim}}^\pi(S^*;T)$（Pearson/Spearman/MAE/Bias）。
2. **Level 2（Oracle 1 主指标）**：`critic` vs `sim`（总误差）、`critic` vs `wm`（模型内 Sobolev 误差 A）、`wm` vs `sim`（现实鸿沟 B）各自的 Sign Agreement、Spearman、Magnitude Ratio 中位数及 IQR、RMSE。
3. **Level 3（Oracle 2 与 Q3 归因）**：对比 $D_{\text{sim,critic}}^{\text{learned}}$ vs $D_{\text{sim,cont}}^{\text{true}}$，检验 $e_{\text{state}} = D_{\text{WM,state}} - D_{\text{sim,state}}$ 与 $e_{\text{critic}}^{\text{true}} = D_{\text{WM,critic}}^{\text{learned}} - D_{\text{sim,cont}}^{\text{true}}$ 的相关性。
4. **Level 4（跨 Critic 矩阵，Q5 预检）**：用同一真实 $D_{\text{sim,cont}}$ 离线评估全部 7 个 `opcl_proto` Critic（`b0@1000/1500/2000`, `B0/B1/B2/B3@2300`）+ `critic_pre/post`。

## 6. 预注册判定树

| 结果形态（clip 聚类 CI 与 LOCO 稳健） | 裁定 | 后续方向 |
|---|---|---|
| Oracle 1 & 2 中 `critic` vs `sim` sign $\ge 0.80$、Spearman $\ge 0.70$、幅值比 $\in [0.5, 2.0]$ | **H-CV1 成立**（Critic 梯度可靠） | Critic-aware WM Learning / 自适应 Horizon |
| `critic` vs `sim` 失准（sign $< 0.65$ 或 Spearman $< 0.40$ 或幅值畸变 $> 2.5\times$），且由 **(A) 模型内误差** 主导 | **H-CV2 成立**（Sobolev 失配） | Value-Gradient-Consistent Critic Learning |
| `critic` vs `wm` 较准，但 `wm` vs `sim` 显著分歧（由 **(B) 现实鸿沟** 主导） | **H-CV3 成立**（WM 幻觉流形） | Simulator-Grounded / Model-Aware Critic Learning |
| (A) 与 (B) 均大，或 Oracle 2 显示仅在共享 Critic 下显准、无误差负相关 | **H-CV4 复合失效** | 同步解决 Critic 真实锚定与 WM–Critic 校准 |
| 主方向族 gated $n < 10$ 或覆盖 clip $< 4$ 或 `restore_state` 不可复现 | **INCONCLUSIVE** | 停止外推，排查接触/确定性/$\epsilon$ |

## 7. 分阶段预算与准入门槛

- **Phase 0（纯 CPU，0 GPU）**：全量 16 起点 × 7 Critic 模型内审计 + 无穿地方向生成 + CPU 单元测试。**准入门槛**：单元测试全过、FK 验算零穿地、确定性选点与方向落盘。
- **Phase 1a（单卡 RTX 5090，≤0.35 GPU-h，待授权）**：4-env `restore_state` 确定性烟测（要求 `repeat_range == 0.0`）+ 4 起点 $\epsilon$/$T_{\max}$ 收敛探针。
- **Phase 1b（单卡 RTX 5090，≈2.1 GPU-h，Phase 1a 通过后再授权）**：Oracle 1（~1.25 GPU-h）+ Oracle 2（~0.87 GPU-h）主采集 + 纯 CPU 全矩阵离线评估。
- **Phase 1c（可选，≤0.35 GPU-h，待授权）**：CRN 观测噪声敏感性消融。
