# UNDER_RESPONSE_HISTORICAL_EVIDENCE

日期：2026-11-18。本报告回答：**已有证据究竟支持什么**。原始数据均可追溯（文件路径+数字）。

## 1. 核心量化事实：单步动作响应幅值低估 ~3.7×，方向基本正确

原始数据（主仓库）：`logs/true_action_jacobian_compare_s4.json`、`_s5.json`、`true_action_jacobian_gain_s4/s5.json`；工具 `tools/true_action_jacobian_audit.py`。

协议：随机基动作（std=2）+ 随机单位方向 d，δ=0.5，同一起点在 Isaac 中并行执行 a±δd 与 a（单步、共享 reset、无恢复问题），70 维 frame-local 状态空间中的响应向量。

| 指标 | s4 (batch64, seed17) | s5 (batch64, seed17) |
|---|---|---|
| true_sensitivity_mean（仿真器） | 0.472 | 0.481 |
| WM baseline `loso4_polyupdates2_s1_1000` | ratio **0.266**, ls_gain 2.98, cos 0.828 (p10 0.683) | ratio **0.272**, ls_gain 2.91, cos 0.808 |
| action_cf（counterfactual 训练） | ratio 0.274 | ratio 0.281 |
| huber05 | ratio 0.226 | — |

**判读**：
- WM 单步动作响应幅值 ≈ 仿真器的 27%（欠响应系数 ~3.7），方向余弦 0.81–0.83。
- 用 counterfactual/Huber 等 geometry 机制训练后 ratio 基本不动（0.23–0.28）——**现有 geometry loss 没有修复幅值失配**。
- 该数字解释了 train.py:1557 的历史注释 "systematic WM under-response to local action perturbations"。

## 2. 全局 gain 校准已试过并失败（不再重复）

`docs/current_frontier.md:170`：

- 新增 `--wm_action_effect_gain`，按仿真器最小二乘 gain（≈2.9）放大想象动作；subject4 seed1 smoke survival 上升，但 1000 步 rollout 相对 baseline **clean/perturb −8.38%/−10.09%**，胜出 1/4 与 0/4。
- 结论（frontier 原文）："全局 action gain 与真实 policy action 语义错位，关闭 scalar calibration；后续若继续只能做 state/action-conditioned calibration。"
- 机制补充（本审计 SOURCE_AUDIT §4）：gain 只作用于 WM 输入动作，观测/reward/BC anchor 仍用原始动作 → 想象闭环内部语义错位。

## 3. "局部几何改善 ≠ 部署改善" 已被多次独立确认

`docs/current_frontier.md`：

- L168：Huber checkpoint 的 action-Jacobian 均值 2.850→2.998，但 rollout 下降；"局部 action sensitivity 不是充分的 deployment proxy"。
- L385（geometry gate 校准）：Gram/cosine 门禁候选全过，但 fixed-policy 500 步 return 更差；"vector geometry 是必要诊断而非充分提升门禁"。
- L439：真实的 on-policy 反事实几何数据直接注入 WM 训练 **决定性变差**（height failure 84.4%），分支关闭；此类数据只做诊断。
- L137-139：action-recovery（逆动力学辅助）在 subject5 上 +25%，但 seed 复核失效（+0.16%）；"可迁移 action geometry + task-compatible gradient" 是收益条件，但高方差。
- L443：多方向类型审计（random/policy-centered/burn-in on-policy）后结论——"现有证据支持 over-smoothing / loss-induced dynamics distortion，而非 value-vs-replay 梯度冲突"。
- 当前最强候选（frontier 头部）：**occupancy-conditioned WM replay**（数据侧、策略分布感知的采样加权），subject4 三 seed +7.8%/+16.1%——与"欠响应源于数据可辨识性"假设同向。

## 4. OPCL 线的 GPU 试点（交接文档之后已完成，状态 INCONCLUSIVE）

位置：`~/Documents/Codex/2026-11-14/new-chat-2/outputs/`：
`opcl_control_gradient_evaluation_report.md`、`opcl_control_fidelity_reproducibility_fix.md`、`opcl_b0_b1_b2_control_fidelity_comparison.md`、`opcl_control_fidelity_results/`。

- 已做：B0/B1/B2 WM + B1 policy，8 个训练内 clip，H=8/32，纯折扣 return（无 critic），两方向（joint 0 单关节/连续 8 帧），ε=0.01/0.03，各 3 重复。GPU 137.8 s（后修复累计 2396.5/5400 GPU 秒）。
- **质量门禁未过**：run1_subject2 同基准动作跨调用回报跳变 0.007283（除以 2ε = 0.364 的导数污染）；部分条件 ε 反号；显式状态恢复 ≠ 接触求解器内部状态恢复（NewtonCfg.deterministic_mode 默认 not_guaranteed）。
- 修复（已验证 3 起点）：`deterministic_mode='run_to_run'` + `MJWarpSolverCfg.disable_sensors=True`；跨进程/顺序差异归零；但 fallAndGetUp1 H32 连续方向在确定性模式下仍随 ε 反号（+.147/−.183/+.037），fail-closed 门禁按规则剔除。
- 描述性数字（明确不可用于排名）：H8 每-clip MAE ≈0.031（三模型几乎相同），H32 ≈0.09–0.10；方向可判定率 H8 25-29/31，H32 14-18/32。
- **B2 与 B1 非单变量对照**（comparison 报告 [FACT]）：B1 = 额外采样数据 + 额外动力学 loss；B2 = Process loss，数据、损失、后续策略都可能变化。

## 5. 综合判断：已有证据支持的结论

1. 欠响应存在、系统性强、单步幅值 ~3.7×，方向大体正确 → **主要误差形态是幅值/增益失配，而非方向混乱**。
2. 幅值失配**不能**用全局标量 gain 修复（已失败）；也不能被现有 geometry loss 修复。
3. 局部几何指标的改善与 rollout/控制收益解耦——**"更好的局部 Jacobian"本身不构成提升机制的充分证据**；任何新机制必须同时报 rollout。
4. 数据侧（occupancy/可辨识性）是目前唯一有正收益信号的干预方向。
5. 仿真器端方向导数测量的可行性受接触求解器非确定性限制；确定性模式修复有效但有残余 ε 敏感性，fail-closed 门禁必须前置。
6. 尚无任何证据表明欠响应**通过哪条梯度路径、在哪些方向上**影响 Policy Optimization——这正是本轮 CRGF 评估体系要补的洞（见 EXPERIMENT_PROTOCOL 与 EVALUATION_REPORT）。
