# CONTROL_SENSITIVE_LEARNING_PROPOSAL

日期：2026-11-18。候选机制提案（按证据排序，不是菜单堆砌）。每条注明：机制定义、前置条件、预期信号、证伪条件、与历史已证伪方向的差异。

## 前置共识（来自历史证据，不可绕过）

1. 局部几何改善 ≠ 部署改善（三次独立证伪：Huber、Gram 门禁候选、counterfactual 注入）。
2. 全局标量 gain 修复已失败（rollout −8~10%）。
3. 短时程 policy 梯度由 critic 主导（本轮 H=8 占比 77–98%）——**任何 WM 动力学机制若不同时处理/固定 critic，效果会被淹没**。
4. B2 vs B1 非单变量；历史 OPCL loss 无已证收益。

因此提案的顺序是：**先测量与归因（P0，GPU 授权后）→ 数据侧（P1）→ 条件化校准（P2）→ 最后才是新 loss（P3）**。


## 路线 A 增量结论（2026-11-18 gain 响应扫描，CPU，`results/crgf/gain_sweep_summary.json` + `figures/fig_gain_sweep.png`）

WM 侧开环等价 gain ∈ {1.0, 2.0, 2.9}（WM 输入动作缩放、reward 保持原始动作——与 train.py `wm_action_effect_gain` 语义一致）：

1. **|D| 对 gain 严格线性**（H=8 delta：D_state 0.0018→0.0036→0.0053；D_critic 0.0126→0.0252→0.0361）——欠响应在 return 级同样是纯幅值乘子。
2. **路径构成对 gain 不变**：H=8 的 critic 占比 0.965/0.965/0.956——标量校准同等放大两条路径，**无论 gain 多准都无法让 dynamics 路径在短时程与 critic 竞争**。
3. 三模型响应曲线重合（±10%）——再次确认分支差异小于结构差异。

**Route A 判定更新**：标量 gain 校准只能修幅值（且历史实验已证明全局修幅值在 rollout 上 −8~10%），**不能修路径构成**。若 Phase 1/2 的 D_sim 真值确认 H5（critic 路径误差主导），则 Route A 单独不足以改善控制相关保真度——机制设计必须同时处理 imagination 中的价值信号（或绕开它：数据侧 Route B）。

## P0 测量包（无训练，纯授权后评估）

- 三路径 × {own, shared critic} × {random, repeated, delta, ascent} × H∈{8,32,64} 的 D_WM vs D_sim。
- 判定：(a) critic vs dynamics 的误差占比与符号一致性；(b) H4 检验——delta/ascent 方向的 fidelity 是否系统性不同于 random；(c) H3 检验——比值随 H 的走势；(d) H3'——per-clip fidelity 是否预测 per-clip ΔJ agreement（对接历史 501/1000/1500/2000 配对数据）。
- 证伪条件：若门禁剔除率 > 60%（sim 端不可测），本路线降级为"有限扰动响应"报告（Codex 试点建议），P1/P2 直接基于单步协议继续。

## P1 数据侧：Targeted Action Perturbation（首选机制）

- 定义：采集时对当前策略动作注入小幅定向扰动（ε ~ 0.5·action_scale·unit，覆盖单关节与低维组合），使 Δa→Δs′ 真实可观测；数据量与训练预算严格等量对照。
- 与历史"counterfactual 注入"的差异：历史失败是把 FD 数据**作为 WM 训练目标**注入（L439，损失形态问题：84.4% height failure）；P1 只改**采集分布**，WM 训练目标仍是普通 L_dyn。
- 对照组：B0 原始采集 / 无差别随机动作扰动 / 定向局部扰动（敏感度选择版可选）。
- 预期信号：ratio 0.27→显著上升 + rollout 不变差 + fidelity 提升。
- 证伪：ratio 改善但 rollout 不变差 → 回到"几何改善不部署"怪圈，记录 NO-GO。

## P2 条件化校准：wm_action_effect_gain 的可学习版本

- 定义：gain(s, j) = MLP_head(s_features, j) 或状态聚类标量，仅作用于想象 WM 输入动作（与现有 wm_action_effect_gain 同位），**但同步修正观测/奖励侧的动作语义错位**（历史失败主因），或将 gain 实现为 WM 输出端的等效缩放（a_model 语义保持一致）。
- 先跑 gain 响应曲线（纯评估，无训练）：gain ∈ {1, 1.5, 2, 2.9, 4} × {全局, per-dim Oracle} 的 fidelity 与（授权后）rollout——若 oracle per-dim 校准显著改善 fidelity 而 rollout 不改善，说明幅值不是瓶颈，P2 NO-GO。
- 证伪条件同 P1。

## P3 Control-sensitive Learning（最后）

- 候选形式：L_control = E_d[Huber(D_WM(d) − D_sim(d))]，但 (a) 只在 curated 方向集（δa + 高敏感状态，用 `_rollout_sensitivity` 筛）上算；(b) D_sim 标签离线缓存（复用 counterfactual 数据管线）；(c) 与"等成本额外 L_dyn"对照（P1 数据的 loss 版）。
- **不建议的变体**：全局方向集硬上（FD 噪声淹没）；不加 rollout 门禁的任何版本；把 sign agreement 当唯一目标（幅值信息丢失）。
- 前置条件：P0 显示 dynamics 路径（非 critic）在 policy 相关方向上有稳健失配，且 P1/P2 不足。

## Critic 路径的处理（横切所有提案）

- 若 P0 证实 critic 主导且 critic 误差大：优先动作是**价值学习改进**（λ-return 结构、缩短 bootstrap horizon、detached value target），而不是动 L_dyn；本仓库已有 `policy_wm_horizon_decay`、`policy_wm_reliability_quantile` 等机制可复用。
- 评估协议必须固定 critic（shared-501）才能看清 WM 层差异——这已在本轮执行。

## 与论文叙事的衔接

若 P0/P1 成立，论文核心主张可以立住："预测精度与控制相关保真度可分离；我们给出三路径分解的测量方法，指出短时程策略梯度被 critic 项主导这一被普遍忽略的事实，并用数据侧干预（而非 loss 工程）改善控制相关保真度。" 若 P0 显示 critic 路径问题不可修，则研究问题本身升级为" imagination 中的价值估计误差如何主导 model-based policy gradient"——同样可发表，但与 OPCL 无关，需用户确认转向。
