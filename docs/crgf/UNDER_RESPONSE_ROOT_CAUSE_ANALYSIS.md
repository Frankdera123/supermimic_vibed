# UNDER_RESPONSE_ROOT_CAUSE_ANALYSIS

日期：2026-11-18。对六个竞争假设的当前证据状态。判定规则：每个假设列出支持/反对证据与"决定性实验"，不提前下结论。

## H1 训练数据动作扰动覆盖不足（可辨识性）

- **支持**：欠响应是系统性的（ratio≈0.27 稳定跨 subject/checkpoint）——与"策略数据只覆盖动作流形一维曲线、偏离方向欠定"的预测一致；确定性 policy + 平滑先验下，欠响应正是数据稀疏方向的典型解。occupancy-conditioned replay（策略分布感知的数据加权）是全仓库唯一有稳定正收益的干预（+7.8%/+16.1%）——数据分布相关。
- **反对**：欠响应是**幅值**失配而非方向混乱；若纯粹欠定，方向余弦应更差。
- **决定性实验**：路线 B（targeted action perturbation 数据采集，GPU 训练授权）——定向扰动数据 vs 等量普通数据，看 ratio 与 rollout 的分离。
- **状态**：最可能的主因之一，待决定性实验。

## H2 网络学到了系统性敏感度缩放偏差（幅值校准）

- **支持**：ratio 0.27 极其稳定；least_squares_gain ≈ 2.9-3.0 跨 subject 一致——像一个可校准的标量偏差。
- **反对**：全局 gain≈2.9 修复使 rollout 变差 8-10%（frontier L170）；局部几何 loss（Huber/counterfactual/contrastive）没能改变 ratio。
- **决定性实验**：state/action-conditioned 校准（per-dim 或状态聚类 gain）的 fidelity 响应曲线——若标量吃掉大部分 |D| 误差而 rollout 仍不变差，则 H2 成立且可用；否则 H2 只是表层描述。
- **增量证据（2026-11-18 gain 响应扫描）**：|D| 对 gain 严格线性、critic 占比对 gain 完全不变（H=8: 0.965/0.965/0.956）——标量校准在结构上不可能改变"critic 主导"的路径构成，Route A 的作用边界被定量锁定为"只修幅值"。
- **状态**：作为"表层描述"成立；作为"修复路径"已被标量版本证伪，条件化版本待测。

## H3 单步 Jacobian 基本正确、长时连乘放大误差

- **支持**：监督窗口 8 步 vs 使用窗口 32 步的错位（train.py --wm_window 8 vs --horizon 32）；frontier L443 记录的 horizon 累积敏感度与 fixed-policy return 同向退化；本轮 H=32 下 21% 起点硬终止（想象越走越偏）。
- **反对**：单步幅值欠响应本身就不是"基本正确"（3.7×），H3 的前提要弱化为"方向基本正确、幅值欠响应在长时被复合"。
- **决定性实验**：horizon 扫描 {8,32,64} 的 D_WM/D_sim 比值曲线（GPU），如果比值随 H 系统性发散 → H3。
- **状态**：很可能与 H1/H2 复合；实验设计已含。

## H4 误差主要位于 policy 不会访问的方向

- **支持**：历史审计的扰动方向是随机动作（action_scale=2）+ 随机方向，而 policy 动作集中在流形上——under-response 可能在"死方向"上无关紧要；B0 在 501 单步 pair 上 ΔJ 预测 97.5% sign agreement 说明 policy 实际走过的方向上模型是可用的。
- **反对**：方向集审计（frontier L443）显示 policy-centered 方向的累积敏感度反而最低（1.42 vs random 2.66）——policy 方向上模型"看不到"响应，与欠响应同向。
- **决定性实验**：delta/ascent 方向 vs random 方向的 fidelity 对比（GPU，本评估体系方向集已内建）。**这是本轮设计的核心增量**。
- **状态**：开放，且是判定"欠响应是否重要"的关键——若 H4 成立，修复欠响应的优先级下降。

## H5 主要误差来自 Critic Bootstrap 而非动力学

- **支持**：历史 4.1（bootstrap sign 72.5% vs 纯回报 97.5%）；**本轮定量：H=8 时 policy 梯度的 77–98% 是 critic 路径**；own-vs-shared critic 交叉显示 critic 更换改变 |D_critic| 达 4.2×，远大于 WM 分支间差异。
- **反对**：H=32 dynamics 主导（critic 占比降到 0.19–0.72）；critic 是在 WM 想象上训练的，critic 误差部分也是 WM 误差的下游。
- **决定性实验**：三路径 × (own/shared critic) × D_sim 真值分解（GPU）。若 D_critic 的 FD 真值本身信噪比高而 D_critic^WM 偏差大 → critic 训练是瓶颈（在 WM 内训练的 critic）；反之 dynamics。
- **支持（增补，θ 层面）**：policy 参数梯度的逐路径分解（pol_path_shared501_perstart.csv）：H=8 时 policy 梯度 97–99% 来自 critic 分量，且 cos(g_state, g_critic)≈0（B2 为 −0.02），6–8/14 起点为负（梯度冲突）；H=32 才部分对齐（cos 0.40–0.48）。**短时程 policy update 实际上由 critic 驱动且不听 dynamics 分量的**。
- **状态**：**头号候选，证据链已在 action 空间与 policy 参数空间两级闭合**——在得出任何"修 WM 动力学"的结论前必须先排除 critic。
- **D_sim 真值判定（H=8，2026-11-18）：H5 原始形式被推翻**——D_critic 的 fidelity 是三路径中最好的（sign 0.83–1.00、ratio 0.92–1.37、Spearman 0.81–1.00）；**H4 获得支持（良性方向）**——dynamics 路径误差集中在 random/repeated 方向（sign 0.67–0.83），policy 相关方向反而最好（0.89–1.00）。详见 EVALUATION_REPORT §6。
- 附加注意：proto 的 critic 带 V_MAX clamp（proto train.py:1144），饱和邻域梯度结构不同，分解时单独报告。

## H6 Reward 直接动作项与状态路径的误差来源不同

- **支持**：二者机制确实不同（直接项不经 WM）。
- **反对**：本轮实测策略动作下 D_direct ≈ 0（确定性策略相邻动作差太小）→ 在当前 policy 动作分布下 H6 无实际影响；随机动作协议下才可测。
- **状态**：机制上真、实证上当前不重要；保留为负对照。

## 综合工作假说（当前最优解释，可被推翻）

欠响应 = **H1（数据可辨识性）决定其存在与方向结构 + H2（平滑先验/损失形态）决定其幅值形态**；其对 policy optimization 的影响被 **H5（critic 路径主导短时梯度）** 部分掩盖——真正的传播通道在 H≥32 的 dynamics 主导区 + policy 相关方向（H4 待测）。因此修复优先级：(1) 先用 sim FD 分解确认 H5/H4；(2) 数据侧干预（H1）优先于新 loss（路线 C 前置条件）；(3) 任何 loss 类机制必须同时报告 fidelity 与 rollout，且先过"局部几何 ≠ 部署"的历史教训。

## 与 OPCL 的关系声明

本轮证据不支持"为保留 OPCL 设计而强行保留 Process/Outcome loss"：B2 相对 B1 无可判定梯度收益、想象回报略差、且非单变量对照。若最终机制是数据侧或校准侧，OPCL 的两个 loss 应按其独立贡献重新评估，而不是作为默认组件保留。

## 附：跨时域 D_sim 真值判定摘要（2026-11-18，详见 EVALUATION_REPORT §7）

- **H5 最终判定**：原始形式推翻（critic 在 H=8 与 H=32 ascent 方向最准）；新增未预料信号——delta（真实更新）方向的 D_critic 在 H=32 sign 0.33-0.67（n=3，三模型一致）→ "critic 失准"是方向依赖的，恰在真实更新方向 × dynamics 主导时域。
- **H4 最终判定**：支持并加强——H=32 random/repeated 方向进一步恶化（state sign 0.50-0.625；critic ratio 0.32-0.62）而 policy 方向保持 1.0。
- **H1-H6 综合裁定**：欠响应（单步 0.27）在 return 级 policy 方向校准消失（1.2-1.4 过响应），其威胁被 H4 的方向隔离与多步复合双重削弱；当前唯一悬而未决的威胁信号是 delta-critic@H32（待加密验证）。

## 终局：delta-critic@H32 加密验证推翻警告信号 → 全局 NO-GO 成立（2026-11-18）

加密验证（35 起点，n=15）：三模型 delta-critic sign 全部 1.000（ratio 0.81-1.13）——早先 n=3 的 0.33-0.67 为小样本假象。**预注册分支判定：全局 NO-GO 成立**（针对"欠响应经梯度通路威胁 policy optimization"这一命题；精确范围与证据链见 EVALUATION_REPORT §8）。H1-H6 终局裁定：H1 对单步欠响应成立但被 H4 方向隔离与多步复合削弱其在 policy 通路上的影响；H2 成立但无害于 policy 通路；H3 未获支持（长时程恶化仅在离策略方向）；H4 支持并加强；H5 推翻；H6 实证上无影响。
