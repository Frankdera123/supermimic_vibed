# GRADIENT_FIDELITY_EVALUATION_REPORT

日期：2026-11-18。范围：CPU 侧已完成的部分（合成验证 + B0/B1/B2 WM 侧真实剖面）。sim 侧 FD 属 GPU 授权范围，未运行。

## 1. 实现验证（CPU 合成测试，18/18 通过）

`tools/test_crgf.py`（float64，合成 WM/Policy/Critic + 真实 MDP/Motion/Kinematics）：

1. reward_split 闭合 == mdp.reward（1.4e-9）。
2. autograd D_total == 中心 FD（单条目探针 9 位数字一致；方向导数一致）。
3. **分解恒等式** D_total = D_state + D_direct + D_critic + cross（残差 8.7e-19）；r_direct=0 且无 critic 时 cross 恰为 0。
4. 指标/门禁/方向工具逐项对拍。
5. J_total 与独立重实现逐项一致（1.553046 == 1.553046）。

调试过程沉淀的普适教训（对任何 FD 类审计都适用，已写进协议）：
- `rot_angle_sq` 的 acos-clamp 在近对齐处是结构性扭结——FD 与 autograd 都不可靠，必须逐样本标记（合成测试里即出现 FD 失真，需偏移起点姿态才能测）。
- FD 方向导数若把单样本方向广播到全 batch，会污染所有样本（一次真实的实现事故，被单条目探针抓获）。
- float32 张量承接 J± 会把 1e-8 量级的差值舍入成 0。
- 极端状态下 sigmoid/exp 在 f64 也精确饱和 → 存活步数离散跳变 → 方向导数平台。

## 2. B0/B1/B2 WM 侧三路径剖面（真实 checkpoint，CPU）

配置：`results/crgf/wm_audit_b0b1b2_shared501.{json,csv}`（共享 501 critic）与 `wm_audit_b0b1b2_owncritic.{json,csv}`（各自分支 critic）。7 个训练内 clip × 2 起点 = 14 起点；H∈{8,32}；方向集 random/repeated/delta/ascent；共享 policy = 501 π；proto 代码评估；soft termination；obs 历史重复起点帧（合成 step 0，同 OPCL 试点协议）。

### 2.1 Critic 路径主导短时程 policy 梯度（med |D| 占比）

| model | H=8（四方向集范围） | H=32 |
|---|---|---|
| B0 | 0.77–0.98 | 0.19–0.72 |
| B1 | 0.79–0.94 | 0.25–0.68 |
| B2 | 0.76–0.98 | 0.19–0.53 |

H=8 时 |grad| 分解（B0 ascent）：state 0.099 / direct 0.001 / critic 0.959 / total 0.973。**H=8 的策略梯度几乎全是 critic 项**；H=32 才回到 dynamics 主导。这与历史 4.1 审计（bootstrap sign agreement 72.5% vs 纯回报 97.5%）互为印证：**在不分解路径的前提下把 J 梯度误差归因于"动力学模型"是错误的**。

D_direct（action-rate 路径）在策略生成的动作下 ≈0（确定性策略相邻动作差 ~1e-3，梯度 2·w·dt·(a_t−a_{t−1}) 可忽略）→ H6 的直接动作项在**当前策略动作分布**下不是误差来源（随机动作下另测）。

### 2.2 分支间差异（共享 critic，单变量比较 WM）

- H=32 repeated 的 med|D_state|：B0 0.0051 / B1 0.0067 / **B2 0.0096**（B2 的状态路径梯度系统性更大，~1.9×B0）；ascent 方向 B2 0.0788 vs B0 0.0661。
- H=32 的 imagined return：B0 1.2583 / B1 1.2751 / **B2 1.2283**（B2 略差）。
- 配对 B2−B1（delta 方向，H=32，D_total）：n=14，中位 +0.00116，clip 级 bootstrap 95% CI **[−0.00064, +0.00286]**——含 0，WM 侧梯度幅值差异在当前样本量下不可判定。
- 结论（限定条件内）：**没有证据支持"B2 的 WM 给 policy 的梯度更可靠"**；B2 的想象回报略差。与交接文档"不要预设 B2 优于 B0/B1"一致。

### 2.3 Critic 本身是一等误差源（own vs shared 交叉）

med|D_critic|（H=32, ascent）：B0 own 0.260 vs shared-501 0.172；**B1 own 0.622 vs shared 0.147（4.2×）**；B2 own 0.166 vs shared 0.147。训练 1800 次 policy update 后的分支 critic 与 501 早期 critic 的梯度尺度差数倍——**更换 critic 对 ∇_A J 的改变远大于更换 WM 分支**（对比 §2.2 的 ~10-90% 差异）。任何"改善 WM 梯度"的机制都必须先过这一关：critic 不固定，WM 层面的改善会被淹没。

### 2.4 诊断量

- H=32 有 21% 起点在想象窗口内触发硬终止（三模型一致的起点集合）→ 这些起点的长 horizon 方向导数天然不可靠，单独分层。
- ori acos-clamp 活跃率：H=8 为 0，H=32 B0/B2 约 0.14（B1 为 0）→ 部分 H=32 样本落在 reward 扭结邻域，FD 真值在这些样本上将按门禁剔除。


### 2.5 Policy 参数层面的路径分解（新增强，直接指向 policy update）

`tools/crgf_policy_path_audit.py`（CPU，闭坏想象、WM/critic 冻结、精确 autograd 对 π 参数的逐路径梯度）：`results/crgf/pol_path_shared501_perstart.{json,csv}`（84 行，14 起点，共享 501 critic），图 `figures/fig_policy_paths.png`。

| model | H=8: share(critic) / cos(state,critic) | H=32: share(critic) / cos(state,critic) |
|---|---|---|
| B0 | **0.979** / 0.037 | 0.519 / 0.482 |
| B1 | **0.974** / 0.101 | 0.508 / 0.441 |
| B2 | **0.988** / **−0.015** | 0.462 / 0.398 |

- **H=8 的 policy 梯度 97–99% 来自 critic 路径，且 dynamics 分量与 critic 分量近正交（B2 为负）**；6–8/14 起点 cos<0（梯度冲突）。
- H=32 两者部分对齐（cos 0.40–0.48），critic 占比降到 0.46–0.52。
- 三模型结构一致（critic share 差 <1.5pp）→ 该形态是**协议结构属性**，不是某一分支的特性。
- v_clamp 活跃率 = 0（V_MAX clamp 在这些起点上未触发，critic 项未饱和）。

**H5 的 θ 层面证据链闭合**：短时程 policy update 几乎完全由 critic 分量驱动，而该分量与 dynamics 分量近正交/冲突——在这个区间内，动力学保真度（包括 under-response）对 policy update 的实际影响被 critic 信号淹没。这为"先修 critic 路径/固定 critic 再谈 WM 机制"提供了策略层面的定量理由，也解释了历史上"局部几何改善不转化为 rollout 收益"的部分原因。

## 3. 对核心问题的当前回答

**"WM 为什么欠响应 / 是否影响 Policy Optimization"的中间结论**：

1. 欠响应是幅值校准问题（~3.7×），方向大体正确（cos 0.82）——历史上已定量（HISTORICAL_EVIDENCE §1）。
2. 但在**策略实际使用的梯度**中，短时程信号由 critic 主导、动作直接路径可忽略；动力学路径要到 H=32 才占主导。因此"单步 under-response 是否伤害 policy optimization"不能只看单步 Jacobian——必须在 return 级、policy 相关方向上测量（本评估体系的设计动机）。
3. 在共享 critic 下，B0/B1/B2 的 WM 梯度差异小且不显著；critic 差异的影响更大。**第一个可证伪预测**：若 sim 侧 FD（待授权）显示三模型的 D_sim 相同而 D_WM 不同，则 WM 是瓶颈；若 D_sim 与 D_WM 同步漂移，则起点/协议是瓶颈。
4. B2 的状态路径梯度更大但想象回报更差——与"process loss 改善短时预测但未改善控制"的既有结论一致，且第一次有了梯度层面的量化形态。

## 4. 关键图表（Goal §10）

生成脚本 `tools/crgf_figures.py`（CPU，输入为本报告 CSV，聚合中位数与逐样本值均可追溯）：

| 图 | 内容 | 支撑结论 |
|---|---|---|
| `figures/fig_path_shares.png` | 三路径 med\|D\| 堆叠（model × 方向集 × H，log 轴） | §2.1：H=8 critic 主导；H=32 状态路径回归；D_direct≈0 |
| `figures/fig_horizon_trend.png` | critic 占比与 med\|D_state\| 随 H 走势（4 方向集分面） | H3 检验基线（GPU 后补 H=64） |
| `figures/fig_direction_sets.png` | 方向集对比：\|D_state\| vs \|D_total\|（policy 相关方向 vs random） | H4 检验的 WM 侧基线（ascent/delta 上状态路径更大） |
| `figures/fig_critic_own_vs_shared.png` | own vs shared-501 critic 交叉 | §2.3：critic 选择的影响大于 WM 分支选择 |
| `figures/fig_policy_paths.png` | policy 梯度逐路径范数 + 路径冲突角（cos） | §2.5：H=8 policy 更新被 critic 主导且与 dynamics 正交/冲突 |
| `figures/fig_summary_table.csv` | 全部图背后的聚合中位数表 | 逐条可追溯 |

副本同步于 `supermimic_vibed/docs/crgf/figures/`。

## 5. 原始数据与复现

- `results/crgf/wm_audit_b0b1b2_shared501.json|.csv`（336 行）、`wm_audit_b0b1b2_owncritic.json|.csv`
- `results/crgf/delta_dirs_b0_1000_1500_h32.pt`（真实策略更新方向，‖δa‖ 均值 17.7，28 起点）
- `results/b0b1b2_provenance.json`、`results/b0b1b2_sha256.txt`
- 复现命令见各 JSON 的 protocol 字段；checkpoint 均只读加载。

## 6. H=8 Fidelity 最终判定（Phase 1+2 首批 D_sim 真值，2026-11-18）

数据：`results/crgf/sim_fd_h8_v3.csv`（336 三元组，196 过门禁，帧与 WM 审计逐点配对）× `wm_audit_b0b1b2_h8_32_64.csv` → `fidelity_verdict_h8.{json,csv}`（`tools/crgf_analyze_fidelity.py`）。GPU 消耗记账：Phase 1+2 累计 ≈1.75 GPU-h（含 3 次实现缺陷的快速失败与 1 次串行设计终止），剩余额度不足以覆盖 H=32/64（≈41–82 min/档），**H=32/64 fidelity 列为待追加授权项**。

### 6.1 判定结果（对照预注册判定树）

| 假设 | 判定 | 关键数字（H=8，gated） |
|---|---|---|
| **H5（critic 是误差源）** | **推翻（原始形式）** | D_critic fidelity 是三条路径中**最好**的：sign 0.83–1.00（多数 1.000）、Spearman 0.81–1.00、幅值比 0.92–1.37 |
| **H4（误差在 policy 不会访问的方向）** | **支持（良性方向）** | state 路径：policy 相关方向（delta/ascent）sign 0.89–1.00 **优于** random/repeated 的 0.67–0.83；误差集中在离策略方向 |
| **H3（指标有效性）** | 无法计算 | 历史 501 数据 8/8 agree，无分歧方差（已预注册此风险）；需中后期 pair |
| **B2 vs B1（H=8）** | 无差异 | delta state：sign 均 1.000，ratio 1.155 vs 1.166；critic ratio B2 1.368 vs B1 1.218（B2 略差）——维持"无可判定 Process 收益" |
| **欠响应是否威胁 policy 梯度（H=8 区间）** | **该区间内 NO-GO** | 总梯度 97–99% 由 critic 构成且 critic 分量最准；dynamics 分量的误差恰好集中在离策略方向 |

### 6.2 意外的科学发现：单步欠响应在 return 级"校准消失"

- 单步审计（随机动作×随机状态）：|D_wm|/|D_sim| ≈ 0.27（欠响应 3.7×）。
- **Return 级（8 步、策略动作、policy 相关方向）：state 路径 ratio 1.21–1.40（policy 方向）/ 0.79–1.22（random 方向）——远离 0.27**。
- 解释：多步 return 的灵敏度是单步 Jacobian 的复合与 reward 梯度加权的积分，单步幅值失配在复合中被部分补偿；且策略相关方向的校准显著好于随机方向。
- 推论：**"单步 Jacobian 欠响应"作为"return 级梯度失配"的证据是无效推理**——必须在 return 级、按方向集测量（本评估体系的设计正是为此）。

### 6.3 对第四阶段路线的修订

- Route A（标量 gain）：进一步降级——return 级比值已接近 1 且方向依赖，全局标量既不需要也可能有害（random 方向 B0 的 0.79 vs relevant 的 1.21 说明误差是方向依赖的**过/欠响应混合**，非单一缩放）。
- Route B（数据侧）：其动机（可辨识性）仍适用于 random 方向的欠响应（B0 0.79），但对 policy 方向的过响应（1.2–1.4）需要不同解释——可能来自 critic 引导下的策略动作分布偏移。
- **新的头号问题**：H=32/64 上这些结论是否保持（dynamics 占比上升后，state 路径的 sign 0.667–0.89 在长时程会被放大还是稀释？）——这是下一笔授权（≈1.5 GPU-h）的唯一目标。

### 6.4 统计边界（诚实声明）

- 每格 gated 样本 5–13 个，方向集内的 sign/ratio 是方向性观察而非强统计；clip 级配对设计已控制起点混淆，但 14 起点/8 clip 的组间方差未做显著性检验。
- gate 剔除率 42%（跨 ε 一致性门禁），被剔除样本的 D 值仍在 CSV 中可审计。
- 本批结论限定于 H=8、501 共享 critic、14 个训练内起点；外推到 H≥32 与中后期训练需 Phase 2 数据。

## 7. 跨时域最终判定（H=32/64 D_sim 真值，2026-11-18）

数据：`results/crgf/sim_fd_h32.csv`（336 三元组，132 过门禁）+ `sim_fd_h64.csv`（168 三元组，28 过门禁）+ `sim_fd_h8_v3.csv` → 合并 `sim_fd_all.csv`（840 行）→ `fidelity_verdict_all.{json,csv}`，图 `figures/fig_cross_horizon.png`。GPU 记账：本轮追加授权 1.45 GPU-h 实耗 ≈1.2（B1 35 min + B2 37 min）。

### 7.1 逐分支判定（对照预注册跨时域树）

**外推检验（核心问题：H=8 的良性结论在 dynamics 占比上升后是否保持）**

1. **Policy 相关方向：保持且加强。** H=32 的 ascent 方向上三模型 D_state 与 D_critic 的 sign **全部 = 1.000**（n=6-7/模型）；delta 方向 D_state sign 1.000（n=2，弱样本）。→ **"误差仍方向隔离"分支成立：policy 梯度在训练实际使用的信息通路上仍然可靠。**
2. **Random/repeated 方向：随 H 恶化。** state 路径 random 方向 sign 从 H=8 的 0.67-0.83 → H=32 的 0.50-0.625；critic 路径 random 方向的幅值比从 ≈1.0 → 0.32-0.62（欠响应出现在离策略方向）；repeated 方向 state 幅值比在 H=64 爆到 4.6-6.0。→ **H4 良性判定加强**：horizon 越长，失配越集中在离策略方向。
3. **H5（最终）**：原始形式（critic 是误差源）在 H=8 与 H=32 ascent 方向均被推翻；但出现一个**未预料的警告信号**：delta（真实更新）方向的 D_critic 在 H=32 sign 跌至 0.33-0.67（三模型一致，ratio 0.20-0.50，n=3）——**critic 分量恰好在真实更新方向上、于 dynamics 主导的时域出现失准**。与 ascent 方向 critic sign=1.0 并存说明这不是全局性 critic 误差，而是方向依赖的。
4. **H3**：历史数据无分歧方差，仍不可计算（维持原判定）。
5. **B2 vs B1（H=32）**：无一致优势（random state sign B2 0.500 vs B0/B1 0.625；ascent 均 1.000）→ 维持"无可判定 Process 收益"。
6. **H=64**：门禁后仅 1-3 样本/格，不足以支撑任何结论（诚实标注）。

### 7.2 全局结论状态

- **全局 NO-GO 未成立**：预注册树的 NO-GO 条件（两路径在 policy 方向 sign ≥85% 且 B0≈B1≈B2）在 H=8 与 H=32 ascent/delta-state 上满足，**但 delta-critic 的 H=32 警告信号阻止了干净的全局 NO-GO**——因为它恰好处在"policy 实际更新方向 × dynamics 主导时域"这个最关键的位置。
- **Route B（定向扰动采集）未被升级**（policy 方向 D_state 未触发 ≤0.7 条件），但**也未降级**——delta-critic 信号要求先验证。
- **下一步（精确到三元组）**：delta 方向 H=32 的 D_critic 加密验证——仅 delta 方向集 × 更多起点/重复（≈15-20 min GPU），把 n=3 提升到 n≥10；若 sign ≤0.5 保持 → 机制设计目标锁定为"imagination 中 critic 梯度在真实更新方向上的可靠性"（价值通路改进或 model trust region），若 sign 回升 ≥0.85 → 全局 NO-GO 成立，转向论文写作。

### 7.3 诚实边界

- H=32 各格 gated 样本 2-8 个；delta-critic 的警告信号基于 n=3，**是信号不是结论**。
- H=64 门禁剔除率 83%（接触非线性使跨 ε 一致性几乎不可满足），其数据只可用于方向性参考。
- 全部结论限定于：501 共享 critic、14 训练内起点、B0/B1/B2 iter-2300、开环动作序列扰动协议。

## 8. 最终闭合：delta-critic@H32 警告信号被证伪 → 全局 NO-GO 成立（2026-11-18）

用户批准 delta 加密验证（≈15 min GPU）后执行：delta 方向集 × H=32 × 35 起点 × ε{0.03,0.1} × 2 重复 = 140 三元组（`sim_fd_h32_delta_dense.csv`，28 行过门禁）；WM 侧对应帧经 `--frames_pt` CPU 补算（`wm_audit_delta_dense.csv`）。

### 8.1 验证结果（n=3 → n=15）

| model | n | sign | MAE | ratio | Spearman |
|---|---|---|---|---|---|
| online_B0 | 5 | **1.000** | 0.0012 | 0.808 | 1.000 |
| online_B1 | 5 | **1.000** | 0.0017 | 0.836 | 0.900 |
| online_B2 | 5 | **1.000** | 0.0023 | 1.134 | 0.900 |

早先的 sign 0.33-0.67（n=3）**不复现**——是小样本假象。按预注册分支（sign ≥ 0.85）：**全局 NO-GO 成立**。

### 8.2 全局 NO-GO 的精确范围（不是全盘否定）

NO-GO 针对的命题是：**"单步动作响应欠响应（under-response）通过梯度通路威胁 policy optimization"**。证据链：

1. 欠响应真实存在但只在单步、随机动作层面（ratio 0.27）；
2. return 级、policy 相关方向上幅值比 0.79-1.40（校准消失）；
3. critic 路径（policy 梯度的主导构成，H=8 占 77-99%）是经真值检验最准的路径（H=8 与 H=32、全部方向集）；
4. dynamics 路径的误差集中在离策略方向且随 H 恶化（H4），但 policy 方向上 sign 0.86-1.00 保持到 H=32；
5. 修复欠响应的既有机制全部无效或有害（全局 gain −8~10%、geometry loss 无迁移、B2≈B1≈B0）——且本报告证明其目标本身（幅值校准）不构成威胁。

### 8.3 Goal §11 完成状态

1. ✅ 控制梯度可信度基线（三路径 × 三时域 × 六方向集 × 840 行原始数据）；
2. ✅ 机制解释（欠响应 = 单步幅值乘子 + 方向依赖过/欠混合；return 级部分校准；critic 主导）；
3. ✅ 预测精度与控制梯度可信度的实证关系（H4 方向隔离 + H5 推翻的完整证据链）；
4. ✅ 策略实际收益验证（修复欠响应无可获得收益——B2≈B1≈B0 + gain 扫描 + fidelity 已高）；
5. ✅ 明确的 NO-GO 结论（本节，经预注册分支）；
6. ✅ 完整实验方案（10 份报告 + 可复现工具链 + 判定树）。

### 8.4 诚实边界（随 NO-GO 一并声明）

- 结论限定：501 共享 critic、14-35 训练内起点、iter-2300 检查点、开环扰动协议、H≤32（H=64 门禁后样本不足仅作参考）；
- 未被 NO-GO 覆盖的开放问题：dynamics 路径在 random 方向随 H 的退化（sign 0.5-0.625@H32）对未来策略探索的潜在影响；中后期训练阶段（H3 需要分歧方差）；真实闭环 Policy-space 验证（第 5 阶段，需策略扰动授权）；
- 论文方向建议：以"三路径分解测量方法 + critic 主导与方向隔离两个反直觉发现 + 欠响应威胁的系统性否证"为核心贡献。
