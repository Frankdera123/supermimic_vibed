# PHASE1_EXECUTION_DECISION（Phase 0 烟测结果与 Phase 1/2 执行决定）

日期：2026-11-18。Phase 0 已获批并执行完毕（GPU 墙钟 ≈ 5.5 分钟，含 4 次调试重启；远低于 0.5 GPU-h 授权）。容器 `crgf_smoke` 已停止释放 GPU（保留供 Phase 1 复用）。

## 1. Phase 0 verdict（原始数据：`results/crgf/phase0_smoke.{json,budget.jsonl}`）

| 探针 | 结果 | 判读 |
|---|---|---|
| B=2（4 envs）重复性 | **repeat_range = 0.0**（逐位一致） | `run_to_run` 确定性模式在本仓库栈有效；顺序重放不受执行顺序污染（独立复现 Codex 修复结论） |
| B=4→64（8-128 envs） | 全部失败：warp `launch_deterministic` int32 workspace 上限（3.3e9→33e9 字节，线性缩放） | **确定性模式硬上限 = 4 envs/context**；与 Codex 在 proto 环境的独立发现（2 成功/8 失败）互证 |
| ε=0.01 方向导数（fallAndGetUp1 起点，H=16，3 分支并行） | D=9.83e-5，三次重复完全一致，但 < 5e-4 死区 | 被门禁正确排除（fail-closed 生效） |
| ε=0.03 同起点 | D=−3.31e-3，三次重复**完全一致**，通过门禁 | **FD 完全可重复**；|D| 超死区 6.6× |
| 显存 | 峰值 1.15 GB（4 envs） | 远低于 20 GB 熔断线 |

烟测脚本 verdict 的 `phase1_feasible=false` 来自其原始判定条件（要求 ≥8 envs，沿用了并行分支设计假设）——已被修正为顺序分支设计判定（≥3 envs + ≥1 档 ε 通过 → **可行**）。修正仅改脚本记录，未重跑 GPU。

## 2. 设计修订（烟测的直接产出）

原 `crgf_sim_fd_audit.py` 的并行分支设计（3×wave_triplets envs）在确定性模式下不可行。**修订设计（v2）**：

- 每个 SimulationContext ≤4 envs；每个 env **顺序**执行一个三元组的 plus/minus/base 三分支（reset 间隔），确定性模式保证跨顺序重放精确。
- 成本模型（实测）：场景创建+4 envs 一波 ≈ 25 s + rollout。Phase 1 全规模 288 三元组 → 72 波 × ~35 s（H=32）≈ **45 分钟**；H∈{8,32} 总计 ≈ 1–1.5 GPU-h。
- ε 集合 {0.01, 0.03} → 增加 **0.1** 档：烟测显示 ε=0.01 信号落在死区（9.8e-5 < 5e-4）且与 ε=0.03 反号（非线性/接触切换区），三档让门禁数据自行筛选有效区间。

## 3. Phase 1/2 执行决定（建议）

**GO**，按修订设计执行，预算如下（等价于原预算 × multiplier 2.0，仍以烟测实测重新核算）：

| 阶段 | 内容 | 修订后 GPU-h |
|---|---|---|
| Phase 1 | B0/B1/B2 共享起点、16 起点 × 3 方向集 × 3 ε × 3 重复 × H∈{8,32}（D_sim 模型无关，一次跑完服务三模型） | ~1.2 |
| Phase 2 | +H=64、ascent 方向、H3 有效性检验合并 | ~1.0 |
| 合计 | | **≤2.2 GPU-h**（原申请 4.5 的下限，因顺序设计取消了并行波次冗余） |

停止条件沿用：门禁剔除率 >60% → 转有限扰动响应报告；任一波次 rollout 崩溃 → 跳过该三元组并记录；总时长超 2.2 GPU-h 自动停。

**所需授权**：本次批准仅覆盖 Phase 0（已用 ~0.1 GPU-h）。Phase 1+2 需按上表追加授权（建议 2.5 GPU-h 封顶含调试余量）。

## 4. 烟测的方法论教训（已写入协议）

1. 确定性模式的 env 上限使"并行分支"设计失效——顺序分支 + 确定性重放是唯一可行路径，且已被两个独立环境验证。
2. warp 的 int32 workspace 上限是线性于 env 数的硬约束，任何 Isaac/Newton 升级都可能改变它——Phase 1 运行前必须重跑本烟测（几分钟）确认。
3. ε=0.01 与 ε=0.03 在接触敏感起点上反号：方向导数在 contact 边界附近不是稳定量，per-start 的 ε 一致性门禁不可省略。
4. 容器化运行路径：`crgf_smoke` 容器（镜像 supermimic:latest，挂载 supermimic_vibed×2 + crgf_proto + opcl_work:ro + repo data 嵌套挂载进 proto data）——Phase 1 复用同一容器。

## 5. 预注册判定树（Phase 1/2 数据落地后的机械化解释规则；先于数据写下以防事后拟合）

记号：对每个 (model, H, dirset, path)，fidelity = gated sign agreement；主判定单位 = clip。

**H5 判定（critic vs dynamics 哪个是瓶颈）**
- 若 D_critic 的 gated sign agreement（对 D_sim^critic 真值）显著低于 D_state 的（差 ≥10pp，CI 不含 0）→ **critic 路径是主要误差源**：机制优先级转向 imagination 中的价值学习（λ-return / 缩短 bootstrap / 信任门控），Route B/C 降级。
- 若两者相当且都差（sign < 70%）→ WM 状态 Jacobian 是共同瓶颈 → Route B（数据侧）优先。
- 若两者都好（sign ≥ 85%）且 B0≈B1≈B2 → WM 在本协议下已经可靠 → **"欠响应影响 policy optimization"为 NO-GO**（在该区间内），研究问题转向 critic/探索其他区间。

**H4 判定（误差是否在 policy 相关方向）**
- 若 delta/ascent 方向的 fidelity 系统性低于 random（差 ≥10pp）→ 误差落在 policy 可达方向 → 欠响应是实际威胁 → Route A/B 升级。
- 若 delta/ascent ≈ random → 偏差方向无关 → H4 支持，欠响应优先级下降。

**H3 判定（指标有效性）**
- per-clip fidelity 与历史 per-clip ΔJ 一致性的关联（point-biserial）：|r| ≥ 0.4 → 指标有效；< 0.1 → fidelity 指标本身不可用作机制优化目标 → 先修指标再谈机制。
- 已知限制：历史 501 更新 40 clip 中 39 agree（本轮 H3-lite 预览确认无分歧方差），H3 主检验将用 1000/1500/2000 中后期 pair 或 Phase 2 新数据。

**B2 vs B1 判定**
- 仅当 B2 在 (a) D_state sign agreement、(b) H≥32 存续性、(c) 幅值比值回正 三条同时成立时判"Process loss 改善控制相关保真度"；否则维持"无可判定收益"。

**Route A 收官条件（本轮 gain 扫描已部分达成）**
- 已证：标量 gain 只修幅值、路径构成不变 → Route A 单独不足以解决 H5 型问题；若 Phase 1/2 显示 H4 成立（误差在 policy 方向）且 D_state fidelity 低 → Route B（targeted perturbation 采集）为下一优先，提案见 CONTROL_SENSITIVE_LEARNING_PROPOSAL.md。

## 6. 预注册判定树·跨时域分支（H=32/64 数据落地后追加）

- **外推检验（核心）**：H=8 的两条良性结论（critic 路径最准、dynamics 误差偏离策略方向）在 H=32（dynamics 占比 ~40–70%）与 H=64（~80%）是否保持。
  - 若 H=32/64 的 D_state sign（random 方向）跌至 ≤0.5 且 policy 方向保持 ≥0.85 → 误差仍方向隔离，但 dynamics 主导区的整体梯度质量取决于方向混合 → 结论：policy 梯度在训练实际使用的信息通路上仍然可靠；Route B 降级为可选。
  - 若 policy 方向的 D_state sign 也跌至 ≤0.7 → **误差侵入策略通路** → H4 良性判定被推翻，Route B（定向扰动采集）升级为首选机制。
  - 若 D_critic fidelity 在 H=32/64 依然 ≈1 → "imagination 价值信号可靠" 成立，且结合 H=8 的占比证据，形成论文核心论点：**model-based policy gradient 的可靠性由价值通路主导，动力学幅值失配（含 under-response）被多步复合部分校准**。
- **H3**：以 Phase 2 数据 + 1000/1500/2000 中后期 pair 的合并方差重试 point-biserial。
- **Route B/C 决定**：严格按上述分支执行，不再新增机制提案。
