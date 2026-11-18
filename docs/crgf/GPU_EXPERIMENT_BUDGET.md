# GPU_EXPERIMENT_BUDGET（预算申请——未授权不执行）

日期：2026-11-18。申请范围：只读评估（无训练、无策略参数扰动、无 checkpoint 写入）。单卡 RTX 5090，与历史授权不互相抵扣。

## Phase 0：确定性烟测与批规模探针（0.5 GPU-h）

- **状态：已批准并执行完毕（2026-11-18，GPU 墙钟 ≈5.5 分钟）。verdict：4 envs 确定性逐位一致（repeat_range=0.0）；≥8 envs 撞 warp int32 workspace 上限；ε=0.03 方向导数三次完全一致并通过门禁。详见 PHASE1_EXECUTION_DECISION.md。Phase 1/2 设计已修订为"≤4 envs + 顺序分支 FD"，修订后预算 ≤2.2 GPU-h（原 4.5 的下限）。** 执行脚本已就绪：`tools/crgf_phase0_smoke.py`（单命令，内置 25 分钟硬超时、20 GB 显存熔断、批规模 2→64 递进与 ε 稳定性探针，产出 verdict.phase1_feasible 与 budget_multiplier）。

- 目的：验证 `deterministic_mode=run_to_run + disable_sensors=True` 在本仓库环境的行为；探测确定性模式下的最大稳定 batch（Codex 在 proto 环境遇到 int32 workspace 限制，8 env 失败、2 env 成功——本仓库可能不同，必须实测）。
- 内容：2/4/8/16/32/64 env × 重复基线回报一致性；一个已知 contact 重起点（复用 run1_subject2 型起点）跨 ε 的 D_sim 稳定性。
- 停止条件：若确定性模式 >4 env 不可用 → 采用 2-env 串行波次，Phase 1 预算 ×3，重新申请。
- 失败保护：所有阶段先写 budget ledger（`crgf_sim_fd_audit.py` 内建 JSONL），超时即停。

## Phase 1：B0/B1/B2 三路径 FD 试点（≈1.5 GPU-h）

- 规模：8 clips × 2 起点 = 16 起点 × 3 方向集 × 2 ε × 3 重复 × H∈{8,32}。
- 仿真 rollout 数：Σ 16×3×2×3×3分支×H = 288×3×40 = **34,560 控制步**（H=8+32）；确定性批 8 env 时 ≈ 432 个波次。
- 峰值显存：Codex 实测 driver 峰值 4.5 GB（普通）/ 10.7 GB（确定性小批）；预算上限 16 GB。
- 产出：per-(start, direction, eps) 的 D_sim(state/direct/critic) + 门禁标签；与已有 WM 侧 CSV 合并 → fidelity 全表。

## Phase 2：全协议 + H=64 + H3 有效性检验（≈2.5 GPU-h）

- 增加 H=64（三模型共享同一 D_sim，成本不变）、ascent 方向（从 WM CSV 读）、own-critic 第二轮（WM 侧 CPU 重跑即可，sim 侧无需重复）。
- H3 检验：把 fidelity 与历史单步 update pair 的 per-clip ΔJ 数据合并（纯 CPU）。
- rollout 数：Σ 16×4×2×3×3×(8+32+64) ≈ **76,000 控制步**。

## Phase 3（可选，≈1.5 GPU-h）：per-component 与状态分层

- 70 维逐分量增益（component_gain 扩展到多步）+ 按失败相位/接触状态分层。
- 触发条件：Phase 1/2 的比值分布显示明显状态依赖才执行。

## 总预算与停止条件

| 阶段 | GPU-h | 累计 | 主要停止条件 |
|---|---|---|---|
| P0 | 0.5 | 0.5 | 确定性模式不可用 → 降级串行，重新申请 |
| P1 | 1.5 | 2.0 | 门禁剔除率 >60% → 转"有限扰动响应"报告 |
| P2 | 2.5 | 4.5 | fidelity 无方向集差异（H4 否）→ 修订机制优先级 |
| P3 | 1.5 | 6.0 | — |

- 总计 **≤6 GPU-h**，峰值显存 ≤16 GB，单卡串行，不与其他任务并行抢卡。
- 每阶段产出独立落盘（CSV+JSONL ledger），任何阶段失败不回滚已完成阶段。
- 全程无参数更新；唯一写操作是 `crgf_proto/results/` 下的评估产物。

## 申请清单（供用户一键授权）

1. Phase 0 烟测（0.5 GPU-h）——先行，通过后再谈其余。
2. Phase 1 试点（1.5）。
3. Phase 2 全协议（2.5）。
4. Phase 3（可选，1.5）。
5. （独立申请，另行论证）路线 B 的定向扰动采集与等量对照训练：预计 4-6 GPU-h 训练 × 3 对照组——**不在本次申请内**。

## 追加申请（2026-11-18，第 6 次提交）：H=32/64 fidelity

基于 v2.1 实测速率（H=8: 7.6 s/波，≈0.28 s/批步）重新核算：

| 运行 | 内容 | 实测外推 |
|---|---|---|
| Run B1 | H=32 × 6 方向集（含 per-model ascent）× ε{0.03,0.1} × 2 重复 = 672 三元组 | ≈44 min |
| Run B2 | H=64 × 3 基础方向集 × 同协议 = 336 三元组 | ≈43 min |
| 合计 | | **≈87 min ≈ 1.45 GPU-h** |

停止条件不变（墙钟熔断 + 门禁 fail-closed + 部分数据落盘）。**科学目标**：H=8 的判定（H5 推翻/H4 良性）在 dynamics 占比从 3%→40–80% 后是否保持——这是"区间限定 NO-GO"能否升级为全局结论的唯一实验。
