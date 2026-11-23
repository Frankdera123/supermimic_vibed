# CVF1A_BUDGET_PLAN（预算与吞吐计划，预注册降档阶梯）

日期：2026-11-20。全部估算基于 CVF-0 实测 ledger（`results/cvf/phase1b/*.budget.jsonl`），
并在 Phase B smoke 后以实测吞吐重估。上限≠承诺耗时；预算用不完不补实验。

## 1. 实测吞吐基准（CVF-0，单卡 RTX 5090）

| 来源 | 规模 | wall time | 吞吐 |
|---|---|---:|---|
| Oracle 2（H32 prefix + T96 cont，B=4 envs） | 16 起点 × 18 rollouts × 128 transitions = 36,864 env_transitions | 3,042 s | **12.1 transitions/s** |
| Oracle 1（状态扰动 16-step prefix + T 续推） | 16 起点 × 多方向 | 3,883 s | 参考值 |

口径：`env_transitions` = 环境步数（每环境每步 1 个）；`simulator_batch_steps` =
`env.step` 次数 × B_env。单环境同步 step 下两者相等，两组数字分开累计、交叉核对
（历史 4× 口径错误不得重现）。

## 2. Phase B 采集规模与降档阶梯

基础规模（44 起点 = Train 24 + Val 8 + Test 12）：

| 配置 | 每起点 rollouts | 每 rollout transitions | 总 transitions | B=4 估时 | B=8 估时(×0.6) |
|---|---:|---:|---:|---:|---:|
| R0 全量：4 族 × 2ε × ± + 2 base | 18 | 128 | 101,376 | 2.33 h | 1.40 h |
| R2 二级族单 ε（ascent/random 仅 ε=0.015） | 14 | 128 | 78,848 | 1.81 h | 1.09 h |
| R3 T_cont=64（branch 96 步） | 18 | 96 | 76,032 | 1.74 h | 1.05 h |

- 烟测（≤0.25 GPU-h）：确定性（B∈{4,8,16} 的 repeat_range、批/单样本一致性）、
  state restore、o_hist 落盘、reward 重建、survival indexing、吞吐实测。
- **执行规则（冻结）**：烟测选出"通过确定性检查的最大 B"；
  ① 若 B=8 通过且吞吐 ≥1.5× → R0（预算内，优先全量）；
  ② 否则按序尝试 R2 → R3；仍超 2.2 GPU-h 预估 → Train 缩至 10 clips（20 起点）并标注 pilot；
  ③ T=128 扩展仅当采集完成后推算全量 ≤2.2 GPU-h（数据合同 §4.4）。
- 每 batch 落盘 + ledger 行 + 预算检查；wall fuse 每 batch 边界生效。

## 3. 各阶段预算

| 阶段 | 上限 | 计划 | 依据 |
|---|---:|---:|---|
| Phase A CPU readiness | 0 GPU-h | 0 | 本文档；全部 CPU |
| Phase B 采集 | 2.5 | 1.1–2.6（阶梯） | §2 |
| Phase C Critic pilot | 1.5 | ≈0.4：C0 目标生成（WM 384 rollout ≈0.05 h）+ 3 组 × ≤1000 步训练（分钟级）+ Val 探针 | Critic 938→256→128 极小 |
| Phase D 评价（额外 GPU） | 1.5（与 E 共享） | ≈0.15：D3 的 WM rollout + 必要的补充 sim 查询；D1/D2/D4 纯 CPU | 存量数据复用 |
| Phase E 单次 Policy Update | ↑ | ≈0.6：imagination rollout（WM，cheap）+ 5 变体 × 20 起点 × 96 步 sim eval = 9,600 transitions ≈0.22 h（B=4）+ E3 条性 10 步重复 ≈0.3 | §E 设计 |
| Phase F 预批准补充 | 0.5 | seed 复验 / α 换档 / anchor 对照，按需 | 协议 §1/§2 |
| **总计** | **6.0** | 预期 2.4–3.9 | — |

## 4. Ledger 与停机规则

- `results/cvf/sacv/budget/budget_ledger.jsonl`：每行 `{ts, phase, task_id, command, seed,
  env_transitions, simulator_batch_steps, gpu_hours_delta, vram_peak_mb, exit_status,
  artifacts_sha256}`。GPU 任务记录 start/end timestamp、CUDA device、数据规模、exit status。
- 阶段门禁：Phase B 完成 → 核对独立 split 完整性 + reliability report → 才进 Phase C；
  Phase C 通过标准（协议 D6）满足才进 E；任一门禁失败立即停机记录，不跨门禁。
- 到达任一上限（GPU-h / wall-clock 8h / 资源安全边界）：保存状态、写 OVERNIGHT_PROGRESS、停止。
- 异常停机条件：NaN、OOM、持续无效梯度、原始资产 SHA256 变化、评价合同不一致
  （→ 有限 CPU 诊断后再决定）。
- 不使用其他用户/项目占用的 GPU（当前 RTX 5090 空闲，3.2/32 GB 已用，0% util，
  快照时间 2026-11-20 02:13 前后）。
