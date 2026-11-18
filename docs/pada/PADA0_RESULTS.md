# PADA0_RESULTS

日期：2026-11-18。协议：`PADA0_EXPERIMENT_PROTOCOL.md`（预注册于任何数据采集之前，未改动判定标准）。审计：`PADA_SOURCE_AUDIT.md`。原始数据与全部 SHA256：`results/pada/`。

## 判定（预注册树）：**INCONCLUSIVE**

（PASS 被条件 3/4 阻断；NO-GO 不成立。依据见 §6，逐条对照见 §7。）

## 1. 实际使用的真实 policy pairs

全部来自 OPCL 训练管线自产自验的 `diag_pairs`（`VALID_SINGLE_POLICY_UPDATE`，`pi_updates N→N+1`，`pi_normalizer_changed=false`，strict 加载通过，SHA256 在 `results/pada/provenance.json`）：

| 角色 | pair | pi_updates | SHA256(前8) |
|---|---|---|---|
| M0 + 训练方向 | **online_B0 @2100** | 1599→1600 | b951c51e |
| Test B（同 run 后续单步 update） | online_B0 @2300 | 1799→1800 | 70ff7577 |
| Test C（跨 run / OPCL 机制） | online_B2 @2100 | 1599→1600 | 7640f3be |

未使用任何 synthetic pair（15 个真实 pair 可用，无需制造）。所有 policy-shift 方向为 **open-loop shared-observation delta**（π_old 在 M0 闭环生成的 obs 序列上 teacher-forced 差分），如实标注，未声称 closed-loop 验证。相邻单步 delta 的 teacher-forced 幅值 ≈ per-step |δa| 0.51 vs |a| 9.39（≈5.4%，审计 §5.4），非噪声级。

## 2. 公平性核验（两臂唯一差异 = direction）

- 起点：12 个 probe starts（6 clips × 2 frames）两臂逐一相同；eval 6 个 held-out starts（3 clips，与 probe clips 不相交：fight1_s3 / sprint1_s2 / walk3_s5）。
- base actions：π_old 在 M0 闭环 H=32，两臂与全部 eval 集共用。
- ε_probe = 1.0（单位化方向）；ε_eval ∈ {0.03, 0.1}（两臂共用同一盲测真值）。
- probe 数：每臂 12×8×2 分支 × H32 = **6144 batch-steps/臂**（严格相等）；eval 共 396 分支 = 12672 batch-steps（共享）。
- sim 配置：确定性模式、4 envs、xi nominal、eval_mode、reset_noise=False（与 CRGF Phase 0/1/2 完全一致）。
- 微调：init=M0、Adam lr 1e-4、1500 步、batch 1024×8 步窗口、同种子索引流（窗口池两臂各 4800，相等，断言在代码中）、normalizer 冻结。22/22 CPU 合成测试通过（`pada_proto/tools/test_pada.py`），含公平性不变量与配对 schedule 一致性。

## 3. Simulator budget

| 项 | 数值 |
|---|---|
| 采集会话 | 195 waves，24960 batch-steps（=99840 env-steps），1810 s ≈ 0.5 GPU-h |
| 总 GPU（含 sanity、方向生成、微调、评估×2） | **1.0 GPU-h / 上限 4.0** |
| 门禁（fail-closed，sim-only，模型无关） | H32 样本级 20/96 通过（75 d_range 失败、1 deadband）；两时域并集 35/96 |

执行偏差声明（协议 §0 允许的执行层面修正，不涉指标/判定）：
1. 评估第一次运行因 30 min 熔断设置过紧被主动终止（结果丢弃），重启时把主判定模型排前、熔断放宽到 80 min；所有模型用同一最终运行评估。
2. 评估工具修复了 3 处实现 bug（维度切片、动态 horizon、重复计数），修复后以 smoke 数据回归验证后再跑正式评估；正式数据未受这些 bug 影响（首次运行产物已丢弃）。

## 4. 主要结果：Level-3 gated sign agreement（D_total）

| 方向集 | H | M0 | M1_500 | M1_1000 | M1_1500 | M2_500 | M2_1000 | M2_1500 | n |
|---|---|---|---|---|---|---|---|---|---|
| eval_random（控制组） | 32 | 0.33 | 0.33 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 | 3 |
| eval_pol_seen（Test A） | 32 | 0.67 | 0.44 | 0.33 | 0.22 | 0.56 | 0.67 | 0.67 | 9 |
| eval_pol_B（unseen 同run） | 32 | 0.50 | 0.25 | 0.00 | 0.25 | 1.00 | 0.75 | 0.75 | 4 |
| eval_pol_C（unseen 跨run） | 32 | 1.00 | 0.50 | 0.25 | 0.50 | 1.00 | 1.00 | 1.00 | 4 |

配对 bootstrap（M2_1500 − M1_1500，95% CI，B=10000）：

| 方向集 @H32 | mean diff | CI95 | n |
|---|---|---|---|
| eval_pol_seen | **+0.444** | **[+0.111, +0.778]**（不含 0） | 9 |
| eval_pol_B | +0.500 | [0, 1]（含 0） | 4 |
| eval_pol_C | +0.500 | [0, 1]（含 0） | 4 |
| eval_random（控制） | −0.667 | [−1, 0] | 3 |

D_state、D_critic、Level-3 H=8 全表见 `results/pada/eval/eval_summary.json`（关键形态：H8 上两臂差异方向一致但幅度小且不显著；D_state 上 M0 普遍 1.00，微调普遍降低）。

## 5. Level-1 / Level-2 / 数据效率

**Level-1（普通 prediction，两者皆受损）**：

| 指标 | M0 | M1_1500 | M2_1500 |
|---|---|---|---|
| one-step eval base RMSE | 0.259 | 0.362 | 0.368 |
| free-running H=8 RMSE | 0.442 | 0.675 | 0.765 |
| free-running H=32 RMSE | 0.373 | 0.777 | 0.912 |
| one-step 自臂 probe RMSE | 0.437 | 0.049 | 0.043 |

两臂微调均以 6144 条 probe 转移为唯一数据（无 anchor 数据可用——原始 replay buffer 未随 checkpoint 保存，审计 §6.1），把各自 probe 数据记忆到 RMSE<0.1，同时全局 prediction 与对 M0 已有的 fidelity 均受损（灾难性遗忘）。M2 的 L1a = 1.42×M0（PASS 条件 2 的 ≤2× 约束满足，但这是"没坏到阈值"，不是"没坏"）。

**Level-2（state FD，ε=0.1，gated）**：M0 轨迹级 cosine 全面最高（0.48–0.92）；policy 集上 M2 略高于 M1（H8: B 0.457>0.378、C 0.569>0.499；H32: seen 0.365≈0.363、B 0.408 vs 0.421（M2 略低）、C 0.425>0.390）；71 维逐通道 sign agreement 三模型 ~0.83–0.88 无分离。方向性 state 响应层面的优势远小于 return 级。

**Level-4（等预算效率）**：两臂 B 相等（6144/6144 batch-steps），ΔFidelity(M2−M1)@H32 policy 三集方向一致为正且跨快照稳定（§4 表）；但 ΔLevel-1 为负（两者都受损，M2 略更差）。"每查询保真收益"的符号取决于指标层——这正是协议 PASS 条件 5（可分离性）所预言的形态，已在数据中出现。

## 6. 为什么是 INCONCLUSIVE（而不是 PASS 或 NO-GO）

**支持 M2 的证据（方向一致）**：
1. 主指标（D_total sign @H32）上 M2 > M1 在全部 3 个 policy 方向集成立（+0.44/+0.50/+0.50），且在 500/1000/1500 三个微调快照上全部保持；
2. 优势迁移到 unseen policy directions（Test B 同 run 后续 update、Test C 跨 run）——不是仅记忆训练方向；
3. pol_seen 的配对 CI 不含 0；
4. 控制组（eval_random）上 M2 不占优甚至更差 → M2 的收益特异于 policy-relevant 方向族，与"方向选择"机制一致，而不是全局微调效应（若是全局效应，随机集也应受益）。

**阻断 PASS 的证据（预注册条件 3/4 未满足）**：
1. H32 门禁后样本量不足：pol_B n=4、pol_C n=4（要求 ≥8）；gate 通过率 20.8%（H32 样本级）< 50%，失败主因 d_range（ε∈{0.03,0.1} 间 D 估计不稳定）；
2. unseen 集的配对 CI 含 0（n=4 无统计力）；
3. Level-1 的灾难性遗忘（两臂）意味着"probe 收益"是在"全局退化背景"下测得的净效应——即使统计显著，也无法直接解读为正式训练场景下的收益（replay buffer 缺失限制，审计 §6.1 已预声明）。

**NO-GO 不成立**：M2>M1 在主指标上方向一致、跨快照稳定、seen 集 CI 排除 0——数据不满足 NO-GO 所要求的"充分样本下 M2≤M1"。

## 7. 预注册判定逐条对照

| 条件 | 结果 | 判定 |
|---|---|---|
| PASS-1 pol_seen 主指标 CI 不含 0 | +0.444, CI [0.111, 0.778] | ✓ |
| PASS-2 ≥2 快照符号稳定 & L1a ≤2×M0 | 3 快照全正；1.42× | ✓ |
| PASS-3 每集 gated n≥8 | pol_B/pol_C H32 n=4 | ✗ |
| PASS-4 unseen 迁移 CI 不含 0 | CI 含 0（n=4） | ✗ |
| PASS-5 Level-1 与 fidelity 方向不完全等价 | L1 降、fidelity 升（M2>M1） | ✓ |
| INCONCLUSIVE 任一 | policy 集 gated n<8；gate<50% | ✓ 触发 |
| NO-GO 全部（含数据充分前提） | 前提不满足 | ✗ |

**最终：PADA-0 = INCONCLUSIVE。** 不进入正式训练。

## 8. 若要得到可判定答案，需要的最小追加（需授权，未执行）

1. **PADA-0b（纯评估加密，≈0.6 GPU-h）**：eval 方向 4→10/集、starts 6→12、ε 增至 {0.03, 0.1, 0.2}，把 H32 每集 gated n 提到 ≥10；probe/微调完全不动。判定仍按本协议 §7。
2. **锚定微调（≈0.3 GPU-h，协议修订）**：微调数据加入小比例 M0 自身生成数据作 anchor（两臂同量），消除灾难性遗忘背景，使"净收益"可直接解读。需要正式修订 §5 并说明理由。
3. 若 0b 后 CI 仍不含 0 且方向为正 → 才值得谈 PADA-Data 的最小可行版本；若方向翻转或归零 → NO-GO 收线，转论文写作（CRGF 线的 delivery 已齐）。

## 9. 产物索引

- `pada_proto/`：bridge + 5 工具 + 22 项 CPU 测试（全绿）+ README（复现命令）
- `results/pada/`：`directions_meta.json`、`probes.pt`、`eval_data.pt`、`collect.budget.jsonl`、`ckpts/M_*.pt`（10 快照）、`eval_rows.csv`（644 行）、`eval_summary.json`、`level1.json`、`provenance.json`（全部 SHA256）、`gpu_usage_log.json`
- 原始 policy pair：`~/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts/*/diag_pairs/`（容器内 `/workspace/opcl_work/...`）
