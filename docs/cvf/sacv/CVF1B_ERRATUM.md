# CVF1B_ERRATUM（CVF-1b/2a 2026-11-20 结果勘误与修正重跑）

日期：2026-11-23。影响范围：`CVF1B_CRITIC_RESULTS.md`、`CVF1B_MECHANISM_ANALYSIS.md`、
`CVF_OVERNIGHT_RESEARCH_REPORT.md`、`CVF_NEXT_DECISION.md`、`OVERNIGHT_PROGRESS.md`
会话 3 中关于 CVF-1b 的全部数值与机制结论，以及 `results/cvf/sacv/{datasets,checkpoints,
metrics}/` 下 **不带 `v2/` 的** sacv 产物。

## 1. 结论先行

**2026-11-20 的 CVF-1b 对照实验（C-init / C0 / C1 / C2）无效。** 实现没有按预注册数据合同
评价 critic：合同规定 critic 在**前缀末端状态 x_H** 上评价（policy update 做 bootstrap 的
位置），实现却把 critic 放在了**续推结束状态 x_{H+96}** 上，而标签是为 x_H 定义的。因此
已发布报告中的组间比较、"根因分析"和"下一步建议（CVF-1b′）"都不成立，不应被引用。
Phase B 采集的原始轨迹不受影响（采集侧烟测全部通过，见 §4）。

## 2. 缺陷清单（代码审计 + 数据复核确认）

| # | 位置 | 缺陷 | 后果 |
|---|---|---|---|
| 1 | `sacv_build_dataset.py` `branch_records`、`sacv_train_critic.py` `state_tensors`、`sacv_dump_preds.py`、`sacv_eval.py` | critic 输入状态取自 `snaps_cont["cont_96"]`（x_{H+96}）而非 `snaps_prefix["prefix_32"]`（x_H） | 训练与评价都在错误的状态上进行；C0 的 WM 续推也从错误状态出发 |
| 2 | builder 的 pair 组装 | `"Delta_Y"` 字段写入的是 G⁺−G⁻（ΔG），不是 Y⁺−Y⁻ | 评价参考（stats）用的是 ΔG；`sigma_dy`（C2 的尺度）由 ΔG 而非 ΔY 计算 |
| 3 | `CVF1B_MECHANISM_ANALYSIS.md` §1 / `label_snr_decomposition.json` | "非 circular 复评"与主评价用了同一个参考向量（因 #2），不是独立验证；尾项权重用了 γ³² 而非 γ⁹⁶ | "尾项占 99.5%+ / 163–203×"**错误**。修正值：Train 上尾项方差为 ΔG 的 27–30×（delta 族）、5×（ascent）；Val/Test 上 <0.25× |
| 4 | `losses/targets.py` `survival_at` | 返回 c_{T−1} 而非 c_T | 末步终止时尾项被错误保留（影响面小） |
| 5 | builder | 所有时域的尾项都用 V_ref(x_{H+96}) | 仅影响 T=32/64 目标 |
| 6 | 训练 | 协议要求的"训练前测量 value/pair loss 数量级与梯度范数"未执行；α 尺度直接取 1/σ² | C2 的 pair 权重实为 α_eff=74.4，而非协议定义的值 |
| 7 | 协议 §1 的尺度定义本身 | "若 L_pair ≫ L_value 则除以 σ²"在逻辑上自相矛盾（除以 <1 的数会放大） | — |

## 3. 数据复核（同一批原始数据，CPU 重算，C-init）

| | 错位（已发布） | 正确对齐 x_H |
|---|---|---|
| Val 成对响应 vs ΔG₉₆：sign / Spearman / 幅度比 | 0.729 / +0.504 / 1.152 | 0.646 / +0.559 / 0.800 |
| Test 成对响应 vs ΔG₉₆：sign / Spearman / 幅度比 | 0.625 / +0.381 / 0.455 | 0.653 / **+0.119** / **0.292** |
| Test delta 族 Spearman（delta_2100 / delta_2300） | −0.02 / +0.52 | **+0.03 / −0.04** |
| Test 标量：Pearson vs Y₉₆ / RMSE | 0.846 / 1.257 | **0.969** / 0.684 |

正确对齐后在全新 clip 上独立复现了 CVF-0 的核心事实：标量价值准（Pearson 0.94–0.97），
但 policy-reachable 方向上的成对响应秩相关接近零、幅度明显欠响应。

## 4. 不受影响的部分

- Phase A 全部产物（母本核验、合同、划分、单元测试）。
- Phase B 原始采集：确定性、跨环境污染、reward 重建、G 重算等烟测均通过，
  `results/cvf/sacv/datasets/raw/` 的 11 个 batch 有效，可被修正版直接复用。
- CVF-0 的全部报告与 deep analysis（未使用 sacv 工具）。
- CVF-2a 本来就未运行（NOT RUN），但其"门禁失败"的依据是无效数字，需按修正版重新判定。

## 5. 修正版（v2）的预评估承诺

写于评估任何微调 critic 的 Val 结果**之前**（已看到的只有 Train 上的初始化测量）。

**修正内容**：critic 一律在 x_H 上评价；`Delta_G` 与 `Delta_Y` 分开存放；c_T = ∏_{k<T}(1−d_k)；
每个时域用各自的 V_ref(x_{H+T})；v1 数据集 fail-closed 拒载；新增回归测试
（`test_pairs_v2_state_label_alignment`、`test_survival_after_last_step_termination`）。
v1 产物原地保留不删除。

**初始化测量（Train，C-init，T=96）**：L_value = 0.649，L_pair = 0.099（参考 ΔY），
critic 参数梯度范数 g_value = 13.73，g_pair = 0.0265（比值 1/519）。σ(ΔY_train) = 0.319。

**α 尺度（修正协议缺陷 #6/#7）**：按 Goal 原文"先测量 loss 数量级与梯度范数，再冻结尺度
定义"，冻结为**梯度范数匹配**：`pair_scale = g_value⁰ / g_pair⁰`（Train、初始化、一次冻结），
于是 α 即 pair 项相对 value 项的梯度占比。主配置 **α = 0.3**。

**预先固定的比较**（不随结果调整）：
- 主比较：C-init vs C0@500 vs C1@500 vs C2(α=0.3)@500，在 Val 上做模型选择，
  Test 只评一次、不用于任何选择。
- 敏感性（只报告、不参与选择）：C2 的 α ∈ {0.1, 1.0}；旧尺度定义 1/σ²(ΔY) 下的 C2；
  250 / 1000 步 checkpoint。
- 评价参考：主参考 ΔG₉₆（无 learned bootstrap）；次参考 ΔY₉₆。
- 分层：ALL；无单侧终止；strict-gated（跨 ε 一致）。reachabledelta_2100/2300）。
- 额外报告：弦平均 autograd 方向导数（policy 消费的是梯度）。
- D3（WM 想象状态域）与 D4（完整 total gradient）本轮**未实现**，如实标记 NOT EVALUATED；
  因此 Phase E 门禁条件 ④ 不可能满足，Phase E 不启动。

**样本量声明**：Val 48 对 / 8 起点，Test 72 对 / 12 起点；strict-gated 层更小。
任何结论最多是 PILOT 级，不构成正式 PASS。

---

## 6. 预评估承诺的履行（2026-11-23 追记）

§5 的承诺在评估任何 Val 结果之前落盘。修正版实验已按承诺执行完毕，结果与裁定见
**[CVF1B_CRITIC_RESULTS_V2.md](CVF1B_CRITIC_RESULTS_V2.md)**（裁定：INCONCLUSIVE，
负面倾向；Sim 标量锚定改善 held-out 标量价值、方向响应无改善、C2 有害）。
承诺之外的执行细节：训练输入预计算使 1000 步训练从 ~57 分钟降到 ~2 分钟；
`glm53-flash` 子智能体、PADA-0A 与 concurrent-exploration 的核对由路线整理任务完成。
