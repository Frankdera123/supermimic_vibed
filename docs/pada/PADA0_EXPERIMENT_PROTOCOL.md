# PADA0_EXPERIMENT_PROTOCOL（预注册）

日期：2026-11-18。本文件在**任何 PADA-0 数据采集之前**写定并落盘。判定标准、指标定义、样本量、种子与预算全部预注册；实验后不得回头修改。唯一允许的协议偏离：执行层面失败后的**样本量削减**（只允许减少，不允许更换指标或判定方向），任何偏离须在 RESULTS 中逐条列出。

## 0. 研究问题（唯一）

> 在相同真实 simulator query budget 下，policy-directed probes 是否比 random probes 更高效地改善 World Model 对 policy-relevant dynamics 的建模？

## 1. 实验对象与 provenance

| 对象 | 取值 | 来源 |
|---|---|---|
| M0（原始 WM） | online_B0 diag pair @ iter 2100 的 `wm` | `server_artifacts/online_B0/diag_pairs/pair_it_002100.pt`，加载时记录 SHA256 |
| 训练 policy pair | 同 bundle 的 `pi_old` / `pi_new`（pi_updates 1599→1600，`VALID_SINGLE_POLICY_UPDATE`） | 同上 |
| 代码变体 | OPCL proto（WM 35 维加速度级、OBS 93、goal 状态相对） | `--proto_root /workspace/opcl_work/supermimic_opcl_proto` |
| critic（Level-3 D_critic 用） | 同 bundle 的 `critic`（= post-update） | 同上 |
| motion | `data/lafan1_g1`（train clips），xi 固定 nominal（friction=1，offsets=0），eval_mode | 与 CRGF 一致 |

pair 使用前置检查（fail-closed，逐 pair）：`same_policy_update_verified==true`、`pi_normalizer_changed==false`、strict state-dict 加载、SHA256 记录。任何检查失败 → 弃用该 pair 并在 RESULTS 声明。

## 2. 公平性约束（不可违反）

Random 臂与 Policy 臂**唯一允许的差异是 probe direction**：

- 相同起点（12 个 probe starts，两臂逐一相同）；
- 相同 base action（π_old 在 M0 中闭环 rollout 的 H=32 动作序列，逐 start 相同）；
- 相同扰动幅值 ε_probe=1.0（单位化方向的统一缩放，见 §4）；
- 相同 probe 数量（每 start 8 个方向 × plus/minus 两分支）；
- 相同 H=32、相同 sim 配置（确定性模式、4 envs、xi nominal）、相同种子集合；
- 微调协议逐项相同（init=M0、lr、步数、batch、窗口、损失、采样索引流由同种子驱动、快照点）；
- 相同盲测集（eval 数据两臂共用，采集中两臂均不可见其内容——eval 方向集在采集前固定）。

## 3. 方向集定义

方向均为 (H×29) 展平后 L2 单位化的动作序列方向。

### 3.1 训练 probe 方向（每臂 12 starts × 8 dirs = 96）

- **random（臂 A）**：`random_dir_seqs`（per-step 模式，seed=2301+start_idx×8+j），逐 (start, j) 独立。
- **policy-shift（臂 B）**：`δa = π_new(o) − π_old(o)`，shared-obs 序列由 π_old 在 M0 中闭环 rollout 生成（`crgf_delta_action_directions` 口径）。生成点：每 probe start 所属 clip 内取 8 个不同 frame（与 probe frame 不同、margin 内），每个生成点 1 个方向，归属到同 clip 的 probe start。open-loop 口径，如实标注。
- 若某生成点的 δa 范数 < 1e-6（退化），弃用并顺延 frame，记录弃用数。

### 3.2 盲测方向集（在 6 个 held-out eval starts 上评估，clips 与 probe starts 不相交）

| 集合 | 来源 | 方向数/start | 角色 |
|---|---|---|---|
| `rand_test` | 全新种子（seed=2401...）random 方向 | 4 | 控制组（两臂都不该偏袒） |
| `pol_seen` | §3.1 训练 policy 方向中 seed 抽取 4 个 | 4 | Test A：训练方向（held-out starts） |
| `pol_unseen_B` | online_B0 pair @ **2300**（同 run，+200 updates） | 4 | Test B：同 checkpoint 后续单步 update |
| `pol_unseen_C` | online_B2 pair @ **2100**（跨 run，OPCL 机制） | 4 | Test C：不同机制 run 的单步 update |

unseen 集的生成点：各自 pair 的 teacher-forced delta，在 eval clips 上生成（每个 eval clip 4 个 frame），方向向量固定后跨全部 eval starts 重复使用（与 CRGF `get_dirs` 同口径）。

### 3.3 关键对照（防作弊设计）

- Test A vs Test B/C 的差 = memorization 分量。PASS 条件要求 unseen 上仍成立（§7）。
- `rand_test` 上的 M1 vs M2 检验"policy probes 是否以牺牲其他方向为代价"。
- probe 数据（ε_probe=1.0 的两臂轨迹）本身作为 Test A 的第二形式（训练点原位评估，仅描述性，不进判定）。

## 4. Probe 与 eval 采集协议

### 4.1 Probe 采集（训练数据，两臂各一份）

- starts：12 个 = 6 clips × 2 frames（seed=2300 确定性抽取，64-frame margin，CRGF 口径）。
- base action：π_old 在 M0 闭环 H=32（noise-free obs，`base_action_rollout`）。
- 分支：`a+ = base + ε_probe·d`，`a− = base − ε_probe·d`，ε_probe=1.0，H=32。
- 记录：每步 (s, s1, a, a_prev, f, r_state, r_direct, soft_term, hard_done)，s1 为 71 维 flatten。
- 数据量/臂：12×8×2×32 = **6144 转移**；sim 成本 12×8×2×32 = 6144 batch-steps。

### 4.2 Eval 采集（共享盲测真值，一次采集，模型无关）

- starts：6 个 held-out = 3 clips × 2 frames（同法，seed=2300，clip 池与 probe 不相交）。
- base：同 §4.1 口径在 eval starts 上 rollout，**2 次重复**（门禁基线重复性）。
- 方向分支：4 方向集 × 6 starts × 4 dirs × ε∈{0.03, 0.1} × ±分支，H=32，1 rep。
- 记录：逐步全轨迹 + reward 分量（H=8 的 return FD 由前 8 步折扣部分和离线计算，H=32 用全程）。
- sim 成本：4×6×4×2×2×32 = 12288 + 基线 384 batch-steps。

### 4.3 查询预算记账

sim query 以 batch-step 计（每 batch-step = 4 envs × 1 控制步）。两臂 probe 预算严格相等（6144 batch-steps/臂）。eval 为共享真值，不计入单臂。全部采集写 `results/pada/*.budget.jsonl` ledger（沿用 CRGF 事件格式）。采集脚本墙钟熔断 75 min/会话。

## 5. M1/M2 微调协议

| 项 | 取值 |
|---|---|
| init | M0（两臂同一权重副本） |
| 数据 | 各臂 probe 转移，8-step teacher-forced 窗口（起点均匀于轨迹内） |
| 窗口池大小 | 12×8×2×(32−8+1) = 4800 窗口/臂 |
| loss | `z = wm.targets(Ŝ, S1) / wm.out_norm.std()`（35 维，normalizer 冻结），mean(z²)，无 component 权重 |
| optimizer | Adam lr=1e-4，(0.9, 0.999)，无 weight decay |
| 步数 | 1500 步，batch=1024 窗口 |
| 采样索引流 | 同种子（seed=2500）驱动两臂 → 同 index 序列作用于同大小池 |
| 快照 | 每 250 步 + 最终（M1_250…M1_1500，M2 同） |
| normalizer | in_norm/out_norm 冻结（不 update） |

等级敏感性：主判定用 1500 步快照；500/1000 步快照用于稳定性检查（PASS 条件 2）。全部微调在 GPU（<0.5 GPU-h），两臂同卡串行。

## 6. 指标定义（四层）

记号：模型 M∈{M0, M1, M2}；测试方向集 T∈{rand_test, pol_seen, pol_unseen_B, pol_unseen_C}；sim 真值只在 eval 采集上定义。

### Level 1 普通 prediction（每模型，不区分方向集另有标注）

- L1a one-step：eval base 轨迹全部转移的 teacher-forced 一步误差，报告 35 维归一化 RMSE（`targets` 空间）与 pose 级误差（root pos/rot、q、qd，Bryson 权重 `weighted_sq_error` 口径）。
- L1b H=8 / H=32：eval starts 从真实初始状态沿 base 动作 free-running 误差（逐步记录，报告 step≤8 与 step≤32 均值）。

### Level 2 方向性 state 响应（每模型 × 每方向集）

对 (start, dir, ε)：sim 响应 `Δs_sim(t) = s±(t) − s∓(t)`（逐步），WM 响应 `Δs_wm(t)` 由 M 对 a± 开环 rollout。逐步配对后聚合 step∈[1,8] 与 [1,32]：

- cosine(Δs_wm, Δs_sim)（逐步后平均与轨迹级聚合两种）；
- magnitude ratio ‖Δs_wm‖/‖Δs_sim‖（中位数）；
- relative error ‖Δs_wm−Δs_sim‖/‖Δs_sim‖；
- sign agreement（按 root 高度通道 + 71 维逐通道两个口径）；
- RMSE。
门禁：采用 CRGF `reliability_gate` 阈值，作用于 **Level-3 return FD**（见下）；Level-2 的门禁用同一 (start,dir,ε) 的 Level-3 gate 标签（fail-closed：无 gate 通过则该样本不计入）。

### Level 3 control-relevant gradient fidelity（CRGF 复用，主指标所在层）

- D_sim(path)：return 级中心差分 `[J(A+εd) − J(A−εd)]/(2ε)`，path∈{state, direct, critic, total}，J 定义同 CRGF（γ=0.99，soft-term 折扣）；H=8 与 H=32 两个时域；critic 路径用真实 sim 末状态 × pair critic。
- D_WM(path)：`open_loop_paths` + `path_directional_derivatives`（精确 autograd）。
- 指标（gated only）：sign agreement、magnitude ratio（中位）、spearman、RMSE——`fidelity_stats` 逐 (M, T, H, path) 报告。
- 门禁：`reliability_gate`，基线重复 = base×2，D 池化 = 2 个 ε。gate per (start, dir)，跨 path 共享同一标签（与 CRGF 一致以 state path 门禁为主标签）。

### Level 4 data efficiency（核心比较）

固定预算 B（两臂各 6144 转移、6144 sim batch-steps）：

- 主对照：ΔFidelity(M2 vs M0) vs ΔFidelity(M1 vs M0) 在每个 (T⊆policy 集, H) 上的差；
- 效率比：`ΔFidelity / sim_batch_steps`（两臂 B 相等 → 退化为 ΔFidelity 比较，比值仅用于 RESULTS 表述）；
- 步数曲线：M1/M2 快照 500/1000/1500 的 fidelity 轨迹（同 B，不同微调节拍）。

**主指标（判定用，预注册）**：Level-3 gated sign agreement of D_total @ H=32，辅以 Level-3 gated sign of D_state @ H=32 与 Level-2 gated cosine @ [1,32]。三者报告，判定顺序为主指标优先，主指标 INCONCLUSIVE 时用辅指标标注降级（不得升级）。

## 7. 预注册判定（PADA-0）

记 `S(M,T)` = 主指标（gated sign, D_total@H32）；配对单位 = (start, dir)；检验 = 配对 bootstrap（B=10000 重采样 starts×dirs）95% CI + 双侧符号检验。

### PASS —— 需同时满足全部 5 条：

1. **优势存在**：S(M2,T) − S(M1,T) > 0 且 CI 不含 0，T = pol_seen；
2. **稳定性**：上述优势在 ≥2 个微调快照（500/1000/1500 中任二）保持符号；且 M2 的 L1a one-step 误差 ≤ 2× M0（无灾难性遗忘）；
3. **样本量充分**：每个 policy 测试集 gated n ≥ 8（否则该集降级 INCONCLUSIVE）；
4. **unseen 迁移**：S(M2,T) − S(M1,T) > 0 在 T = pol_unseen_B **或** pol_unseen_C 上成立（CI 不含 0；两集至少一集）；
5. **可分离性**：Level-1 改善方向与 control-relevant fidelity 改善方向**不完全等价**——即存在至少一个对照使两者符号不同（例：M2 在 fidelity 上胜 M1 而 L1b@H32 无改善，或 M1 在 L1 上改善 ≥ M2 而 fidelity 反向）。

### INCONCLUSIVE —— 出现任意一条：

- policy 集 gated n < 8 或 gate 通过率 < 50%；
- 两臂微调均发散（L1a 误差 > 5× M0）或均无变化（|Δfidelity| 全集 < 0.01 且 CI 全含 0）；
- sim 噪声导致 D 池化不可靠（gate 失败原因占比 > 60% 且集中于 d_range）；
- 任一 PASS 条件的数据不足以判定（缺 pair、缺方向、采集中断且无法在预算内重跑）。

### NO-GO —— 同时满足：

- 数据充分（每集 gated n ≥ 8、gate 通过率 ≥ 50%）；
- S(M2,T) ≤ S(M1,T) 在 pol_seen **且**两个 unseen 集上成立（至少不劣）；
- 配对 CI 排除 "M2 > M1"（单侧）在上述全集成立；
- 快照曲线不支持任何时点反转。

→ NO-GO 则停止 PADA，不进入正式训练，不堆 directional loss。

## 8. 预算与执行顺序

| 步骤 | 预算 | 产物 |
|---|---|---|
| 0 CPU 合成测试 | 0 | test_pada 全绿 |
| 1 GPU sanity（1 triplet 端到端 + gate 冒烟） | ≤10 min | sanity verdict |
| 2 方向集生成（CPU/GPU WM rollout） | ≤15 min | dirs `.pt` + provenance |
| 3 probe+eval 采集（单次 sim 会话） | ≈2.1 GPU-h | 轨迹/转移 + ledger |
| 4 微调 M1/M2 + 快照 | ≤0.5 GPU-h | 快照序列 |
| 5 WM 侧评估 + 离线指标 | ≤0.5 GPU-h | CSV/JSON |
| 合计 | **≤3.7 GPU-h（硬上限 4.0）** | |

超预算处理：预计超出即停止并报告，不允许自行扩大。单卡 RTX 5090，串行。

## 9. 研究纪律（重申，判定时逐条自检）

- 不把 shared-obs delta 声称成 closed-loop 验证；
- 不把同方向 train/test 当 generalization（Test A/B/C 分离）;
- 不把更多样本的收益说成 direction 选择收益（两臂严格等量）；
- 不把 Level-1 MSE 改善说成 control 改善（Level-3/2 才算）;
- 不把相关性说成 policy performance 因果收益（PADA-0 完全不测 policy return）；
- 结果为负 → 明确 NO-GO；结果为正但证据不足 → INCONCLUSIVE，不硬凑 PASS。
