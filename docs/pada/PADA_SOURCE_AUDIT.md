# PADA_SOURCE_AUDIT

日期：2026-11-18。审计目的：为 PADA-0（Policy-Shift Data Efficiency Audit）盘点 `dev` 分支现有 CRGF 工具链与数据资产，明确"可直接复用 / 已有数据可复用 / 数据缺失 / 必须新增"四类清单。所有结论给出代码位置或文件路径，无法确认的一律标注 **UNKNOWN**，不允许猜测。

## 0. 结论速览

| 问题 | 答案 |
|---|---|
| 真实相邻单步 policy pair 是否存在？ | **是**。15 个，全部经过训练管线自身验证（`VALID_SINGLE_POLICY_UPDATE`），非 synthetic |
| PADA-0 是否需要新造 synthetic pair？ | **不需要** |
| 已有 delta 方向资产能否直接当 PADA 训练方向？ | 不能直接用主资产：`delta_dirs_b0_1000_1500_h32.pt` 来自相距 **500 iteration** 的 pair（非相邻），只能作参考/额外测试集 |
| WM 微调所需原始 replay buffer 是否存在？ | **不存在**（checkpoint 不含 buffer）→ M1/M2 定义为 M0+probe 微调，见 §6 缺口 1 |
| probe 采集是否需要写新 sim 工具？ | 是（记录轨迹/转移的工具不存在），但完全复用 CRGF 的 sim 协议与核心库，见 §3 |

## 1. 可直接复用的代码

### 1.1 CRGF 核心（`crgf_proto/supermimic/crgf.py`，421 行）

- `Protocol`（crgf.py:54）：双代码变体（主仓库 / OPCL proto）运行时适配——obs 签名、goal 签名、`wm_step` kwargs、critic 输入。**PADA 必须用 proto 变体**（见 §4），此适配器直接复用。
- `open_loop_paths`（crgf.py:146）：WM 开环 rollout + 三路径（state/direct/critic）图构建 → Level-3 的 WM 侧 D_state/D_critic/D_total 直接复用。
- `path_directional_derivatives`（crgf.py:240）：精确 autograd 方向导数。
- `random_dir_seqs` / `normalize_dir_seqs` / `delta_dir_seqs`（crgf.py:269-296）：random/delta 方向集生成。
- `fidelity_stats`（crgf.py:331）：gated sign/mae/rmse/ratio/spearman → Level-2/3 指标直接复用。
- `reliability_gate`（crgf.py:360）：fail-closed FD 门禁（基线重复范围 + D 池化范围 + deadband）。
- `base_action_rollout`（crgf.py:392）：base action 序列（π 闭环于 WM）→ probe/eval 共用。
- `reward_split` / `soft_term`（crgf.py:97-114）：与训练一致的 reward 分解。

### 1.2 CRGF 工具

- `crgf_proto/tools/crgf_delta_action_directions.py`：真实 policy pair → shared-obs sequence → `δa = π_new(o) − π_old(o)` 方向场。PADA 的 policy-shift 方向生成**直接以此为基础**（输入换成 diag pair 的 `pi_old`/`pi_new`）。
- `crgf_proto/tools/crgf_sim_fd_audit.py`（v2 顺序分支设计）：确定性 sim（≤4 envs）+ plus/minus/base 顺序分支执行 + 墙钟熔断 + budget ledger。PADA probe/eval 采集工具**沿用该模式**，但需要新写（现有工具只记录 return 标量，不记录轨迹/转移，见 §3.4）。
- `crgf_proto/tools/_crgf_bootstrap.py`：isaaclab stub + MJCF 路径修复，直接复用。
- `crgf_proto/tools/crgf_wm_directional_audit.py`：WM 侧审计逻辑，PADA 以库函数方式调用 crgf.py，不重复实现。

### 1.3 proto 变体包（`~/Documents/Codex/2026-11-14/new-chat-2/work/supermimic_opcl_proto`，容器内 `/workspace/opcl_work/supermimic_opcl_proto`）

diag pair 的 π/WM 与该变体匹配，必须经 `--proto_root` 加载：

- WM = **35 维加速度级输出**（dv, dw, dqd；proto networks.py:53 `WM_OUT = 3+3+29`），位姿半隐式积分；与主仓库 70 维全增量 WM 不同（CRGF 源审计 §5）。
- `OBS_DIM = 93`（无 6D 参考相对姿态）；goal = 状态相对 `(actor, wm_goal)`，`GOAL_FRAMES=10` 硬编码。
- proto `WorldModel.targets(S,S1)` 给出网络输出空间的回归目标；`weighted_sq_error`（Bryson 权重）可用于 Level-1。
- 两种变体差异表：见 `docs/crgf/GRADIENT_FIDELITY_SOURCE_AUDIT.md` §5（已核实，不再重复）。

### 1.4 WM 训练损失形态（`crgf_proto/supermimic/train.py`）

- `wm_update`（train.py:1782）核心 = teacher-forced rollout + 归一化误差（`_wm_err_weighted`，train.py:1392：`z = state_error/std`，std=`out_norm.std()`，floor 由 `ERR_FLOOR`）。
- PADA 微调**镜像此损失形态**（proto 变体下用 `targets` 空间归一化），但只含核心项——所有 aux 机制（prioritized sampling、realism discriminator、consistency risk 等数十项）不进入 PADA-0，且两臂严格一致，不影响内部有效性。

## 2. 可直接复用的数据资产

### 2.1 真实相邻单步 policy pair（PADA 的核心输入）★

位置：`~/Documents/Codex/2026-11-14/new-chat-2/work/opcl_autonomous_20261117/server_artifacts/{online_B0,online_B1,online_B2,online_B3,b0_resume}/diag_pairs/pair_it_00{2100,2200,2300|1000,1500,2000}.pt`（容器内路径 `/workspace/opcl_work/opcl_autonomous_20261117/server_artifacts/...`）。

| run | 性质 | pair 迭代 | pi_updates 区间 | 验证状态 |
|---|---|---|---|---|
| online_B0 | baseline（无 OPCL/B1 机制） | 2100/2200/2300 | 1599→1600 / 1699→1700 / 1799→1800 | `VALID_SINGLE_POLICY_UPDATE` |
| online_B1 | compute-matched 对照 | 同上 | 同上 | 同上 |
| online_B2 | OPCL run | 同上 | 同上 | 同上 |
| online_B3 | OPCL 变体 | 同上 | 同上 | 同上 |
| b0_resume | 共同祖先 run | 1000/1500/2000 | 499→500 / 999→1000 / 1499→1500 | 同上 |

- 四个 online run 全部 resume 自 `b0_resume_2000.pt`（`args.json: resume`，四个 run 一致），即 2100 处的 pair 是"同一祖先 + 各自 100 次 update 后"的相邻单步。
- 每个 pair bundle 内含：`pi_old`、`pi_new`（**恰好一次 policy update 前后**）、`wm`（同迭代 WM）、`critic_pre`/`critic_post`、normalizers、完整 `operation_order`（`collect → wm_update → snapshot_pre_policy_update → policy_update_single_step → critic_update → snapshot_post_policy_update`）、组件级 SHA256（`component_sha256`）、`same_policy_update_verified=true`、`pi_normalizer_changed=false`（B2@2100 实测）。
- **合计 15 个真实单步 pair，足够覆盖：训练 pair（1）+ Test B（同 run 后续 update）+ Test C（跨 run / 更早训练阶段）**。
- 加载验证：已用 proto `Policy(93, 730, hist=10)` 与 proto `WorldModel()` strict 加载 pair_it_002100 成功（2026-11-18，容器内）。

### 2.2 已有 delta 方向资产（参考/补充测试集，不作主训练方向）

- `crgf_proto/results/crgf/delta_dirs_b0_1000_1500_h32.pt`：**来自 b0_resume ckpt_001000 vs ckpt_001500，相距 500 iteration，非相邻单步**（meta 记录，SHA256 在案）。28 个方向（H=32），flattened delta norm mean 17.71。→ 只可作"远距离 update"参考测试集。
- `crgf_proto/results/crgf/delta_dense_starts.pt`：35 个 (clip, frame) 起点（CRGF delta 加密运行所用）。
- `crgf_proto/results/crgf/crgf_dirs_all.pt`：CRGF 的 random/repeated/ascent 方向集存档（PADA 不直接使用，random 由同版 `random_dir_seqs` 现场生成）。

### 2.3 Provenance 与 motion 数据

- `crgf_proto/results/b0b1b2_provenance.json` + `b0b1b2_sha256.txt`：B0/B1/B2/origin501 checkpoint 来源与 SHA256（其中 B0/B1/B2 last.pt 的 SHA256 与 §2.1 server_artifacts 的文件对应，已在 CRGF 阶段核验）。
- `data/lafan1_g1`：motion 库（train clips），容器内经 mount 可用；`crgf_proto/data -> ../data` symlink 按 README 建立后 CPU 路径可用。

## 3. 数据缺口（PADA-0 必须新增）

1. **H 步方向性 probe 转移数据不存在**：现有工具要么记录标量 return（`crgf_sim_fd_audit`），要么是 1 步反事实对（repo-root `tools/generate_counterfactual_dataset.py`，单动作 ±δ、非 H 步方向序列、不记录逐步轨迹）。→ 新写 `pada_proto` 采集工具（复用 §1.2 的 sim 协议）。
2. **逐状态方向导数真值（Level-2）不存在**：CRGF 只测了 return 级 D_sim；PADA-0 的 state-response FD 由新采集的 plus/minus 轨迹离线计算。
3. **原始 replay buffer 不存在**（见 §6 缺口 1）。
4. **diag pair 的 teacher-forced delta 幅值此前未测过**（CRGF 只测过 500-iter pair）。本审计已补测（见 §5.3）。

## 4. Policy pair 的具体来源与合法性（对应任务 §3 问题 5/6/7）

1. **来源**：OPCL autonomous 训练管线（2026-11-17）的 `diag_pairs` 机制：训练循环在指定迭代处按固定操作顺序 snapshot 前后策略，bundle 记录 `pi_updates_before/after` 相差恰好 1、`same_policy_update_verified=true`。**这不是我们事后构造的 pair，而是训练管线自己产出并自我验证的真实单步 update**。
2. **是否相邻单步**：是（上表 pi_updates 区间逐个核对 manifest）。
3. **方向可否严格来自真实 pair**：可以。方向 = `δa_t = π_new(o_t) − π_old(o_t)`，两策略 teacher-forced 于同一 shared-obs 序列（π_old 在指定 WM 中闭环 rollout）。**注意口径**：这是 open-loop shared-observation delta，不是 closed-loop rollout delta；CRGF 与本审计均如实标注，不声称 closed-loop 验证（研究纪律 §14）。
4. **幅值可行性（本审计实测，CPU，4 clips × H=32，B2@2100 pair）**：per-step |δa| mean ≈ 0.506，typical |a| ≈ 9.39 → 相对幅值 ≈ **5.4%**。相邻单步的 action delta 是系统性、非噪声级的方向场，PADA-0 的 policy-shift direction 有意义。

## 5. 已核验的环境与执行事实

- GPU：RTX 5090（32 GB），nvidia-smi 正常；默认 conda python 无 isaaclab。
- **执行容器**：docker `crgf_smoke`（image `supermimic:latest`），mounts：repo → `/workspace/supermimic`，OPCL work → `/workspace/opcl_work`，repo data → proto `data/`。栈：torch 2.11.0+cu130 + isaaclab + isaaclab_newton + newton + warp（已验证 import）。
- 确定性约束（CRGF Phase 0 实测，`docs/crgf/GPU_EXPERIMENT_BUDGET.md`）：`deterministic_mode="run_to_run"` + `disable_sensors=True` 下 **≤4 envs** 逐位一致；≥8 envs 撞 warp int32 上限。速率 ≈ 0.26–0.28 s/batch-step（4 envs）。
- sim FD 先例：H=32、336 三元组 ≈ 35 min；delta 加密 140 三元组 ≈ 15 min（budget ledger 在 `crgf_proto/results/crgf/*.budget.jsonl`）。

## 6. UNKNOWN 与缺口声明（fail-closed）

1. **原始 WM 训练数据（replay buffer）未随 checkpoint 保存** → 无法执行"M0 + 原数据 + probe"的三方训练。PADA-0 采用 M0 上 probe-only 微调（两臂同 protocol），回答的是"给定 M0，probe 数据分布的边际价值"。此偏离对两臂对称，不影响 random vs policy 的内部有效性，但**外部解释范围受限**：不得把结论外推为"完整训练管线中 probe 数据的收益"。【已确认缺失，非 UNKNOWN】
2. diag pair 各 bundle 的 `pi_normalizer_changed` 只核对了 B2@2100（=false）；使用其它 pair 前须逐个核验（若 normalizer 变了，π_old/π_new 的比较需用各自 normalizer）→ 代码中强制检查，遇违规即标 UNKNOWN 并弃用该 pair。【待运行时核验】
3. probe ε=1.0（per-step ≈19% 相对幅值）下 H=32 轨迹是否大量触发 termination，进而影响 probe 数据量 → 由 GPU sanity check 回答。【待运行时核验】
4. proto WM 在 probe-only 微调下 fidelity 是否会移动（两臂皆然）——这是实验问题本身，不是审计能回答的。【实验目标】
5. b0_resume pair@1000 的 `wm` 与 online run 的 WM 架构一致性（同 repo 产出，预期一致；加载时 strict 校验，不一致即弃用）。【待运行时核验】

## 7. PADA-0 最小新增代码清单（全部在 `pada_proto/`，不修改 `crgf_proto/`）

| 文件 | 职责 | 复用 |
|---|---|---|
| `pada_proto/supermimic_bridge.py` | 容器/主机路径、proto_root 装载、pair bundle 加载+校验（normalizer/SHA256/验证标记） | `_crgf_bootstrap` |
| `pada_proto/tools/pada_gen_directions.py` | policy-shift 方向集（真实 diag pair → shared-obs delta）+ random 方向集；输出带完整 provenance 的 `.pt` | `crgf_delta_action_directions` 逻辑、`crgf.random_dir_seqs` |
| `pada_proto/tools/pada_sim_collect.py` | 一次 sim 会话采集：probe（两臂）+ eval（共享盲测集）；记录逐步状态/转移/reward 分量；budget ledger | `crgf_sim_fd_audit` 模式、`TrackingSim`、`reward_split` |
| `pada_proto/tools/pada_finetune_wm.py` | M0 → M1/M2（同种子同 schedule 微调），快照序列 | train.py 损失形态镜像 |
| `pada_proto/tools/pada_eval.py` | Level-1/2/3/4 全部指标（WM 侧 autograd + 离线 sim 真值），产出 CSV/JSON | `crgf.open_loop_paths`、`fidelity_stats`、`reliability_gate` |
| `pada_proto/tools/test_pada.py` | CPU 合成测试（公平性不变量、指标正确性、方向单位性、配对 schedule 一致性） | `test_crgf` 风格 |
| `pada_proto/README.md` | 复现说明 | — |

## 8. 审计结论

PADA-0 **具备全部关键前提**：真实相邻单步 policy pair 充足（15 个、自验证、带 SHA256）、CRGF 三路径与门禁体系可整体复用、sim 协议与速率已知、运行容器就绪。缺口（轨迹级采集工具、微调/评估脚本）均为薄封装层，不需要重新实现任何 CRGF 已有能力。唯一的结构性限制是 replay buffer 缺失（§6.1），已在协议中作为预注册偏离声明处理。
