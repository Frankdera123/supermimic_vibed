# CVF1A_READINESS_REPORT（Phase A 完成报告）

日期：2026-11-20（夜间会话）。执行模式：**APPROVED=false → 仅 CPU 资产检查、源码审计、
数据合同、实现准备、协议设计**；所有 GPU 阶段（B/C/D/E）未执行，等待项目负责人授权。

## 1. 仓库与母本资产状态（fail-closed 核验，全部通过）

- Git：`dev @ 6e9b4e7`（PADA-0 报告提交），无 upstream；领先 `origin/main@0525a23` 3 个未推送
  提交。`git fetch --dry-run` 网络超时（SSH 慢/无网），远端同步状态无法即时确认——
  **ALLOW_GIT_PUSH=false，本轮不做任何推送**，仅记录。
- 母本身份链（`results/cvf/sacv/manifests/mother_asset_manifest.json`，15/15 检查 VERIFIED，
  耗时 0.5 s，CPU）：

| 角色 | 载体 | SHA256 | 核验 |
|---|---|---|---|
| WM@2100 | `ckpt_002100.pt["wm"]` == `pair_it_002100.pt["wm"]` | `43cd7f5a…60f5` | 组件哈希复现一致 |
| pi_old@2100 | 仅 `pair_it_002100.pt["pi_old"]`（ckpt 内 pi 为 pi_new） | `c4810c4d…f6b0` | 组件哈希复现一致 |
| C-init=B0@2100_post | `pair_it_002100.pt["critic_post"]` == `ckpt_002100.pt["critic"]` | `65118115…3f54` | 组件哈希复现一致 |
| pair 文件 | `pair_it_002100.pt` / `pair_it_002300.pt` | `b951c51e…3495` / `70ff7577…cd25c` | 与训练时 `diag_pair_manifest.json` 自记录一致 |

  配置核验：H=32、γ=0.99、λ=0.95、hist=10、seed=0、lr=2e-4、critic_epochs=5、
  critic_minibatches=4、OBS 93、Critic 938→[256,128]→1 + RunningNorm、no_critic=false。
  `same_policy_update_verified=true`、`pair_validation_reason=VALID_SINGLE_POLICY_UPDATE`、
  pi_updates 1599→1600、`pi_param_l2_diff=0.0886`。
  附注：`bundle["critic"]` == critic_pre（ascent 方向构造与 CVF-0 phase0 完全同口径）。
- CVF-0 原始数据：`results/cvf/phase1b/`（oracle1/oracle2 全套 batch+sim 文件）哈希与
  `da01_inventory_contract.json` 记录一致（4/4）→ 历史诊断回归集可用。
- 资产结论：**母本可直接进入 Phase B/C，无 BLOCKED 项**。真 replay buffer 原始 dump 不存在
  （与 CVF-0 相同，不依赖）；`results/pada/collect/` 数据来自其他 probe 政策，标记为
  off-policy 偏差数据，本轮不用。

## 2. 数据划分（已冻结）

`CVF1A_SPLIT_MANIFEST.json`（seed=2701，docs/results 双落盘）：
- 池 40 clips − CVF-0 已用 10 = 30 可用（0 个过短）；
- **Train 12 / Val 4 / Test 6 clips（= 24/8/12 起点）**，每 clip 2 起点（间距 ≥64 帧），
  horizon 可行性 `f0+32+96 ≤ clip_end−1` 全部满足；
- Test 覆盖 6 类运动（含 sprint1/fightAndSports1 单例），Train 覆盖 8 类；8 clips 预留；
- 渗漏检查（tests family 13）通过：无 clip 跨 split、无 CVF-0 泄漏、csv sha256 在册。

## 3. CPU 单元测试（Goal §10 的 15 类，16 个测试）

`cvf_proto/sacv/tests/test_sacv.py` — **16/16 PASS**：

| # | 合同 | 测试 |
|---|---|---|
| 1 | 折现 return 数学（闭式/变奖励） | test_discounted_return_math |
| 2 | 软终止 survival indexing（c_t·r_t 对齐训练口径） | test_soft_survival_indexing |
| 3 | 硬终止/截断 + c_T=0 时尾项消失 | test_hard_termination_truncation_and_tail |
| 4 | 正负 pair 恒等（反对称、零差） | test_plus_minus_pair_identity |
| 5 | Pair loss 解析验证（完美拟合=0、梯度闭式=autograd） | test_pair_loss_analytic_and_grads |
| 6 | B=1 vs B=N 一致 | test_batch_size_consistency_b1_vs_bn |
| 7 | Critic autograd vs 中心差分（解析 MLP + **真实 C-init 938 维**，float64） | test_critic_autograd_vs_finite_difference |
| 8 | 无跨 batch reduction（成分不变性） | test_no_cross_batch_reduction |
| 9 | C1/C2 样本与标签同一性（C2=C1+α·pair，α=0 退化为 C1） | test_c1_c2_identity_and_stream |
| 10 | WM 冻结 | test_freeze_wm_parameters |
| 11 | Policy 冻结 | test_freeze_policy_parameters |
| 12 | Normalizer 统计不变性（eval 模式不更新、批量无关） | test_norm_statistics_invariance |
| 13 | Train/Val/Test 渗漏 + horizon 可行性 | test_split_leakage_and_feasibility |
| 14 | Policy optimizer 状态一致性（cold-optimizer 共享初始化） | test_policy_optimizer_state_consistency |
| 15 | 母本 checkpoint 与历史数据 SHA256 不变性 | test_checkpoint_and_data_hash_invariance |
| — | 指标模块 sanity（sign/RMSE/ratio/spearman/配对符号检验/γ^H c_H 权重） | test_metrics_sanity |

修复记录：float32 下 FD 一致性检查受舍入放大支配（h=1e-4 时误差 ~1e-4），改为 float64
（float64 下 5e-12）；生产训练精度（bf16 AMP）属另一域，已注明。

## 4. 实现准备（已就绪、未执行的 GPU 工具）

| 工具 | 阶段 | 状态 |
|---|---|---|
| `sacv/tools/verify_mother_assets.py` | A | **已执行**（VERIFIED） |
| `sacv/tools/build_split_manifest.py` | A | **已执行**（冻结） |
| `sacv/tools/sacv_collect.py` | B | 编译通过；smoke（确定性/restore/companion 置换/reward 重建/G 重算/吞吐 R0-R2-R3 预估）+ collect 全流程 |
| `sacv/tools/sacv_build_dataset.py` | B→C | 编译通过；raw→pairs/scalars + V_ref 目标 + 可靠性报告 |
| `sacv/tools/sacv_train_critic.py` | C | 编译通过；C0/C1/C2 冻结合同训练 + WM 目标预生成 + ckpt 元数据 |
| `sacv/evaluation/sacv_eval.py` | D | 编译通过；D1/D2/D4 指标 + ε-gate 分层 + fixture 冒烟通过 |
| `sacv/evaluation/metrics.py` | D | 已测试（test_metrics_sanity） |
| `sacv/losses/{targets,value_losses}.py` | C/D | 已测试（families 1-9） |
| CVF-2a（Phase E）工具 | E | **NOT RUN / 未实现**——按协议 §4 为条件性任务，待 Phase C/D 门禁结果后再实现（避免过早绑定未确定的 Critic 选择） |

## 5. 继承事实 → 本轮协议的关键绑定（摘要）

完整依据见 `CVF1A_EXPERIMENT_PROTOCOL.md` §0；要点：
1. 主失败面 = vel_reachable 方向响应（sign 0.31–0.38，p≈0.45）；标量校准已达标，不动。
2. 成功判据锚定反事实上限 CF D（sign 0.958/Sp +0.954/ratio 0.98），必须超过 A(0.562)/C(0.604)
   且保留 Critic 幅度贡献（prefix-only 仅 0.08×）。
3. Sim 锚噪声（±不对称 94.9%）→ ±ε 成对差分 + 双层（ALL/Strict-gated）报告 + 逐 clip 分层。
4. 所有训练目标强制拼 γ^T c_T V_ref 尾项（T=32 丢 71%、T=96 丢 ~38% 尾部）。
5. Route A（WM→Critic 蒸馏）已证伪 → C0 仅作 matched-target 对照。
6. 口径继承：c_t·r_t、B 归约无 bug、grad_raw vs clamp 双报告、Oracle D 仅诊断。

## 6. 预算（详见 CVF1A_BUDGET_PLAN.md）

Phase B 采集 44 起点：R0 全量 2.33 GPU-h（B=4 实测吞吐 12.1 transitions/s 外推）→
预注册降档 R2/R3 至 1.7–1.8 GPU-h；B=8 确定性通过则 ~1.1–1.4。Phase C ≈0.4，D ≈0.15，
E ≈0.6。总上限 6.0 GPU-h，预期 2.4–3.9。

## 7. 风险与未决项（进入 Phase B 前的诚实边界）

1. **吞吐外推**基于 CVF-0 B=4 实测，新 clip（fight/fallAndGetUp 类）可能更早 termination
   （缩短 rollout，利好）或步态分叉更频繁（Strict-gated 通过率低，历史 3–4/16）；
   smoke 后按阶梯重估。
2. **方向生成依赖 WM 闭环 rollout 的确定性**（bf16 GPU vs CPU 数值差异会改变 A0/delta 方向）；
   已在 smoke 中通过 base_repeat_range 检查兜底；方向生成落盘（dirs + A0 进 batch 文件），
   可完整复算。
3. **Pilot 样本量**：Val 仅 4 clips/8 起点；D6 全部硬指标在 Val 上功效有限，预期结论多为
   `PILOT_PROMISING`/`INCONCLUSIVE` 层级——已预注册，不为 PASS 而扩 Test 使用。
4. **C0 的 WM 目标生成**使用与训练一致的 bf16 路径（GPU），与 C1/C2 的 CPU-side sim 标签
   精度域不同——已在协议 §1 注明为合同一致性风险项，Phase C 启动前以 float32 复算抽查。
5. Adam state 预检：`ckpt_002100.pt` 无 optimizer 键 → Phase E 按 **cold-optimizer** 预案
   （协议 §E1 已预注册，全程标注）。
6. `git fetch` 网络不可达 → 远端同步未验证（不影响本地实验；不做 push）。

## 8. 裁定

**READY（conditional）**：数据合同、划分、实现准备、CPU 测试全部完成且通过；
Phase B 烟测待 `APPROVED=true` 后启动。未开展阶段（B/C/D/E/F）一律 NOT RUN，
无任何虚构结果。
