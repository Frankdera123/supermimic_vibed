# 研究路线图（Research Roadmap）

更新：2026-11-23。仓库：SuperMimic 非官方复现的 fork（`dev` 分支）。
本文回答三个问题：我们问过什么问题、得到了什么答案、下一个最值得做的问题是什么。
逐线的完整证据在各线目录（导航见 [README.md](README.md)）；本文只保留
经过核对（含数据抽验）的结论与跨线综合。

## 0. 背景体系（两条代码变体，勿混用）

| 体系 | WM | OBS | critic | H / trunc | 用途 |
|---|---|---|---|---|---|
| 主仓库 `supermimic/` | 70-D 全增量 MLP | 99 | 无（默认） | 64 / 4 | 上游复现 + concurrent 线 |
| OPCL proto（仓库外） | 35-D 加速度级 | 93 | LN+GELU [256,128]，938 输入 | 32 / — | OPCL/CRGF/PADA/CVF 四条线 |

共享母本：`online_B0@2100`（WM + pi_old + critic_post，组件级 SHA256 在
`results/cvf/sacv/manifests/mother_asset_manifest.json`，fail-closed 核验 15/15）。

## 1. 五条研究线：问题 → 答案 → 成本

### L0. Concurrent WM/policy exploration（2026-11-05~07）

- **问题**：让 policy 的 return 梯度直接参与 WM 训练（更紧的耦合），能否稳定超过
  replay-only WM + 截断 BPTT？
- **答案**：**不能**。~80 个编号消融条目（`concurrent_exploration.md`，1886 行）中，
  直接梯度耦合的全部变体（scalar/projection/conflict-gate/norm-matching/EMA/
  trust-region/L2-anchor/capacity/短前缀）、EMA-target WM、action-conditioned
  residual adapter、ensemble-disagreement gate、rollout 校准 gate（阈值不迁移）
  全部未稳定超过 uniform H64/trunc4 baseline；文档为此定义了 "model exploitation"
  分类。**正向残留**：leading recipe = grouped WM×2 → policy×2 + aux-ramp 0.05 +
  独立 batch 重采样（subject4/5 留出集 +10~22%）；机制证据指向"顺序策略适应"而非
  更好 WM 拟合。
- **保留意见**：①仅 subject4/5 两个留出 split；②compute-matched 时单次更长训练
  反超双更新（experiment_matrix.csv −8~−10%）——收益是"每迭代内的顺序适应"，
  不是计算量；③**待核实**：早期 300 轮对照可能受 `wm_only_iters` 覆盖影响
  （argparse 默认 500 = 前 500 轮无 policy 更新，train.py:347；线内配方自称默认 5，
  concurrent_exploration.md:1096）——需用 `logs/<run>/args.json` 逐 run 复核。
- **成本**：~数十次 300–5000 轮训练（无统一 ledger；单次 1000 轮 ≈0.05–0.15 GPU-h，
  见 budget_audit_key_runs.csv）。

### L1. OPCL（2026-11-14~17）

- **问题**：outcome+process 耦合损失（L_out/L_proc）能否改善 WM 的控制相关性？
- **答案**：**无已证收益**。B2（OPCL）对 B1（预算匹配对照）配对差 CI 含 0；
  B2 对 B1 非单变量（多了 OPCL 触发/采集分支）；OPCL 自己的 GPU 试点
  INCONCLUSIVE（显式 state restore ≠ 接触求解器内部状态恢复）。
- **遗产**：B0/B1/B2/B3 的 iter-2300 checkpoints 成为 L2–L4 的共享母本；
  proto 代码变体（35-D WM/OBS 93/critic/H32）成为后续所有实验的底座。
- **成本**：约 2300 iterations × 4 branches（原始文档在仓库外）。

### L2. CRGF（2026-11-17~18）

- **问题**：单步动作欠响应（ratio 0.27，约 3.7×）是否经梯度通路威胁 policy
  optimization？
- **答案**：**全局 NO-GO（该命题成立）**：欠响应真实存在但主要发生在 policy 不访问
  的方向（H4 加强）；**H5"critic 是误差源"被推翻**（own-critic 反而 4.2× 更好）。
- **核对发现的脚注**（本轮 agent 逐文档核对）：①NO-GO 判据只统计了 delta-critic
  路径，同批 state 行 sign 0.71–0.86 未进判定（`fidelity_verdict_delta_dense.csv`）；
  ②H32 gate 剔除率 60.7%、H64 83.3%、delta 加密 80%，越过预注册的 >60% 降级线
  但只有 H64 被降级标注；③CRGF 的 survival 索引与训练源码差一步
  （c_{t+1}·r_t vs c_t·r_t，`crgf_proto/supermimic/crgf.py:186-188`，docstring 与实现
  不一致）；④ledger 合计 ≈3.5 GPU-h vs 文档 ≈3.3（±0.2 未解释）。
- **成本**：≈3.3–3.5 GPU-h（授权上限 4.15；Phase 3 未执行）。

### L3. PADA-0 / 0A（2026-11-18）

- **问题**：相同真实 simulator query 预算下，policy-directed probes 是否比 random
  probes 更高效改善 WM 的 policy-relevant 建模？
- **答案**：**INCONCLUSIVE**；提交版的"方向性证据偏 YES"被 PADA-0A 离线审计
  （未提交，本轮已补交）**降级为不成立**：critic 路径主导、单 clip 脆弱（gated 的
  54% 来自 walk3_subject5）、幅值畸变 2–10×、clip 聚类后 CI 全部含 0、state 路径
  反向（seen −0.333，CI 不含 0）。**PADA-Data 立项依据不存在**。
- **审计发现的方法论坑**（已被后续线继承为检查单）：梯度求导点错误（A+εd vs 中心
  差分，修正后一个符号结论翻转）、survival detach 口径、pooled-ε 混尺度、
  样本聚类未处理、provenance 漏记 ckpt SHA256。
- **成本**：1.0 GPU-h（0A 纯 CPU 446 s）。

### L4. CVF：CVF-0 → CVF-1a/1b（v1 无效 → v2）（2026-11-19~23）

- **CVF-0 问题**：学习到的 critic 的状态价值梯度在多大程度上忠实于真实 simulator
  的闭环回报梯度？误差来自模型内 Sobolev 失配（A）还是 WM→Sim 现实鸿沟（B）？
  **答案**：标量准（r 0.987–0.990）但方向响应失败（vel_reachable sign 0.31–0.38）；
  vel_reachable 的 MSE 中 e_B 占 97–98%，但**符号失败是 critic 特有**（WM 0.69–0.81）；
  共享 learned critic 造成评价混淆（e_state 与 e_critic_true 正相关 +0.413）；
  反事实上限 CF-D（同一 32 步 WM prefix + 真 96 步 Sim continuation）= sign 0.958 /
  Sp +0.954。成本 1.93 GPU-h。
- **CVF-1a/1b 问题**：用真实 Simulator 的标量价值与有限尺度成对差分监督微调 critic
  （C1 标量 / C2 +pair / C0 WM 对照，全部从 C-init 出发），能否改善 control-relevant
  响应并传导到 policy？
  **v1（11-20）无效**：critic 被评在 x_{H+96} 而全部标签定义在 x_H；`Delta_Y` 字段
  存的是 ΔG。已发布结论撤回（[CVF1B_ERRATUM.md](cvf/sacv/CVF1B_ERRATUM.md)）。
  **v2（11-23）修正重跑，方法成立**（状态对齐回归测试、预评估承诺、Test 一次性盲评）：
  - **INCONCLUSIVE（负面倾向）**。Sim 标量锚定（C1）真实改善 held-out 标量价值
    （Test RMSE 0.684→0.497，−27%；Pearson 0.975；bias 减半）；但 T=96 的
    reachable 方向响应无任何改善（Test 配对 CI 全跨 0；Val 的 +3.1pp 不迁移）；
    显式 pair 监督（C2，梯度平权）明确有害且有毒剂量效应（标量 RMSE +103–107%）；
    C0（WM 监督）≈ 无操作。
  - **机制修正**：T=32 的 reachable 成对响应本来就准且跨 split 一致（C-init sign
    0.875/0.792，sp +0.92/+0.60）；T=96 的 ΔG 在新 clip 上与短时域信号失相关
    （Test reach sp(G32,G96)=+0.45 而 sp(G32,G64)=+0.90）。**瓶颈不在 critic 的局部
    拟合，而在 96 步续推差分目标在新 clip 上的可靠性（噪声天花板）**。v1 的
    "尾项淹没 99.5%"是测量错误（修正后 Train delta 族 27–30×、Val/Test <0.25×）。
  - **Phase E**：两次门禁失败，NOT RUN（0 额外 GPU）。
- **成本**：Phase A 0 + B 2.28 + C(v1) 0.68 + C/D(v2) 0.08 ≈ **3.0 GPU-h**；
  有效数据资产：44 起点 / 78,848 transitions（`datasets/raw/`，可复用于任何后续线）。

## 2. 跨线综合：什么是确定的

1. **"更紧耦合"被五条线一致否定**：policy 梯度改写 WM（L0 全部变体、L1 OPCL、
   L2 NO-GO）、WM 续推当监督（L4 C0≈无操作、CVF-0 CF-B）、probe 定向 WM 微调
   （L3）——没有任何一条在预注册判据下稳定超过 replay-only WM + 截断 BPTT。
   Dreamer 式分离（WM 从 replay 拟合、policy 在冻结 WM 上想象）被反复独立验证。
2. **真实存在、可复现的收益只有两个，都很小**：①同迭代内第二次 policy update
   （L0 leading recipe，subject4/5，非 compute-matched）；②Sim 标量锚定改善
   critic 的 held-out 标量精度（L4 v2，Test −27% RMSE）。两者都不触及
   "WM 梯度质量"这个原始动机。
3. **反复出现的失败模式（方法论层）**：评价混淆（共享 critic、口径漂移、求导点、
   状态对齐）、±ε 分岔噪声导致的 gated 样本不足（3–8 对）、单 clip 脆弱性、
   compute/预算匹配缺失。四条线各自独立踩中至少一个，且全部是**用离线审计/勘误**
   而非在线监控发现的。
4. **方法论教训（已制度化）**：预注册 + 门禁 + 盲测的框架有效（两次阻止了错误结论
   被推进到 Phase E）；但**实现与合同的逐项对照回归测试必须在实验前存在**
   （CVF-1b v1 的状态错位正是漏了这一层；现已补
   `test_pairs_v2_state_label_alignment` 等回归测试）。

## 3. 下一步决策（按信息价值/成本排序）

| # | 任务 | 成本 | 为什么 |
|---|---|---:|---|
| 1 | **T=96 噪声天花板量化**：逐 clip 把 ΔG₉₆ 与 ΔG₃₂/₆₄、接触切换记录关联，估计"可预测成分"占比，产出 96 步目标可靠性地图 | 0 GPU-h（纯 CPU，数据在盘） | 决定 CVF 线是否值得继续：若可预测成分低，96 步成对目标在分布外 clip 上不可学习，应改写研究问题而非继续调监督 |
| 2 | **wm_only_iters 历史复核**：用 `logs/<run>/args.json` 复核 L0 早期 300 轮对照的 policy 更新状态 | 0 GPU-h（CPU 脚本扫描） | 若成立，L0 的部分早期负结果需要降级标注（影响路线图 L0 的置信度） |
| 3 | **标量锚定的下游验证（Phase E-lite）**：C1@1000（Test 标量 −27%、方向无害）作单次 policy update 对照 | ≈0.6 GPU-h | 检验"更好的标量价值"是否转化为更好的真实 policy update；工具已就绪（`sacv_policy_update.py`，MJWarp B≤4 分批） |
| 4 | **接触对齐/分段续推目标改造** | ≈2.5 GPU-h | 仅当 #1 显示失相关集中在接触事件后才立项 |

被否决的方向（勿重开）：policy 梯度改写 WM 的任何变体（L0/L1/L2）、WM 续推监督
（L4 C0）、probe 定向 WM 微调（L3）、T=96 pair 监督加权重（L4 v2 剂量效应）。

## 4. 仓库状态

- 分支 `dev`（跟踪 `frankdera/dev`）；研究资产按线组织：`docs/{crgf,pada,cvf}/` +
  concurrent 线根目录散页（索引见 [docs/README.md](README.md)）；
  `results/{crgf,pada,cvf}/` 原始产物与 ledger；`{crgf,pada,cvf}_proto/` 工具链。
- 大体量 `.pt` 证据不入库（政策），SHA256 在各 manifest/registry；
  小体积方向集与冻结 split manifest 入库。
- 已知未整理项：`docs/` 根 ~336 个 JSON 保持原位（concurrent_exploration.md 的
  相对引用底座，移动会断链）；`logs/`（267GB，912 runs）未入库也未离档。
