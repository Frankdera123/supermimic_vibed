# OVERNIGHT_PROGRESS（CVF-1a 夜间自主研究进度日志）

模式：**APPROVED=false**。本轮仅执行 Phase A（CPU）：资产检查、源码审计、数据合同、
实现准备、协议设计。Phase B/C/D/E/F 全部 **NOT RUN**，等待项目负责人显式授权。

---

## 会话 1：2026-11-20（Phase A 执行）

### 完成事项
1. **仓库核验**：`dev @ 6e9b4e7`，未跟踪：`crgf_proto/data, cvf_proto/, docs/cvf/,
   docs/pada/PADA0A_OFFLINE_AUDIT.md, pada_proto/tools/{pada_eval_matched_fd,test_pada_audit}.py,
   results/cvf/, results/pada/offline_audit*/`。`git fetch` 网络超时 → 远端同步未验证；
   不 push（ALLOW_GIT_PUSH=false）。
2. **CVF-0 事实继承**：终报 + deep_analysis 12 篇 + CSV 全部读取提炼（关键修正：
   vel_reachable MSE 中 e_B 占 97–98% 但符号失败为 Critic 特有；CF D 上限
   sign 0.958/Sp+0.954/ratio 0.98；T=32 尾项占 71%；Sim 锚 ±不对称 94.9%；
   Route A（WM 蒸馏）证伪；B=1 vs B=N 无 reduction bug；口径 c_t·r_t 正确继承）。
3. **资产盘点**（Explore agent + 本地核验）：CVF-0 Oracle 原始数据在
   `results/cvf/phase1b/`；母本在仓库外 `server_artifacts/online_B0/`；40 个 LAFAN1 clips；
   真 replay buffer dump 不存在（本轮不依赖）；GPU RTX 5090 空闲（3.2/32GB，0%）。
4. **母本 fail-closed 核验**：`verify_mother_assets.py` → **15/15 VERIFIED**
   （文件级 + 组件级 SHA256 复现；ckpt_002100["wm"]==bundle["wm"]、
   ckpt_002100["critic"]==critic_post、pi_old 仅在 pair bundle；bundle["critic"]==critic_pre）。
   manifest：`results/cvf/sacv/manifests/mother_asset_manifest.json`。
5. **数据合同**：`CVF1A_DATA_CONTRACT.md`（扩展状态 x=(s,h,a_prev,f,ξ)、Dv/Dp、
   G_T/Y_T 双目标、c_t·r_t 口径、T∈{32,64,96}+T=128 条件扩展规则、ALL vs Strict-gated
   分层、env_transitions vs simulator_batch_steps 双计数交叉核对）。
6. **划分冻结**：`build_split_manifest.py` → `CVF1A_SPLIT_MANIFEST.json`
   （Train 12 / Val 4 / Test 6 clips，24/8/12 起点，seed=2701，CVF-0 10 clips 全排除，
   8 clips 预留，horizon 可行性 100%）。
7. **实验协议**：`CVF1A_EXPERIMENT_PROTOCOL.md`（C-init/C0/C1/C2 冻结合同、α∈{0.1,0.3,1.0}
   + σ_ΔY 尺度冻结、遗忘探针、D1–D6 评价、PASS 判据、Phase E 单步更新设计、
   机制判别树映射 Goal §9 情况 A–F）。
8. **预算计划**：`CVF1A_BUDGET_PLAN.md`（CVF-0 实测 12.1 transitions/s 基准、
   R0/R2/R3 降档阶梯、停机规则、ledger 格式）。
9. **核心数学实现 + 测试**：`sacv/losses/{targets,value_losses}.py`、
   `sacv/evaluation/{metrics,sacv_eval}.py`、`sacv/tools/{sacv_collect,sacv_build_dataset,
   sacv_train_critic}.py`（GPU 工具仅编译，未执行）；`test_sacv.py` **16/16 PASS**
   （含真实 C-init autograd vs FD float64 检查、真实数据 SHA256 不变性、渗漏检查）。
10. **评估管线冒烟**：`sacv_eval.py --fixture 200` schema 冒烟通过。

### 偏离与问题记录
- 计划偏移：无（Phase A 全部按 Goal §4 执行；A4 的 T=128/256 触发条件式延后至 Phase B
  实测吞吐后决策，已在合同 §4.4 预注册）。
- 测试修复：float32 FD 舍入问题（→float64）；两处测试变量遮蔽 `T`（模块别名被局部 int
  覆盖）；metrics 期望值（median ratio=1.0 非 0.95）。
- 工具草稿期错误在编译前修正：死代码、raw state dict 误用为 module、companion 置换逻辑。
- 发现：`ckpt_002100.pt` 无 optimizer state 键（68MB）→ Phase E 预案改为
  cold-optimizer experiment（协议已预注册标注）。
- 风险登记：Val 仅 8 起点 → D6 硬指标功效有限，结论预期为 PILOT_PROMISING/INCONCLUSIVE
  层级；C0 目标生成 GPU(bf16) vs sim 标签精度域差异需 Phase C 前抽查。

### 资源消耗
- GPU：**0.00 GPU-h**（CPU only：资产核验 0.5s、MotionLib+划分 2.9s、测试套件 ~90s）。
- Wall clock：~1.5 h（含 deep_analysis 全文审读与继承）。

### 门禁状态
- Phase A 合同测试：**通过**（核心数学合同成立，无需暂停）。
- Phase B smoke：**等待 APPROVED=true**。
- 预算累计：0 / 6.0 GPU-h。

### 下一步（授权后执行序）
1. `sacv_collect.py --mode smoke --envs 4 --max_minutes 15` → 确定性/一致性/吞吐检查；
2. 依 smoke 选 B（4/8/16）与阶梯（R0/R2/R3）→ 正式采集 Train→Val→Test（每 batch 落盘+ledger）；
3. `sacv_build_dataset.py` 三 split → horizon 冻结复核（合同 §4.4）→
4. `sacv_train_critic.py` C1/C2/C0（1000 步 pilot，α=0.3 主）→ Val 探针 →
5. Phase D 评价 → 判定树（SUPPORTED/PILOT_PROMISING/INCONCLUSIVE/CONTRADICTED/BLOCKED）→
6. 条件性 Phase E；全程 OVERNIGHT_PROGRESS 与 budget_ledger 持续更新。

---

## 会话 2：2026-11-20 04:00–（APPROVED=true，Phase B 执行）

### 授权与环境恢复
- 项目负责人显式授权："后续所有使用GPU都不需要授权，继续goal" → APPROVED=true 已入 ledger。
- **容器 GPU 修复**：`crgf_smoke` 容器 NVML 掉线（宿主驱动更新后旧容器典型症状，cuda False），
  `docker restart` 恢复 RTX 5090 直通（cuda True）。
- **工具修复**（编译期发现）：`crgf` import 应为 `supermimic.crgf`（命名空间包）；
  **xi 口径对齐 CVF-0**：WM `wm_step`/`open_loop_paths` 与 `critic_input` 必须用
  `xi_n = normalize_xi(xi)`（[(1-0.75)/0.45, 0×29, 0×3]），env 物理侧用原始 xi_one——
  与 `cvf_phase0_audit.py:125` 一致；builder 改为按 start 元数据路由 split；
  C0 组 wm_t 合并 bug 修复。

### Phase B 执行记录
1. **Smoke（B=4，627.7s，7,424 transitions）全过**：base_repeat_range=0.0（位级）、
   restore_repeat_range=2.38e-7、companion_perm_slot0=0.0（无跨环境污染）、
   reward 重建 1.49e-8、G 重算 1.85e-6。有效吞吐 ~11.8 transitions/s（与 CVF-0 12.1 一致）。
2. **B=8 尝试失败**：MJWarp `deterministic_mode="run_to_run"` 的 sort/reduce workspace
   int32 溢出（3.26e9 字节）。判定为确定性模式在 8 环境下的固有局限，不可安全绕过
   （调接触池会破坏与 CVF-0 可比性）→ 按预注册规则回退 B=4。显存占用 ~19.5GB/32GB。
3. **R0→R2 降档（预算纪律）**：R0 全量外推 2.4–2.5 GPU-h，加已耗 0.25（smoke 0.20 +
   B8 中止 0.05）将破 Phase B 2.5 上限 → 切换预注册 R2（ascent/random 单 ε，
   delta_2100/2300 双 ε），44 起点 × 14 rollouts × 128 = 78,848 transitions ≈ 1.9 GPU-h。
4. **R2 实测首批**：622.1s/批（含冷启动重编译 ~4 分钟），7,168 transitions，
   rep_rng=0.0。11 批预计 ~115 分钟，fuse 135 分钟，总预算预计 ~2.45 ≤ 2.5 ✓。

### 工具链补齐（等待采集期间）
- `sacv_stats.py`：D6 判据自动核算（C2 vs C1 五项门限）+ 配对 bootstrap/符号检验
  （ALL 与 Strict-gated 双层）+ horizon 冻结诊断（合同 §4.4）。
- `sacv_dump_preds.py`：批式 per-pair critic 预测导出（sacv_preds_v1 schema）。
- `sacv_policy_update.py`：Phase E 完整实现——忠实复刻原训练 update 路径
  （imag_step 无动作噪声、opt_step norm1.0+value0.1、γ^32·c₃₂·V.clamp(10) bootstrap）、
  cold-optimizer 预案标注、组间共享想象轨迹（manual_seed 重置 → 强配对）、
  **MJWarp B≤4 分批**（CHUNK=4 想象 6 批 / 评价 5 批）、cos(update, P0-update) 诊断。

---

## 会话 3：2026-11-20 05:00–08:00（Phase B 完成 → C/D 执行 → E 门禁）

### Phase B 完成
- R2 采集 **11/11 批全部完成**：44 起点、78,848 transitions（== batch steps）、
  6,594s = **1.83 GPU-h**，速率 11.96/s，rep_rng 全 0.0，fuse 未触发。
  节奏注记：Train/Val clips ~2.8 分钟/批，Test 剧烈 clips（fight/sprint 类）慢 5–10 倍。
- 三 split 数据集全绿：Train 432 / Val 144 / Test 216 pairs（3 时域）；
  reward 重建 ≤3e-8、G 重算 ≤3.8e-6、base_repeat_range 全 0.0。

### Horizon 冻结（合同 §4.4）
规则 1 全时域满足（Train G64~G96 sp=0.977）；规则 2（尾项占比 <0.5）在 T=96 为
1.27/5.66 **失败——如实记录**（尾项经 V_ref r=0.987 传导为稳定共同项；配对差分中
大部分相消；CF D 上限本身即 T_cont=96 口径）。**冻结 T=96**（可比性+尾部信息；
Test 未参与，属预注册事前调整）。

### Phase C（0.68 GPU-h）
- C0 WM 目标预生成（144 pairs，同尾项合同）；C1/C2(α=0.3×pair_scale=247.9)/C0
  各 500 步（匹配批/流/lr；σ_ΔY=0.0635 冻结）。ckpt s0250/s0500 含全部元数据。
- 过程修复：**setdefault 缓存低效**（每样本重 load 100MB batch 文件——真正的速度
  杀手）与 **epoch 重排后切片游标 off-by-logic**（step3 空批崩溃）；批式前向改造。

### Phase D（CPU）与裁定：INCONCLUSIVE（负面倾向）
- **没有任何微调组超过 C-init**（Val ALL：C-init sign 0.729/sp +0.504 最高；
  gated sp +0.929 最高）；C1 vs C-init gated 配对检验方向为 C-init 显著更好。
- C2 在 Val gated 明确退化（sign 1.000→0.875、sp 0.881→0.333）；D6 门限 2/3/5 失败。
- Test 盲测 C2 略优（0.639/+0.501）但与 Val 方向相反——flip-flop，不作为证据。
- 非 circular 复评（ΔG⁹⁶ 口径）结论不变——排除评价伪影。
- **决定性根因**：Train pair 标签方差 99.5%+ 为 V_ref 尾项（delta 族 163–203×）
  → 微调是"向 C-init 自我回归"，ΔG 新信号 <1%。CVF-0"拼尾项"教训的适用边界
  （标量校准成立、差分监督有害）已划清。

### Phase E：NOT RUN
门禁条件 ②（无 Val 改善候选）⑤（无正向机制信号）失败 → 按预注册纪律不启动，
0 GPU。工具已就绪供下一轮使用。

### 资源结算
- GPU：0.196(smoke) + 0.05(B8 尝试) + 0.2(R0 中止) + 1.832(R2) + 0.05(C0 gen)
  + 0.63(3×500 步训练) ≈ **2.96 / 6.0 GPU-h**；wall ≈5h（04:00–08:00）。
- 全部原始数据、ckpt（含 parent/split hash）、metrics、ledger 落盘，sha256 入 registry。

### 交付
最终报告集：`CVF1B_CRITIC_RESULTS.md`、`CVF1B_MECHANISM_ANALYSIS.md`、
`CVF2A_POLICY_RESULTS.md`（NOT RUN）、`CVF_OVERNIGHT_RESEARCH_REPORT.md`（Q1–Q8）、
`CVF_NEXT_DECISION.md`（CVF-1b′ 修正标签合同重训，≈2.5 GPU-h）。
`final_summary.json` 已更新。
