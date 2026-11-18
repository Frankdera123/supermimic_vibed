# RESEARCH_DECISION

日期：2026-11-18。本文件是本轮 CRGF 准备工作的决策总结：做了什么、发现了什么、下一步需要什么。

## 1. 已完成（CPU/只读，全部可追溯）

1. **两线证据整合**：supermimic_vibed 的 under-response 审计原始数据（ratio 0.27 / gain 2.9 / cos 0.82）+ frontier 主日志结论 + OPCL 线 GPU 试点（INCONCLUSIVE → 确定性修复）。确认 B0/B1/B2 的真实来源、配置 diff、SHA256（iter 2300，同源 501，B2 vs B1 非单变量）。
2. **隔离评估体系**：`crgf_proto/`（不动原仓库）；`supermimic/crgf.py` 双协议适配（主仓库 + OPCL proto）；三路径分解（state/direct/critic + 精确 cross 恒等式）；四种方向集（含真实策略更新 δa）；fail-closed 门禁；18/18 CPU 合成测试。
3. **第一份真实 B0/B1/B2 三路径梯度剖面**（CPU，无仿真器）：
   - **H=8 的 policy 梯度 77–98% 来自 critic bootstrap**（首次定量）；H=32 dynamics 才主导。
   - own-vs-shared critic 交叉：critic 更换改变 |D_critic| 达 4.2×，远大于 WM 分支差异 → **H5（critic 误差）升级为头号候选**。
   - B2 状态路径梯度更大但想象回报更差；B2−B1 配对差 CI 含 0（n=14）→ 无"B2 梯度更可靠"的证据。
   - D_direct（action-rate）在策略动作下 ≈0 → H6 当前无实证影响。

## 2. 对核心问题的当前回答

**WM 为什么欠响应**：数据可辨识性（H1）+ 平滑先验/损失形态（H2）的复合，幅值形态稳定（~3.7×）、方向大体正确；全局 gain 与现有 geometry loss 均不能修复（历史已证伪）。

**是否影响 Policy Optimization**：被 critic 路径部分掩盖——短时程梯度主要是 critic 项；动力学路径的影响要在 H≥32、policy 相关方向（H4）上验证。这把问题从"WM 准不准"重新定义为"**return 级、policy 相关方向上的梯度保真度 + imagination 中价值估计的质量**"。最终判定需要 GPU 的 D_sim 真值（预算已备）。

## 3. 下一步（等授权）

1. **Phase 0 烟测**（0.5 GPU-h）：确定性模式 + 批规模探针——最小成本、最高信息量，建议首先授权。
2. Phase 1/2（4.0 GPU-h）：B0/B1/B2 三路径 FD 全协议 + H3 有效性检验。
3. 依据结果走 P1（数据侧定向扰动）或 critic 路径改进；P3（新 loss）只在 P0-P2 证据支持下进入。

## 4. 研究方向调整声明

按 Goal §11 的许可：若 GPU 阶段证实 H5（critic 主导且不可修），研究问题应转向 **"imagination 中的价值估计误差如何主导 model-based policy gradient"**——此时 OPCL 的 Process/Outcome loss 不再是核心组件，继续保留它们反而混淆归因。若 H4 成立（误差在死方向上），欠响应修复的优先级下降，"可靠性门控"类机制（本仓库已有的 horizon decay / reliability quantile 家族）优先级上升。两种转向都会先向用户说明理由再执行。

## 5. 产物索引

- 代码：`crgf_proto/supermimic/crgf.py`、`crgf_proto/tools/crgf_{wm_directional_audit,delta_action_directions,sim_fd_audit,test_crgf}.py`、`tools/_crgf_bootstrap.py`
- 数据：`crgf_proto/results/crgf/`、`results/b0b1b2_provenance.json`、`results/b0b1b2_sha256.txt`
- 报告：本目录 9 份（同步副本：`supermimic_vibed/docs/crgf/`）
- 原始依据：`logs/true_action_jacobian_*.json`、`docs/current_frontier.md`、OPCL outputs（路径见 HISTORICAL_EVIDENCE）

## 6. 图表与执行准备（2026-11-18 增补）

- §10 关键图表已产出（CPU，`tools/crgf_figures.py`）：三路径占比、horizon 走势、方向集对比、critic 交叉，见 `results/crgf/figures/` 与 `docs/crgf/figures/`，聚合表 `fig_summary_table.csv` 逐条可追溯。
- Phase 0 烟测脚本已就绪并注册：`tools/crgf_phase0_smoke.py`（单命令、硬超时、显存熔断、自动产出 `phase1_feasible` 判定）。**授权申请已提交，等待用户批准；GPU 保持未运行。**

## 7. Phase 0 已执行，Phase 1/2 决定已提交（2026-11-18 增补）

烟测 verdict：4 envs 确定性逐位一致；≥8 envs int32 上限；FD 三次重复完全一致（ε=0.03）。设计修订为顺序分支 FD，Phase 1+2 预算 ≤2.2 GPU-h。执行决定见 PHASE1_EXECUTION_DECISION.md（GO，待追加授权）。

## 8. Phase 1+2 执行准备全部就绪（2026-11-18 增补）

- `tools/crgf_sim_fd_audit.py` **v2 顺序分支设计**完成重写（单 context ≤4 envs 复用、每 env 顺序执行 plus/minus/base、确定性重放防污染、per-eps 失败归因、墙钟熔断）——编译通过。
- WM 侧补跑 H=64（`wm_audit_b0b1b2_h8_32_64.csv`）并保存全部方向集原始单位向量（`crgf_dirs_all.pt`，含 per-model ascent）。
- `tools/crgf_analyze_fidelity.py` 判定管线完成：fidelity 统计（gated）+ H3（对历史 per-clip ΔJ 一致性的 point-biserial 检验）+ H4（policy 相关 vs random 方向配对比较）+ H5（critic vs state fidelity）+ B2−B1 配对 bootstrap，一键出 verdict。
- **Phase 1+2 授权请求已提交，等待用户批准。批准后两条命令完成全部剩余判定**：sim FD（容器内）+ 分析管线。

## 9. Phase 1+2 已执行：H=8 fidelity 判定完成（2026-11-18）

用户批准 Phase 1+2（≤2.2 GPU-h）后，v2.1 批量化顺序分支设计经 4 次实现缺陷修复（env 批形状、缓冲区维度、critic 输入切片、种子/帧配对）后成功产出 **H=8 的 D_sim 真值对照**（336 三元组，196 过门禁，帧与 WM 审计逐点配对）。

**判定（按预注册判定树，详见 EVALUATION_REPORT §6）**：
- **H5 推翻**：critic 路径是 H=8 最准的梯度路径（sign 0.83–1.00、ratio≈1）；
- **H4 支持（良性）**：dynamics 路径误差集中在离策略方向（random 0.67–0.83）而 policy 相关方向最好（0.89–1.00）；
- **H=8 区间内"欠响应威胁 policy 梯度"= NO-GO**；单步欠响应（0.27）在 return 级校准消失（ratio 0.79–1.40）；
- **B2 vs B1 无差异**（H=8）；**H3 不可计算**（历史数据无分歧方差）。

**GPU 记账**：Phase 1+2 累计 ≈1.75/2.2 GPU-h。**待追加项**：H=32/64 fidelity（≈1.5 GPU-h）——dynamics 占比上升后 6.2 节的结论是否保持，是下一步的唯一实验目标。

## 10. 跨时域判定完成（2026-11-18）

H=32/64 追加授权执行完毕（B1 35 min + B2 37 min = 1.2/1.45 GPU-h）：H=32 全 6 方向集 336 三元组（132 过门禁）+ H=64 基础 3 集 168 三元组（28 过门禁）。

**最终判定（预注册树逐分支）**：
1. **Policy 相关方向可靠结论保持并加强**：H=32 ascent 三模型 D_state/D_critic sign 全 1.000，delta D_state sign 1.000；
2. **H4 加强**：random/repeated 方向随 H 继续恶化（state sign 0.50-0.625；critic ratio 0.32-0.62）；
3. **新警告信号**：delta（真实更新）方向 D_critic@H32 sign 0.33-0.67（n=3，三模型一致，ratio 0.20-0.50）——恰在"真实更新方向 × dynamics 主导时域"；
4. **H=64 门禁后样本不足**（1-3/格），不作结论；
5. **全局 NO-GO 未成立也未否定**：delta-critic@H32 信号要求先验证（delta 方向加密运行 ≈15-20 min GPU，把 n=3 → n≥10），然后按 §6 预注册分支二选一：机制目标锁定价值通路（sign ≤0.5 保持）或全局 NO-GO 成立转论文写作（sign 回升 ≥0.85）。

**Goal 状态**：§11 的六项完成标准中，1（基线）、2（机制解释：欠响应=单步幅值乘子+方向依赖，return 级部分校准）、3（实证关系：H4 方向隔离 + 路径分解）已达成；4/5/6 收敛于单一待验证信号（delta-critic@H32）与后续 Route B/C 执行。

## 11. Goal 收敛（2026-11-18 最终）

delta 加密验证获批并执行（GPU ≈15 min）：n=15，三模型 delta-critic sign 全部 1.000——警告信号证伪。**预注册分支判定：全局 NO-GO 成立**（命题："欠响应经梯度通路威胁 policy optimization"）。

### §11 六项完成标准对照

1. ✅ 控制梯度可信度基线：三路径 × H{8,32,64} × 6 方向集，840+140 行原始 D_sim 数据；
2. ✅ 机制解释：欠响应 = 单步幅值乘子（0.27）+ 方向依赖过/欠混合，return 级部分校准（0.79-1.40），critic 主导（77-99%）；
3. ✅ 预测精度 vs 控制梯度可信度：可分离（H4 方向隔离 + H5 推翻）；
4. ✅ 策略实际收益验证：修复欠响应无可获得收益（fidelity 已高 + B2≈B1≈B0 + gain 扫描）；
5. ✅ 明确 NO-GO（EVALUATION_REPORT §8，含精确范围）；
6. ✅ 论文方案：三路径分解测量方法 + 两个反直觉发现（critic 主导、方向隔离）+ 系统性否证。

### 交付物终版清单

- 报告 10 份（`docs/crgf/`，两仓库同步）；原始数据 `crgf_proto/results/crgf/`（840+140 行 sim FD、三级 WM 剖面、gain 扫描、判定 JSON）；工具链 8 个（`crgf.py` 核心 + 7 工具 + 18 项测试）；图表 6 张；SHA256 与代码 diff。

### 唯一遗留（不阻塞 Goal 收敛）

- H=64 门禁后样本不足（参考级）；H3 需中后期 pair 分歧方差；闭环 Policy-space 验证（第 5 阶段）与 Route B/C 的机制实现需新的授权与项目周期——均已在报告中作为开放问题声明。
