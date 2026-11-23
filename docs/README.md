# docs/ 目录索引

更新：2026-11-23。本文件是 `docs/` 的导航层：四条已完成/进行中的研究线、根目录数据
文件家族、以及命名约定。研究总览与路线决策见
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)。

## 研究线目录

| 目录 | 研究线 | 时间 | 体系 | 最终裁定 | 入口文档 |
|---|---|---|---|---|---|
| （根目录散页） | Concurrent WM/policy exploration | 2026-11-05~07 | 主仓库 `supermimic/`（H64/trunc4/no_critic） | 耦合类全部负向；leading recipe = grouped WM×2→policy×2（见下方保留意见） | [concurrent_exploration.md](concurrent_exploration.md)（1886 行编号消融日志）、[leading_recipe_memo.md](leading_recipe_memo.md)、[coupling_decision_tree.md](coupling_decision_tree.md)、[experiment_matrix.csv](experiment_matrix.csv) |
| [crgf/](crgf/) | CRGF：控制相关梯度保真度 | 2026-11-17~18 | OPCL proto（35-D WM / OBS 93 / critic / H32） | 全局 NO-GO（该命题）；H5"critic 是误差源"被推翻；gate 剔除率越线未按协议降级（见路线图脚注） | [crgf/RESEARCH_DECISION.md](crgf/RESEARCH_DECISION.md)、[crgf/GRADIENT_FIDELITY_EVALUATION_REPORT.md](crgf/GRADIENT_FIDELITY_EVALUATION_REPORT.md) |
| [pada/](pada/) | PADA-0：policy-directed probe 数据效率 | 2026-11-18 | OPCL proto（B0/B2 母本） | INCONCLUSIVE；PADA-0A 离线审计把"方向性证据偏 YES"降级为不成立；PADA-Data 不获支持 | [pada/PADA0_RESULTS.md](pada/PADA0_RESULTS.md)、[pada/PADA0A_OFFLINE_AUDIT.md](pada/PADA0A_OFFLINE_AUDIT.md)、[pada/PADA_RESEARCH_DECISION.md](pada/PADA_RESEARCH_DECISION.md) |
| [cvf/](cvf/) | CVF-0 → CVF-1a/1b/2a：critic 价值梯度保真 → Sim 锚定微调 | 2026-11-19~23 | OPCL proto（online_B0@2100 母本） | CVF-0：H-CV2+评价混淆；CVF-1b v1 **无效已勘误**；v2 修正重跑：INCONCLUSIVE 负面倾向（Sim 标量锚定改善 held-out 标量价值；T=96 方向响应不可被监督改善） | [cvf/CVF0_FINAL_REPORT.md](cvf/CVF0_FINAL_REPORT.md)、[cvf/sacv/CVF1B_ERRATUM.md](cvf/sacv/CVF1B_ERRATUM.md)、[cvf/sacv/CVF1B_CRITIC_RESULTS_V2.md](cvf/sacv/CVF1B_CRITIC_RESULTS_V2.md)、[cvf/sacv/OVERNIGHT_PROGRESS.md](cvf/sacv/OVERNIGHT_PROGRESS.md) |

OPCL 本身（2026-11-14~17，B0/B1/B2/B3 训练）的原始文档与代码变体在仓库外
（`~/Documents/Codex/2026-11-14/new-chat-2/work/`）；仓库内的记录散见
[crgf/GRADIENT_FIDELITY_SOURCE_AUDIT.md](crgf/GRADIENT_FIDELITY_SOURCE_AUDIT.md)、
[cvf/CVF0_SOURCE_AUDIT.md](cvf/CVF0_SOURCE_AUDIT.md)、[pada/PADA_SOURCE_AUDIT.md](pada/PADA_SOURCE_AUDIT.md)
与 `crgf_proto/results/b0b1b2_provenance.json`。核心事实：B2（OPCL）对 B1（预算匹配对照）
无已证收益（配对差 CI 含 0）；B0/B1/B2 的 iter-2300 checkpoints 是后续三条线的共享母本。

## 根目录散页：实验证据文件家族（2026-11-05~07，concurrent 线）

`docs/` 根目录的 ~336 个 JSON / 8 个 CSV / 1 个 TXT 是 concurrent 线的**逐实验评估转储**
（每个文件 = 一次 clean/perturb 配对评估），按文件名前缀分家族。`concurrent_exploration.md`
的编号条目引用它们（同目录相对路径），因此**保持原位、不重组**；新增索引如下。

| 前缀家族 | 内容 | 对应文档章节 |
|---|---|---|
| `paired_sched_*`、`sched*`、`schedsample05_*`、`fixed_eval_sched*`、`promo_*`、`recheck_sched*`、`smoke_sched*` | WM-aux ramp / 采样调度家族 | concurrent_exploration.md 中段 |
| `freeze*`、`freezeerr*`、`policy_lr*`、`paired_sched_updates2`、`dual_q_*`、`policy_target_*`、`pt_seed*` | policy 更新数 / 学习率 / 目标策略家族 | 同上 + experiment_matrix.csv |
| `ess_*`、`riskgrad_*`、`riskmpc_*`、`rombrl_*` | risk / ESS gate / MPC / ROMBRL 家族 | rombrl_next_experiment.md、shadow 系列 |
| `cons_*`、`blend*`、`geom*`、`heightgate*`、`h4_*`、`cf_h4*`、`cf_height*` | 耦合系数 / 残差 / 几何 / 高度失败家族 | coupling_decision_tree.md、current_frontier.md |
| `clipcond*`、`contact*`、`phasecond*`、`failurestart*`、`fallgrace*`、`goal20*`、`hardclip*`、`rootoutput*`、`kinematic_*`、`joint_replace*`、`clip_*`、`oracle_*` | motion/contact/条件化消融 | external_mechanism_* 、current_frontier.md |
| `model_realism_*`、`motion_*_audit`、`subject[45]_motion_cluster*`、`ensemble_coverage*`、`shadow_*`、`dual_q_*` | WM 真实性 / ensemble / shadow-policy 审计 | shadow_policy_experiment_spec.md、strategic_probe_spec.md |
| `mnm_*`、`failure_*`、`gated_recovery*`、`fixed_eval_termination*` | norm-matching / 失败恢复家族 | mnm_gap_analysis.md、current_frontier.md |
| `budget_audit_key_runs.csv`、`experiment_matrix.csv`、`kinematic_component_weights.*`、`wm_weights_dyn.txt` | 预算 / 结果矩阵 / 权重汇总 | README、current_frontier.md |
| `*.pt`（~50 个，被 `.gitignore` 的 `*.pt` 规则忽略，仅存本地） | 固定起点评估的状态文件 / adapter 快照（可再生成） | — |

## 约定

- 研究线的预注册协议、结果、决策三件套命名：`<LINE><STAGE>_EXPERIMENT_PROTOCOL.md` /
  `<LINE><STAGE>_RESULTS.md` / `<LINE>_RESEARCH_DECISION.md`；源码审计为
  `<LINE>_SOURCE_AUDIT.md`。
- 推翻性更正一律以独立勘误文件发布（先例：[pada/PADA0A_OFFLINE_AUDIT.md](pada/PADA0A_OFFLINE_AUDIT.md)、
  [cvf/sacv/CVF1B_ERRATUM.md](cvf/sacv/CVF1B_ERRATUM.md)），不覆写原始结果文件。
- 大体量 `.pt` 证据不入库（`.gitignore` 的 `*.pt`），SHA256 记录在对应 manifest/
  provenance/registry；小体积方向集有显式否定规则例外。
- `results/` 与 `docs/` 的对应：`results/<line>/` 存原始产物与 ledger，
  `docs/<line>/` 存可读报告。
