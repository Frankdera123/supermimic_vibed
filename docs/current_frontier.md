# SuperMimic concurrent WM-policy：current frontier

## 已经稳定排除的方向

- 直接把 imagined policy-return 梯度写入 nominal WM：scalar coupling、gradient projection、conflict gate、norm matching、EMA、trust region、L2 anchor、低容量 residual、短 task prefix 和延迟启动都没有形成长期稳定收益。
- deterministic/noisy collection 混合：subject5 的 0.5 mixture 在 clean/perturb 均下降。
- 原始 disagreement penalty、半数 imagined-start uncertainty filter 和全程 ensemble fallback：只能作为安全诊断，不能作为性能机制。
- 通过目录名推断预算：已发现多个目录名与真实 checkpoint iteration 不一致，所有比较必须使用 `tools/checkpoint_budget_audit.py`。

## 当前可靠的经验

- 4-step teacher-forced WM + 8-step free-running auxiliary、权重 `.05`、ramp 300 是目前最稳定的 WM-policy 协同候选。
- `policy_updates=2` 在固定 interaction budget 下通常提高适应速度；subject4 三个 seed 的 12 个 paired 条件中，双更新+ramp 全部胜出。
- total optimizer steps 严格匹配后，subject4 与 subject5 方向不一致：subject4 seed0 单更新约高 13.3%，subject5 两 seed 汇总近似打平且方向翻转。因此不能宣称全局 compute 优势。
- nominal-prefix uncertainty gate（前 32 步 nominal，后段 train-calibrated ensemble gate）在 subject4 两 seed合并只比普通 ramp 高约 0.3%，未通过收益 gate。

## 下一步实验入口

详细的分支关闭条件和下一条 implementation order 见 [`docs/coupling_decision_tree.md`](coupling_decision_tree.md)。

1. 保持 nominal WM replay-only，不再让 policy-return 直接改写 nominal 参数。
2. 只在独立、可校准的对象中引入 coupling：优先尝试 target construction 或 replay selection，而不是继续调 gate threshold。
3. 每个新机制先做 300-step smoke；必须同时记录 WM err8、真实 sim reward、固定起点 rollout 和预算审计。
4. 正式比较至少覆盖两个 seeds，并同时报告 interaction、total optimizer steps 和 wall-clock 三条轴。
5. 若一个机制只能提高 imagined return、却不能提高真实 rollout，归类为 model exploitation，不进入主线。

## 必备审计

- `tools/checkpoint_budget_audit.py`：checkpoint iteration、sim steps、policy/WM optimizer steps、total steps、wall-clock。
- `tools/fixed_policy_eval.py`：相同起点、相同物理扰动、clean/perturb 配对真实 rollout。
- `tools/teacher_forcing_audit.py`、`tools/imagined_support_audit.py` 和 ensemble audit：分别检查监督目标迁移、support drift 和 uncertainty calibration。
- teacher-forced/free-running consistency（`.01`）完成 smoke 与两个 seed 的 1000-iteration 训练，WM 误差基本持平、训练期 sim reward 略高，但尚无稳定真实 rollout 增益；暂不纳入主线。
- termination-aware WM target（`.05`）在两个 seed 的固定起点 500-step rollout 合并约 +0.2%，方向相反且 first-failure 不稳定；停止调参，保留接口作终止校准诊断。
- EMA WM 供 policy imagination（β=.995）在训练期指标上变好，但两个 seed 的固定起点 rollout 合并下降约 10.2%，且 wall-clock 变慢；明确排除。

## 当前最有希望的 coupling

- occupancy-conditioned WM replay（weight=.5, sigma=.05, floor=.25）是目前最强候选：subject4 三 seed 的 clean/perturb 六个 paired rollout 全部胜出，均值提升约 +7.8%/+16.1%；预算、sim steps 和 optimizer steps 完全匹配。
- 该机制只改变真实 replay WM loss 的 detached sampling weight，policy 仍通过 imagined rollout 更新，避免 EMA dynamics mismatch。
- 下一步优先在 subject5 held-out split 复核，再做小范围 `weight/sigma` 敏感性。
- subject5 跨 split 复核否定了 occupancy replay 的全局泛化：单更新下降 5.2%，双更新下降 19.0%。目前只能作为 subject4 上的强候选，不能替换全局 leading recipe。
- mild occupancy（weight=.25/floor=.75）更稳：subject5 clean/perturb 约 +5%，但 subject4 clean -2.6%、perturb +8.5%；仍无跨 split 一致收益。
- mild occupancy 的下一步不是继续扫标量，而是用实时 WM validation error 做退化检测与 uniform fallback；当前证据仍不足以设为默认。
- validation-error gated occupancy（ratio=1.05）修复 subject5 clean/perturb，但 subject4 clean 下降 7.6%；它是 context-specific adaptive coupling，尚未泛化。
- 独立 replay validation EMA gate 已实现。subject5 clean/perturb +20.6%/+18.0%，subject4 clean/perturb -4.7%/+6.6%；比同 batch gate 更有证据基础，但仍是 context-dependent。
- 独立 validation ratio 放宽到 1.10 的两个 split smoke 均退化；当前不继续放宽 gate，1.05 仅作为诊断阈值。
- subject5 seed1 复核反转了 validation-EMA 的 seed0 收益（clean/perturb -10.6%/-8.1%），因此该 coupling 目前明确是高方差诊断线，不是 leading recipe。
- batch32、4 evaluation seeds 的方差审计使 validation-EMA occupancy 的合并结果约 -3.5%；此前单次 +20% 已撤回，coupling 线暂不再扩展。

## External mechanism gap

外部检索显示，当前 action-distance occupancy 不是严格的 policy-aware model learning。TOM（Transition Occupancy Matching）用 transition discriminator + dual Q 估计 `(s,a,s')` policy footprint，再做 importance-weighted WM regression；下一条研究线应先实现 classifier/odds audit，再考虑替换启发式 occupancy 权重。
- 新增 transition classifier audit；新旧 replay footprint AUROC 约 0.81–0.90，但时间片标签仍混入 recency，尚不能作为 TOM 权重。
- age/coverage-matched transition classifier（隐藏绝对 frame/age）AUROC 仍约 0.92–0.95；支持做 discriminator-odds 离线 WM weighting smoke，但尚未接入训练。
- 已完成 discriminator-odds 离线 WM weighting 接线：`tools/transition_occupancy_audit.py --save_model` + `--wm_transition_weight_checkpoint`，subject5 seed0 matched classifier（AUROC 0.936）30-iteration smoke 正常；这只是 plumbing/梯度隔离验证，尚无真实 rollout 结论。
- subject5 三个训练 seed的 odds weighting 结果为 clean/perturb `+6.65%/+6.29%`（12 条件胜出 10）；但 subject4 seed1 的四个 paired 条件为 `-4.07%/-4.38%`（仅 2/4 胜出，matched AUROC 0.840）。因此该机制明确是 split-dependent diagnostic，不进入 leading recipe。
- subject4 的 odds scale=.25 仍为 clean/perturb `-3.44%/-2.18%`、各 1/4 胜出，排除单纯“权重过强”；停止 scalar sweep，转向独立 calibration 或真正的 transition-occupancy target。
- subject4 held-out temperature calibration（T=.485）反而在 8 个 paired 条件 0/8 胜出，clean/perturb `-7.03%/-7.95%`；classifier-odds 线暂时关闭，下一步只考虑 genuine dual-Q/Bellman-flow occupancy target。
- 新增 `tools/dual_q_occupancy_audit.py`：SMODICE/TOM chi dual objective 在 subject4/5 seed1 的 effective sample fraction 约 `0.477/0.487`，P95 权重 `2.60/2.47`、最大 `13.19/15.07`，直接接入 WM 风险过高；当前只作 audit，后续必须换成当前 policy action 的 `q(s')`。
- `--use_policy_action` 已实现严格的当前 policy `q(s')` 输入；subject4 5/10/20 epochs ESS `0.892/0.885/0.859`，但 40/80 epochs 降至 `0.548/0.329`，subject5 80 epochs ESS `0.358`。下一步需 early stopping 或 target-network regularization，暂不接 WM。
- target-Q（tau=.001–.05）仍只有 ESS `.33–.38`；Q L2=.01 仅到 `.355`，gradient clip 反而 `.300`。dual-Q 当前保留为 early-stopped audit，不接 WM。
- The policy-aware simulator formulation in Dann, Mansour & Mohri
  ([arXiv:2605.29032](https://arxiv.org/abs/2605.29032)) reinforces the current
  boundary: use critic-error regions to acquire new factual transitions, rather
  than letting a policy-return gradient rewrite the factual WM. The next
  principled arm is therefore an isolated strategic-error probe buffer, not
  another replay-weight or actor-gradient coefficient.
- The storage layer is now available as
  `supermimic.strategic_probe.StrategicProbeBuffer`: a separate ring buffer
  with probe scores, source ids, state save/load, and optional score-prioritized
  sampling. Phase 0 is now executable behind
  `--strategic_probe_capacity` plus a small optional
  `--strategic_probe_weight`: replay transitions are scored by detached
  strategic error and policy compatibility, then fitted with an additive
  one-step factual loss. This is a replay bootstrap; it does not yet learn an
  error critic or collect extra simulator actions.
- Phase 1 offline critic scaffold is now implemented in
  `supermimic.strategic_critic.StrategicErrorCritic` and
  `tools/train_strategic_critic.py`: it fits detached `log1p` factual WM error
  and ranks bounded action perturbations with a held-out RMSE report. A smoke
  fit on 512 replay rows completed with train/validation RMSE
  `.339/.300` (log-error units). This is candidate ranking only; simulator
  execution and paired control evaluation remain required before promotion.
- Candidate execution calibration is currently negative: 32 critic-ranked
  actions on the smoke checkpoint had score/real-error correlation `.176`,
  and mean normalized real error was essentially unchanged versus the base
  replay action (`275.30` vs `275.39`). Treat offline critic regression as
  scaffolding only; the next iteration needs a larger policy-occupied split
  and multi-step validation before using it for active collection.
- Same-state counterfactual collection is now available through
  `tools/collect_strategic_dataset.py --action_candidates K`. A 320-label,
  four-action smoke fit had validation RMSE `1.448`; Isaac ranking was still
  slightly worse than base (`294.35` vs `294.20`) with correlation `-.264`.
  This validates the data path but closes the current small-checkpoint critic
  as a candidate for active collection.
- The active path now accepts `--critic_checkpoint` in the collector and
  `--strategic_probe_dataset` in the trainer. On a mature subject4 checkpoint,
  two critic seeds reduced one-step real WM error by about 15% across 64
  ranked actions (`285.23→240.57`, `295.45→251.37`). A matched one-iteration
  continuation restored 224 probes and had e8 `.259` versus `.295` control.
  This supports the mechanism as a probe-data generator, not yet as a policy
  improvement.
- Matched 100-iteration probe-weight sweep: weight `.1` degraded clean return
  `.2238` vs control `.2516` despite lower WM e8; weight `.01` gave clean
  `.2543` vs `.2516` and perturb `.2499` vs `.2406`, with identical end
  survival. The active probe loss is therefore highly weight-sensitive; `.01`
  is the only candidate retained for a multi-seed gate, while `.1` is closed.
- The shared-start 500-step gate did not promote `.01`: seed0 return `.0468`
  vs `.0484` control and seed1 `.0396` vs `.0397`, all with zero end survival.
  Critic target normalization and candidate-range clipping now guard the
  observed OOD score extrapolation, but no long-horizon control gain is
  established.
- Adding candidate action-deviation penalty `50` reduced selected-action RMS
  `8.00→6.69` and normalized deviation `.470→.395`; the diagnostic candidate
  error also fell, but score/real-error correlation was `-.059` and no policy
  rollout was run. This remains a selector ablation until reward/safety
  relevance is included.
- Reward-gated strategic collection (`critic_reward_weight=2`, action penalty
  `50`, probe weight `.01`) repaired the `.01` probe's long-horizon clean
  regression in a matched 500-step test: `.04840` vs `.04841` control, with
  first failure `52.56` vs `52.63`. Under perturbation it scored `.04887` vs
  `.04827`, with first failure `53.06` vs `52.50`. End survival was zero and
  all runs were height failures, so this is a stability result without a
  promotion-level gain; multi-seed validation remains necessary.
- 10-epoch policy-action dual-Q table 已接入 `--wm_dual_q_weight_checkpoint`；subject4 30-iteration smoke 正常（wm_err8=.351），但尚无真实 rollout 结论。
- subject4 完整 1000-iteration dual-Q rollout 相对 nominal：clean/perturb `-1.44%/-2.34%`，各 2/4 paired 胜出；比 classifier-odds 稳定但仍无正收益，不进入 leading recipe。
- 分段刷新 dual-Q（300→600→900，每段重算 replay table）仍为 clean/perturb `-3.29%/-3.42%`、仅 1/4 胜出，说明 stale weights 不是主因；dual-Q target 与 tracking WM objective 本身不匹配。
- 新增 `tools/model_realism_audit.py`：real-vs-nominal-WM classifier held-out AUC 为 subject4 `0.624`、subject5 `0.707`（各 4096 real/fake transitions），支持下一步 realism-gradient 小 penalty，但必须先独立 calibration。
- `--policy_realism_checkpoint` + weight=.01 已完成 subject4 30-iteration smoke，无 shape/gradient 错误；尚无长训真实 rollout 结论。
- realism penalty=.01 的 subject4 1000-iteration rollout 在 4/4 paired 条件落后，clean/perturb `-13.49%/-14.92%`；关闭 naive fixed-discriminator penalty，若继续 MnM 需 fresh online fake + joint discriminator/lower-bound。
- realism penalty=.001 仍为 clean/perturb `-2.07%/-2.17%`、各 2/4 胜出；排除单纯幅度问题，固定 offline discriminator 路线关闭。
- online real-vs-WM discriminator（每次 WM update 用当前 replay real/fake 联合训练）在 subject4 四个 paired seeds 上 clean/perturb `-9.50%/-9.08%`、0/4 胜出；固定与在线 realism penalty 路线均关闭。
- 新增 `--policy_wm_error_gate_ratio`：当当前 WM err8 高于 EMA 比例时跳过第二次 policy update。subject4 四个 paired seeds 上 clean/perturb `-3.97%/-4.20%`、均 1/4 胜出；比 realism penalty 温和但未超过固定双更新，暂不启用默认。
- `policy_awbc_weight=.01` 的真实 replay-action anchor 在 subject4 四个 paired seeds 上 clean/perturb `-4.67%/-2.55%`，胜出 `2/4` 与 `1/4`；简单 advantage-weighted behavior anchor 未解决当前 policy-action 与 replay target 的错位，暂不启用。
- 新增 `--wm_reward_weight`，按 batch 内 one-step tracking reward 对真实 WM replay loss 做 detached weighting，借鉴成功交互轨迹更新思路；subject4 完整 1000 轮（weight=.5）在四个 paired seeds 上 clean/perturb `-15.53%/-17.63%`、0/4 胜出。训练期 sim reward 的 smoke 增益属于 model exploitation，关闭该路线。
- 梯度审计支持 nominal WM 后，subject4 nominal 的 task/replay cosine 为 `-0.047/-0.040/-0.050`（horizon 4/8/16）；reward-weighted WM checkpoint 变为 `-0.083/-0.086/-0.085`，16-step task-on-replay projection 从 `-0.016` 增至 `-0.030`。这解释了 reward weighting 的真实 rollout 退化：它加剧了 policy-task 与 replay-fidelity 冲突。
- 新增 action-recovery auxiliary（`--wm_action_recovery_weight`）：由 WM 预测的 next state 反推 replay action，梯度通过预测状态回到 WM，inverse head 只作训练辅助。subject4 weight=.1 基本持平（clean/perturb `+0.69%/+0.02%`，各 2/4 胜）；subject5 weight=.1 提升 `+15.13%/+16.17%`、8/8 胜，weight=.05 提升 `+25.64%/+25.83%`、8/8 胜。普通 held-out WM error 反而变差，说明可能是 action geometry 改善而非 factual fidelity 改善；暂列 leading exploratory coupling，需第二训练 seed 和 subject4 weight 复核。
- subject5 action-recovery `.05` 的严格训练-seed复核：seed1 相对同 seed baseline 曾为 `+25.64%/+25.83%`，但 seed2 仅 `+0.16%/+0.54%`、各 2/4 胜。该机制为高方差、context-dependent action-geometry diagnostic，撤回 leading recipe，不替换固定双更新 baseline。
- action-recovery audit：subject5 seed1 held-out action-MSE/WM-error correlation `0.371` vs seed2 `0.051`；对应 gradient alignment horizon16 task/replay cosine/projection `0.192/.075` vs `0.059/.022`。这支持“可迁移 action geometry + task-compatible gradient”是收益条件，但目前仅为 checkpoint audit，尚未形成在线 gate。
- contrastive action-recovery（`wm_action_contrastive_weight=.05`）在 subject5 两个训练 seeds 上 clean/perturb 分别 `+12.89%/+12.65%`（4/4、4/4）与 `+5.61%/+3.90%`（3/4、3/4）；但 subject4 seed1 四个起点全部下降 `-9.77%/-10.89%`。相比回归式更稳定，却仍是 split-dependent，暂不进入默认 recipe。
- 对 contrastive action encoder 做动作 `/20` 归一化的 30-step smoke 无数值问题，但 WM err8 高于原始动作输入；该变体改为显式 `--wm_action_contrastive_normalize_actions`，未做完整 rollout，不纳入结果比较。
- subject4 contrastive action-recovery weight=.01 也全部退化：clean/perturb `-10.78%/-11.87%`、0/4 胜；降低权重不能修复 split mismatch，contrastive action-discrimination 暂时关闭为 split-dependent 研究分支。
- contrastive transition 改用 normalized frame-local delta（`--wm_action_contrastive_delta`）修正维度后可运行，但 30-step subject4 smoke 的 WM err8=`0.5565`，明显高于原 absolute-state contrastive smoke 约 `.324`，未进入完整 rollout。
- 2026-11-07 schedule-sampling × action-recovery interaction: seed1、100 iterations 的 `wm_schedule_sampling_prob=.5`、consistency=`.1`、action-recovery=`.05` smoke 数值稳定，训练末 sim reward=`.378`；但 fixed-policy 300-step clean/perturb return 仅 `.0856/.0871`、survival=`0/0`，首次 failure=`51.5/52.4`。该组合没有把两个辅助目标转化为控制收益，且 tilt error/angular speed 上升，关闭为 promotion 路线。
- 2026-11-07 evaluator recheck: 上述 batch16 结果与历史 batch64 记录不可直接比较；同一旧 schedule checkpoint 在当前 evaluator、batch64 下为 clean/perturb `.06908/.06942`。匹配的 schedule×action-recovery checkpoint 为 `.07079/.07115`，均 survival=`0`，只带来约 2.5% 的小幅 return 差异，不能宣称收益。后续 fixed-policy 结论统一使用当前 evaluator、batch64，并将这次 recheck 作为新的同版本基线。
- evaluator 现在支持 `--write_starts`/`--starts_file`，把 motion clip 和 offset 固定后再比较多个 checkpoint；paired 64-start 复核给出 baseline/action-recovery clean `.06908/.07079`、perturb `.06942/.07115`，确认小幅差异不是随机起点造成的。该接口成为后续 fixed-policy promotion 的强制评估路径。
- 2026-11-07 schedule-sampling × `policy_updates=2`: seed1、100 iterations 在训练期 sim reward 末值仅 `.332`，paired fixed-policy 300-step clean/perturb 从 baseline `.06908/.06942` 降至 `.05846/.05816`（约 `-15%`），survival 仍为 `0`。额外 policy update 在当前 schedule recipe 下加剧漂移，关闭该组合。
- 当前 evaluator 下的 schedule vs consistency-only matched 对照：schedule clean/perturb `.06908/.06942`，consistency-only `.06697/.06729`，短期 return 约提升 `3%`，但两者 survival 都是 `0`；schedule 的首次 failure 反而提前约 `2.6` 步，说明它改善的是累计 tracking reward，不是终止稳定性。
- 2026-11-07 `policy_before_wm` 时序消融：schedule 配置下，paired clean/perturb return 从 `.06908/.06942` 降至 `.06265/.06293`（约 `-9%`），首次 failure 提前约 `3.2` 步，tilt failure 上升到 `7.8%`。先更新 policy 再更新 WM 不是稳定化方向，关闭该变体。
- 2026-11-07 `interleave_policy_wm` 时序消融（`policy_updates=2`, `wm_batches=2`）：paired clean/perturb return 为 `.05950/.05991`，相对 schedule baseline `.06908/.06942` 下降约 `14%`；首次 failure 提前约 `2.2` 步，perturb tilt failure 达 `7.8%`。交错更新没有修复 policy/WM 漂移，关闭该变体。
- 2026-11-07 schedule × multi-step auxiliary: `wm_aux_window=8`, weight `.05`, ramp `100` 与 schedule sampling 联合训练后，paired clean/perturb return 为 `.06855/.06772`，相对 baseline `.06908/.06942` 略降；tilt failure 上升到 `7.8%/9.4%`，angular speed 约 `3.12`。更长 WM 监督没有转化为控制稳定性，关闭该组合。
- 2026-11-07 schedule × termination loss: `wm_termination_weight=.05` 的 paired clean/perturb return 为 `.06309/.06347`，相对 baseline 下降约 `9%`；survival 仍为 `0`，tilt failure 升至 `21.9%/25.0%`。schedule 没有修复 termination auxiliary 的 split-specific 失稳，关闭该组合。
- 2026-11-07 schedule × on-policy replay: `policy_before_wm`、每轮刷新 8 步、`onpolicy_replay_fraction=.25` 的训练期 WM loss/err8 爆到 `72.96/161.62`；paired clean/perturb return 为 `.06491/.06546`，相对 baseline 下降约 `6%`，tilt failure 升至 `53.1%/51.6%`。少量当前 policy 数据直接混入 WM batch 会严重破坏 factual dynamics，关闭该组合。
- 2026-11-07 保守 on-policy 复核：将 fraction 降到 `.05` 并启用 `onpolicy_separate_wm_update` 后，WM 仍漂移（末端 loss/err8=`22.65/40.36`），但 paired return 与 baseline 基本持平，clean/perturb `.06954/.06997` vs `.06908/.06942`，survival 都为 `0`。这只是中性稳定性信号，没有足够收益进入主线；保留为后续独立 fresh-data 更新的参考点。
- 修复了 `onpolicy_separate_wm_update` 的 fresh-error 诊断：独立 fresh batch 之前被错误标记为 `fresh_n=0`，日志恒为 `0.000`；现在 25-iteration smoke 正确报告 `fresh_wm_err=1.272`。该修复只影响诊断/可选 fresh-only 权重，不改变历史训练结果。
- 当前 evaluator 下的 seed0、300-iteration cross-seed 复核：schedule 相对同 seed uniform WM，clean return `.06948` vs `.06178`（+12.5%），perturb `.06866` vs `.05928`（+15.8%）；tilt failure 从 `17.2%/15.6%` 降至 `7.8%/9.4%`，但 survival 仍为 `0`。这确认 schedule sampling 的 tracking/姿态改善在第二个 seed 可重现，不过尚未解决最终 height termination。
- 2026-11-07 schedule × 小幅 height penalty: `policy_height_penalty=.01` 的 paired clean/perturb return `.06914/.06957`，仅约 `+0.1%/+0.2%`；height failure 没有下降，survival 仍为 `0`，tilt failure 轻微上升。小幅 height 梯度也不能修复最终 termination，关闭该变体。
- fixed evaluator 现在输出每个 start 的 clip/offset 与 failure subtype。对 64 个 paired starts 的初步聚合显示，schedule 的 clean failure 主要是 height（约 `61/64`），tilt failure 集中在少数 clip（clip 9、12）；这说明剩余问题更像 motion-clip/state-conditioned failure，而不是全局 height 标量。下一步应做 clip/state 分层的 WM consistency 或 data collection，而不是继续扫全局 penalty。
- 固定高风险 clip 的扩大评估没有发现单一机制收益：`fallAndGetUp1_subject4` 上 schedule/consistency clean return `.1192/.1182`、survival 都 `.0625`；`fallAndGetUp2_subject3` 上 `.2170/.2201`、survival 都 `.25`，perturb 方向相同。clip 条件确实改变 failure mix，但 schedule 在两个 clip 上方向相反，不能用 clip-specific promotion 取代全局验证。
- 新增 termination-risk-weighted consistency：`wm_consistency_risk_weight=1.0` 按 teacher 状态的 soft termination 提高 free-running WM loss。seed1 paired clean/perturb return 从 `.06908/.06942` 降至 `.06236/.06232`，tilt failure 从 `4.7%` 升至 `7.8%`，survival 仍为 `0`。直接强调临近 termination 的状态会放大局部偏差，暂不采用。
- 新增可选 `--wm_action_contrastive_corr_gate`，仅在 batch 内 contrastive action loss 与 WM transition error 相关性超过阈值时启用；subject4 threshold=.1 的 300-step smoke 末端 WM err8=`.2005`、sim reward≈`.369`，未显示优于原始 contrastive/baseline，暂不做完整 rollout。
- same-state counterfactual contrastive negatives（`--wm_action_contrastive_counterfactual`）在 subject4 `.05` 上比 batch-negative 稳定：clean/perturb `-1.65%/-2.00%`，胜出 `2/4` 与 `1/4`；仍未超过 baseline，但显著减轻原 contrastive 的 `-9.77%/-10.89%` 退化。
- subject5 seed2 的严格复核给出 clean/perturb `+3.02%/+5.65%`，胜出 `3/4` 与 `4/4`；因此 same-state counterfactual 比 batch-negative 更值得保留，但仍受 split 影响，不能替换 leading recipe。
- 修正 counterfactual 分支：动作归一化 flag 现在同时作用于正/负 action encoder；`corr_gate` 在 counterfactual 下只用正样本半段与 factual WM error 做相关性门控，避免 `2V` 对 `V` 的维度错误。
- 新增 `tools/action_contrastive_audit.py`，同时审计 WM-generated counterfactual 与真实 replay next-state 的 action 识别。same-state counterfactual 在 subject4/5 audit split 的真实 transition accuracy 约 `0.978/0.980`，普通 batch-negative 约 `0.980/0.988`；两者均很高，且不能解释 rollout 的正负差异。因此 contrastive accuracy/margin 不是充分的 deployment proxy，后续必须继续检查 task-gradient compatibility，而不是仅优化识别率。
- gradient alignment audit（batch=64）同样不能作为 deployment gate：subject4 baseline/counterfactual 的 horizon16 cosine 为 `-0.077/+0.088`，但 counterfactual rollout 仍略降；subject5 baseline/counterfactual 为 `+0.154/+0.075`，candidate rollout 上升。故 task/replay cosine 只能作解释性诊断，不能直接筛选 coupling。
- 新增 `--wm_huber_delta`，尝试对 normalized WM transition residual 使用 smooth-L1，动机来自近期 policy/WM co-training 的 noise-tolerant loss。subject4 seed1、delta=.5 的 300-step smoke 末端 survival `0.077` vs MSE `0.055`，但完整 1000-step paired rollout clean/perturb 变为 `-2.28%/-1.39%`，仅 1/4 起点胜出。smoke gain 未迁移到真实控制，关闭该 scalar Huber 线，不继续扫 delta。
- 对 baseline、counterfactual 和 Huber checkpoint 做了同 replay 的 `tools/jacobian_audit.py`。Huber subject4 的 action-Jacobian 均值从 `2.850` 提到 `2.998`，但 rollout 下降；counterfactual subject4 为 `2.916`、subject5 为 `2.953`，相对各自 baseline `2.850/2.974` 方向不一致。因此局部 action sensitivity 也不是充分的 deployment proxy，不能继续用 scalar geometry regularizer 代替真实 rollout。
- 新增 `tools/true_action_jacobian_audit.py`：在 Isaac 中从相同状态执行 `a+δ/a−δ`，直接比较真实与 WM 一步 action effect。subject4/5 的 baseline WM 方向 cosine 约 `0.823/0.809`，但 sensitivity ratio 仅 `0.274/0.284`；counterfactual 只升至 `0.285/0.292`，仍系统性低估幅值。Huber subject4 更低至 `0.232`。这提示真正的缺口是 simulator-grounded action-effect magnitude calibration，而非继续加强 self-supervised action discrimination。
- 新增 `--wm_action_effect_gain`，仅在 imagined policy rollout 中按 simulator-grounded least-squares gain（约 2.9）放大 WM 输入 action；replay WM 与真实收集不变。subject4 seed1 smoke survival 上升，但完整 1000-step rollout 相对 baseline clean/perturb `-8.38%/-10.09%`，胜出 `1/4` 与 `0/4`。全局 action gain 与真实 policy action 语义错位，关闭 scalar calibration；后续若继续只能做 state/action-conditioned calibration。
- true-action audit 增加 action magnitude sweep：subject4 action std=2/10/20 时，baseline WM 的 least-squares gain 分别约 `2.98/6.78/8.25`，真实方向 cosine 从 `.828` 降至 `.553/.447`。动作幅度越大，WM 偏差越非线性、方向也越不可靠；这直接证明 global gain 不适用，下一步必须做 action-magnitude/state-conditioned calibration 或真实 counterfactual supervision。
- action-norm quartile audit（常见 std=2 区间）显示 gain 仍随 norm 变化：subject4 baseline 四分位约 `2.93/3.21/3.53/3.27`，subject5 约 `2.73/3.23/3.36/3.42`；cosine 同时随 norm 下降。故即使在非饱和动作范围也不存在可靠固定系数，calibration 至少需要 action-magnitude/state 条件。
- 外部原始文献进一步支持多尺度评估：`Does LeWorldModel Preserve Local Action Geometry?`（arXiv:2610.31716）使用多扰动尺度的 action-Jacobian Gram，而非单一 sensitivity 标量。SuperMimic 的 delta/action-scale sweep 已观察到同样的 scale dependence；后续 simulator audit 应报告多尺度 Gram discrepancy/方向误差，不能用一个 global gain 概括。
- 新增 `tools/multiscale_action_gram_audit.py`，按 `δ=.1/.5/1.0`、每状态 4 个 action directions 比较 simulator/WM response Gram。subject4 baseline discrepancy `0.347/0.342/0.313`，counterfactual `0.351/0.370/0.337`；subject5 baseline `0.261/0.343/0.281`，counterfactual `0.288/0.365/0.301`。counterfactual rollout 虽在 subject5 为正，但 Gram 多数尺度变差；Huber Gram 略好却 rollout 为负。多尺度几何是必要诊断，但不是 control-return proxy。
- 原型化 `--wm_action_delta_calibration`：从 simulator finite-difference audit 生成 70-D local state-error component gains，并只作用于 imagined transition，保持 policy action 不变。修复一次 `NUM_J` 未定义后，subject4 300-step smoke 可运行但明显退化（survival `.023`、sim reward `.376`，MSE smoke 为 `.055/.379`）；不继续完整训练，关闭该简单 component-wise calibration。
- 新增 `tools/generate_counterfactual_dataset.py`，并生成可复用的 simulator ground-truth 数据：`data/counterfactual_s4_seed23.pt` 与 `data/counterfactual_s5_seed23.pt`。每个包含 256 个状态、4 个 action directions、`δ=.1/.5/1.0` 的真实正负 next states（约 2.1 MB）。下一步可在固定数据上训练 state/action-conditioned intervention adapter，避免重复 simulator 采样。
- 新增 `tools/train_intervention_adapter.py`：simulator counterfactual adapter 的 one-step RMSE 在跨 split 上改善（s4→s5 `4.116→3.639`，s5→s4 `4.145→3.490`），但直接接入 imagined rollout 即使 residual blend=.1 仍出现 `pi nan`。说明 one-step intervention accuracy 不足以保证递归稳定；关闭直接 inference adapter，后续应把 counterfactual 数据作为 WM 训练期监督，而不是部署时外接修正器。

- 新增 `--wm_counterfactual_dataset/--wm_counterfactual_weight`，把 simulator 生成的 `a±δd` 真实 next-state 作为 WM 训练期辅助监督。使用 subject4 测试运动生成的数据仅作 plumbing smoke（不作泛化证据）：`.05` 使 `sim_rew≈.375`、WM loss 升至 `.679`，明显不如同配置 MSE smoke；降到 `.005` 后 `sim_rew=.3793`、`imag_survival=.0464`，仍未超过 baseline。结论：该辅助项可稳定接入，但当前 one-step counterfactual loss 与递归 policy objective 不一致，暂不进入 leading recipe；保留数据生成器与训练开关，后续若继续应改为 train-split 数据、state/action-conditioned weighting 或短期 rollout consistency。
- 将 counterfactual 监督进一步拆成只拟合 `a+δd` 与 `a−δd` 的 action-effect 差分（`--wm_counterfactual_effect_weight`，训练集数据、权重 `.005`）后，300-step smoke 得到 `sim_rew=.373`、`imag_survival=.075`，低于 MSE smoke `sim_rew=.379`。因此退化并非只来自 absolute one-step loss；单独局部 action-effect 拟合也没有转化为稳定递归控制收益。当前两种 counterfactual auxiliary 均不进入 leading recipe，下一步若重启应直接采用短 rollout / task-gradient consistency，而非继续调 one-step scalar 权重。
- 首次直接测试短 horizon task-gradient residual coupling：`joint_residual_hidden=128`、`joint_wm_policy_weight=.1`、`joint_wm_replay_weight=1`、`joint_wm_policy_horizon=4`、replay window=4。300-step subject4 smoke 得到 `sim_rew=.370`、`imag_survival=.038`、`wm_err8=.255`，低于 MSE baseline smoke（`.379`、`.055`），且吞吐下降。短 horizon task gradient + replay anchor 仍不能稳定 concurrent 更新；后续应转向显式 multi-step real-transition target 或 trust-region/implicit update。
- 测试了 `joint_residual_max_norm=.1` + `joint_gradient_conflict_gate`（同样的 4-step task/replay 配置）。subject4 300-step smoke 显著崩坏：`sim_rew=.276`、`wm_err8=.547`、吞吐 `1.32 it/s`。因此当前 residual task gradient 即使加输出 trust bound 与冲突门控，也会破坏 nominal WM；简单 trust-region 变体关闭。
- 测试 delayed EMA world model：`--policy_ema_model --policy_ema_start 50 --wm_ema_beta .99`，policy 只通过 EMA target WM 反传，online WM 继续 replay-only。subject4 300-step smoke 得到 `sim_rew=.369`、`imag_survival=.069`、`wm_err8=.248`，低于 MSE baseline `.379/.055`；同时 `_imag_step` 因 requires_grad 分支触发 Dynamo recompilation，吞吐降至 `1.92 it/s`。EMA-only policy imagination 暂不纳入主线。
- 文献检索得到一个更明确的方向：Policy-Aware Simulator Learning 将 model exploitation 写成 model/policy minimax game，并用 Error-MDP 中的 strategic one-step error 做 active data selection（[arXiv:2605.29032](https://arxiv.org/abs/2605.29032)）。SuperMimic 现有 `--wm_prioritized` 只是按 factual WM error 做 replay priority；subject4 300-step smoke 为 `sim_rew=.375`、`imag_survival=.060`，低于 MSE baseline `.379/.055`，因此不能把普通 prioritized replay 当作 policy-aware selection。
- subject4 seed2 的 action-recovery `.05` 完成 1000-step 复核。训练期 `sim_rew=.423`；fixed-policy 100-step evaluator（batch=64、相同 starts）clean return `0.2345→0.2593`，perturb `0.2338→0.2579`，约 +10%，但 survival 均为 `.109`、failure rate 均为 `.891`。它改善 tracking return，却未改善摔倒率；因此保留为“tracking-only/context-dependent”分支，不进入默认 recipe。
- action-recovery subject4 seed2 的机制审计：candidate factual one-step WM error `0.1775`（P90 `.3182`），baseline `0.1614`（P90 `.2768`），即普通 dynamics fidelity 变差；candidate inverse action-MSE 与 WM error 相关性 train/audit=`.144/.181`。gradient alignment candidate horizon4/8/16 cosine=`.084/.070/.096`，baseline=`.042/.107/.145`。结合 fixed-policy return +10% 但 survival 不变，结论是该 auxiliary 学到局部 tracking/action geometry，而非稳定的 factual dynamics；暂不推广为 concurrent 主线。
- 新增 `tools/strategic_error_audit.py`：在相同 simulator 状态上执行 checkpoint policy action，同时比较真实 simulator next state 与 WM next state，直接测量 policy-selected strategic error。action-recovery candidate 将 subject4 strategic error `1.314→1.048`、subject5 `2.035→1.128`，说明其确实改变了 policy footprint；但这不保证 survival。
- 新增 `--policy_strategic_error_weight`，在最近 on-policy transitions 上以 action-proximity gate 惩罚 policy-selected WM error。`.01` subject4 smoke 将 strategic error `1.872→1.360`，但 1000-step fixed-policy 真实评估 clean/perturb return `0.2558→0.2514`、`0.2543→0.2525`，survival 均 `.109`。因此 strategic error 适合作为 policy-aware 数据选择/验证信号，直接加到 policy loss 暂时关闭。
- 新增 `--wm_onpolicy_error_weight`，仅在 isolated on-policy WM update 中按当前 model error 做 detached 高误差加权，作为 Error-MDP active-selection 的最小近似。subject4 300-step smoke（refresh16、on-policy fraction=.5、weight=.5）得到 `sim_rew=.370`、`wm_err8=.357`、吞吐 `2.12 it/s`，明显差于 baseline；因此“按当前误差放大 recent on-policy transitions”不是充分的 strategic selection，暂时关闭。
- 新增 `--wm_action_recovery_fidelity_gate`，按 factual WM error 对 action-recovery auxiliary 做反向 detached weighting，试图避免 inverse loss 牺牲基本 dynamics。subject4 weight=.05、gate=1 的 300-step smoke 为 `sim_rew=.374`、`imag_survival=.055`，低于 baseline；因此“只在低 factual-error transition 上做 action recovery”也暂时关闭。
- 新增 `--wm_action_recovery_onpolicy_only`，只在 isolated/recent on-policy WM batches 上施加 action-recovery auxiliary，避免旧 replay 分布干扰。subject4 300-step smoke（refresh16、fraction=.5、weight=.05）退化到 `sim_rew=.355`、`wm_err8=.375`、吞吐 `2.10 it/s`；因此 recent-policy-only inverse geometry 更不稳定，暂时关闭。
- 对历史 `wm4_teacher_onpolicy_nominal_s0_1000` 做固定起点复核：尽管训练期 `sim_rew=.428`，其真实 clean return 仅 `.230`，低于同 seed nominal baseline `.295`；survival 均 `.109`，failure 类型从全 height 转为含 tilt failure。该 run 的 interaction budget 也约为 baseline 的 2.1 倍（669,952 vs 320,512 sim steps）。结论：高训练期 reward 是 model exploitation，fresh on-policy replay 不能直接视为有效 concurrent 机制。
- 复核现有 `--wm_rollout_sensitivity_weight=.1`：按多步 imagined rollout 的 tracking-reward sensitivity 做 detached WM replay weighting。subject4 300-step smoke 得到 `sim_rew=.375`、`imag_survival=.046`、吞吐 `1.81 it/s`，低于 baseline；多步 sensitivity weighting 暂时关闭。
- 将 `tools/strategic_error_audit.py` 稳定化：policy action 与 WM step 均在 `no_grad` 下执行，物理步进显式使用 detached action，并支持 `--output <json>`。subject4 seed2 batch=8 smoke 成功输出 policy-selected strategic error（均值约 `1.275`）及 action-error correlation，后续可直接作为 policy-aware 数据选择的机器可读输入。
- 新增 `tools/policy_aware_selection_audit.py`，离线比较 uniform、policy compatibility、`compatibility × factual WM error` 和可选 ensemble disagreement 选择。subject4/5 各 2048 样本中，`compatibility × error` 的 top-25% 平均误差分别为 uniform 的约 `2.64x/2.33x`，说明它确实富集 policy-compatible 高误差 transition；但 disagreement 几乎不改变排序。
- 将该选择分数接入 `--wm_policy_aware_weight` 做历史 replay-only WM 拟合。subject4 seed2 的 300-step smoke：weight=`.5` 时 sim reward `.369`、固定起点真实 return `.167`；weight=`.1` 时 sim reward `.400`、真实 return `.195`；baseline 分别约 `.379/.224`，survival 均未改善。结论：selection signal 有诊断价值，但直接按 strategic error 加权会加剧训练期 reward 与真实控制的脱钩，当前关闭，不进入 leading recipe。
- 又加入 `--wm_policy_aware_floor` 与 `--wm_policy_aware_validation_ratio`，测试 floor=`.75`、weight=`.1`、validation ratio=`1.02`。subject4 300-step smoke 的 sim reward `.393`，但固定起点真实 return `.190`，仍低于 baseline `.224`，survival 不变；因此简单 uniform floor/WM-error validation gate 也不能修复 selection-induced control gap，policy-aware 权重继续保持关闭。
- 新增 policy-only reliability gate：`--policy_wm_reliability_quantile` 按 factual replay WM error 对 imagined policy gradient 做 detached 样本门控，WM 仍保持 uniform replay。subject4 median gate、floor=`.5` 的 300-step smoke 真实 fixed-policy return `.174` vs baseline `.224`，survival 不变。结论：高 factual-error start 可能包含必要的纠偏信号，简单“只信低误差 WM 区域”的 policy gate 也关闭。
- 新增 `--policy_proximal_weight`，对每次 policy update 的起始 action 加 pre-update policy proximal trust-region。subject4 seed2 的 300-step smoke 中 weight=`.1/.01` 的真实 fixed-policy return 分别为 `.203/.174`，均低于 baseline `.224`，survival 不变；因此起始动作级 proximal 约束会削弱有效纠偏，当前关闭。
- 新增 `--adaptive_action_noise`：用 frozen ensemble disagreement 只调节真实 simulator 的 action exploration，WM 仍按 uniform factual replay 拟合。subject4 300-step smoke（scale=`.5`）clean fixed-policy survival 从 `.094` 提到 `.125`、failure `.906→.875`，return `.224→.214`；perturb survival 未提升（均 `.094`），且训练吞吐约从 `3.5` 降到 `2.3 it/s`。这是目前唯一同时降低 clean failure 的主动收集候选，但仍需同 seed/多 split 复核，暂不进入默认 recipe。
- subject5 同配置复核（同 split nominal baseline ensemble）return `.190` vs `.243`、survival 均为 `0`，未复现 subject4 的 clean survival 信号；因此 adaptive noise 目前判定为 split-specific，关闭。ensemble loader 同时支持 nominal/residual 成员，保留作后续 uncertainty/active-collection 基础设施。
- 新增 `--ensemble_policy_softmin_temperature`，对 nominal/ensemble 的完整 imagined return 使用 soft-min，直接做 policy-side minimax，而不是额外惩罚 disagreement。subject4 τ=`.05` 的 300-step smoke 真实 fixed-policy return `.177` vs baseline `.224`、survival 不变，吞吐约 `1.1 it/s`；因此 robust return soft-min 当前关闭。
- 新增 `--wm_reward_mismatch_weight`，按 WM 预测与真实 replay 的 task reward/soft-termination mismatch 对 factual WM loss 加权。subject4 weight=`.5` 的 300-step smoke 训练 sim reward `.384`，但真实 return `.181` vs baseline `.224`、survival 不变；因此 task-aware mismatch weighting 仍不能阻止闭环 exploitation，当前关闭。

- 2026-11-06 adaptive recoverability-gated exploration: subject5, nominal WM ensemble, `adaptive_action_noise=.5` with `adaptive_noise_recoverability=.2` (300-step smoke). Training sim return reached `.372`, but fixed-policy eval (batch32, 100 steps) fell from baseline `return_norm=.2428` to `.1694`; survival remained `0`, first failure step worsened `48.34→39.19`. Closing this variant; recoverability gating of exploration noise is not a candidate.
- 2026-11-06 TD-error-weighted WM replay: added a matched critic-only comparison (`critic_epochs=1`, `critic_minibatches=1`) on subject4/5. `wm_td_weight=.5` improved 100-step fixed-policy return relative to the matched critic baseline (subject4 `.1775` vs `.1646`, subject5 `.1617` vs `.1500`), but remained below the established no-critic dual-update baselines (`.2236/.2428`) and survival stayed `0.094/0`. Height-failure behavior also worsened on subject5. Treat as critic-conditioned diagnostic only; do not add a critic or TD weighting to the leading recipe.
- 2026-11-06 explicit multi-step real-transition target: added `--wm_multistep_weight`/`--wm_multistep_horizon`, which recursively rolls the WM over recorded replay actions and anchors the final prediction to the actual simulator state after 2 steps. Subject4 300-step smoke (`weight=.1`, horizon=2) ran stably, but fixed-policy return fell `0.2236→0.1629`; survival stayed `.094` and first failure moved `46.88→39.19`. The extra recursive target therefore amplifies short-horizon fitting error under the current normalized loss; close this simple version rather than tuning scalar weight.
- 2026-11-06 directional disagreement exploration: added `--adaptive_action_direction`, differentiating frozen ensemble disagreement with respect to the action and querying the real simulator along that direction. Subject4 300-step smoke (direction=`.1`, nominal ensemble) reduced training throughput to ~4.0 it/s and fixed-policy return fell `0.2236→0.1798`; survival stayed `.094` although mean first failure moved `46.88→42.75` and failure mix shifted. Directional disagreement is not a reliable active-collection target under this setup; keep the interface for diagnostics, disabled by default.
- 2026-11-06 sparse directional exploration: added `--adaptive_action_direction_fraction` to apply disagreement-gradient action probes only on a subset of collected transitions. Subject4 300-step smoke with direction=`.1`, fraction=`.25` improved over all-step direction (`return .1872` vs `.1798`) but remained below baseline `.2236`; survival stayed `.094` and height failures increased. Rare probes reduce the damage but do not create a useful coupling signal; directional active collection remains disabled.
- 2026-11-06 low-weight multi-step target check: reducing the explicit 2-step real-transition target from `.1` to `.01` improved its smoke return (`.1802` vs `.1629`) but remained below the no-target baseline `.2236`; survival stayed `.094` and failure mix shifted toward tilt. This rules out simple over-weighting as the main issue; the recursive target form is closed for now.
- 2026-11-06 detached-intermediate multi-step target: tested the DAgger-like variant that rolls the WM through its own intermediate state but detaches that state before the next action, then fits the real 2-step target (`weight=.1`). Subject4 300-step smoke gave fixed-policy return `.1668` vs baseline `.2236`, survival unchanged `.094`, and more tilt failures. Detaching the recursive path does not repair the target mismatch; close this branch.
- 2026-11-06 policy-side failure-shaping recheck: current subject4 seed2 with root-height penalty `.05` reduced height failure `.875→.656` but raised tilt failure `.031→.25` and return `.2236→.1840`; adding tilt penalty `.02` further shifted failures (`height=.50`, `tilt=.469`) and return `.1541`. This confirms the apparent height benefit is failure-mode trading, not a better controller; both penalties remain disabled.
- 2026-11-06 full actor-critic concurrent baseline: enabled the default critic (`critic_epochs=5`, `critic_minibatches=4`) while retaining WM imagination and two policy updates. Subject4 300-step smoke fixed-policy return was `.1591` vs no-critic baseline `.2236`; survival stayed `.094` and tilt failure increased. A critic is therefore not a free fix for the policy/WM mismatch; keep the no-critic recipe and do not pursue TD weighting as the main coupling.
- 2026-11-06 BPTT state-only truncation: tested `policy_bptt_state_only` with truncation=4, preserving policy history/action gradients while detaching only WM states. Subject4 300-step smoke fixed-policy return `.1609` vs baseline `.2236`, survival unchanged `.094`; the extra recurrent gradient path worsened control, so the default full-history detach remains.
- 2026-11-06 ramped root-height penalty recheck: ramping `.05` over 300 iterations reduced subject4 seed2 height failure `.875→.75`, but increased tilt failure `.031→.219` and reduced return `.2236→.1590`. The schedule does not remove the failure-mode tradeoff; root-height policy shaping is closed for the current recipe.
- 2026-11-06 imagined action-noise regularization: tested `imag_action_noise=.05`, injecting Gaussian action perturbations only inside policy WM rollouts so the actor optimizes a local neighborhood return. Subject4 300-step smoke fixed-policy return fell `.2236→.1708`; survival and failure rates were unchanged. This simple smoothing does not prevent exploitation and remains disabled.
- 2026-11-06 direct minimax residual test: enabled `joint_adversarial_residual` with a 128-hidden residual, 4-step policy task horizon, and replay-fidelity anchor. This is the closest implementation to adversarially training WM against policy exploitation. Subject4 300-step smoke was unstable (`sim_rew=.249`, throughput ~2.65 it/s); fixed-policy return `.1312` vs baseline `.2236`, survival unchanged `.094`. The naive reversed task gradient is strongly destructive; future minimax work needs a constrained critic/lower-bound objective rather than sign-flipping the residual gradient.
- 2026-11-06 norm-bounded minimax residual: changed `joint_adversarial_residual` to scale the reversed task gradient relative to the replay gradient via `joint_gradient_norm_ratio`, then tested ratio=`.1`. Subject4 300-step smoke remained unstable (`sim_rew=.304`, throughput ~2.69 it/s); fixed-policy return `.1294` vs `.2236`, survival `.094`. Gradient magnitude was not the sole failure; naive residual minimax is closed.
- 2026-11-06 residual validity audit: `tools/validity_classifier_audit.py` found a useful offline signal in the failed adversarial residual checkpoint: a held-out env classifier-selected subset had residual improvement rate `.622` and mean error delta `-.0103`, versus overall improvement rate `.474`. However, the simple existing disagreement threshold (`.00168`) used as an online residual gate made training much worse (subject4 300-step `sim_rew=.272`, throughput ~1.67 it/s, fixed-policy return `.0995` vs `.2236`). Thus residual validity is state-dependent, but disagreement threshold is not a deployable gate; a learned classifier would need calibration and online integration before further residual training.
- 2026-11-06 learned residual-validity artifact: extended `tools/validity_classifier_audit.py` with `--save_model` and saved `logs/adversarial_validity_gate.pt`. The held-out calibration remains positive (selected rate `.340`, residual improvement `.622`, mean delta `-.0103`), so the signal is worth preserving for a later calibrated online gate; no training claim is made yet.
- 2026-11-06 online learned validity gate: loaded `logs/adversarial_validity_gate.pt` into imagination and continued the adversarial residual checkpoint for 30 iterations. The gate ran successfully, but subject4 fixed-policy return was `.1317` (survival `.094`, failure `.906`, first failure `42.8`) versus ungated adversarial `.1294`; this is statistically indistinguishable and still far below the no-residual baseline `.2236`. The classifier signal does not yet rescue concurrent residual training; keep it as an audit artifact, not the leading recipe.
- 2026-11-06 matched full-WM concurrent smoke: removed residual adapter and let the complete WM receive the policy-imagination gradient, with replay anchor weight 1 and joint policy weights `.001` and `.0001`. Both 300-iteration subject4 runs had stable training (`sim_rew≈.384-.389`, `wm_err8≈.048`) but identical fixed-policy return `.1840` (survival `.094`) to the matched no-joint 300-iteration control `.1840`. This weak coupling is effectively inert at these scales; it neither improves nor destabilizes the controller, so it is not a useful concurrent-training gain yet.
- 2026-11-06 experiment-audit correction: all earlier 300-iteration smoke runs used the default `wm_only_iters=500`, so their policies had not updated; their fixed-policy comparisons cannot support conclusions about concurrent policy/WM coupling. A valid 600-iteration no-joint subject4 control (same large config) reached fixed return `.21495`.
- 2026-11-06 structured output-layer coupling: added `joint_wm_policy_last_layer`, routing the policy imagination gradient only to `wm.net[-1]` while replay gradients update the full WM. The large 64-env/64-step run exposed an autograd-memory OOM at the first policy update, so it is not yet scalable at that configuration. A matched small run (16 env, horizon32, `wm_only_iters=100`, 300 iterations) completed: structured coupling fixed return `.1796` vs no-joint `.1526`, with survival `.125` in both; this is a promising but single-seed small-budget result requiring larger-memory-efficient implementation and confirmation.
- 2026-11-06 memory-efficient structured backward: rewrote `joint_wm_policy_last_layer` to compute task/policy gradients and replay gradients separately with `retain_graph=False`, then manually merge them; this avoids the previous AOTAutograd donated-buffer error and large rollout-graph OOM. The 16-env/horizon32 smoke completed through policy updates. Its fixed return was `.1677` (survival `.125`, height failure `.813`, tilt `.188`) versus matched no-joint `.1526`; the positive direction is smaller than the earlier buggy implementation's `.1796`, but remains a modest single-seed signal. Large-scale confirmation is still pending.
- 2026-11-06 structured last-layer second seed: corrected memory-efficient implementation, same 16-env/horizon32/300-step setup. Seed0 fixed return `.1637` vs matched no-joint `.1586` (+3.2%); seed2 remains `.1677` vs `.1526` (+9.9%). Both have identical survival `.125`, with failure-mode redistribution. The effect is directionally consistent but small and not yet sufficient for the main recipe.
- 2026-11-06 structured coupling coefficient sweep: with seed0 and the corrected small setup, increasing `joint_wm_policy_weight` from `.001` to `.005` reduced fixed return to `.1537` vs no-joint `.1586` (survival unchanged `.125`; height failure `.563`). The weak positive at `.001` is therefore coefficient-sensitive; `.005` is disabled and `.001` remains diagnostic only.
- 2026-11-06 action-path coupling: added `joint_wm_policy_action_path`, projecting policy gradients only onto the WM first-layer columns that receive the 29-D action input; replay gradients still update all WM parameters. Corrected small runs at weight `.001` improved fixed subject4 return for seed0 `.1586→.1849` (+16.6%) and seed2 `.1526→.1748` (+14.5%), with survival unchanged `.125`. This is the strongest consistent concurrent signal so far, though still from the reduced 16-env/horizon32 budget and requires larger-budget confirmation.
- 2026-11-06 matched 600-step action-path confirmation: reduced setup, seed0, action-path `.001` reached fixed return `.1771` versus matched no-joint `.1670` (+6.0%), survival `.125` in both. The gain shrinks from the 300-step comparison (`+.0263`) but remains positive after a longer policy-training budget; action-path coupling remains the leading experimental arm pending a larger-env run.
- 2026-11-06 env-scale confirmation: action-path `.001` at 32 envs, horizon32, 600 iterations completed without memory issues (`wm_err8=.216`, `sim_r=.396`). Fixed subject4 return was `.19695` vs matched no-joint `.19535` (+0.8%), survival `.094` for both. The stronger small-batch gains do not survive this scale change; action-path coupling remains stable but currently neutral at the larger batch.
- 2026-11-06 action-path norm-matched screening: added `joint_wm_policy_action_norm_ratio`, which scales the action-path task gradient to a fixed fraction of the replay-gradient norm. At 32 envs, ratio `.1`, 300 iterations, training remained stable (`wm_err8=.285`, `sim_r=.375`); fixed return `.1618` is recorded as a screening point, not a matched claim because the 32-env no-joint control at the same 300-step budget was not run.
- 2026-11-06 matched action-path norm ratio: at 32 envs and 300 iterations, gradient-norm ratio `.1` fixed return `.1618` versus no-joint `.1686` (−4.0%), survival `.094` for both. The norm-normalized variant is therefore not an improvement and remains disabled; fixed scalar action-path `.001` is still the more stable candidate.
- 2026-11-06 delayed action-path coupling: fixed structured gating to honor `joint_wm_policy_start_iter` and `joint_wm_policy_period`. At 32 envs/300 iterations, delaying action-path `.001` until iteration 200 improved fixed return to `.1759` vs matched no-joint `.1686` (+4.3%), survival `.094` for both. This is stronger than coupling from the first policy update and motivates a 600-step delayed run.
- 2026-11-06 delayed long-budget check: 32 envs, 600 iterations, action-path `.001` enabled only after iteration 300 reached fixed return `.1907` versus matched no-joint `.1953` (−2.4%), survival `.094` for both. The 300-step delayed gain does not persist at longer budget; delayed coupling is closed for now.
- 2026-11-06 sparse action-path long-budget check: `joint_wm_policy_period=4`, weight `.005`, 32 envs/600 iterations reached fixed return `.1796` vs no-joint `.1953` (−8.1%), survival `.094` for both. The encouraging 300-step result (`.1860` vs `.1686`) does not persist; sparse coupling is closed for the current recipe.
- 2026-11-06 action-path conflict-gate long check: 32 envs/600 iterations with `joint_gradient_conflict_gate` reached fixed return `.1757` vs no-joint `.1953` (−10.1%), survival `.094` for both. Conflict gating improves short-horizon screening (`.1799` vs `.1686`) but not long-budget control; no action-path coupling variant currently survives the 600-step scale test.
- 2026-11-06 action-path task-gradient EMA: smoothing the action-path task gradient with `joint_gradient_task_ema=.9` and weight `.005` at 32 envs/300 iterations produced fixed return `.1680` vs no-joint `.1686` (−0.4%). EMA removes the short-run gain rather than fixing the long-budget drift; keep disabled.
- 2026-11-06 cross-subject generalization audit: the 32-env/600-step action-path checkpoint that was near-neutral on subject4 (`.19695` vs `.19535`) fell to `.1776` on unseen subject5, below no-joint `.2067` (−14.1%) and with zero end survival. The coupling is therefore not transferable across subjects; do not treat its subject4 short-run gains as a general method.
- 2026-11-06 mixed-subject coupling check: training on subjects1–4 (32 envs/300 iterations) and evaluating subject5 reduced the action-path generalization gap: action-path `.1636` vs no-joint `.1550` (+5.5%), although both had zero end survival. On subject4 the same pair was `.1833` vs `.1857` (−1.3%). Multi-subject data helps transfer slightly but does not produce a robust control gain.
- 2026-11-06 leading-recipe policy-update audit: with `policy_updates=2` (instead of the recent one-update screening), 32-env/300-step action-path coupling fixed return `.1641` versus matched no-joint `.1906` (−13.9%). Thus the action-path signal does not survive the actual double-policy-update recipe; the main concurrent coupling direction is closed pending a fundamentally different stability mechanism.
- 2026-11-06 target-model lifecycle utility: added `supermimic/policy_target.py`, a deep-copied frozen target wrapper with hard-sync, EMA-sync, and independent replay audit gating. Unit tests cover architecture-preserving construction, parameter/buffer sync, EMA updates, and pass/fail gating. This is infrastructure only; it is not yet wired into the training loop or claimed as an algorithmic result.
- 2026-11-06 target-model audit integration: added `--policy_target_audit_max_error` and `--policy_target_audit_interval`. When enabled, the policy-sensitive EMA target is evaluated on a freshly sampled replay batch after each WM update; imagination uses it only while the normalized one-step audit error is below the threshold, otherwise it falls back to nominal WM. The default remains disabled (`0`), and no performance claim is made until a matched policy-updates=2 run is completed.
- 2026-11-06 first audit-gated target experiment: matched 32-env/300-iteration, `policy_updates=2`, subject4 training split. With audit threshold `.5`, fixed subject4 return was `.1788` versus nominal `.1751` (+2.1%), survival `.09375` for both; a final replay proxy audit error was `.228`, below threshold, so target use was active. The matched subject5 cross-subject run reversed the result: `.1549` versus nominal `.1684` (−8.0%), with zero end survival in both. This is a split-sensitive diagnostic, not a promoted recipe; the target gate needs a stronger policy-sensitive adapter and longer-budget validation.
- 2026-11-06 audit-gated target long-budget check: extending the same subject4 arm to 600 iterations reversed the short-run result: target `.1857` versus matched nominal `.2174` (−14.6%), survival `.09375` for both and earlier first failure (`43.1` vs `47.1`). One-step replay auditing is therefore insufficient to guarantee long-horizon control; EMA-only target coupling is closed pending a multi-step held-out trust-region adapter.
- 2026-11-06 multi-step target audit gate: added `--policy_target_audit_horizon`; the gate now requires the maximum normalized error over an independently sampled replay rollout horizon to pass the threshold. A horizon-2/50-iteration Docker smoke completed without runtime errors (`wm_err8=.526`), so the mechanism is ready for the next adapter experiment; default horizon remains 1 for backward compatibility.
- 2026-11-06 offline action-conditioned adapter audit: added `tools/target_adapter_audit.py`, fitting a residual adapter on half of replay environments and evaluating a disjoint contiguous 4-step audit split. On the subject4 600-step nominal checkpoint, three seeds with trust weight `.1` worsened audit error by `+8.0%`, `+6.3%`, and `+20.0%`. The naive adapter overfits environment-specific replay and is not ready for online policy coupling; future adapter splits must be motion/subject stratified and use a stronger trust region.
- 2026-11-06 time-stratified adapter follow-up: splitting each environment by time and shrinking the adapter hidden width to 32 changed the picture but not the conclusion. Subject4 600-step checkpoint had relative audit changes `−6.4%`, `−9.4%`, `+6.7%` across three seeds; subject5 300-step checkpoint had `−12.8%`, `+11.5%`, `−8.9%`. Two of three seeds improve on each split, but variance is large and this is teacher-forced fidelity only; keep it as an offline lead, not an online method.
- 2026-11-06 free-running adapter audit: the same subject4 time-split hidden-32 adapter produced essentially no free-running improvement across three seeds (`+0.005%`, `+0.007%`, `−0.004%`) despite teacher-forced changes. Thus the adapter is fitting local one-step residuals without reducing rollout drift; do not promote it to online coupling.
- 2026-11-06 multi-step adapter objective: changed the offline adapter trainer to optimize a 4-step free-running loss. At the previous lr/trust setting it became unstable (held-out error increased about 17×); a conservative `lr=1e-5`, trust weight `1.0` stabilized it, but free-running error changes across three subject4 seeds were only `−0.015%`, `−0.008%`, `−0.013%`. Direct multi-step residual fitting therefore does not provide a useful coupling signal by itself.
- 2026-11-06 ensemble uncertainty audit: bootstrap disagreement was not a reliable multi-step error proxy. On subject4, one-step correlation was `.130` but final 4-step correlation was `−.186`; on subject5, the corresponding values were `.349` and `.132`. A global disagreement gate therefore has no stable cross-subject semantics and remains diagnostic only.
- 2026-11-06 horizon-2 target gate check: subject4 32-env/300-step target run improved fixed return `.1751→.2038` (+16.4%) and survival `.09375→.125`, but the matched 600-step run reversed to `.1834` versus nominal `.2174` (−15.7%), with survival `.09375` for both. Multi-step audit gates improve short-run selection but do not prevent long-budget policy/WM drift; target-gate coupling is closed for now.
- 2026-11-06 target-gate observability: checkpoints now record independent audit pass/fail counts in addition to the last audit error and enabled flag, so future target experiments can report fallback coverage rather than infer it from the final model state.
- 2026-11-06 pessimistic task-model adapter cross-split check: the subject4
  recipe (`adapter_hidden=32`, pessimistic return, horizon-4 audit, 25%
  relative trust margin, `policy_updates=2`) was replicated on the subject5
  training split for 300 iterations. At 500-step fixed evaluation, return
  fell from `.04433` nominal to `.03094` (−30.2%), end survival from `.0078`
  to `0`, and first failure moved earlier by `11.40` steps. The bounded
  task-model adapter is therefore split-sensitive and is closed as a promoted
  coupling arm. `checkpoint_budget_audit.py` confirms both runs used the same
  `sim_steps=25024`, `policy_steps=580`, and `wm_steps=600`; the adapter only
  increased wall time (`0.020h` vs `0.011h`).
  The matched run also passed 211/600 independent audits (35.2%) and failed
  389/600; its final adapter norm was `3.37` with audit error `89.0`. Similar
  fallback coverage on subject4 did not prevent its long-budget reversal, so
  audit fallback frequency is not itself a deployment-quality signal.
- 2026-11-06 strategic-error mechanism audit: on the same held-out subject5
  states, the nominal policy had mean/p90/p99 WM error `1.479/1.667/1.678`
  with action RMS `.358`; the adapter policy rose to
  `1.820/2.087/2.198` and action RMS `.477` (error/action correlation `.159`).
  The adapter therefore expands the policy footprint into regions where the
  nominal world model is less accurate, supporting active factual probes
  chosen by strategic error rather than further direct adapter coupling.
- 2026-11-06 sparse target EMA: added `--policy_target_ema_interval` and tested horizon-2 audit with EMA sync every 10 WM updates. Subject4 300-step fixed return fell `.1751→.1584` (−9.5%), survival stayed `.09375`, and failures shifted from height to tilt with an earlier first failure. Slowing target synchronization is not a stability fix; keep interval 1/default and close this schedule variant.
- 2026-11-06 action-recovery gradient-alignment gate: added `--wm_action_recovery_alignment_min`, which suppresses the inverse-action auxiliary when its WM gradient cosine with factual replay is negative. A subject4 300-step smoke improved only `.1751→.1811` (+3.4%, survival unchanged), while the matched subject5 run fell `.1684→.1461` (−13.3%, survival 0). Gradient alignment filtering does not remove split sensitivity; close this gate.
- 2026-11-06 action-recovery gradient projection: added `--wm_action_recovery_project_grad`, projecting the auxiliary WM gradient onto the nonnegative factual-gradient direction while retaining inverse-head updates. Subject4 300-step fixed return was `.1704` versus nominal `.1751` (−2.7%), with no survival/failure-mode improvement. The stricter projection also fails to provide a coupling gain and is closed.
- 2026-11-06 target adapter scaffold: extended `PolicyTargetWorldModel` with an optional zero-initialized low-rank adapter, separate adapter parameter access/norm, and EMA logic that never overwrites adapter parameters. The default wrapper remains adapter-free; three unit tests verify zero-init nominal equivalence, frozen nominal parameters, and adapter preservation across EMA.

### 2026-11-06 — policy-target adapter runtime closure

The opt-in `PolicyTargetWorldModel` adapter path now accepts the full `WorldModel.step(..., use_residual=...)` interface. A Newton/RTX smoke run with `policy_target_adapter_hidden=16` and replay audit threshold 100 completed 4 iterations, saved `last.pt`, and recorded `policy_target_enabled=True`, 4 audit passes, 0 failures, plus adapter and optimizer state. A stricter threshold-2 smoke correctly kept the target disabled (0 passes, 8 failures). This validates lifecycle and gating behavior only; it is not a performance result.

### 2026-11-06 — pessimistic target and relative audit gate

The target-coupling arm now supports `--policy_target_pessimistic`, which takes the minimum of the task-model imagined return and the detached nominal/EMA reference return, and `--policy_target_audit_ratio`, which rejects a task adapter when its independent replay error exceeds the nominal target by the configured relative margin. A 4-iteration Newton smoke with horizon-2 audit, pessimistic return, and 25% relative margin completed with 4 audit passes/0 failures and persisted adapter plus optimizer state. This remains a wiring/safety validation, not a long-budget performance result.

### 2026-11-06 — first 300-iteration adapter smoke (subject4/train_not4)

Run `logs/policy_target_s4_300` used the leading 4+8-step WM recipe, `policy_updates=2`, horizon-4 independent audit, pessimistic target return, and a 25% relative audit margin. It completed 300 iterations without runtime failure. The target gate passed 223 audits and failed 377; final state was disabled with audit error 181.65, so fallback coverage was substantial rather than continuously trusting the adapter. The adapter state norm reached 4.81. Fixed-policy evaluation at 100 steps/seed123 gave return_norm 0.2011 versus 0.1538 for the archived `target_nominal_s4_300` checkpoint (same evaluator), with both failing before 100 steps. This is a preliminary, checkpoint-level signal only: the baselines are not yet a matched new pair, and the frequent gate failures mean it does not qualify as a promotion.

Subject5 fixed evaluation of the same 300-iteration checkpoints (100 steps, seed123, `lafan1_g1_test_s5`) gave return_norm `0.1969` for the adapter run versus `0.1531` nominal; mean first failure step `43.1` versus `37.8`. Both had 100-step survival 0. This cross-motion result is directionally consistent with subject4, but remains a short-horizon paired checkpoint comparison rather than a 600-iteration promotion test.

The 500-step fixed evaluation weakens the short-horizon interpretation. On test_s4, adapter return_norm was `0.04785` vs `0.04548` nominal, but mean first failure was earlier (`54.3` vs `60.2`) and it introduced 3.1% tilt failures. On test_s5, return_norm was `0.04387` vs `0.03334` and first failure improved (`44.7` vs `40.9`), but both had zero end survival and the adapter still introduced 3.1% tilt failures. Thus the 100-step gain does not establish robust long-horizon stability.

### 2026-11-06 — adapter parameter trust region

Added `--policy_target_adapter_max_norm`, an explicit post-step projection on the task adapter. A 100-iteration Newton smoke with cap 1.0 completed; checkpoint global adapter norm was exactly 1.0, with 59 audit passes/41 failures. This confirms cumulative adapter drift can be bounded independently of replay gating; it is not yet a performance result.

The 100-iteration capped smoke did not improve the fixed subject4 evaluator: return_norm `0.1690` versus archived nominal `0.1751`, equal end survival 0.09375, and slightly earlier mean failure (`41.8` vs `42.6`). The cap is therefore a stability control, not evidence of a gain.

### 2026-11-06 — rejected adapter proposals now roll back

The replay audit is now a true acceptance test: a failed audit restores the last accepted adapter state instead of merely disabling the next rollout. A 20-iteration strict-ratio Newton smoke completed with 11 accepted/9 rejected audits; checkpointed adapter and last-good states matched at norm 1.0, confirming rollback state is persisted.

### 2026-11-06 — 300-iteration rollback arm

The full subject4 run with adapter cap 1.0, pessimistic return, 25% relative audit margin, and reject-and-rollback completed 300 iterations. It recorded 246 accepted and 354 rejected audits; final adapter norm was 1.0. At 500-step fixed evaluation it scored return_norm `0.04205` vs `0.04548` nominal, mean first failure `50.3` vs `60.2`, and 3.1% tilt failures. Rollback improves proposal safety semantics but does not rescue long-horizon control; this arm is rejected for promotion.

### 2026-11-06 — low adapter learning-rate arm

Added `--policy_target_adapter_lr` and ran subject4 for 300 iterations with adapter LR `1e-5`, cap 1.0, pessimistic return, relative audit margin 25%, and rollback. Audits: 204 accepted/396 rejected; adapter norm 1.0. At 500-step fixed evaluation, return_norm was `0.04802` vs `0.04548` nominal and first failure `59.2` vs `60.2`; however 3.1% tilt failures remained and end survival was 0. This is closer to nominal stability but still not a robust gain.

### 2026-11-06 — low-LR cross-split replication

A second 300-iteration run (seed1, same train split) with adapter LR `1e-5`, cap 1.0, pessimistic objective and rollback yielded 199 accepted/401 rejected audits and ended disabled (last audit error 367.8). On the subject5 500-step fixed evaluator, return_norm was `0.03559` vs `0.03441` nominal and first failure was `40.8` vs `39.4`; both had zero end survival, and the adapter again introduced 3.1% tilt failures. The small return/first-failure improvement is not enough to overcome failure-mode regression.

### 2026-11-06 — temporal checkpoint audit of low-LR arm

On subject5 500-step evaluation, the low-LR run was non-monotonic across checkpoints: iteration100 return `0.03518` (tilt failure 9.4%, first failure 41.8), iteration200 `0.03268` (no tilt failures, first failure 38.4), iteration300 `0.03559` (tilt failure 3.1%, first failure 40.8). All had zero end survival. This indicates the small return changes are training noise/failure-mode redistribution rather than a stable adapter trajectory.

### 2026-11-06 — default-path regression check

A 20-iteration Newton smoke with all policy-target flags at defaults completed normally (`sim_r=.382`, `wm_err8=.964`). The checkpoint contained no adapter, no target optimizer, and `policy_target_enabled=False`, confirming the new rollback/LR code is opt-in and does not alter the default recipe lifecycle.

### 2026-11-06 — horizon-specific ensemble coverage calibration

Extended `tools/ensemble_coverage.py` with `--horizons` so disagreement thresholds are calibrated separately at each rollout step. Using archived subject4 checkpoints (`target_nominal_s4_300` and `_600`) with window 8, the median-disagreement selection retained 74.6%/70.7%/69.7% of audit windows at horizons 1/4/8, while `error<=2` coverage fell from 100%/98.9% to 83.8% at horizon 8. The selected mean error rose from 0.118 to 1.440. This supports horizon-specific uncertainty scaling and confirms that a one-step threshold cannot be reused for long imagined rollouts. Raw calibration output is in `docs/ensemble_coverage_s4_horizon.json`.

### 2026-11-06 — horizon-specific ensemble policy gate

Added `--ensemble_policy_gate_horizon`: with `--ensemble_policy_gate_fraction`, imagined starts can now be ranked by disagreement at a calibrated rollout step instead of the previous all-step mean. A 20-iteration Newton smoke using archived subject4 members, fraction=.5 and horizon=4 completed successfully with logged gate fraction 0.500. This is infrastructure validation; no performance claim yet.

### 2026-11-06 — horizon-4 ensemble gate long-budget screening

A matched 300-iteration run using archived ensemble members, `ensemble_policy_gate_fraction=.5`, and the new `ensemble_policy_gate_horizon=4` completed with gate fraction exactly 0.5. At 500-step fixed evaluation, subject4 return_norm was `0.04502` vs `0.04548` nominal, but first failure moved earlier (`51.7` vs `60.2`); subject5 return_norm was `0.04304` vs `0.03441`, first failure `43.9` vs `39.4`. Both subjects had zero end survival and no consistent failure-rate improvement. Horizon-specific gating is functional but not yet a control improvement.

### 2026-11-06 — horizon-8, 75%-coverage ensemble gate

A milder uncertainty gate (`ensemble_policy_gate_horizon=8`, fraction=.75) completed 300 iterations. At 500-step evaluation, subject4 return_norm was `0.04442` vs `0.04548` nominal and first failure was earlier (`53.0` vs `60.2`); subject5 was `0.03703` vs `0.03441`, first failure `42.6` vs `39.4`, but introduced 3.1% tilt failures. End survival remained 0 for both subjects. Relaxing coverage did not turn the calibrated gate into a robust control improvement.

### 2026-11-06 — thresholded uncertainty exploration

Added `--adaptive_action_noise_threshold` so ensemble-driven real-data exploration activates only above a normalized disagreement threshold. A 300-iteration subject4 run with scale=.5 and threshold=1.5 completed. At 500-step evaluation, subject4 return was `0.04443` vs `0.04548` nominal with earlier failure (`51.3` vs `60.2`); subject5 was `0.03816` vs `0.03441`, first failure `41.5` vs `39.4`, and 3.1% tilt failures. End survival remained 0, so thresholding does not yet turn uncertainty-guided collection into a robust gain.

### 2026-11-06 — symmetric directional probes

Added `--adaptive_action_direction_symmetric`, sampling random ± signs for ensemble-gradient action probes to avoid systematic bias. A 300-iteration subject4 run with direction=.1 completed. At 500-step evaluation, subject4 return_norm was `0.04626` vs `0.04548` nominal but first failure was earlier (`53.1` vs `60.2`); subject5 was `0.03934` vs `0.03441`, first failure `41.9` vs `39.4`, with 3.1% tilt failures. End survival stayed 0, so zero-mean directional probing does not yet provide a robust gain.

### 2026-11-06 — sparse symmetric directional probes

The symmetric probe was restricted to 25% of collected actions. At 500-step evaluation, subject4 return_norm improved to `0.05062` vs `0.04548` nominal and first failure moved later (`63.0` vs `60.2`), with no tilt failures; subject5 was `0.03941` vs `0.03441`, first failure `42.1` vs `39.4`, but introduced 3.1% tilt failures. End survival remained 0 on both. This is the first probe variant with a subject4 long-horizon return/first-failure improvement, but it is not transferable enough to promote.

### 2026-11-06 — sparse symmetric probe second seed

The candidate sparse symmetric probe was replicated with seed1. At 500-step evaluation, subject4 return_norm fell to `0.03860` vs `0.04548` nominal, first failure `47.9` vs `60.2`, and 3.1% tilt failures; subject5 fell to `0.03337` vs `0.03441`, first failure `37.5` vs `39.4`, with 9.4% tilt failures. End survival remained 0. The seed0 subject4 gain (`.05062`) is therefore seed-specific; close this probe for promotion.

### 2026-11-06 — probe action-support audit

`tools/policy_support_audit.py` on nominal and both sparse-symmetric probe checkpoints found action-distance/error correlations near zero (nominal `-.031`, seed0 `-.052`, seed1 `-.020`). Probe checkpoints had lower factual WM error, but their policy/replay action distances were slightly larger (mean `0.0189/0.0183` vs `0.0155`) without selecting lower-error support regions. This suggests the seed0 control gain was not explained by a simple action-support shift; probe quality remains uncalibrated.

### 2026-11-06 — counterfactual action-geometry audit of sparse probe

Simulator finite-difference audit (subject4 test, batch8, delta=.1, action_scale=2) compared nominal and seed0 sparse-symmetric probe. The probe reduced least-squares action gain bias from `4.21` to `3.64` (true simulator sensitivity normalized to 1), but also reduced action-vector cosine from `.825` to `.804` and increased normalized predicted sensitivity ratio `.185→.210`. Thus it changes action geometry in a mixed way: gain magnitude moves closer to the simulator while directional fidelity worsens. This partially explains why the control return signal did not transfer to subject5.

The same action-geometry audit on the seed1 sparse probe reproduced the pattern: gain `4.21→3.44`, cosine `.825→.823` (p10 `.727→.720`), and sensitivity ratio `.185→.229`. Thus the probe consistently changes magnitude toward the simulator while worsening or failing to improve directional fidelity; it is not a seed0-only artifact.

### 2026-11-06 — multiscale Gram audit of sparse probes

Across simulator action perturbations δ=.1/.5/1.0, seed0 sparse probe Gram discrepancy was `0.411/0.381/0.396` vs nominal `0.410/0.361/0.369`, while seed1 was `0.403/0.351/0.367`. Response-vector cosine changed only slightly and sensitivity ratios increased from nominal `.159/.163/.164` to `.195/.201/.202` (seed0) and `.215/.221/.221` (seed1). The probe does not consistently improve multi-dimensional action geometry; its return signal should not be promoted.

### 2026-11-06 — offline vector-valued counterfactual adapter

The intervention trainer now preserves state/δ/direction groups and optionally adds paired signed-effect and normalized multi-direction Gram losses (`--effect_weight`, `--gram_weight`). On the subject4 counterfactual set, ordinary MSE reached holdout RMSE `0.687` and cross-subject5 RMSE `2.810`; the vector objective reached `0.679` and `3.135`, respectively. Thus the vector losses improve in-domain grouped holdout error slightly but worsen cross-subject transfer in this first offline test. The adapter remains diagnostic only and is not wired into the online training loop.

### 2026-11-06 — isolated uncertainty probe-buffer collection

Added `--adaptive_probe_onpolicy_only` so ensemble-driven exploration can be confined to the fresh `onpolicy_buffer`; historical replay remains uniform and the fresh data enters through the isolated WM update. A matched subject4 300-iteration smoke (`onpolicy_refresh_steps=8`, replay fraction `.25`, thresholded adaptive noise `.5/1.5`) completed, but fixed 500-step evaluation was worse than the no-probe control on both splits: subject4 return `0.03406` vs `0.03874` and first failure `46.2` vs `50.3`; subject5 return `0.02647` vs `0.03268` and first failure `36.6` vs `40.8`. End survival was zero for all four cases. The probe-buffer separation is functional but the uncertainty-triggered noise target is rejected for promotion.

### 2026-11-06 — online counterfactual Gram supervision

Extended the simulator counterfactual WM auxiliary with opt-in `--wm_counterfactual_gram_weight`, computing normalized response-vector Gram loss across all available action directions at each sampled state. A matched 300-iteration subject4 run with one-step weight `.01`, effect weight `.1`, and Gram weight `.1` improved subject4 return `0.04834→0.05144` and first failure `55.0→63.2`, but degraded subject5 return `0.04173→0.03871` and first failure `44.7→41.3`; end survival remained zero. The vector objective is therefore useful as an auditable training signal but is split-sensitive and not promoted.

### 2026-11-06 — held-out geometry gate calibration

Extended `tools/multiscale_action_gram_audit.py` with a reference/candidate gate using per-scale Gram discrepancy and response-vector cosine tolerances. On the held-out subject5 simulator split, the Gram candidate passed all three scales against the effect reference (Gram deltas `−.0055/+ .0036/−.0047`, cosine deltas `+.0008/+.0015/−.0018`), even though its fixed-policy 500-step return was worse. Vector geometry is therefore a necessary diagnostic but not a sufficient promotion gate; it must be combined with real-rollout failure checks.

### 2026-11-06 — cross-motion counterfactual Gram supervision

Used the broader `counterfactual_train_not4_seed23.pt` collection instead of subject4-only counterfactuals, with the same `.01/.1/.1` one-step/effect/Gram weights. Against the subject4-only Gram checkpoint, subject4 500-step return fell `0.05144→0.04431` (first failure `63.2→49.8`), while subject5 rose slightly `0.03871→0.03887` (first failure `41.3→41.4`); end survival remained zero. Broader counterfactual coverage improves transfer only marginally and sacrifices the source split, so unweighted cross-motion mixing is not a promotion candidate.

### 2026-11-06 — balanced cross-motion mixture

Added `tools/merge_counterfactual_datasets.py` to merge compatible datasets by matching δ pairs and concatenating sampled states. A 1:1 subject4/train-not4 mixture with the same Gram recipe produced subject4 return `0.05144→0.05101` and subject5 `0.03871→0.03764`, with a 6.25% tilt-failure rate on subject5. Simple balancing does not recover transfer and is closed; motion-conditioned weighting remains hypothetical rather than an observed gain.

### 2026-11-06 — source-separability diagnostic for motion weighting

A 5-fold logistic probe on the two counterfactual collections found that `xi` alone cannot distinguish subject4 from train-not4 (AUC `0.50`), while the initial state vector separates them perfectly (AUC `1.00`). Thus an `xi`-conditioned weighting rule has no identifiable source signal in these data, whereas state-conditioned weighting would risk learning a domain shortcut. No weighting model was added on this evidence.

The reusable `tools/counterfactual_source_probe.py` also measures normalized true response geometry. Its source AUC is `0.184` (orientation-inverted, hence highly separable), confirming that action-response geometry itself is strongly split-specific. Any response-conditioned weighting must therefore be evaluated on a third motion split rather than tuned on these two domains.

With subject5 added as a third split, pairwise response-geometry probes remain strongly separable: AUC `0.166` for subject4↔subject5 and `0.174` for subject5↔train-not4 (orientation-inverted values about `0.83`). `xi` remains at chance. This supports response-conditioned weighting only as a domain-specific hypothesis, not as a universal correction.

The source-probe utility now reports pairwise metrics automatically for three or more datasets, avoiding an unstable high-dimensional multiclass score. It reproduces chance `xi` AUC for all pairs, state AUC `1.00/1.00/.895`, and response AUC `.166/.184/.174` for the three pairings.

### 2026-11-06 — weak Gram-weight arm

Reduced `--wm_counterfactual_gram_weight` from `.1` to `.02` while keeping the one-step/effect weights fixed. Subject4 return fell from `.05144` to `.04546`; subject5 rose from `.03871` to `.04302`, but tilt failures increased to `18.75%`. Lowering the Gram weight therefore trades source return for a new failure mode and does not satisfy the dual-split gate.

Added `--json_out` to `tools/fixed_policy_eval.py` and a strict `tools/fixed_policy_gate.py`. On the subject5 `.1` versus `.02` Gram evaluation, the gate correctly rejected `.02` solely because tilt failure rose `3.125%→18.75%`, despite its higher return and later mean first failure. Failure-mode rejection is now reproducible rather than an informal review step.

### 2026-11-06 — Gram-only counterfactual arm

Removed the paired effect term while keeping `gram_weight=.1` and one-step weight `.01`. Subject4 return fell `0.05144→0.04657`; subject5 rose `0.03871→0.04240`, but subject5 tilt failures increased `3.125%→6.25%`. The effect term is not the sole source of split sensitivity; the Gram-only arm also fails the strict gate.

### 2026-11-06 — counterfactual component error audit

Decomposed held-out one-step counterfactual RMSE for effect+Gram, Gram-only, and weak-Gram checkpoints over position/rotation/linear velocity/angular velocity/joint position/joint velocity. Across subject4 and subject5, differences were small; the dominant error remained joint velocity (`4.41–4.58` in normalized units), while position and rotation stayed near `0.004/0.03`. The Gram-induced tilt trade-off is therefore not explained by a single exploding pose component, supporting a multi-step failure-aware gate rather than more component-weight scans.

### 2026-11-06 — Gram seed replication

Replicated the subject4-only effect+Gram arm with seed1 under identical settings. Against seed0's `0.05144/0.03871` returns, seed1 scored `0.04031` on subject4 and `0.03211` on subject5, with tilt failures `6.25%` and `9.375%` and earlier first failures. The apparent seed0 subject4 gain is therefore seed-specific; close the Gram branch for promotion.

### 2026-11-06 — deterministic counterfactual sampling

Added opt-in `--wm_counterfactual_deterministic`, which cycles state/δ/direction indices instead of randomly resampling them. On seed1, this improved subject4/subject5 returns from `0.04031/0.03211` to `0.04460/0.04192`, but remained below seed0's `0.05144/0.03871` and introduced `6.25%` tilt failures on both splits. Deterministic sampling reduces variance but does not pass the strict dual-split gate; retain it as infrastructure for a future held-out target gate.

Added independent `--wm_counterfactual_batch`; a full 256-state cyclic batch on seed1 scored `0.05026`/`0.03624` on subjects4/5, with subject5 tilt failures `15.625%`. Full coverage does not stabilize the target and can amplify its domain mismatch, so batch-size scaling is closed.

### 2026-11-06 — one-step counterfactual baseline

The same 300-iteration recipe with only the ordinary counterfactual one-step loss (`weight=.01`, no effect/Gram terms) scored `0.03195` on subject4 and `0.02776` on subject5, with tilt failures `21.875%` and `25%`. Multi-direction supervision is therefore materially better than one-step fitting on seed0, but the seed1 failure shows that the current target construction remains unstable rather than universally useful.

## 2026-11-06 policy-occupancy counterfactual branch

The counterfactual generator previously centered perturbations on Gaussian random actions (`tools/generate_counterfactual_dataset.py`). It now accepts `--policy_checkpoint` and uses the checkpoint policy action at the sampled simulator state, with the same zero-history/current-observation context as the strategic-error audit. A matched 300-iteration run with one-step=.01, effect=.1, Gram=.1 on `counterfactual_policy_s4_seed23.pt` was evaluated under the fixed-policy gate: subject4 return `0.03805`, mean first failure `52.94`, tilt failure `0`; subject5 return `0.03156`, mean first failure `43.78`, height failure `96.9%`, tilt failure `9.4%`. The earlier random-action counterfactual candidate was stronger on this seed (subject4 `0.05144`, first failure `63.19`; subject5 `0.03871`, first failure `41.31`). Conclusion: action-occupancy mismatch is a real data-quality issue, but policy-centered one-step/Gram supervision alone is not a promoted recipe; close this branch pending multi-step on-policy counterfactuals or a better state-distribution construction.

## 2026-11-06 multi-step on-policy counterfactuals

`tools/generate_counterfactual_dataset.py` now supports `--policy_burnin_steps`: it rolls the saved policy in the simulator, copies the resulting physical state into the counterfactual slots, and measures plus/minus transitions around the actual policy action. A burn-in-4 dataset was generated with 256 states and 4 directions. The matched 300-iteration effect+Gram run was decisively worse: subject4 return `-0.00028`, mean first failure `36.78`, height failure `84.4%`, tilt failure `18.8%`; subject5 return `-0.00687`, mean first failure `31.88`, height failure `84.4%`, tilt failure `21.9%`. Thus even genuine on-policy local geometry is not a safe WM training target under the current loss weighting; close this injection branch and use such data for diagnostics only until a consistency-preserving objective is designed.

## 2026-11-06 gradient/sensitivity cross-check

`gradient_alignment_audit.py` now accepts `--seed` so task/replay gradient geometry can be repeated. Across seeds 0, 7, and 13 (horizons 4/8/16), cosine means were random-counterfactual `[.043,.061,.045]`, policy-centered `[.041,.025,-.019]`, and burn-in-4 on-policy `[.015,.095,.218]`; standard deviations were large. There is no robust negative-gradient conflict explaining the rollout failure. Separately, `horizon_jacobian_audit.py` gave cumulative action-sensitivity means 2.66 (random), 1.42 (policy-centered), and 1.12 (burn-in-4), while fixed-policy returns degraded. Current evidence favors over-smoothing / loss-induced dynamics distortion over a simple value-vs-replay gradient conflict.

## 2026-11-06 sensitivity-preservation auxiliary

Added `--wm_counterfactual_norm_weight`, matching the normalized magnitude of the predicted symmetric action effect to the simulator effect magnitude (clamped SmoothL1). A direct weight of `.1` was numerically unstable and produced NaNs by iteration 50. A stabilized `.01` run stayed finite, but its 100-step cumulative action sensitivity was only `0.93`, still below the random-counterfactual reference `2.66`; the term did not restore useful sensitivity. Treat direct absolute norm matching as unstable/insufficient. A future version should use a frozen reference WM and relative bounded ratios, not raw simulator effect magnitudes.

## 2026-11-06 same-state relative sensitivity gate

Added `tools/relative_sensitivity_audit.py`, which uses identical replay starts, identical reference-policy actions, and identical random probes for all checkpoints. Relative to `cf_gram_s4_300` (mean sensitivity 2.675), policy-centered was 0.728x (p10 .473), burn-in-4 on-policy was 0.651x (p10 .389), and the stabilized norm trial was 0.477x (p10 .269); all failed the bounded [0.67, 1.5] ratio gate. This confirms the sensitivity collapse is a checkpoint-level effect, not a sampling artifact.

## 2026-11-06 frozen-reference relative sensitivity

Added `--wm_counterfactual_reference_checkpoint` and `--wm_counterfactual_relative_weight`, which compare the current predicted symmetric action-effect norm to a frozen reference WM on the same counterfactual batch using a bounded log-ratio loss. A 100-iteration test with relative weight `.01` remained finite but had same-state sensitivity ratio mean `0.429` (p10 `.244`, p90 `.654`) against the reference, failing the `[.67, 1.5]` gate. Thus a weak relative penalty does not prevent early sensitivity collapse; do not promote or simply increase its weight without first controlling update scale.

## 2026-11-06 matched-stage sensitivity correction

The relative gate must compare checkpoints at the same training stage. Comparing 100-step candidates against the 300-step `cf_gram_s4_300` reference falsely exaggerated collapse: a no-counterfactual 100-step baseline had ratio `0.409x` to that older reference. Re-running against the matched `baseline100` reference gave scaled counterfactual ratio `0.877x` (p10 .482, p90 1.311) and frozen-relative ratio `0.881x` (p10 .632, p90 1.162). Mean sensitivity is therefore broadly preserved at 100 steps; only the lower tail remains outside the strict per-sample gate. Future gates must use matched iteration/seed baselines and report both mean and tail metrics.

## 2026-11-06 matched-stage fixed-policy check

The 100-step low-weight counterfactual candidate was compared with the 100-step no-counterfactual baseline under the same 500-step fixed-policy evaluation. Subject4: baseline return `.03064`, first failure `44.66`; scaled candidate `.02562`, first failure `35.25`. Subject5: baseline `.02880`, first failure `41.38`; scaled candidate `.02756`, first failure `38.44`. Thus matched-stage sensitivity means near `.88` do not translate into improved control; the low tail and multi-step rollout still matter.

## 2026-11-06 lower-tail hinge attempt

Added `--wm_counterfactual_relative_floor` for a one-sided log-ratio penalty that only acts below a reference sensitivity floor. A matched 100-step test with floor `.8` and relative weight `.01` reduced the sensitivity ratio mean to `.699` (p10 `.463`, p90 `.945`), worse than the unconstrained relative trial (`.881`). The hinge did not isolate a benign tail; it interacted with WM fitting and lowered the whole distribution. Close this formulation.

## 2026-11-06 one-step-only counterfactual ablation

A 100-step matched ablation with simulator one-step loss only (`wm_counterfactual_weight=.001`, no effect/Gram) was also worse than the no-counterfactual baseline: subject4 return `.02733` vs `.03064`, first failure `40.75` vs `44.66`; subject5 `.02334` vs `.02880`, first failure `34.94` vs `41.38`. The harm is therefore not specific to the Gram term; direct counterfactual injection itself is not currently compatible with the concurrent recipe.

## 2026-11-06 policy-side intervention adapter check

Tested the existing frozen intervention adapter as a policy-side imagined-transition correction only (`--wm_intervention_adapter_checkpoint`, blend `.1`), with no counterfactual loss injected into the main WM. The 100-step run immediately became invalid in imagination (`v=inf`, imagined reward about `-7.2`, survival `0`), despite finite simulator reward. The adapter correction is therefore out of distribution or numerically unsafe in this interface; do not enable it without a bounded residual/trust-region wrapper.

## 2026-11-06 bounded adapter correction

Added `--wm_intervention_adapter_max_ratio` to cap corrected local-delta norm relative to the nominal WM delta. With blend `.1` and ratio cap `1.5`, imagination remained finite (unlike the uncapped adapter), but fixed-policy results were mixed: subject4 return `.03106` vs matched baseline `.03064`, first failure `47.22`; subject5 `.01956` vs `.02880`, first failure `32.47`. The cap prevents numerical blow-up but does not produce a robust controller; do not promote.

## 2026-11-06 adapter validity gate

Added `--wm_intervention_adapter_gate_ratio`, which rejects (falls back to nominal WM) corrections whose norm ratio exceeds a threshold. At gate ratio `1.2`, the imagined rollout stayed finite, but fixed-policy returns were worse than matched baseline: subject4 `.02316` (first failure `43.5`) and subject5 `.01740` (first failure `34.16`). A validity gate alone does not recover control; close direct adapter correction.

## 2026-11-06 validity-classifier transfer check

Re-audited the existing residual validity classifier across held-out simulator environments. Training selected 25% of states with 97.9% residual-improves fraction; held-out selection rate rose to 34.0% but improves fraction fell to 62.2% (delta residual-minus-nominal `-0.0103`). The classifier has weak transferable signal for residual WM, but it is not an adapter gate and is not strong enough evidence to justify adding another gate to the intervention adapter.

## 2026-11-06 intervention adapter offline domain audit

A direct one-step audit of the s4-trained adapter found strong but domain-dependent behavior: improvement on `counterfactual_s4_seed23` was `98.97%` of perturbations (mean normalized error change `-2.86`), on s5 `87.52%` (`-1.65`), but on the broader `train_not4` dataset only `79.51%`, with mean error change `+1.03` (worse). The adapter contains useful local information, but its validity is source-dependent; this explains why ungated imagined use is unsafe and motivates adapter-specific domain/validity features rather than more global norm caps.

## 2026-11-06 adapter-specific validity classifier

Added `tools/intervention_validity_audit.py`, labeling each adapter correction by whether it lowers the true one-step simulator error. A classifier trained on s4 labels transferred poorly: held-out AUC was `.529` on s5 and `.401` on train_not4 (near or below chance), despite a high s4 positive rate (`.990`). The deployment-time 170-D features therefore do not support a reliable cross-domain adapter gate. Close adapter gating unless new domain diversity or richer state/context features are collected.

## 2026-11-06 multi-domain adapter

Merged s4/s5/train_not4 counterfactuals (768 states) and trained a mixed adapter. Its offline one-step RMSE was strong on held-out rows (`.524`) and s5 evaluation (`.492`), but policy-side imagined use with gate ratio 1.2 remained mixed: subject4 return `.02914` / first failure `46.91`, subject5 `.02257` / first failure `36.44` versus 100-step baseline returns `.03064/.02880`. Domain diversity improves local fitting but does not make direct imagined correction robust.

## 2026-11-06 policy-target action-conditioned adapter

Ran the first matched 100-step policy-only target experiment with a zero-initialized action-conditioned adapter (`hidden=32`, parameter norm cap `.1`, audit gate enabled). Nominal WM factual updates were unchanged. Fixed-policy results were subject4 `.02999` / first failure `48.81` and subject5 `.01963` / first failure `38.72`, versus matched baseline `.03064/.02880`. The target adapter does not transfer across motions; close this arm unless the target is conditioned on richer domain/context features.

## 2026-11-06 policy-target audit failure mechanism

The policy-target adapter's final state showed `enabled=false`, audit error `130.45` with the original small adapter settings; reducing adapter LR to `1e-5` and parameter cap to `.01` still ended disabled with audit error `313.34` (109 passes, 91 failures). The adapter is not merely over-updating gradually: its two-step replay audit is intrinsically violated under this target construction. Do not spend more time on LR/cap scans.

## 2026-11-06 EMA target isolation

A matched 100-step policy-target run with no adapter (EMA target only, beta .995, independent replay audit) isolated the wrapper effect. It was also worse than baseline: subject4 return `.02290`, first failure `39.97`; subject5 `.01980`, first failure `35.59` versus baseline `.03064/.02880`. Thus the degradation is not solely from the trainable target adapter; the current EMA target policy-side coupling itself is not robust.

## 2026-11-06 full EMA target correction

The earlier EMA-only result was partly confounded by the absolute audit threshold: the target was disabled after audit error exceeded 100. A corrected run used a very high absolute threshold plus a 25% relative audit ratio; the target stayed enabled for all 200 audits. At 100 steps it briefly improved returns (`.03345/.03153` vs baseline `.03064/.02880`), but the 300-step continuation collapsed to `.01850/.01659`, with subject5 tilt failure `46.9%` and first failure `26.59`. Full EMA target coupling is therefore short-horizon overfit, not a robust recipe.

## 2026-11-06 scheduled policy-target fallback

Added `--policy_target_until_iter`, forcing policy imagination back to nominal WM after a chosen early phase. With EMA target enabled only through iteration 100, the 300-step seed0 run improved both held-out returns versus matched baseline300: subject4 `.03245` vs `.02376`, subject5 `.02931` vs `.01419`. A seed1 replication was mixed: subject4 `.03192` vs `.02838` improved, but subject5 `.02294` vs `.02485` degraded. This is the first long-budget target variant with a dual-subject seed0 gain, but it is not yet robust; it requires more seeds and a better fallback criterion than a fixed iteration cutoff.

## 2026-11-06 dynamic relative audit failure

Tested a dynamic fallback with no fixed cutoff: target enabled whenever held-out replay error was no worse than nominal (`policy_target_audit_ratio=0`). On seed1 all 600 audits passed and target stayed enabled, yet 300-step fixed-policy returns collapsed to `.01733/.01190` (subject4/5). Replay fidelity is therefore not a sufficient proxy for policy imagination quality; the scheduled cutoff remains the only currently useful fallback signal, though it is seed-sensitive.

## 2026-11-06 cutoff-length sensitivity

On seed1, reducing scheduled target usage from 100 to 50 iterations improved the trade-off but did not make it dual-split positive: subject4 return `.03165` (baseline `.02838`), subject5 `.02387` (baseline `.02485`). The early target window helps subject4 consistently, while subject5 remains slightly worse; cutoff length alone is not a robust fallback criterion.

## 2026-11-06 cutoff-length sensitivity, seed0 replication

The same `policy_target_until_iter=50` recipe was rerun at 300 iterations with seed0. Fixed-policy evaluation gave subject4 return `.03386` and subject5 `.02995`, versus matched baseline300 `.02376/.01419`; first-failure means were `51.59/46.41` steps. This makes the early-window lead consistent across the two tested seeds for both subjects, but the runs still have zero end-of-horizon survival and nontrivial tilt failures; two seeds are insufficient for promotion. Keep the schedule as the current leading hypothesis and require more seeds plus a deployment-relevant fallback signal.

## 2026-11-06 matched seed2 cutoff replication

The required same-seed baseline changes the cutoff conclusion. At seed2, `policy_target_until_iter=50` returned `.03174/.02549` on subject4/5, while the matched nominal baseline returned `.03836/.03096`; first-failure means were `49.66/41.69` versus `54.09/41.91`. Thus the fixed early target window is positive on seed0, mixed on seed1, and negative on seed2. Close fixed-cutoff promotion; any further target coupling needs a fallback driven by deployment-relevant policy value or short real-simulation performance.

## 2026-11-06 training-signal audit

Added `tools/policy_target_signal_audit.py` and exported the three-seed joins in
`docs/policy_target_signal_audit_s4.csv` and `docs/policy_target_signal_audit_s5.csv`.
Training-time metrics do not identify the deployment regression: on seed2 the target run has higher final `sim_rew` (`.313` vs `.295`) and `imag_rew` (`.242` vs `.219`) than its nominal baseline, yet fixed-policy return is lower on both held-out subjects (`.0317/.0255` vs `.0384/.0310`). Replay/imagination metrics are therefore unsuitable as the sole fallback signal; a short held-out simulator probe is required if target coupling is revisited.

## 2026-11-06 short real-simulation checkpoint probe

Ran seed3 with `--ckpt_every 50` and evaluated all six checkpoints on identical fixed starts. The paired subject4/subject5 returns were: iter50 `.03514/.03102`, iter100 `.03900/.03003`, iter150 `.02413/.02003`, iter200 `.02402/.01246`, iter250 `.03997/.03339`, iter300 `.02452/.01968`. The deployment signal is strongly non-monotonic, and the sharp drop at iter150 is visible in both subjects while training metrics remain finite. This validates the proposed probe mechanism, but also shows that one probe can be noisy; a practical fallback needs repeated short probes or a confidence bound before disabling the target.

## 2026-11-06 repeated probe confidence check

Repeated the checkpoint probe with batch128 and two independent start seeds (123, 456). Mean returns for iter100/200/250 were respectively `.04227/.02671`, `.02459/.01286`, and `.04300/.02794` on subject4/5; pooled standard deviations were `.00778/.00590/.00754` across the four subject/seed values. The iter200 collapse remains much larger than probe noise, so a repeated probe can reliably detect this kind of degradation. Use at least two start seeds and a minimum-effect threshold before disabling a target.

## 2026-11-06 executable probe gate

Added `tools/policy_probe_gate.py`, which consumes paired `fixed_policy_eval.py` JSON files and rejects a candidate when the lower confidence bound of its relative return change falls below the allowed degradation margin. On matched seed2 baseline/target checkpoints, two subject probes gave relative changes `-.1399` and `-.1397`; with a 5% allowed drop the gate correctly rejected the target (`lower bound -.1400`). Unit tests cover both accept and reject paths.

The same gate also accepts the seed0 target: paired subject4/5 relative changes were `+.464/+ .785`, with lower confidence bound `+.310` under the same 5% margin. Thus the gate distinguishes the observed positive and negative seed cases using the intended paired real-simulation evidence.

## 2026-11-06 external probe control integration

Added `--policy_target_control_file` to `supermimic/train.py`. During imagined rollout, the trainer reads this optional file; explicit contents `disable`, `off`, `false`, or `0` force nominal WM, while a missing/empty file preserves the configured target behavior. This provides a process boundary for a probe runner to disable target coupling without interrupting training. The control path is intentionally one-way in the safety direction: malformed or partially written content cannot enable a target that was otherwise disabled.

`policy_probe_gate.py` now accepts `--control-file` and atomically writes `enable` or `disable`, so a rejected paired probe can directly trigger the trainer's nominal fallback. The control-file path has a dedicated unit test.

Added `tools/run_probe_gate.py` to orchestrate paired evaluation over multiple motion splits and start seeds, then invoke the gate. An end-to-end smoke run on the seed0 subject4 pair completed successfully and wrote `logs/runner_smoke/target.control = enable` after a `+49.0%` paired return change.

Added `tools/watch_probe_gate.py`, which watches `ckpt_*.pt` publication, invokes the runner for each new checkpoint, and can stop on the first rejected gate. This is the first complete online orchestration path from checkpoint publication to external target disable.

An end-to-end watcher smoke run processed seed3 checkpoints 50/100/150/200 and then rejected checkpoint 250 after a paired short probe showed `-12.6%` relative return; it wrote `logs/watcher_smoke/target.control = disable`. The runtime control transition is therefore verified, not only unit-tested.

Ran the first online training experiment with the control file enabled (seed4, 200 iterations, checkpoints every 50). Paired subject5 probes at iters 50/100/150/200 all remained positive (`+80.3%`, `+74.8%`, `+14.8%`, `+19.4%` versus the fixed baseline), so the control stayed `enable`. This validates the non-interrupting path in a live training run; it is a short subject5-only probe and is not evidence that target coupling is generally safe.

The strict post-run audit of that same seed4 run used 500-step probes on both subject4/5 and two start seeds. The final online target was worse than the same-seed 200-step nominal baseline on all four paired probes; relative changes were `-.355, -.330, -.322, -.332`, with mean `-.335` and lower bound `-.349`. The gate wrote `disable`. This is a decisive horizon-mismatch finding: the 100-step subject5 watcher did not detect a degradation that was clear at 500 steps. The production gate must use a long enough horizon or a validated multi-horizon rule.

Encoded the horizon requirement in the tooling: `policy_probe_gate.py`, `run_probe_gate.py`, and `watch_probe_gate.py` now default to `--min-steps 500` and reject shorter JSON probes before computing a decision. This prevents the failed 100-step configuration from silently returning `enable` in future runs.

Ran a second live experiment (seed5, 200 iterations) with the enforced 500-step watcher. The first checkpoint's subject4 probe measured `-.0621` relative return against the reference, crossing the 5% margin; the watcher wrote `disable` while training continued through iteration 200. This verifies the intended early protection behavior under the long-horizon gate, including non-interrupted continuation after target shutdown.

Completed the same-seed audit for seed5 with a 200-step nominal baseline. The post-disable online policy was worse on all four 500-step probes (subject4/5 × seeds123/456): relative changes `-.337, -.399, -.355, -.400`, mean `-.373`, lower bound `-.404`; the gate wrote `disable`. Disabling the target prevents further imagined coupling but cannot undo policy updates already made before the first probe. A useful deployment policy therefore needs either a much earlier probe, a rollback checkpoint, or a conservative shadow-target mode that never updates the live policy before validation.

Implemented rollback control: a gate can write `rollback:<checkpoint>`; the trainer refreshes this command outside `torch.compile`, restores policy/critic and available optimizer states, and disables target coupling. A resumed-training smoke test emitted the rollback confirmation while restoring iter150 state.

Fixed a watcher ordering bug found during rollback validation: the current candidate was previously selected as its own rollback target. The watcher now snapshots prior accepted checkpoints before marking the candidate seen; a rerun showed candidate100 invoking the runner with checkpoint50 as the rollback source.

Compared the rollback smoke checkpoint numerically: its policy parameter distance to the rollback source was relative `0.52%`, versus `24.21%` to the rejected checkpoint. The rollback is therefore materially restoring the accepted policy state, with only the one subsequent nominal update accounting for the small residual.

The control-level audit confirms the same result on identical 500-step starts: accepted source return `.027862`, rejected checkpoint `.025055`, and post-rollback return `.027864` on subject4. The rollback recovered the accepted control performance to numerical precision in this smoke continuation.

Completed an end-to-end watcher rollback trigger with synthetic accepted/rejected candidates: ckpt50 was accepted, ckpt100 was rejected at `-32.3%` on subject5, and the actual control file contained `rollback:.../ckpt_000050.pt`. This verifies the full gate-to-watcher-to-trainer rollback token path.

## 2026-11-06 shadow branch runtime smoke

The shadow-policy prototype completed a 20-iteration Isaac Lab runtime smoke
with `--shadow_policy`, publishing both `ckpt_000020.pt` and
`shadow_ckpt_000020.pt`. The shadow artifact audit confirmed
`shadow_policy=true`, the source checkpoint link, and a nonzero policy distance
from live. A resumed
one-iteration run consumed `shadow_promote` and printed the promotion event
before the next update. This verifies the runtime branch isolation, artifact
publication, and control-file promotion path; it is not a performance result.

A 100-iteration shadow smoke then produced paired live/shadow artifacts at
iterations 50 and 100. On identical 100-step starts (batch 16, seed 123), the
iteration-100 live policy returned `.1687` with end survival `.0625`, while the
shadow policy returned `.1643` with zero end survival. This is one short smoke,
not a recipe comparison, but it confirms that shadow divergence is visible to
the real simulator probe and should be gated before promotion.

The follow-up low-rate shadow run (`--shadow_policy_lr_scale=.1`) clarified the
parameter behavior: at iteration 60 the live policy norm was about `9684.7`,
while the shadow norm was `58.3`; the shadow branch was not the exploding
branch. Its 100-step paired return was `.2089` versus live `.0505` on the same
starts, although both had zero end survival. This is a diagnostic smoke only,
but it motivates keeping a separate shadow learning-rate control and reporting
both branch norms in audits.

A matched nominal 60-iteration run with the same tiny configuration but without
`--shadow_policy` reached the same live policy norm, `9684.7`. The live
parameter explosion is therefore caused by the under-trained smoke setup, not
by interaction with the shadow branch. This confirms the low-rate shadow result
as a diagnostic safety signal rather than a recipe ranking.

The probe gate now also checks `survival_at_end` when the evaluator provides it.
On the paired smoke probes, the default-rate shadow was rejected despite only a
`-2.6%` return delta because end survival dropped by `.0625` (above the `.05`
allowance); the low-rate shadow passed with zero survival drop. This prevents a
return-only gate from accepting a candidate that falls more often.

The 300-iteration low-rate shadow run was evaluated at 500 steps on the same
starts. Shadow/live return pairs were `.0347/.0247` at iteration 100,
`.0452/.0248` at iteration 200, and `.0260/.0333` at iteration 300; all six
policies had zero end survival in this tiny setup. The early shadow advantage
therefore reversed by iteration 300, reinforcing the need for repeated
checkpoint probes and no one-time promotion assumption.

The real shadow-mode watcher was then run on the iteration-300 artifacts. It
automatically paired `shadow_ckpt_000300.pt` with `ckpt_000300.pt`, measured a
`-21.85%` return change at 500 steps, and wrote `shadow_discard` (with no live
rollback token). This verifies the full same-iteration comparison and safe
rejection path on actual simulator output.

Finally, a real sequence watcher processed shadow iterations 100, 200, and
300. The first two paired probes passed (`+40.25%` and `+82.28%` return), while
iteration 300 failed (`-21.85%`); the final control file was
`shadow_discard`. This confirms checkpoint-by-checkpoint promotion/discard
behavior rather than a one-off gate invocation.

Added `--shadow_policy_updates` so the isolated branch can match the live
policy update cadence. A 100-iteration smoke with `wm_batches=2`,
`policy_updates=2`, and `shadow_policy_updates=2` completed successfully and
logged both `policy_param_norm` and `shadow_policy_param_norm`; no runtime or
checkpoint publication errors occurred.

Generated a kinematics-based per-joint WM component vector with
`tools/derive_kinematic_component_weights.py` (`q` range `.234–2.50`, `qd`
range `.673–1.78`) and completed a 30-iteration GPU smoke using
`--wm_component_weights`. The run remained numerically stable (`wm_err8=7.56`);
this is plumbing evidence only, with no control claim yet.

Added `--wm_component_weights_file` so generated vectors can be passed without
embedding 70 comma-separated values in a shell command. A 10-iteration Isaac
Lab smoke loaded `logs/kinematic_component_weights.csv` through this option and
completed cleanly.

A 300-iteration run with the same grouped `wm_batches=2` and
`policy_updates=2` schedule reached `wm_err8=.590`. Its single 500-step probe
gave return `.0303` with zero end survival. Without a matched unweighted run
under the exact same seed and budget, this remains a diagnostic datapoint rather
than evidence for promotion.

The matched unweighted 300-iteration baseline then gave a paired 500-step
return of `.0357` versus `.0303` for the kinematic-weighted model (−15.3%),
with zero end survival for both. The physically motivated weighting is
therefore not a leading candidate under this budget; keep it as a controlled
WM-only diagnostic and do not combine it with policy-coupling changes.

A resume smoke from that checkpoint consumed `shadow_promote`, synchronized the
shadow Adam states into live, and completed the next iteration without optimizer
or parameter-group errors. The promotion event was emitted before training
continued.
### Kinematic-cluster reset sampling (new opt-in arm)

`TrackingSim` now supports `--reset_sampling_mode kinematic` with
`--reset_kinematic_clusters K`. It clusters one-second reference bins using
root-frame velocity/angular velocity, joint position, and scaled joint velocity,
then samples clusters uniformly and bins uniformly within a cluster. The
default `return` mode is unchanged, and the cluster path does not use episode
returns, so it is a clean data-coverage ablation. Static compilation and the
buffer tests pass. A container-backed runtime smoke with 8 environments, 4
iterations, and 8 clusters completed successfully and wrote
`logs/kinematic_reset_smoke/ckpt_000004.pt` plus `last.pt`; the remaining work
is a matched-budget performance comparison against the default return sampler.

That paired 20-iteration probe is now complete with seed 7 and identical
optimizer/data budgets. Kinematic sampling reduced the final-5 WM err8 mean
from 6.07 to 3.57, but did not improve training sim reward (0.320 vs 0.332).
The fixed 100-step evaluation from the same 40 starts also favored the default
sampler (return norm 0.0931 vs 0.0824; both had zero end survival). This closes
the kinematic reset sampler as a leading performance mechanism under this
small matched budget; retain it as a coverage diagnostic and do not combine it
with the concurrent WM-policy claims.

### Action-recovery cross-seed audit

The existing 1000-iteration `wm_action_recovery_weight=.05` runs cover two
seeds on each train split with paired 500-step fixed-policy JSONs. On subject4,
candidate-versus-baseline return changes were approximately `+5.5%` and
`-5.9%`; on subject5 they were approximately `+3.6%` and `+4.2%` for the
available paired probes, while other start seeds remained mixed. The candidate
also worsens ordinary held-out WM error in the earlier audit. This shows that
action recovery can change policy geometry, but it is not yet a stable
concurrent WM-policy mechanism. Keep it for controlled diagnostics and require
a deployment-aware repeated probe before any promotion.

The probe runner now accepts repeated `--steps` values and requires every
horizon to pass independently before writing `enable` or `shadow_promote`.
This prevents a 100-step gain from masking a 500-step regression. An end-to-end
container replay on the reset-sampler pair produced relative drops of `-19.1%`
at 100 steps and `-23.0%` at 500 steps, and wrote `disable`; both horizon
decisions were independently rejected.

`watch_probe_gate.py` now forwards repeated horizons to the runner, so the same
multi-horizon rule applies during online shadow training rather than only in a
post-run manual audit. The watcher and runner gate tests pass together.
When multiple horizons are explicit and `--min-steps` is omitted, the shortest
horizon is accepted as the minimum; the historical single-horizon default of
500 steps is unchanged.

The full selected regression suite now passes inside the project container:
49 tests OK (one intentional skip). The host-only run reports one import error
because Isaac Lab is container-provided; it is not treated as a code failure.

An end-to-end shadow watcher replay with the multi-horizon rule also completed:
the iteration-300 shadow candidate was down `-7.2%` at 100 steps and `-26.7%`
at 500 steps, so both horizon gates rejected it and the watcher wrote
`shadow_discard`. The paired live checkpoint remained untouched.

The gate now optionally checks `first_failure_step_mean` through
`--max-first-failure-drop`. On the same shadow replay, the candidate failed
earlier by 2.0 steps at horizon 100 and 9.5 steps at horizon 500; a 5-step
budget therefore rejects the long-horizon probe even when both policies have
the same zero end-survival. The default remains disabled for backward
compatibility.

## Standard-recipe shadow probe (seed0, 100 iterations)

A fresh 32-environment run used the standard 4-step teacher-forced plus 8-step
free-running recipe, `policy_updates=2`, `shadow_policy_updates=2`, and a
shadow learning-rate scale of `.1`. It completed cleanly and published live
and shadow checkpoints at iterations 50 and 100. The multi-horizon watcher
accepted both candidates: iteration 50 changed return by `+4.1%` at 100 steps
and `-4.5%` at 500 steps, while iteration 100 changed it by `+45.3%` and
`+40.5%`; survival was zero for both branches and the first-failure guard also
passed. The final control token was `shadow_promote`.

This is a single-seed, single-split, 100-iteration signal. It validates the
shadow promotion path under the standard recipe, but is not evidence for a
general performance improvement; the required next check is a second seed and
the other motion split.

The required second run used the same budget and recipe with seed1 on the
other training split. Its iteration-50 candidate passed at both horizons
(`+4.9%` at 100 steps, `+10.6%` at 500 steps). At iteration 100, the short
probe was still within tolerance (`-1.4%`), but the 500-step probe fell
`-7.6%`; the multi-horizon rule therefore wrote `shadow_discard`. This is a
concrete example of why the short probe alone is insufficient.

Compatibility replay on the previously accepted shadow checkpoints at
iterations 100 and 200 kept the gate decision unchanged with a 5-step budget;
their first-failure changes were `-4.5` and `-6.125` (later failures), so both
still passed. The new metric therefore rejects the demonstrated bad candidate
without rejecting the existing positive examples.

A third seed on the subject4 split reinforces the same boundary: the final
shadow candidate changed return by `-0.7%` at 100 steps but `-11.5%` at 500
steps, with the first-failure mean 5.25 steps earlier at the long horizon. The
watcher again wrote `shadow_discard`. Across the two rejecting seeds, short
probes alone would have accepted the candidate; the long-horizon rule is
detecting a repeatable failure mode of this early shadow recipe.

The corrected audit now reports layer-wise displacement as well. For seed1 at
iteration 100, encoder and decoder relative distances were `0.194` and `0.212`,
respectively. The displacement is distributed across the policy rather than
concentrated in a single output layer, so a future stabilization should bound
the whole task update or change its objective instead of only freezing the
last layer.

The failure is not explained by a simple parameter-norm blow-up: at iteration
100 the three runs had live/shadow trainable-policy norms of approximately
`51.9/50.9` for seed0, `51.8/50.9` for seed1, and `51.9/50.9` for seed2. The
checkpoint audit initially overstated the distance because it included running
normalizer buffers (`count`/`var`); it now excludes those non-trainable buffers.
For seed1 the corrected relative policy distance is `0.203`. The shadow branch
is therefore numerically bounded while still regressing at 500 steps on two
seeds. Further work should change the task-model objective or fallback
criterion, rather than only rescale the shadow learning rate.

After correcting the audit, all three iteration-100 candidates show the same
bounded separation: relative distance to live is `0.208` (seed0), `0.203`
(seed1), and `0.208` (seed2). This consistency rules out an isolated bad
checkpoint artifact; the long-horizon failures occur despite a repeatable
policy-branch displacement.

An opt-in whole-policy trust-region cap is now available as
`--shadow_policy_max_step_norm`; it projects each isolated shadow optimizer
step in parameter space after the Adam update, while leaving the live branch
and nominal WM unchanged. The cap is disabled by default, has a unit test, and
a 12-iteration container smoke with cap `.05` completed and wrote a normal
training log. This is an implementation arm only; it still needs the same
paired multi-horizon evaluation before any claim.

The shadow branch also supports `--shadow_policy_tilt_penalty`, an additional
imagined root-tilt penalty applied only while updating the isolated branch.
In the matched seed1 cap recipe, `.02` reduced the shadow tilt-failure shift,
but iteration-100 paired returns still fell `−8.5%` at 100 steps and `−8.7%`
at 500 steps. The penalty changes the failure mode without restoring
deployment return; keep it as a diagnostic arm and leave the default at zero.

`tools/shadow_stability_audit.py` now joins shadow/live parameter distance with
paired simulator returns and failure subtypes across checkpoints. The exported
`docs/shadow_stability_audit.json` shows that a relative policy displacement of
about `.20` can accompany either a +40.5% 500-step gain (seed0) or −7.6%/−11.5%
losses (seeds1/2). Parameter distance is not a deployment gate; return and
failure-mode deltas must remain simulator-based.

The optional subtype gate is intentionally conservative: the old seed0
iteration-100 candidate, which gained about 40% return, is rejected under a 5%
height-failure limit because its 100-step height-failure rate increased by 25
percentage points. A return gain with a new failure mode is not promotion-safe
unless the experiment explicitly accepts that tradeoff.

The subtype threshold sweep is archived in `docs/shadow_gate_calibration.json`.
There is no universal threshold that accepts every historical return-positive
candidate while rejecting every failure-mode change: a `.15` limit accepts the
cap iteration-50 point but rejects the standard seed0 iteration-100 point with
height-failure delta `.25`. Subtype limits are therefore safety preferences,
not a hidden replacement for return evaluation.

The gate also exposes `--failure-z` for noisy subtype estimates. With `1.96`,
height/tilt limits use a conservative independent-binomial upper bound while
both raw and bounded deltas are reported; `0` preserves point-estimate behavior.

- The independent seed1 replication stayed neutral: clean `.03915` vs `.03919`
  control and perturb `.03902` vs `.03907`, again with zero end survival and
  height-only failures. With two seeds, reward gating is retained as a
  stability-preserving diagnostic arm, but is not promoted as a control gain.
- The collector now audits the executed immediate reward of each selected
  counterfactual against its base action. On a fixed seed1 candidate set, the
  critic-only selector had mean delta `-0.5985`; reward weights `.5` and `2`
  gave `-0.1694` and `0.0`, respectively, with weight `2` choosing the base
  action for every state. The current reward gate therefore suppresses
  exploration rather than finding positive interventions; future work needs a
  multi-step/advantage target.

- A direct interaction test combined `wm_action_recovery_weight=.05` with the
  reward-gated strategic probe (`probe_weight=.01`, `critic_reward_weight=2`)
  for a matched 100-iteration continuation from the subject4 seed1
  action-recovery checkpoint. Relative to action recovery alone, the combined
  arm fell from `.07223` to `.07003` clean and `.07211` to `.06890` under
  perturbation; first failure moved earlier by `3.25` and `3.31` steps. The
  auxiliaries are therefore not complementary at these weights and the
  combination is closed. The resume path now catches optimizer parameter-group mismatches caused by
  adding a module and resets only the incompatible optimizer state, preserving
  checkpoint weights and replay state.

- A horizon-2 selector was added, rolling the current policy once through the
  WM-predicted next state. On the same seed1 candidate set, reward weights `.05`
  and `.5` still produced no positive immediate reward deltas (means `-.5668`
  and `0.0`). A short raw-reward rollout is therefore insufficient; future
  work needs a bootstrapped value/advantage target.

- Added `tools/evaluate_candidate_rollouts.py` for real-simulator short-horizon
  validation of critic-selected first actions. On 64 seed1 candidates, followed
  by the original policy for 4/8 steps, candidate returns were `-4.244/-1.845`
  versus base `-1.755/-0.574`. The WM-error critic is anti-aligned with control
  advantage in this audit; active action selection from this critic is closed.

- Added `tools/collect_advantage_dataset.py` and `tools/train_advantage_critic.py`
  for simulator-grounded short-horizon labels. At horizon 4, noise `.05` had
  no positive candidate; noise `.005` had `15.2%` positive candidates but only
  `+0.00053` best mean return advantage. The 1280-row advantage critic reached
  validation RMSE `.00193` with correlation `.002`; the signal is too weak for
  active selection without more data or action-gradient search.

- Out-of-sample evaluation of the conservative advantage selector (trained on
  seed456, tested on fresh seed789) produced mean selected advantage `-4.6e-5`
  and only `2.75%` positive selections, despite the base fallback. The local
  `.005` advantage signal does not transfer; keep this branch offline and do
  not connect it to training.

- Added a local action-gradient audit. On 256 seed1 train-split starts, moving
  opposite the WM predicted-reward gradient gave real-return deltas `+.00029`
  (epsilon `.005`, 4 steps), `+.00013` (8 steps), and `+.00026` (epsilon `.01`,
  4 steps); the positive direction was slightly worse. This is a weak but
  directionally coherent diagnostic, requiring held-out/cross-seed validation.
  Legacy data lacks `a_prev`, so the audit currently uses the current action as
  its approximation.

- Cross-seed action-gradient replication from the independent seed0 checkpoint
  preserved the sign: negative gradient `+0.000060` and positive gradient
  `-0.000493` over 4 steps at epsilon `.005` (seed1 negative-gradient gain was
  `+.00029`). The sign is coherent but the effect is too small for promotion or
  direct policy/WM coupling.

- Held-out test_s4 validation of the action-gradient candidate was neutral or
  negative: negative gradient `-7.7e-7`, positive gradient `-.000319` over 4
  steps at epsilon `.005`. The train-split gain does not transfer, so this
  remains a local geometry diagnostic and is not promoted.

- Corrected action-gradient evaluation to store the true initial `a_prev`.
  On held-out test_s4, the negative-gradient candidate then fell `-0.00398`
  over 4 steps. The earlier near-zero/positive result came from substituting
  current action for `a_prev`; close this line for promotion and keep the
  corrected data path for audits.

- Long-budget paired fixed-policy evaluation of the promoted schedule checkpoints on the same 64 starts (300-step rollout) confirms the cross-seed split. Seed0 (`promo_sched_s0_300`) reaches clean/perturb return `0.06948/0.06866`, mean first failure `49.7/49.1`, height failure `95.3%/93.8%`, tilt failure `7.8%/9.4%`; seed1 (`promo_sched_s1_300`) reaches `0.05821/0.05857`, first failure `44.0/44.3`, height failure `96.9%/95.3%`, tilt failure `3.1%/4.7%`. End survival is `0` for all four conditions. Thus schedule sampling remains useful for short-term tracking and reduces some tilt failures, but the 300-step stability bottleneck is still early height failure and the seed split is persistent; no promotion claim is made.

- Added optional `--policy_failure_start_fraction`: a detached sampler arm that allocates a fraction of imagined policy starts to replay states ranked by normalized root-height/root-tilt termination proximity. A seed1, 100-iteration schedule+consistency smoke with fraction `0.25` improved paired 64-start clean/perturb return from `0.06908/0.06942` to `0.07298/0.07413`, delayed mean first failure from `47.1/47.3` to `49.7/50.8`, and reduced tilt failure `4.7%→3.1%`; height failure remained `95.3%→96.9%` and end survival stayed `0`. This is a promising recovery-training signal but not yet a promotion result; a matched 300-iteration seed1 run is in progress.

- The 300-iteration seed1 failure-focused sampler (`policy_failure_start_fraction=.25`, ranking max(height, tilt) proximity) improved paired 64-start return `0.05821/0.05857 → 0.06171/0.06158` and reduced height failure `96.9%/95.3% → 92.2%/92.2%`, but increased tilt failure `3.1%/4.7% → 7.8%/7.8%`; end survival remained `0`. A height-only ranking ablation at 100 iterations was neutral/slightly negative (`0.06894/0.06901` vs baseline `0.06908/0.06942`) and did not reduce height failure. The long-run gain therefore comes from mixed boundary-state exposure, but its tilt trade-off is unresolved; keep as a diagnostic candidate, not a promotion.

- Balanced boundary-start sampling (half top height-error starts, half top tilt-error starts; fraction `.25`) was neutral in the seed1 100-iteration smoke: paired clean/perturb return `0.06894/0.06901` versus baseline `0.06908/0.06942`, first failure `47.8/48.0` versus `47.1/47.3`, height failure `96.9%/96.9%`, tilt failure `3.1%/3.1%`, and survival `0`. The mixed max-risk sampler's short-term gain therefore does not transfer to a simple balanced quota at this budget; keep both as diagnostic branches.

- Added `--policy_failure_start_ramp_iters` to introduce failure-focused starts gradually. A seed1 100-iteration mixed-risk smoke ramping fraction `.0→.25` over 100 iterations produced clean/perturb return `0.07029/0.06982` versus baseline `0.06908/0.06942`, with first failure `47.6/47.7` versus `47.1/47.3`; height failure rose to `98.4%/98.4%`, tilt was `4.7%/3.1%`, and survival stayed `0`. The curriculum is a small return-only signal and does not fix the failure mix at this budget.

- Added replay cause instrumentation: `env.step` now stores separate `term_height` and `term_tilt` labels, `ReplayBuffer` keeps them with backwards-compatible zero defaults, and `tools/analyze_replay_causes.py` reports cause counts from checkpoints. A 5-iteration Isaac smoke completed and the checkpoint loaded successfully; its short buffer had no terminal events, as expected. This enables the next curriculum experiments to use observed failure causes rather than inferred aggregate evaluator rates.

- Instrumented seed1 baseline (100 iterations, 16 envs, schedule+consistency) produced 1,024 replay transitions and 22 terminal transitions. All 22 were `term_height`; `term_tilt=0`. This is a useful diagnosis: simulator training data is height-dominated, while fixed-policy evaluation still shows a small tilt-failure mode. Replay-cause balancing therefore cannot rely on terminal counts alone; the next curriculum must include nonterminal tilt-near states or controlled perturbations to expose that mode.

- Pure tilt-proximity sampling (`policy_failure_start_tilt_only`, fraction `.25`) in seed1 100-iteration paired evaluation improved clean/perturb return `0.07112/0.07211` versus baseline `0.06908/0.06942`, delayed first failure `48.8/49.5` versus `47.1/47.3`, and reduced tilt failure `3.1%/4.7%` versus `4.7%/4.7%`; height failure increased to `96.9%/96.9%` and survival stayed `0`. This confirms the tilt-near replay arm is learnable and explains the mixed sampler's height/tilt trade-off; it is not a complete recovery solution.

- Alternating recovery sampler (`policy_failure_start_alternate_iters=10`, fraction `.25`) switched between height-near and tilt-near starts every 10 iterations. In seed1 100-iteration paired evaluation, clean/perturb return improved to `0.07153/0.07214`, first failure to `49.2/49.7`, and tilt failure fell to `1.6%/1.6%`; height failure rose to `98.4%/98.4%`, survival remained `0`. Alternation protects the tilt mode but does not recover the dominant height mode at this budget.

- Alternating height/tilt recovery starts every 10 iterations improved seed1 100-iteration clean/perturb return to `0.07153/0.07214` and reduced tilt failures to `1.6%/1.6%`, but height failure rose to `98.4%/98.4%`; survival stayed zero. This confirms the two recovery modes compete when sharing one policy. The current evidence now supports evaluating a state-gated multi-head policy rather than further static sampler mixtures.

- Added `--gate_mode max_error` to `fixed_policy_eval.py` for an inference-level two-policy gate: checkpoint 0 is height-focused, checkpoint 1 tilt-focused, and the action is selected by the larger current normalized height/tilt error. On seed1 paired 64-start 300-step evaluation using the height-only and tilt-only 100-iteration checkpoints, the gate returned `0.06827/0.06852` clean/perturb, first failure `47.6/47.8`, height failure `96.9%/96.9%`, tilt failure `3.1%/3.1%`, survival `0`. It underperformed both specialized policies, so raw current error is not a sufficient recovery selector; do not promote this gate.

- Added `--gate_mode error_trend` to the evaluator, selecting the height or tilt checkpoint by the larger one-step increase in normalized error. On the same seed1 paired starts, clean/perturb return was `0.06825/0.06878`, first failure `48.2/48.5`, height failure `96.9%/96.9%`, tilt failure `3.1%/3.1%`, survival `0`; this is no better than the raw-error gate and below either specialized policy. One-step error trend is also insufficient as a recovery selector.

- Added trace collection (`fixed_policy_eval.py --trace_out`) and `tools/train_recovery_gate.py` to fit an offline selector from paired specialized-policy rollouts. On 64 seed1 starts × 300 steps, a 4-feature gate (height error, tilt error, speed, angular speed) achieved validation accuracy `0.5603`, indistinguishable from the majority baseline `0.5605` (positive rate `0.4367`). Thus current state features do not predict which recovery head wins future return; a useful gate needs longer history or WM-based predictive features.

- Extended the offline recovery-gate audit to an 8-step temporal window (`tools/train_recovery_gate.py --history 8`). Validation accuracy was `0.5578`, below the majority baseline `0.5605`, versus single-step `0.5603`; the paired specialized-policy traces contain no useful selector signal even with short history. A gate likely needs model-predicted counterfactual rollouts or a different training objective, not more hand-selected kinematic history.

- Added `tools/policy_counterfactual_wm_gate.py` to compare two policies under one frozen WM. On 1,024 replay starts (horizon 16), the height-trained WM ranked the tilt-focused policy worse (policy1 win fraction `0.340`, predicted return margin `-0.00107`), while the tilt-trained WM ranked it better (`0.711`, margin `+0.00147`). Real fixed-policy evaluation slightly favored the tilt-focused policy. WM-based head ranking is therefore model-dependent and can be miscalibrated; a gate must use multiple WMs and uncertainty-aware pessimism.

- Extended the counterfactual gate audit to two WMs and a pessimistic min aggregation. Across 1,024 replay starts, the height-trained WM preferred height policy (`0.03151 vs 0.02964`), the tilt-trained WM preferred tilt policy (`0.03011 vs 0.02964`), but the per-start pessimistic aggregation was nearly tied and slightly preferred height: margin `-9.99e-5`, tilt-policy win fraction `0.479`. This does not provide a calibrated selector; uncertainty aggregation removes the WM ranking flip by collapsing the signal rather than improving action choice.

- Added mean-minus-uncertainty calibration to the two-WM counterfactual audit. For score `mean - λ·std`, λ=`0/0.25/0.5/1/2` gave policy1 (tilt) win fractions `0.469/0.472/0.467/0.508/0.521` and mean margins `-1.68e-4/-1.06e-4/-4.30e-5/+8.25e-5/+3.33e-4`. Larger uncertainty penalties slightly shift ranking toward tilt, but the effect is weak and has not been validated in real gated control; no promotion.

- Held-out fixed-start validation (64 new starts, seed 202) preserved the specialized-policy ordering: tilt-focused policy exceeded height-focused return on clean `0.06807 vs 0.06636` and perturb `0.06693 vs 0.06571`, with first failure later by `0.52/0.55` steps. Both had `98.4%` height failure, zero tilt failure, and `1.6%` end survival. The policy ranking transfers modestly across starts, but this does not validate the WM gate ranking, which remains model-dependent.

- Cross-seed 300-iteration replication of mixed failure-focused starts (`fraction=.25`) failed decisively. On the same 64 paired starts, seed0 baseline/risk-start clean return was `0.06948/0.06702` and perturb `0.06866/0.06649`; first failure moved `49.75→45.48` and `49.08→45.19`; height failure became `100%` in both conditions and end survival stayed zero. The seed1 improvement (`0.06171` with lower height failure) is therefore split-dependent. Close this sampler as a promotion route; retain only as a diagnostic for state-distribution sensitivity.

- Cross-seed policy-support audit explains the failure-focused sampler split. Baseline schedule action-distance mean / WM-error mean were seed0 `0.00631/0.594` and seed1 `0.00563/0.270`; risk-start training increased action distance to `0.01158` (seed0) and `0.01141` (seed1), nearly 2x, while WM errors were `0.234/0.265`. Thus risk-start exposure pushes the policy far off replay action support; on seed0, where the baseline WM is already much less accurate, this support shift causes early collapse despite lower one-step WM error. This supports a proximal/support-constrained recovery objective rather than more boundary oversampling.

- Tested support-constrained recovery: mixed boundary starts `.25` plus `policy_bc_weight=.5` on seed0, 100 iterations. Height failure decreased versus its baseline (`96.9%→92.2%` clean, `96.9%→93.8%` perturb), but return fell `0.06603→0.06409` / `0.06629→0.06391`, first failure moved earlier by ~1.8/2.0 steps, and tilt failure rose to `7.8%`. A replay-action anchor controls boundary overshoot but does not produce a useful recovery policy; close this coefficient/configuration.

- State-dependent support constraint (`policy_bc_weight=.5`, `policy_bc_risk_margin=.5`) was tested with mixed boundary starts on both seeds. Seed0 100-iteration smoke improved clean/perturb return `0.06603/0.06629 → 0.06734/0.06708` with height failure `96.9%→95.3%/96.9%`; seed1 was not robust: return `0.06908/0.06942 → 0.06938/0.06922`, while tilt failure jumped `4.7% → 18.8%/20.3%`. End survival stayed zero. State-gated BC is safer than global BC on seed0 but still split-dependent; do not promote.

- Failure-conditioned tilt penalty (`policy_tilt_penalty=.01`, applied only when start tilt proximity exceeds `.5`) improved seed1 100-iteration paired return to `0.07124/0.07119`, first failure `49.4/49.6`, tilt failure `3.1%/3.1%`; however seed0 replication degraded return `0.06603/0.06629 → 0.06134/0.06160` and first failure `47.2/47.6 → 42.6/42.8`, while height failure fell but tilt failure rose. The state gate avoids the global penalty's action explosion on seed1 but remains strongly split-dependent; reject for promotion.

- Dose-response at failure-focused fraction `.10` was more stable than `.25` on seed0 but still not a cross-seed improvement. Seed0 clean/perturb return changed `0.06603/0.06629 → 0.06661/0.06641`, with tilt failure lower but height/survival unchanged; seed1 return fell `0.06908/0.06942 → 0.06609/0.06664`, height failure rose to `96.9%`, and survival stayed zero. Lowering exposure reduces catastrophic support shift but does not produce a transferable recovery signal.

- Exact-start WM calibration is now available: `fixed_policy_eval.py --trace_out` stores initial simulator states, and `policy_counterfactual_wm_gate.py --trace_state` rolls both policies from those same states in frozen WMs and compares predictions to real trace returns. On the same 64 seed1 starts, the pessimistic WM gate predicted the tilt head better on `65.6%` of starts, while the real 300-step return favored tilt on only `42.2%`; predicted-vs-actual per-start margin correlation was `-0.060`, and predicted winner accuracy `0.484`. This closes counterfactual WM ranking as a usable action gate under the current WM; the model's imagined ranking is effectively uncalibrated/anti-correlated.

- Exact-start WM ranking calibration was tested across splits. A linear affine calibration fitted on all 64 seed1 predicted-vs-real margins had negative slope (`-10.92`) and improved raw held-out seed202 winner accuracy from `0.438` to `0.563` (actual tilt win rate `0.578`), but the seed1 raw correlation was `-0.060` and within-seed held-out-half calibration accuracy fell to `0.344`. The apparent held-out improvement is therefore weak and unstable; a one-split affine correction is not safe for control. Calibration needs more independent starts/seeds and a bounded acceptance gate.

- Expanded exact-start calibration to a second held-out split (seed303). Raw WM winner accuracy was `0.375`, predicted/actual margin correlation `-0.075`, and actual tilt win rate `0.516`. The affine correction fitted on seed1 improved seed202 from `0.438→0.563` but only seed303 from `0.375→0.516`, exactly the majority class; no consistent calibration benefit appears across held-out splits. Multi-seed exact-start evidence closes simple post-hoc affine calibration.

- Counterfactual WM horizon audit on the same 64 exact seed1 starts shows that increasing imagination horizon does not recover a usable selector. With the two specialized WMs and pessimistic min aggregation, predicted-vs-real return-margin correlation / winner accuracy were: H=1 `-0.196/0.391`, H=4 `-0.222/0.313`, H=8 `-0.100/0.422`, H=16 `-0.060/0.484`, H=32 `-0.127/0.453`, H=64 `-0.026/0.438`; the real 300-step rollout still favored the tilt head on only `42.2%` of starts. The ranking is anti-correlated at short horizons and merely uninformative at long horizons, so horizon tuning cannot repair the current WM gate. The next research target is multi-step WM fidelity / value calibration during training, rather than another inference-time horizon or uncertainty coefficient.

- Follow-up gradient audit separates the two long-horizon schedule splits. At replay window 4, seed0 schedule checkpoint had positive task/replay WM gradient cosine `0.083/0.143/0.179` at horizons `4/8/16`, while seed1 was `-0.089/-0.039/+0.040`; the seed1 short-horizon conflict is consistent with its earlier 300-step collapse and explains why a free-running auxiliary can improve local WM error without transferring to control. The same seed1 schedule+consistency-risk checkpoint showed positive cosine `0.084/0.097/0.112`, so risk-weighting changes the local gradient geometry but does not guarantee rollout stability. This points to an online compatibility gate on the consistency update itself, rather than another fixed consistency coefficient.

- Grouping the exact-start H=16 counterfactual audit by motion clip shows the WM ranking failure is concentrated in several multi-start clips: clip IDs `3, 6, 7` had predicted-vs-real margin correlations `-0.461, -0.943, -0.720` with winner accuracies `0.25, 0.00, 0.25`; other clips were mixed and mostly under 4 samples. This is diagnostic rather than a promotion result, but it supports clip/state-conditioned calibration and argues against a single global WM gate.

- Implemented an opt-in `--wm_consistency_alignment_min` gate. For each WM batch it compares the factual replay-loss gradient with the free-running consistency gradient on WM parameters; if their cosine is below the threshold, only the consistency term is dropped for that batch. The gate is disabled by default, validates `[-1,1]`, and automatically switches the affected imagination path to eager mode because retained autograd queries are incompatible with AOT donated buffers. A 30-iteration CUDA Isaac smoke with schedule sampling, consistency weight `.1`, and threshold `0.0` completed and saved `last.pt` without runtime errors; the existing 37-test suite and `py_compile` also pass. This is now the next candidate for matched fixed-policy evaluation, with threshold `0.0` as the first conservative test.

- The alignment-gated checkpoint (`smoke_cons_align_s1_30`, 30 iterations, threshold `0.0`) also completed the current fixed evaluator for a 100-step smoke: normalized return `0.1539`, end survival `0`, height failure `0.875`, tilt failure `0.266`, first failure `38.6` over 40 sampled starts. This is only a runtime/sanity result because the checkpoint budget and evaluator horizon differ from promotion runs; it is not evidence of control improvement.

- Matched 100-iteration gradient-compatibility gate test (`wm_schedule_sampling_prob=.5`, `wm_consistency_weight=.1`, threshold `0.0`) was evaluated on the same 40 usable starts from `paired_starts_s1_300.pt` for 300 steps. Against an identical no-gate baseline, seed0 gate changed clean/perturb return `0.0535/0.0534 → 0.0470/0.0464`, with no survival and earlier first failure (`41.0→40.9`); seed1 gate changed `0.0441/0.0436 → 0.0547/0.0541`, delayed first failure `42.0/41.2→44.8/44.4`, and reduced tilt failure `48.4/51.6%→4.7/4.7%`, but increased height failure `54.7/53.1%→96.9/95.3%`. The gate therefore trades tilt failures for height collapse and remains seed-dependent; it is rejected as a promotion recipe, while the implementation is retained as an experimental diagnostic.

- Root-height WM weighting (`wm_root_height_weight=.5`) was matched against the same schedule+consistency baseline on 40 fixed starts, 300-step evaluation. Seed0 clean/perturb return changed `0.0535/0.0534 → 0.0409/0.0408`, height failure improved `85.9/87.5%→79.7/79.7%`, but tilt failure rose `20.3/17.2%→28.1/29.7%` and return dropped. Seed1 changed `0.0441/0.0436 → 0.0513/0.0509`, with height failure `54.7/53.1%→93.8/95.3%` and tilt failure `48.4/51.6%→6.3/4.7%`. No survival appeared. The weight trades tilt for height in a split-dependent way, so scalar root-height reweighting is closed; future height recovery needs a state/action-conditioned objective rather than a global component multiplier.

- Added state-conditioned policy height penalty `--policy_height_penalty_start_margin`, mirroring the existing tilt gate: the imagined root-height penalty is applied only when the initial normalized height error exceeds the configured margin. The argument is validated in `[0,1)`, defaults to the previous global behavior (`0`), and a 30-iteration seed1 CUDA smoke with penalty `.01` and margin `.5` completed; 100-step sanity evaluation gave return `0.1515`, height failure `90.6%`, tilt failure `28.1%`, survival `0`. The implementation is ready for a matched 100/300-step seed0/seed1 comparison; the short smoke alone is not a promotion result.

- The state-conditioned height penalty was completed as a matched test (`policy_height_penalty=.01`, start margin `.5`, same schedule+consistency recipe, 40 fixed starts, 300 steps). It failed decisively: seed0 clean/perturb return `0.0291/0.0271`, height failure `81.3/78.1%`, tilt failure `21.9/21.9%`; seed1 return `0.0331/0.0329`, height failure `20.3/21.9%`, but tilt failure `79.7/79.7%`; survival was zero throughout and first failure was around `31–34` steps. The gate simply exchanged height failures for tilt collapse, worse than the ungated baseline. Close this branch; future state-conditioned recovery must model coupled height/tilt actions rather than gate a single penalty.

- Added an inference diagnostic `--blend_alpha` to `fixed_policy_eval.py`, which continuously mixes the height-focused and tilt-focused policy actions: `(1-alpha)*a_height + alpha*a_tilt`. On seed1's fixed 40-start/300-step set, alpha `.25/.5/.75` produced clean returns `.0706/.0715/.0716` and perturb returns `.0695/.0700/.0700`, versus the height-focused head `.06894/.06901`; all had zero survival on this difficult split, but the blend had only `3.1%` tilt failure. The alpha `.75` blend transferred to held-out seed202: clean/perturb return `.0690/.0687` versus the best specialized head `.0681/.0669`, with survival unchanged at `1.6%` and no tilt failures. Continuous mixing is therefore a stronger multi-head design signal than hard error gating, though it has not yet solved height termination and remains an inference-only diagnostic.

- Extended the blend diagnostic with `--blend_error_gain`, a smooth state-dependent correction around a base alpha using `sigmoid(logit(alpha)+gain*(tilt_error-height_error))`. On held-out seed202 with base alpha `.75`, gain `.5` reduced clean return to `0.0682` (static `.75`: `0.0690`) and gave perturb `0.0687` (static `.75`: `0.0687`), with unchanged `1.6%` survival and no tilt failures. The static mixture is currently better calibrated than this hand-designed dynamic correction; do not treat the error-gain formula as the learned gate. The evaluator now handles scalar/vector alpha correctly and remains backward compatible.

- Static alpha `.75` blend was validated on a second independent split (seed303). Specialized-head clean/perturb returns were height `0.0698/0.0688`, tilt `0.0722/0.0714`; the blend gave `0.0727/0.0718`, a small `0.7%/0.6%` gain over the best head. Survival remained `1.6%` for every policy, with height failure `98.4%` and no tilt failures. Together with holdout202, this supports continuous multi-head mixing as a robust tracking improvement, but it still does not solve the dominant height termination.

- Added `--blend_action_scale` to test whether the blend's residual height collapse is mainly action amplitude. On seed1 fixed starts with alpha `.75`, scale `.8/.9` gave clean/perturb returns `.0720/.0704` and `.0718/.0702`, slightly above unscaled `.0716/.0700`, but survival remained `0`, height failure `96.9%`, and tilt failure `3.1%`. Lower action magnitude improves tracking marginally without changing the terminal failure, so the remaining issue is recovery direction/state distribution rather than global action amplitude.

- WM Jacobian audit adds a mechanistic clue for the next residual-policy design. On seed1 checkpoints, the height-focused WM had action-Jacobian mean `1.640` and one-step-error correlation `-0.252`; the tilt-focused WM had `1.710` and correlation `-0.275`. Thus states with larger local action sensitivity are not the states with larger factual WM error; simply scaling actions or using current WM sensitivity as a recovery gate will select the wrong regions. A useful recovery residual needs an explicit height/tilt failure-state target or simulator-grounded counterfactual label, rather than a Jacobian-magnitude heuristic.

- Generated a simulator-grounded counterfactual dataset (`docs/cf_height_s1.pt`, 256 states × 8 action directions, deltas `.02/.05`) around the height-focused policy. One-step finite-difference effects show root-z sensitivity is tiny (mean absolute derivative `1.06e-4`, P90 `2.47e-4`), while the root-tilt component is much more directly affected (mean `0.431`, P90 `0.889`); the z/tilt response correlation is only about `-0.17`. This explains why one-step height penalties and residual gates keep trading into tilt: height recovery is not a one-step action problem in this simulator. A useful recovery target must predict a short action sequence / latent momentum change, rather than train on immediate root-z error alone.

- A second counterfactual probe after 20 policy burn-in steps (`docs/cf_height_burnin20_s1.pt`, 128 states × 8 directions) confirms the temporal nature of height recovery. Root-z action sensitivity rises to mean absolute `3.23e-4` (about 3× the random-start probe) but remains orders smaller than tilt sensitivity (`0.410`); z/tilt response correlation stays weak (`-0.158` to `-0.214`). A 40-step burn-in attempt hit a simulator device-side assert when many rollouts terminated, so the 20-step dataset is the valid result. Recovery should therefore use a short-horizon action sequence / momentum target sampled before termination, with explicit terminal-state filtering.

- Extended `generate_counterfactual_dataset.py` with `--counterfactual_horizon` and measured repeated action-direction sequences. After the same 20-step policy burn-in, repeating each counterfactual action for 4 steps increased root-z sensitivity from the one-step `3.23e-4` to `2.13e-3` (delta `.02`, ~6.6×; delta `.05`: `1.84e-3`, ~5.7×). Tilt sensitivity remained substantial (`.335/.305`), and z/tilt response correlation became positive (`.34/.41`) instead of the weak negative one-step correlation. This is the clearest evidence so far that height recovery should be learned as a short-horizon action sequence, where the coupled height/tilt effect becomes observable. The dataset is `docs/cf_height_burnin20_h4_s1.pt`.

- Trained a simulator-supervised transition adapter on the 4-step counterfactual response dataset (`cf_height_burnin20_h4_s1.pt`) with paired-effect and response-Gram losses. On held-out states from the same dataset, RMSE fell `0.8822→0.3073`; on an independent seed303 simulator dataset (`cf_height_burnin20_h4_s303.pt`), RMSE fell `0.7207→0.2528`. The adapter therefore learns transferable sequence-level action effects rather than memorizing individual states. This is the first cross-seed evidence for a concrete short-horizon WM correction target; the next step is to adapt the intervention interface from one-step correction to a horizon-4 latent/momentum correction before using it in policy rollouts.

- The WM counterfactual loss now reads the dataset's `counterfactual_horizon` metadata and rolls each plus/minus action through that many WM steps before comparing with the simulator endpoint. This prevents the h4 sequence dataset from being silently misused as a one-step target. A 30-iteration CUDA smoke using `cf_height_burnin20_h4_s1.pt` with one-step/effect/norm counterfactual weights `.05/.05/.05` completed and saved a checkpoint; 100-step sanity evaluation gave return `0.1570`, height failure `96.9%`, tilt failure `4.7%`, survival `0`. Runtime integration is verified, but this short run has no promotion evidence yet.

- Matched 100-iteration h4 counterfactual-supervision run (`wm_counterfactual_weight/effect/norm_weight=.05/.05/.05`) did not pass cross-seed control validation. On 40 fixed starts, seed0 clean/perturb return changed baseline `0.0535/0.0534 → 0.0557/0.0547`, with height failure down to `71.9/73.4%` but tilt failure up to `40.6/40.6%`; seed1 return changed `0.0441/0.0436 → 0.0434/0.0428`, height failure `57.8/57.8%`, tilt failure `53.1/56.3%`. Survival stayed zero. The h4 target improves height in both seeds but destabilizes tilt and is not a promotion recipe at this weight; retain the sequence adapter as a learned-dynamics diagnostic and next test a gated/low-dose sequence target with explicit tilt-preservation constraint.

- Low-dose h4 counterfactual supervision (`.01/.01/.01`) also failed the cross-seed gate, but exposed a sharper split. Seed0 return fell to `0.0343/0.0340` with height failure `31.3/32.8%` and tilt failure `68.8/67.2%`; seed1 return rose to `0.0588/0.0577` with height failure `93.8/93.8%` and tilt failure only `6.3/6.3%`. Survival was zero. Lowering the dose does not solve the incompatibility: it selects different failure modes by seed. The h4 target is useful as a diagnostic and for learning sequence response, but must be gated by an independent transition/failure calibration before policy updates.

- Added `--wm_counterfactual_error_gate`: a detached per-sample reliability weight `exp(-gate * current h4 endpoint error)` for simulator sequence targets. This directly implements the independent calibration idea without changing the policy gradient. The gate validates nonnegative strength, preserves the existing one-step default, and a 30-iteration CUDA smoke with h4 data and gate `1.0` completed successfully; all 37 tests and `py_compile` pass. Full matched control evaluation remains pending.

- Full matched h4 run with `--wm_counterfactual_error_gate 1.0` completed for seeds 0/1. On paired 300-step starts, seed0 clean/perturb return `0.0526/0.0521`, height failure `70.3/73.4%`, tilt failure `34.4/34.4%`; seed1 `0.0565/0.0562`, height failure `90.6/90.6%`, tilt failure `10.9/10.9%`; survival remained zero. The gate improves seed1 return over ungated h4 but does not remove the cross-seed failure-mode split and is not a promotion recipe. Training logs report gate fraction 1.0, so the current error scale makes `exp(-error)` effectively nonselective; future gating requires calibrated normalization/quantiles, not this raw-error form.

- Corrected the interpretation of the gate diagnostic: the CSV `gate_fraction` is the policy reliability gate and does not measure counterfactual weighting. A direct 2-iteration probe reported normalized h4 counterfactual weights `mean=1.000, min=0.012, max=1.518`, so `exp(-error)` is selective after all. The same detached reliability weights now apply consistently to endpoint, effect, and norm counterfactual losses. A 30-iteration CUDA smoke completed (`wm=1.4672`, imagined survival `0.932` at the short smoke horizon), with the gate again showing `min=.012/max=1.518`; 37 unit tests and `py_compile` pass. The previous full-run conclusion about gate fraction being 1.0 should be read as referring only to the unrelated policy gate, not as evidence that counterfactual gating was inactive.

- Tested synchronized reliability weighting (the same h4 gate applied to endpoint/effect/norm losses) for 100 iterations on seeds 0/1. Fixed 300-step evaluation: seed0 clean/perturb return `0.0448/0.0441`, height failure `67.2/62.5%`, tilt failure `45.3/45.3%`; seed1 `0.0376/0.0378`, height failure `90.6/90.6%`, tilt failure `9.4/9.4%`; survival remained zero. This is worse than endpoint-only gating (especially seed1 return), so synchronized weighting is rejected as a policy recipe. Keep the implementation as a diagnostic option; the reliable next direction is calibrating the sequence target itself or separating it from policy-facing WM updates.

- Tested the existing independent recovery-head calibration as a possible h4 risk gate. `tools/train_recovery_gate.py` on `docs/recovery_gate_trace_s1.pt` reached validation accuracy `0.5576` with majority baseline `0.5591`; adding a 16-step history gave `0.5581`, still below/at the baseline. The current state-risk features do not predict which recovery head wins, so this selector cannot safely mediate concurrent policy/WM updates. This closes the present recovery-head gate artifact; a useful future selector must include calibrated multi-step transition features and be trained across motion splits, not only a single trace.

- A stricter recovery-gate audit used all 5 trace features plus first differences and an 8-step causal average, with validation split by whole start (51 train starts / 13 held-out starts). Validation accuracy was `0.541` versus majority `0.525`, only a small absolute gain on one trace and far below a deployable selector margin. This is a weak diagnostic hint rather than evidence for policy coupling; no online update was enabled.

- Tested the closest concurrent variant on the current schedule-sampling baseline: `wm_schedule_sampling_prob=.5`, consistency `.1`, plus action-path-only policy-to-WM gradient (`joint_wm_policy_weight=.001`, replay anchor `1`, horizon `4`) for seeds 0/1. Against matched schedule checkpoints on 64 fixed starts, seed0 clean/perturb return fell `0.0660/0.0652 → 0.0367/0.0358`, with tilt failure rising to `10.9/7.8%`; seed1 fell `0.0691/0.0677 → 0.0604/0.0602`, with tilt failure `23.4/20.3%`. Survival remained zero. The action-path coupling is therefore incompatible with the current schedule recipe at this small horizon/budget; direct concurrent gradient is rejected for this arm.

- Implemented the first model-gradient-style geometry objective: h4 simulator/WM action-effect direction cosine plus log-magnitude loss (`--wm_counterfactual_geometry_weight`, `--wm_counterfactual_geometry_mag_weight`), without policy-return gradients entering the WM. The CUDA smoke passed. Full WM-only geometry runs (both seeds, 100 iterations) failed fixed-policy promotion: seed0 clean/perturb `0.0385/0.0386`, seed1 `0.0569/0.0502`, survival `0` for all; the schedule baseline was `0.0660/0.0652` and `0.0691/0.0677`. A held-out multiscale action audit also showed response-vector cosine falling from baseline `0.485/0.508/0.522` to geometry `-0.046/-0.063/-0.061` at deltas `.1/.5/1.0`. The first geometry implementation is therefore rejected; likely the current `wm.targets` projection is not an adequate local action-effect representation, so future work must target latent/action Jacobians rather than endpoint target differences.

- Added `--wm_counterfactual_geometry_root_only` to restrict h4 geometry alignment to 12-D root position/rotation/linear/angular velocity effects, excluding joint coordinates/velocities. The two-seed 100-iteration run did not pass: seed0 clean/perturb return `0.0404/0.0409` with tilt failure `29.7/28.1%`; seed1 `0.0672/0.0667` with survival `0` and height failure `93.8/95.3%`. Root-only masking helps seed1 relative to full geometry but remains split-dependent and does not improve survival; the geometry-loss branch remains diagnostic only.

- Implemented a MnM-style `--online_realism_policy_transitions` option: after each policy imagination rollout, the online real/fake discriminator is updated on detached multi-step policy transitions as fake and replay transitions as real, then its logits are used immediately for the policy realism penalty. CUDA smoke passed. The matched 100-iteration two-seed run with schedule sampling, consistency `.1`, and realism weight `.001` failed fixed-start promotion: seed0 clean/perturb `0.0531/0.0522`, seed1 `0.0520/0.0515`, survival `0` for all; schedule baseline was `0.0660/0.0652` and `0.0691/0.0677`. Seed1 tilt failure rose to `31.3/35.9%`. Training the discriminator on policy fake transitions is more semantically aligned than the old replay-only update, but the current penalty still destabilizes control; retain the option for future lower-bound calibration, do not promote.

- Tested the conservative MnM variant where online realism logits gate imagined policy gradients instead of adding a meaningful penalty (`online_realism_weight=1e-5`, policy-transition fake updates, temperature `1`). Two-seed 100-iteration fixed-start results were seed0 `0.0409/0.0412` and seed1 `0.0543/0.0532` clean/perturb, survival `0` throughout; seed0 tilt failure reached `25%`. This is worse than schedule baseline, so both current online-realism uses (penalty and normalized gate) are closed for this implementation. The remaining MnM path would require an explicitly estimated lower-bound objective, not a local discriminator heuristic.

- Added the generator side of the online MnM game via `--online_realism_model_weight`: after discriminator training, the WM is also trained to make detached replay fake transitions look real. A 30-iteration smoke was stable. The two-seed 100-iteration run (`model_weight=.001`, schedule `.5`, consistency `.1`, no policy realism penalty) still failed fixed-start promotion: seed0 clean/perturb `0.0480/0.0477`, seed1 `0.0597/0.0590`, survival `0`; the schedule baseline was `0.0660/0.0652` and `0.0691/0.0677`. Generator-side realism alone therefore does not repair the model-policy mismatch; the full lower-bound estimator needs calibrated density-ratio/value terms rather than binary realism loss alone.

- Tested density-ratio reward shaping via `--online_realism_reward_weight`, adding discriminator `log D(fake)` at every imagined step instead of a separate realism penalty. The discriminator was updated on policy-imagined fake transitions. Smoke was stable, but the matched 100-iteration run failed badly: seed0 clean/perturb `0.0395/0.0383`, seed1 `0.0438/0.0425`, survival `0`; seed1 tilt failure reached `56.3/57.8%`. Together with the failed penalty, gate, and generator-side variants, this closes the current binary-discriminator interface. A true MnM reproduction now requires estimating the full transition density ratio/lower bound with a calibrated held-out protocol rather than using raw `log D`.

- Revisited the dual-Q/TOM path with `--use_policy_action` so the Bellman bootstrap uses the current policy action at `s'`, rather than reusing the historical replay action. On the current schedule checkpoint, observed-action weights had effective sample fraction `.468` and P95 `2.80`; policy-action weights became much more concentrated (ESS fraction `.284`, P95 `4.64`, max `5.15`, positive/historical means `1.76/.24`). A 30-iteration WM smoke using these policy-action weights reached `wm_err8=7.34` and simulator return `.336`, below the matched schedule smoke (`wm_err8≈4.6`, return `.384`). The policy-action Bellman correction is therefore high-variance even before a long rollout; retain the audit artifact but do not integrate it yet.

- Added an ESS-constrained `--weight_clip` projection to `dual_q_occupancy_audit.py`. For the policy-action dual weights, clip=`2` reduced ESS concentration from fraction `.284`/P95 `4.64` to `.466`/P95 `2.20` (positive/historical means `1.37/.63`). However, a 30-iteration schedule WM smoke with the clipped weights still reached `wm_err8=7.47`, simulator return `.330`, versus `.384` for the matched schedule smoke. Clipping fixes weight concentration numerically but not the underlying policy-action Bellman mismatch; no long rollout is warranted yet.

- Critical implementation audit: online realism experiments before this point were invalid. The discriminator was constructed and then immediately overwritten by the later default initialization (`online_realism_model=None`), so all prior penalty/gate/reward/generator results actually ran without a discriminator. The input dimension was also stale (`3*71+29` instead of the actual `2*71+29`). Fixed both issues and corrected the policy-transition fake path (`As` is a list). A 5-iteration smoke then executed real discriminator updates successfully. The first valid online realism penalty run (`schedule=.5`, consistency `.1`, policy-transition fakes, weight `.001`, seeds 0/1, 100 iterations) gave fixed clean/perturb returns seed0 `0.0442/0.0439`, seed1 `0.0542/0.0545`, survival `0`; it still underperformed the schedule baseline and increased seed1 tilt failure to `43.8/45.3%`. All earlier online-realism conclusions must be discarded; this corrected run is the authoritative result so far.

- Re-ran the previously invalid generator-side MnM branch after fixing discriminator initialization/dimensions. With a real online discriminator trained on policy-imagined fake transitions and `online_realism_model_weight=.001` (no policy realism penalty), two-seed 100-iteration fixed-start results were seed0 clean/perturb `0.0433/0.0425`, seed1 `0.0414/0.0410`, survival `0`; tilt failure reached `17.2%` on seed1. This valid generator-side result also fails the schedule baseline, so the corrected binary realism game remains unpromoted.

- Re-ran the per-step density-ratio reward shaping after fixing the online discriminator initialization/dimension bug. With actual policy-transition fake updates and `online_realism_reward_weight=.001`, two-seed 100-iteration fixed-start results were seed0 clean/perturb `0.0516/0.0516`, seed1 `0.0495/0.0489`, survival `0`; the schedule baseline remained higher (`0.0660/0.0652`, `0.0691/0.0677`). This valid reward-shaped branch also fails; the corrected generator-side and penalty branches fail as well.

- Completed the last previously invalid binary-discriminator variant: corrected online realism confidence gate (`online_realism_weight=1e-5`, temperature `1`, policy-transition fakes, schedule `.5`, consistency `.1`). It is catastrophically unstable on seed0: by iteration 100 `pi_loss=437`, `v_loss=3.5e8`, imagined reward `-15128`, simulator return `-2.89`; seed1 remains finite but weak (`sim_r=.119`). The gate is therefore rejected and the entire current binary discriminator interface is closed. This run also confirms the corrected discriminator path is genuinely active; the earlier invalid experiments must not be used as evidence.

- Tested a structural local-delta realism discriminator after fixing the online discriminator initialization bug. With discriminator inputs changed from absolute `(s,a,s')` to frame-local state error `delta(s',s)` plus action (99-D), policy-transition fakes, schedule sampling `.5`, consistency `.1`, and realism weight `.001`, the CUDA smoke and two-seed 100-iteration runs were stable but failed fixed-start promotion: seed0 clean/perturb return `0.0357/0.0361`, seed1 `0.0547/0.0546`, survival `0` for all; height failure was `96.9/93.8%` (seed0) and `87.5/85.9%` (seed1). The local-delta representation removes absolute-state/time shortcuts but does not solve the model-policy mismatch. Binary discriminator realism is therefore closed for this implementation; a useful MnM continuation needs a calibrated transition-density-ratio or lower-bound objective.

- Hardened `tools/model_realism_audit.py` with an optional source-transition group split, held-out temperature calibration, and density-ratio ESS statistics. On the promoted schedule checkpoint, the 300-iteration audit gave AUROC `.704`, calibrated real-ratio ESS fraction `.975`, and P95 ratio `1.32`; the matched 1000-iteration checkpoint gave AUROC `.656`, ESS fraction `.455`, and P95 ratio `2.98`. The later checkpoint therefore has a much more concentrated calibrated ratio despite similar separability. This is useful as a deployment drift diagnostic and a possible trigger for conservative WM/policy updates, but is not yet used as a training weight because prior odds-weighted WM runs underperformed.

- Cross-seed drift replication on the second schedule 1000-iteration checkpoint gave AUROC `.690`, calibrated ratio ESS fraction `.451`, and P95 ratio `3.40` (temperature `.679`). The ESS collapse is therefore reproducible across training seeds, while the 300-step checkpoint remains near-uniform. This supports a stage-dependent trust-region hypothesis: late training should reduce or freeze policy/model coupling when calibrated transition-ratio concentration crosses a threshold; implementing that gate still requires an online estimator rather than a frozen audit classifier.

- Added opt-in `--online_realism_ess_gate`: the live transition discriminator now computes calibrated-odds ESS over each imagined policy batch and can skip only that batch's policy gradient when ESS falls below a threshold, while simulator collection and factual WM updates continue. A 5-iteration CUDA smoke with threshold `.4` and policy-transition discriminator updates completed with finite losses and a saved checkpoint. This is a trust-region mechanism, not a claimed performance improvement; the next experiment is a matched short rollout with the gate enabled and a no-gate control.

- Strict matched evaluation of the ESS gate required two controls: discriminator/replay-transition updates were kept active in both arms, and discriminator initialization was made RNG-neutral so optional realism modules cannot alter policy/WM initial weights. At threshold `.4`, seed0 clean/perturb return changed from control `.01175/.01334` to `.03805/.03756`; seed1 changed `.02381/.02399` to `.02503/.02479`. The apparent return gain is therefore small and seed-dependent. The consistent effect is a failure-subtype trade: seed0 height/tilt failure moved `40.6/59.4% → 82.8/18.8%`, seed1 `7.8/92.2% → 76.6/26.6%`; survival stayed `0`. ESS gating suppresses tilt failures by accepting more height collapse, so it is not a promotion recipe without a joint termination-risk gate.

- Added the proposed joint variant `--online_realism_ess_risk_floor`, which keeps gradients for imagined starts whose normalized height risk exceeds the floor when ESS gating opens. A seed0 100-iteration run with ESS threshold `.4` and risk floor `.5` still produced clean/perturb return `.06294/.06337`, height failure `96.9/96.9%`, and tilt failure `4.7/3.1%`; it did not recover height stability. The height-risk exception therefore does not repair the support-gate tradeoff at this formulation; keep both options diagnostic only.

- Completed the pending promotion test for `--wm_consistency_alignment_min=0.0` with the exact promoted schedule configuration (`num_envs=8`, `pi_batch=8`, `warmup_steps=4`, two training seeds, 300 iterations). Relative to the matched schedule checkpoints (`seed0 .06948/.06866`, `seed1 .05821/.05857` clean/perturb), alignment-gated consistency reached only seed0 `.04422/.04398` and seed1 `.03396/.03374`; survival stayed `0` and first failure moved earlier, especially on seed1 (`31.8` steps). The gradient-cosine gate is therefore rejected as a deployment recipe; local task/replay alignment is not a sufficient online criterion for retaining free-running consistency.

- Added two-timescale control through `--policy_lr_scale`, scaling policy and critic learning rates while leaving WM learning rate unchanged. A matched `.25` run over 300 iterations was split-dependent: seed0 clean/perturb `.06412/.06439` versus schedule `.06948/.06866` (worse), while seed1 `.06559/.06508` versus `.05821/.05857` (better). Survival remained `0` in all cases and height failure stayed about `97%`. Lower policy LR changes the tracking-return tradeoff but does not address terminal height stability; retain the option for diagnostics and do not promote a global scale.

- Added `--wm_counterfactual_tilt_weight`, an endpoint-only root-rotation preservation term for h4 simulator counterfactual supervision. The matched `.05/.05/.05` endpoint/effect/norm recipe plus tilt weight `.05` gave seed0 clean/perturb `.0560/.0549` and seed1 `.0408/.0410`; compared with the ungated h4 arms (`.0557/.0547` and `.0434/.0428`), tilt failure settled near `9.4%` in both seeds but height failure remained `90.6–93.8%` and survival stayed `0`. The extra root-rotation term does not robustly improve the height/tilt tradeoff, so retain it as a diagnostic auxiliary only.

- Merged the two available h4 counterfactual domains (`cf_height_burnin20_h4_s1.pt` and `_s303.pt`, 256 states) to test whether the h4 split sensitivity was caused by narrow data coverage. The matched `.05/.05/.05` run gave seed0 clean/perturb `.0391/.0391` and seed1 `.0597/.0613`; height failure stayed `90.6–95.3%`, tilt failure `4.7–12.5%`, and survival remained `0`. More counterfactual domain diversity changes which seed wins but does not produce a robust concurrent recipe; retain merged data for offline diagnostics, not policy-facing training.

- Corrected `tools/train_intervention_adapter.py`'s horizon evaluation: its previous baseline compared a one-step WM delta against a four-step simulator endpoint, so the reported adapter gain was not deployment-consistent. The corrected script rolls the nominal WM for the dataset's `counterfactual_horizon` before computing baseline RMSE. Retraining the h4 adapter from `smoke_sched_heightstart_s1_100` gives train RMSE `0.6044→0.1584`, held-out same-domain `0.7780→0.2493`, and independent s303 evaluation `0.6437→0.1896`. This validates sequence-level response fitting offline, while the existing one-step imagined correction interface remains unvalidated and should not consume this adapter directly.

- Implemented an opt-in h4 block endpoint interface (`--wm_intervention_adapter_horizon`): at each block start, the corrected adapter predicts a repeated-action endpoint delta and its residual is distributed across the block; `--wm_intervention_adapter_max_ratio` caps each local residual relative to the nominal one-step delta. The uncapped `.1` blend was unstable (`imag_r=-6.93`, survival `.03` at 5 iterations), while horizon `4` plus blend `.1` and ratio cap `1.5` stayed finite (`imag_r=.351`, survival `.964` at 30 iterations). However, matched 100-iteration fixed 300-step evaluation remained below schedule: seed0 `.05830/.05840`, seed1 `.04413/.04406`, survival `0`; the strong 30-step sanity signal was therefore an early-budget artifact. Keep the block interface as a validated diagnostic, not a promoted recipe.

- Added an endpoint-only h4 block mode (`--wm_intervention_adapter_endpoint_only`) so the adapter residual is applied once at the end of each four-step imagined block instead of being injected at every intermediate step. This removes the distributed residual's repeated local perturbation and is stable in short runs (seed0 30-iteration fixed 100-step sanity: return `.1627`, survival `.0469`; seed1: `.1718`, survival `.0469`). The failure mode remains terminal height collapse (height failure `60.9%` seed0 and `93.8%` seed1; tilt failure `35.9%` and `1.6%`), so the cross-seed result is not robust and the endpoint-only path is diagnostic only.

- The stale GPU process was identified as our own orphaned offline audit (it had accidentally imported `supermimic.train`) and terminated. The required 600-iteration pessimistic policy-target adapter run then completed at 16 envs (same 600 iterations, horizon 32, two policy updates, adapter cap 1.0, LR `1e-5`). Against the matched nominal 600-step checkpoint on test_s4, clean return was `.05005` vs `.08823` and perturb `.05084` vs `.08854`; target first failure was around `43–44` steps versus `58` for nominal, with end survival no better. The adapter path therefore fails the long-budget promotion test despite earlier 100-step gains; the required target-model coupling branch is closed for this recipe.

- Added `--policy_wm_horizon_decay`, a detached MBPO-style trust curve that attenuates later imagined-return terms according to each start's one-step factual WM error, without changing factual WM fitting. The 50-iteration smoke was numerically stable and nearly neutral (`.17521` vs `.17418` fixed 100-step return). A strict 300-iteration seed0 schedule match then fell to `.05857` versus `.06948` control; height failure changed `95.3%→93.8%` but tilt failure rose `7.8%→10.9%` and first failure moved earlier. Error-based horizon truncation therefore does not fix the current model-policy mismatch; retain the option as a diagnostic and do not promote it.

- Added `--joint_wm_policy_root_outputs`, which masks concurrent policy gradients to the WM's 12 root position/rotation/linear/angular-velocity output rows while factual replay gradients still update the full WM. A 50-iteration seed1 smoke improved fixed 100-step return `.1607→.1741`, but the strict 300-iteration match fell `.03645` vs `.05091`. It traded failure subtype (height `67.2%→89.1%`, tilt `39.1%→12.5%`) without improving survival or return. Root-output-only coupling is therefore another short-budget failure-mode trade and remains diagnostic only.

- Delaying root-output coupling until iteration 150 did not repair drift: the matched seed1 300-iteration run scored `.01441` vs `.05091` control, first failure moved `42.4→29.1` steps, and height/tilt failures became `76.6%/23.4%`. Delayed activation is therefore rejected for this mask as well.

- A low-dose root-output coupling check (`joint_wm_policy_weight=.0001`, seed1, 300 iterations) scored `.03403` vs `.05091` control; first failure was later (`45.6` vs `42.4`) but height failure rose `67.2%→95.3%` and tilt remained `18.8%`. Reducing the coefficient changes timing but does not recover return or survival, so scalar tuning of this mask is closed.

- Tested the previously unvalidated `--joint_replace_wm_step` timing: one standalone WM update is replaced by a single policy-return plus replay-fidelity joint update (`wm_batches=2`, seed1, 50 iterations). Fixed 100-step return was `.1050` vs `.1223` matched control; first failure moved earlier `28.6` vs `34.4` steps and height failure rose `90.0%→97.5%`. Replacing the WM step with a concurrent step is therefore not a stability improvement in the current objective.

- Added `--wm_schedule_sampling_ramp_iters` to gradually increase free-running schedule sampling after WM warmup. A strict seed0 300-iteration ramp `.0→.5` over 100 iterations collapsed fixed return to `.03313` vs constant-`.5` control `.06948`; height/tilt failures shifted to `43.8%/67.2%` and first failure moved to `28.8` steps. The current recipe benefits from early schedule sampling; curriculum ramping is rejected.

- Added `tools/counterfactual_risk_gradient_audit.py` and the opt-in `--wm_counterfactual_risk_gradient_weight`, implementing the Model Gradient idea at the observable h4 task-risk level. On the promoted schedule WM, simulator-vs-WM height/tilt risk-gradient correlation was essentially random on both counterfactual domains (`-.015/. - .036`, sign accuracy `50.8%/48.4%`). A 30-iteration smoke with weight `.01` raised the in-domain audit to correlation `.717` and sign accuracy `75.3%`, and fixed 100-step return improved `0.1347→0.1505`. However, the strict 300-iteration seed1 run fell to `.03296` versus `.05821` schedule control, with failure subtype trading from height `96.9%→64.1%` to tilt `3.1%→37.5%`; survival stayed `0`. The risk-gradient target is therefore a valid diagnostic signal but direct WM optimization still destabilizes the policy, so do not promote it without a coupled tilt/safety constraint.

- Added balanced risk-gradient matching (`--wm_counterfactual_risk_gradient_balanced`), normalizing height and tilt finite-difference gradients separately. The 30-iteration smoke improved fixed return `0.1347→0.1716`; at 300 iterations it nearly matched schedule (`.05737` vs `.05821`), slightly reduced height failure (`96.9%→95.3%`) but doubled tilt failure (`3.1%→6.3%`) and did not improve survival. Balanced matching is a better-conditioned diagnostic than the scalar risk target, yet remains below the promotion bar pending a safety-constrained variant.

- Added `--wm_counterfactual_risk_gradient_tilt_scale` for the safety-constrained screening. At scale `2.0`, a matched 100-iteration smoke scored `.09264` vs `.08814` control and delayed first failure (`27.4` vs `31.5` was not a stable safety gain because the control's failure mix was split-dependent); survival remained zero. This is too short and noisy to promote, so no further scalar tilt sweep is planned.

## 2026-11-07 frozen-WM risk-MPC action correction (paired evaluation)

Added optional deployment-time bounded action correction in `tools/fixed_policy_eval.py`: freeze the checkpoint world model, roll a short repeated-action horizon, differentiate normalized root-height/tilt risk with respect to the action, and apply a bounded gradient step. This tests whether the useful counterfactual risk gradient can be converted into online action recovery without retraining the policy.

Paired evaluation used the same 40 starts from `docs/riskmpc_starts.pt`, checkpoint `logs/riskgrad_bal_s1_30/last.pt`, 100 steps, horizon 4, step 0.5, tilt scale 1.0:

| mode | return_norm | failure | height failure | tilt failure | first failure step |
|---|---:|---:|---:|---:|---:|
| fixed policy | 0.17158 | 100% | 100% | 5% | 39.225 |
| + frozen-WM risk-MPC correction | 0.16823 | 100% | 100% | 0% | 39.200 |

The correction slightly reduced return and did not change the dominant height-failure mode. Treat this as a negative deployment diagnostic; no promotion.

## 2026-11-07 two-timescale policy freeze diagnostic

Added `--policy_freeze_after_iter`, which stops policy/critic optimizer updates after a chosen post-WM iteration while continuing factual and free-running WM updates. This directly tests whether late policy drift amplifies the concurrent WM/policy mismatch.

Strict matched schedule configuration (`num_envs=8`, `pi_batch=8`, `buffer_steps=8`, `wm_batches=1`, schedule sampling `.5`, consistency `.1`, 300 iterations) with freeze after iteration 50:

| seed | mode | return_norm | height failure | tilt failure | first failure step | survival |
|---|---|---:|---:|---:|---:|---:|
| 1 | promoted schedule | 0.05822 | 96.9% | 3.1% | 43.97 | 0%
| 1 | freeze policy after 50 | 0.07003 | 96.9% | 3.1% | 47.66 | 0%
| 0 | promoted schedule | 0.06948 | 95.3% | 7.8% | 49.75 | 0%
| 0 | freeze policy after 50 | 0.07004 | 96.9% | 3.1% | 47.67 | 0%

The seed1 gain is substantial, but seed0 loses failure latency and worsens height failure. Since neither seed survives 300 steps and the effect is split-dependent, this is a timescale diagnostic rather than a promotion recipe. It suggests late policy drift matters in some regimes, but a robust method needs a state/model-health-triggered freeze rather than a fixed iteration cutoff.

## 2026-11-07 WM-error-triggered policy freeze diagnostic

Added `--policy_freeze_wm_error_ratio`, `--policy_freeze_wm_error_beta`, and `--policy_freeze_wm_error_min_iter`. After WM fitting, the current policy update is skipped when `wm_err8` exceeds a running EMA ratio, while later updates may resume. This is an online model-health trigger rather than a fixed iteration cutoff.

A strict seed1 100-iteration smoke with the promoted schedule settings (`wm_only_iters=1`, `buffer_steps=8`, `wm_batches=1`, schedule `.5`, consistency `.1`) and ratio `1.3`, beta `.9`, minimum iteration 25 remained finite but degraded: fixed 300-step return `0.06317` versus matched schedule `0.06908`, first failure `45.61` versus `47.06`, and survival remained zero. The training WM error also rose to `13.65` at iteration 100 versus about `1.42` for the matched schedule. The trigger is therefore too aggressive/self-reinforcing in this formulation: skipping policy updates changes the replay distribution and worsens the WM metric that controls the trigger. Keep the option as an audit mechanism, not a training recipe; a useful adaptive freeze needs hysteresis and a separate held-out model-health metric.

## 2026-11-07 factual-error-gated consistency diagnostic

Added `--wm_consistency_error_gate_ratio` and `--wm_consistency_error_gate_beta`. The gate attenuates the free-running consistency term when the current factual replay error exceeds an EMA ratio, while leaving factual WM fitting and policy updates active.

Matched seed1 100-iteration schedule configuration (`wm_only_iters=1`, `buffer_steps=8`, `wm_batches=1`, schedule `.5`, consistency `.1`) with ratio `1.3`, beta `.9` was finite but degraded: final `wm_err8=12.80` versus matched control `1.42`, imagined survival `.210`, and training simulator reward `.348` versus `.427` control. A 300-step fixed-policy evaluation is not warranted because the model-health metric already failed by an order of magnitude. The batch-error scaling changes the auxiliary loss scale too aggressively; retain this as a diagnostic implementation, not a concurrent recipe. A useful trust region likely needs a normalized held-out error and a bounded update ratio rather than direct per-batch attenuation.

- External objective audit (2026-11-07): the current online-realism branches are not a faithful MnM reproduction. The paper's paired-transition classifier supplies both policy-side augmented reward and a model-side value-plus-log-odds objective; SuperMimic currently has separate realism penalty/reward/generator fragments but no value-plus-log-odds model update. Added `docs/mnm_gap_analysis.md` with the equation-level gap and a diagnostic-first implementation gate. Do not interpret the earlier corrected discriminator failures as closing the full lower-bound method.

- Held-out MnM lower-bound audit (2026-11-07): extended `tools/model_realism_audit.py` to report fake-transition classifier logit correlations with factual replay reward and one-step WM RMSE. On the 1000-iteration schedule checkpoints, group-split calibrated classifiers had AUROC `0.724/0.732`, but logit–reward correlations only `0.054/0.065` and logit–RMSE correlations `0.052/-0.053` (seed1/seed0). The discriminator separates real/model transitions, yet its calibrated odds are not a useful value or fidelity proxy on held-out data. This fails the diagnostic gate for a value-plus-log-odds task adapter; do not connect the lower-bound term to policy/WM training until the classifier is conditioned/calibrated for a predictive target.

- Literature-to-implementation audit (2026-11-07): direct extraction from the supplied `2505.13709v3.pdf` shows ROMBRL is a constrained maximin/Stackelberg objective with a replay-distribution KL trust region, dual budget, and timescale ordering `eta_phi >> eta_lambda >> eta_theta`. The earlier unconstrained adversarial residual branch lacked all three and therefore was not a faithful ROMBRL test. Added `docs/rombrl_next_experiment.md` with a minimal adapter+KL+dual implementation and falsification gates; no new policy-facing update is enabled yet.

- Implemented the first constrained ROMBRL diagnostic (2026-11-07): `PolicyTargetWorldModel` now tracks a held-out replay KL proxy; training exposes `--policy_target_kl_budget` and `--policy_target_kl_dual_lr`, updates a nonnegative dual variable, rejects adapter proposals that exceed the budget, and logs/checkpoints the proxy and dual state. A 10-iteration smoke with budget `.1` was finite (`KL≈7.0e-8`, 8 accepts/2 rejects, imagined survival `.509`). A deliberately tight budget `1e-9` activated the constraint (`dual=1.88e-10`, 2 accepts/8 rejects) while remaining finite. This validates the constrained adapter lifecycle, but the proxy scale is far below `.1`; budget calibration on an independent replay distribution is required before any performance experiment.

- ROMBRL KL-budget calibration and paired test (2026-11-07): budget `1e-8` was the first active regime (10-iteration smoke: KL `9.93e-9`, 4 accepts/6 rejects); `1e-7` was effectively inactive (KL `7.00e-8`, 8 accepts/2 rejects). A matched seed1 100-iteration run at budget `1e-8` activated the constraint heavily (6 accepts/94 rejects, final dual `4.71e-8`, KL proxy `1.34e-7`) but destabilized the model (`wm_err8=13.47`) and underperformed the schedule fixed-policy control (`return .06395` vs `.06908`, first failure `46.34` vs `47.06`, survival 0). The constrained adapter lifecycle is functional, but this adapter/KL proxy is not a control improvement; no long-budget run is warranted until the task-model update is decoupled from the nominal WM error blow-up.

- ROMBRL pessimistic-return ablation (2026-11-07): enabling `--policy_target_pessimistic` on the active KL-budget `1e-8` adapter did not repair the feedback loop. Seed1 100-iteration training ended at `wm_err8=13.71`, imagined survival `.189`, KL proxy `1.53e-7`, dual `5.21e-8`, with 5 accepts/95 rejects. Fixed-policy return stayed `.06395` (height failure `95.3%`, tilt `4.7%`, survival 0), identical to the non-pessimistic KL arm. Taking the minimum nominal/task imagined return is insufficient when the task adapter still shifts the policy's data distribution; close this variant.

- ROMBRL adapter per-step correction cap (2026-11-07): added `--policy_target_adapter_max_delta`, clipping the task adapter's normalized RMS output correction before applying it to the target WM. With KL budget `1e-8` and cap `.05`, imagined survival improved `.244→.312` and `wm_err8` fell `13.47→12.58`, but fixed-policy 300-step return degraded to `.05194` (height failure `96.9%`, tilt `3.1%`, first failure `42.81`, survival 0). This is another model-imagination versus real-control divergence; the cap is diagnostic only and not a promotion recipe.

## 2026-11-07 long-budget schedule baseline resolves much of the early-failure bottleneck

Resumed the existing 64-env schedule checkpoint `logs/schedsample05_s1_1000/last.pt` with its original configuration (`num_envs=64`, `buffer_steps=64`, `wm_batch=32`, `pi_batch=32`, `wm_only_iters=5`) to iteration 3000. Training remained stable; imagined survival rose from about `.4` near iteration 1000 to `.97` at iteration 3000, and final `wm_err8=.228`.

On the exact same 64 fixed starts used for the 1000-iteration checkpoint:

| checkpoint | return_norm | height failure | tilt failure | first failure step | end survival |
|---|---:|---:|---:|---:|---:|
| 1000 iterations | 0.0862 | 95.3% | 4.7% | 52.9 | 0%
| 3000 iterations | 0.3305 | 23.4% | 4.7% | 246.1 | 71.9%

Held-out start splits confirm transfer: survival is `75.0%` on holdout202 (return `.3140`) and `64.1%` on holdout303 (return `.3451`), with no tilt failures on either split. This is the strongest current evidence that early 300/1000-iteration conclusions were budget-limited. It becomes the new long-budget baseline for any concurrent WM/policy variant; aggressive joint gradients must beat this checkpoint, not the under-trained 300-step schedule.

- Long-budget action-path coupling recheck (2026-11-07): started the exact 64-env/horizon-64 schedule configuration that produced the strong 3000-step baseline, with `joint_wm_policy_action_path` and weight `.001`. The coupling reduced throughput from about `5.6` to `3.0 it/s`; by iteration 500, training simulator reward fell from `.322` at iteration 100 to `.016`, while WM `e8` rose to `2.58` and imagined survival plateaued near `.83`. No checkpoint was available before the configured 1000-step save, so the run was stopped at 500 as a negative screening result. The action-path arm should not be extended to 3000 under this large-budget configuration without a new stability mechanism.

- Low-dose long-config action-path screen (2026-11-07): reducing `joint_wm_policy_action_path` weight from `.001` to `1e-5` did not remove the failure on the 64-env/horizon-64 schedule configuration. By iteration 300, training simulator reward fell `.323→.019`, WM `e8` was `2.39`, and imagined survival `.779`; the `.001` run showed the same early collapse pattern. The issue is therefore not simply an overly large coupling coefficient; this action-path direction is incompatible with the large long-budget recipe and remains closed.

## 2026-11-07 long-budget cross-seed replication

Extended the matching 64-env schedule `.05` configuration from existing 1000-iteration checkpoints to 3000 for seed0 as well as seed1.

| seed | checkpoint | return_norm | height failure | tilt failure | first failure step | end survival |
|---|---|---:|---:|---:|---:|---:|
| 0 | 1000 | 0.0913 | 98.4% | 1.6% | 56.3 | 0%
| 0 | 3000 | 0.0678 | 82.8% | 20.3% | 61.2 | 0%
| 1 | 1000 | 0.0862 | 95.3% | 4.7% | 52.9 | 0%
| 1 | 3000 | 0.3305 | 23.4% | 4.7% | 246.1 | 71.9%

The seed1 long-budget gain is not robust: seed0 trades some height failures for a large tilt-failure increase and loses return. This makes failure-subtype and cross-seed gates mandatory for future concurrent-training claims; imagined survival alone is not a sufficient stopping criterion.

## 2026-11-07 seed0 5000-iteration continuation

Extended the seed0 64-env schedule `.05` run from `logs/schedsample05_s0_3000/last.pt` to iteration 5000 (`logs/schedsample05_s0_5000`). The training trace remained finite; imagined survival fluctuated between `.17` and `.70`, ending at `.701`, while final `wm_err8=.437`. On the same 64 fixed starts used for the earlier checkpoints, the real-environment result improved substantially over seed0 at 3000 but still lagged the seed1 long-budget result:

| checkpoint | return_norm | height failure | tilt failure | first failure step | end survival |
|---|---:|---:|---:|---:|---:|
| seed0, 3000 | 0.0678 | 82.8% | 20.3% | 61.2 | 0%
| seed0, 5000 | 0.1853 | 45.3% | 1.6% | 203.0 | 53.1%
| seed1, 3000 | 0.3305 | 23.4% | 4.7% | 246.1 | 71.9%

The seed0 trajectory therefore shows a real late-training recovery, especially by removing the tilt-failure spike, but it does not establish a robust cross-seed recipe: at 5000 it remains below seed1 at 3000 in return, survival, and height-failure rate. Continue using fixed-start, subtype-aware evaluation and do not select checkpoints from imagined survival alone. Evaluation artifact: `docs/schedsample05_s0_5000_eval.json`.

The seed0 5000 checkpoint also transfers consistently to independent fixed-start splits: holdout202 gives return `.1821`, height failure `46.9%`, tilt failure `1.6%`, first failure `192.4`, survival `.5156`; holdout303 gives return `.1939`, height failure `45.3%`, tilt failure `3.1%`, first failure `201.2`, survival `.5156`. Thus the recovery is not confined to the original paired starts, but its absolute level remains below the seed1 3000 baseline. Artifacts: `docs/schedsample05_s0_5000_holdout202_eval.json` and `docs/schedsample05_s0_5000_holdout303_eval.json`.

## 2026-11-07 checkpoint trajectory audit

Evaluated the intermediate seed0 checkpoint `logs/schedsample05_s0_5000/ckpt_004000.pt` on the same fixed starts. Despite its training trace showing imagined survival `.318`, its real 300-step performance was return `.0885`, height failure `87.5%`, tilt failure `6.25%`, first failure `110.9`, survival `6.25%`. The 5000 checkpoint then rose to return `.1853`, height failure `45.3%`, tilt failure `1.6%`, first failure `203.0`, survival `53.1%`. This is a sharp late recovery rather than a smooth improvement, and further supports using real fixed-start/holdout validation for checkpoint selection instead of imagined survival.

## 2026-11-07 cross-start checkpoint comparison

Swapping the two final policies across the paired start files did not change the conclusion: seed1-3000 on the seed0 start set gives return `.3305`, height failure `23.4%`, tilt failure `4.7%`, survival `.7188`; seed0-5000 on the seed1 start set gives return `.1855`, height failure `45.3%`, tilt failure `1.6%`, survival `.5312`. The observed gap is therefore a policy/checkpoint difference rather than an artifact of the nominal start-set label. Artifacts: `docs/schedsample05_s1_3000_on_s0_eval.json`, `docs/schedsample05_s0_5000_on_s1_eval.json`.

Correction: `paired_starts_s0_300.pt` and `paired_starts_s1_300.pt` contain identical `clip` and `off` arrays (their file hashes differ only because of metadata). The preceding cross-start run therefore reproduces the same 64-start benchmark; it is not an independent start-set swap. Retain the numerical evaluations as same-benchmark replication, but do not count them as evidence against start-distribution confounding. Independent evidence remains the separate holdout202/holdout303 files.

## 2026-11-07 fresh random-start generalization audit

Generated a new 128-start set (`docs/fresh_starts_20261107_128.pt`) with seed `20261107`, independent of the prior paired/holdout files, and evaluated both leading checkpoints for 300 steps:

| checkpoint | return_norm | height failure | tilt failure | first failure step | end survival |
|---|---:|---:|---:|---:|---:|
| seed1, 3000 | 0.3281 | 22.7% | 0.8% | 251.0 | 76.6%
| seed0, 5000 | 0.1730 | 46.1% | 3.1% | 191.2 | 52.3%

The same ranking holds on this fresh sample: seed1-3000 remains substantially stronger, while seed0-5000 retains its late-training recovery but has roughly twice the height-failure rate. Artifacts: `docs/fresh_starts_20261107_128.pt`, `docs/schedsample05_s1_3000_fresh128_eval.json`, and `docs/schedsample05_s0_5000_fresh128_eval.json`.

## 2026-11-07 motion-conditioned failure probe

The fresh-start per-clip audit identifies a shared hard motion rather than a global failure. On 32 fresh starts from `fallAndGetUp1_subject5` (clip 10), both policies have zero end survival: seed1-3000 has height/tilt failures `71.9%/28.1%` and first failure `85.6`; seed0-5000 has `93.8%/6.3%` and first failure `74.2`. On `jumps1_subject1` (clip 19), both are comparatively strong: seed1-3000 survival `.875`, seed0-5000 `.8125`. This suggests the next targeted experiment should address recovery-motion conditioning/data coverage, especially the fall-and-get-up subset, rather than adding another global WM-policy coupling term. Artifacts: `docs/fresh_clip10_32.pt`, `docs/fresh_clip19_32.pt`, and the four corresponding `*_clip*_eval.json` files.

## 2026-11-07 kinematic-reset coverage screen

Tested the existing kinematic-cluster reset sampler (`reset_sampling_mode=kinematic`, 16 clusters) for a matched seed1 1000-iteration schedule run. Training remained finite but ended with imagined survival `.087` and simulator return `.271`, versus the return-sampler seed1 1000 trace (`sim_r≈.427` near the end). On the fresh 128-start benchmark the kinematic reset checkpoint achieved return `.0452`, height failure `99.2%`, first failure `39.0`, survival `0%`; on the focused `fallAndGetUp1_subject5` starts it achieved return `.0368`, height failure `84.4%`, tilt failure `15.6%`, survival `0%`. This low-budget coverage intervention is negative and should not be extended to 3000 without a different mechanism; the hard recovery-motion failures are not fixed by uniform kinematic reset coverage alone. Artifacts: `logs/kinematic_cov_s1_1000`, `docs/kinematic_cov_s1_1000_fresh128_eval.json`, and `docs/kinematic_cov_s1_1000_clip10_eval.json`.

## 2026-11-07 failure-focused policy-start screen

Tested `policy_failure_start_fraction=.5` with balanced height/tilt ranking and a 500-iteration ramp for a matched seed1 1000-iteration schedule run. The mechanism stayed finite (`wm_err8=.884`), but the fresh 128-start benchmark was return `.0529`, height failure `87.5%`, tilt failure `27.3%`, first failure `43.5`, survival `0%`. On the focused `fallAndGetUp1_subject5` set it was return `.0405`, height failure `68.8%`, tilt failure `40.6%`, first failure `41.8`, survival `0%`. Relative to the ordinary seed1 1000 checkpoint, it did not improve survival or return and converted some height failures into tilt failures. Do not extend this exact high-fraction balanced sampler; a useful recovery objective needs better phase/clip conditioning and a lower-risk update than simply replay-start oversampling. Artifacts: `logs/failurestart_balanced_s1_1000`, `docs/failurestart_balanced_s1_1000_fresh128_eval.json`, and `docs/failurestart_balanced_s1_1000_clip10_eval.json`.

## 2026-11-07 phase-window screen (`goal_frames=20`)

Expanded the policy target window from 10 to 20 future frames for a matched seed1 1000-iteration schedule run. Throughput fell from about `5.7` to `5.0` it/s during warmup, and training remained finite (`wm_err8=.948`), but real control degraded: fresh 128-start return `.0437`, height failure `93.8%`, tilt failure `7.8%`, first failure `35.8`, survival `0%`; on `fallAndGetUp1_subject5`, return `.0321`, height failure `71.9%`, tilt failure `34.4%`, first failure `33.0`, survival `0%`. A longer raw target window does not provide a useful phase-aware recovery representation under this recipe. Artifact: `logs/goal20_s1_1000` and the corresponding `docs/goal20_s1_1000_*_eval.json` files.

## 2026-11-07 explicit phase-conditioned policy screen

Added an optional `--phase_conditioned_policy` path that appends normalized clip phase as `(sin(2πp), cos(2πp))` to proprioceptive policy observations; evaluation infers the flag from the checkpoint. Existing 37 unit tests still pass. A matched seed1 1000-iteration screen remained finite (`wm_err8=1.058`), but real performance was negative: fresh 128-start return `.0471`, height failure `99.2%`, tilt failure `1.6%`, first failure `39.1`, survival `0%`; on `fallAndGetUp1_subject5`, return `.0367`, height failure `90.6%`, tilt failure `15.6%`, first failure `38.0`, survival `0%`. A raw normalized clip phase is therefore not a sufficient recovery state variable and should not be extended unchanged. Code path is in `supermimic/mdp.py`, `supermimic/train.py`, and `tools/fixed_policy_eval.py`; artifact: `logs/phasecond_s1_1000`.

## 2026-11-07 clip-identity conditioned policy screen

Added an optional `--clip_conditioned_policy` path that appends a one-hot identity for the 40 motion clips to policy observations; existing tests remain green. A matched seed1 1000-iteration run stayed finite (`wm_err8=.911`) but degraded on real evaluation: fresh 128-start return `.0402`, height failure `93.8%`, tilt failure `7.0%`, first failure `34.6`, survival `0%`; on `fallAndGetUp1_subject5`, return `.0297`, height failure `71.9%`, tilt failure `28.1%`, first failure `31.5`, survival `0%`. Explicit clip identity does not rescue the hard motion at this budget and likely increases optimization burden; close this direct one-hot variant. Code path is in `supermimic/mdp.py`, `supermimic/train.py`, and `tools/fixed_policy_eval.py`; artifact: `logs/clipcond_s1_1000`.

## 2026-11-07 dynamics-perturbation diagnostic

Evaluated the leading checkpoints on the same fresh 128 starts with identical friction, joint-offset, and COM perturbations. Seed1-3000 changed only from survival `.7656` to `.7500` (return `.3281→.3192`); seed0-5000 changed from `.5234` to `.5312` (return `.1730→.1745`). On the hard `fallAndGetUp1_subject5` starts, both remained at zero survival under perturbation. The dominant failure is therefore motion-conditioned even in the nominal simulator, rather than primarily a robustness-to-dynamics mismatch; prioritize recovery/contact-specific targets over a generic robust-WM objective. Artifacts: `docs/schedsample05_s1_3000_fresh128_perturb_eval.json`, `docs/schedsample05_s0_5000_fresh128_perturb_eval.json`, and the corresponding `clip10_perturb` files.

## 2026-11-07 recovery-phase termination-grace screen

Added optional reference-aware termination grace: height errors are not terminal while reference root height is below `.35`, and tilt errors are not terminal while reference uprightness is below `.5`. This tests whether intentional fall/get-up phases were being truncated before recovery. The matched seed1 1000-iteration run stayed finite (`wm_err8=.935`, imagined survival `.177`) but real performance remained negative: fresh 128-start return `.0525`, height failure `96.9%`, tilt failure `4.7%`, first failure `44.2`, survival `0%`; focused `fallAndGetUp1_subject5` return `.0392`, height failure `78.1%`, tilt failure `21.9%`, first failure `42.6`, survival `0%`. The grace thresholds do not rescue recovery at this budget. The diagnostic oracle one-step reference-action controller also had zero survival on the hard clip, suggesting a low-level/root-contact feasibility issue rather than only early termination. Code path: `supermimic/env.py`, `supermimic/train.py`, `tools/fixed_policy_eval.py`; artifact: `logs/fallgrace35_05_s1_1000`.

## 2026-11-07 hard-clip feasibility and focused-training audit

The 32 random starts on `fallAndGetUp1_subject5` all failed for both leading policies, but filtering to 32 starts with reference pelvis height above `.5 m` gave seed1-3000 survival `12.5%`, return `.1921`, first failure `136.9`. This confirms a strong start-phase dependence; the entire clip is not uniformly impossible. However, a 1000-iteration single-clip training run using only `fallAndGetUp1_subject5` (`logs/hardclip10_s1_1000`) did not solve it: high-height starts had return `.0470`, height failure `96.9%`, survival `0%`; the original mixed starts had return `.0437`, survival `0%`. Thus simple data concentration is insufficient; the failure likely requires contact-aware state/action modeling or a physically compatible recovery controller. Artifacts: `docs/fresh_clip10_highz32.pt`, `docs/fresh_hardclip10_highz32.pt`, and `docs/hardclip10_s1_1000_*_eval.json`.

## 2026-11-07 reference-contact audit and cleaned-data screen

Forward-kinematics audit of all 40 retargeted motions found below-ground body-link samples in 12 clips. The hard `fallAndGetUp1_subject5` clip has minimum body-link height `-0.1106 m`, at least one link below ground on `6.72%` of frames, and `0.51%` of all body-frame samples. Other fall/get-up clips are also affected (`fallAndGetUp2_subject3`: `18.90%` of frames), whereas ordinary walk/run clips are mostly clean. Full report: `docs/motion_contact_audit.json`.

A cleaned motion directory excluding all clips with any below-ground FK sample (28/40 clips retained) was used for a matched seed1 1000-iteration run (`logs/contactvalid_s1_1000`). On a fresh 128-start benchmark drawn from the cleaned directory, the cleaned-data policy achieved return `.0535`, height failure `97.7%`, first failure `48.6`, survival `0%`; the ordinary full-data seed1-1000 checkpoint on the same starts achieved return `.0669`, height failure `96.9%`, first failure `43.4`, survival `0%`. This short budget does not demonstrate a performance gain, but the contact audit identifies a concrete upstream data-validity defect that should be repaired before interpreting fall/get-up policy results.

## 2026-11-07 contact-lift repair screen

Constructed `data/lafan1_g1_contact_lifted` by vertically translating each clip whose FK body links crossed the ground, using the minimum-link offset plus `1 cm`; the repaired library has global minimum body height `0.01 m`. A matched seed1 1000-iteration run (`logs/contactlift_s1_1000`) was finite (`wm_err8=.983`) but did not improve the fresh 128-start benchmark: lifted-data policy return `.0585`, height failure `86.7%`, tilt failure `20.3%`, first failure `49.5`, survival `.8%`; the ordinary full-data checkpoint on the same lifted starts returned `.0716`, height failure `93.8%`, tilt failure `6.3%`, first failure `45.4`, survival `.8%`. A simple global vertical translation is therefore not a sufficient repair, though the FK audit remains a concrete data-quality issue requiring phase/contact-aware retarget correction rather than policy-only tuning.

## 2026-11-07 frame-wise contact projection screen

Built `data/lafan1_g1_contact_projected` by computing per-frame FK minimum body-link height, adding the smallest nonnegative root-z correction needed to keep every link above `1 cm`, and interpolating that correction back to the original 30 Hz CSV. The repaired library's global minimum body height is `0.0025 m`. A matched seed1 1000-iteration run (`logs/contactproject_s1_1000`, final `wm_err8=.897`) did not improve a fresh 128-start benchmark: projected-data policy return `.0584`, height failure `87.5%`, tilt failure `16.4%`, first failure `47.1`, survival `1.6%`; the ordinary full-data checkpoint on the same projected starts returned `.0769`, height failure `92.2%`, tilt failure `7.8%`, first failure `47.1`, survival `1.6%`. Contact projection alone is insufficient and changes failure subtype without increasing survival.

## 2026-11-07 cleaned-data long-budget test

Extended the contact-valid (28/40 clips, all FK body-link samples above ground) seed1 run from 1000 to 3000 iterations (`logs/contactvalid_s1_3000`). Training imagined survival rose to `.846` and final `wm_err8=.643`, but on the same fresh 128 starts from the cleaned directory the policy achieved return `.1398`, height failure `43.0%`, tilt failure `1.6%`, first failure `211.5`, survival `56.3%`. The full-data seed1-3000 checkpoint evaluated on those same clean starts achieved return `.3570`, height failure `10.9%`, tilt failure `1.6%`, first failure `272.5`, survival `87.5%`. Removing invalid clips is therefore not sufficient and substantially reduces transfer at this budget; the invalid motions may still provide useful pose diversity even though their contact geometry is wrong. Preserve the contact audit as a data-quality gate, but do not promote naive clip deletion as a training recipe.

## 2026-11-07 contact-safe tracking reward screen

Added optional `--contact_safe_reward`, which clamps reference body-link z targets to `0.02 m` inside the tracking reward while retaining the original motion, joints, and root state. The implementation initially exposed an inplace autograd bug; this was fixed with an out-of-place tensor construction, and all 37 tests remain green. A matched seed1 1000-iteration run was finite (`wm_err8=.966`) but negative on real control: fresh 128-start return `.0445`, height failure `94.5%`, tilt failure `8.6%`, first failure `43.5`, survival `0%`; hard clip return `.0331`, height failure `71.9%`, tilt failure `31.3%`, survival `0%`. Clamping only the reward target is insufficient; it changes failure subtype without recovering the motion.

## 2026-11-07 contact-masked reward screen

Added optional `--contact_masked_reward`: on reference frames where any FK body link is below ground, body-link position/orientation/velocity tracking terms are removed from the reward gradient while root tracking remains active. This preserves invalid clips and avoids explicitly chasing impossible body targets. The matched seed1 1000-iteration run remained finite (`wm_err8=1.130`) but was negative: fresh 128-start return `.0460`, height failure `79.7%`, tilt failure `31.3%`, first failure `44.2`, survival `0%`; hard clip return `.0344`, height failure `78.1%`, tilt failure `25.0%`, survival `0%`. The mask shifts height/tilt failure balance but does not recover the motion at this budget. All 37 tests remain green.

## 2026-11-07 dynamics audit and local smoothing screen

Computed 50 Hz reference dynamics in `docs/motion_dynamics_audit.json`. `fallAndGetUp1_subject5` has joint-acceleration peaks about `652 rad/s²` (RMS `26.4`), while its root acceleration peak is `24.3 m/s²`; the dominant issue is fast joint/contact transitions rather than only root translation. Applied a light Gaussian temporal smoothing (`sigma=.8` source frames) to root position, quaternion (renormalized), and joints in the 12 below-ground clips, retaining all motion files (`data/lafan1_g1_contact_smoothed`). A matched seed1 1000-iteration run (`logs/contactsmoothed_s1_1000`) did not improve fresh 128-start control: smoothed policy return `.0539`, height failure `75.8%`, tilt failure `35.2%`, first failure `47.1`, survival `0%`; the ordinary checkpoint on the same smoothed starts returned `.0761`, height failure `90.6%`, tilt failure `9.4%`, survival `0%`. Smoothing shifts the failure subtype but does not solve recovery.

## 2026-11-07 contact-pattern conditioned policy screen

Added optional `--contact_conditioned_policy`, appending a 14-body reference contact mask (`body_z < .05 m`) to policy observations. This is a direct local contact-phase signal; implementation passed all 37 tests. A matched seed1 1000-iteration run remained finite (`wm_err8=1.005`) but did not improve real control: fresh 128-start return `.0527`, height failure `96.9%`, tilt failure `3.9%`, first failure `44.2`, survival `0%`; hard clip return `.0399`, height failure `90.6%`, tilt failure `12.5%`, first failure `43.3`, survival `0%`. A reference contact mask alone is not sufficient; the policy needs actual simulator contact/force feedback or a contact-consistent transition model.

## 2026-11-07 WM risk-MPC recovery probe

Used the existing frozen-WM risk-MPC evaluator to add bounded one-shot action corrections to seed1-3000. On the hard clip, horizons 4/8 and steps `.05/.10` raised return from the policy baseline `.0821` to at most `.0837` and delayed mean failure from about `85.6` to `87.8`, but all configurations still had survival `0%`. On the fresh 128-start benchmark, the best setting (horizon 4, step `.10`) was return `.3295`, survival `.7656`, essentially identical to the uncorrected policy (`.3281`, `.7656`). The nominal WM has no useful recovery gradient for the hard contact transition; model-based short-horizon correction is not a rescue.
## 2026-11-07 continuation: current FK contact observation and realism audit

- Added an opt-in `--current_contact_policy` feature. It appends a 14-dimensional binary body-link contact pattern computed from the **current simulated state** via FK (`body_z < 0.05`) to the policy observation. This is distinct from the existing reference-contact feature and is a low-dimensional contact-feedback ablation; it does not alter reward, termination, or the world model. `supermimic/mdp.py`, `supermimic/train.py`, and `tools/fixed_policy_eval.py` were updated. `py_compile` and the full 37-test suite pass.
- Minimal training (`logs/currentcontact_s1_1000`, seed 1, 64 envs, 1000 iterations) completed. On the fixed 128-start, 100-step evaluation, current-contact policy: normalized return **0.3543**, survival **0.7500**, height failure **0.1797**, tilt failure **0.0703**, mean first failure **88.4**. Same starts with the strongest seed-1 baseline (`logs/schedsample05_s1_3000/last.pt`): return **0.3586**, survival **0.8438**, height failure **0.1484**, tilt failure **0.0078**, first failure **89.4**. The added signal did not help and notably increased tilt failures; do not promote this feature yet.
- A held-out paired world-model realism audit on `logs/schedsample05_s1_3000/last.pt` (`tools/model_realism_audit.py --group_split`) found AUROC **0.604**, fake-transition RMSE mean **0.244**, and classifier-logit/RMSE correlation **0.078**. The model-vs-real distinction is weak and does not currently provide a reliable risk signal for policy gating.

## 2026-11-07 continuation: dynamic-component WM loss weighting (promising)

- The replay audit showed the dominant one-step prediction errors were root angular velocity and joint velocity, rather than contact frames: baseline component RMSE was root angular velocity **0.189**, joint velocity **0.369**; contact-frame aggregate RMSE was **0.231** versus **0.260** on clean frames. This redirected the experiment away from adding more contact features.
- Added an explicit weight file `docs/wm_weights_dyn.txt` for `--wm_component_weights_file`: root angular-velocity components are weighted 3x and joint-velocity components 2x, while other components remain 1x. No model architecture or policy objective was changed.
- A 1000-iteration run reduced those WM errors (root angular velocity **0.146**, joint velocity **0.312**) but was still early and evaluated below baseline. Resuming that checkpoint to 3000 iterations (`logs/wmdyn_s1_3000`) produced a strong result. On the same 128 fixed starts and 100 simulator steps, it achieved normalized return **0.7006**, survival **0.9141**, height failure **0.0469**, tilt failure **0.0391**, mean first failure **94.4**; the strongest baseline achieved **0.3586 / 0.8438 / 0.1484 / 0.0078 / 89.4**.
- On the difficult `fallAndGetUp1_subject5` clip-10 high-z starts (`docs/fresh_clip10_highz32.pt`, 100 steps), the dynamic-weight run achieved return **0.6772**, survival **0.8125**, height failure **0.0313**, tilt failure **0.1563**, first failure **95.6**; the paired baseline achieved **0.3325 / 0.5000 / 0.4688 / 0.0313 / 75.4**. The main gain is preventing height collapse; tilt failures remain a trade-off to investigate.
- Current promoted checkpoint for this branch: `logs/wmdyn_s1_3000/last.pt`. The weighting remains opt-in; the default baseline configuration is unchanged.

## 2026-11-07 continuation: small tilt regularization on dynamic-weight WM

- Resumed `logs/wmdyn_s1_3000/last.pt` for 500 iterations with the same dynamic WM weights and `--policy_tilt_penalty 0.001`, producing `logs/wmdyn_tilt001_s3500/last.pt`.
- On the same 128 fixed starts and 100 simulator steps, the combined run achieved normalized return **0.7355**, survival **0.9375**, height failure **0.0547**, tilt failure **0.0078**, mean first failure **95.4**. Relative to dynamic-weight WM alone (`0.7006 / 0.9141 / 0.0469 / 0.0391 / 94.4`), return and survival improved while tilt failures dropped sharply; height failure changed only slightly.
- On clip-10 high-z starts, the combined run achieved return **0.6916**, survival **0.8750**, height failure **0.0313**, tilt failure **0.0938**, first failure **94.9**. Dynamic-weight WM alone was **0.6772 / 0.8125 / 0.0313 / 0.1563 / 95.6**. This is the strongest current branch, although the clip-10 result is still a 100-step diagnostic rather than a full-clip guarantee.
- Current promoted candidate: `logs/wmdyn_tilt001_s3500/last.pt`, with the two opt-in settings `--wm_component_weights_file docs/wm_weights_dyn.txt --policy_tilt_penalty 0.001`.

## 2026-11-07 continuation: long-horizon validation and tilt-penalty dose response

- Generated `docs/long_starts_64.pt` with starts guaranteed to have at least 500 recorded steps remaining. On this independent long-horizon set, the promoted `wmdyn_tilt001_s3500` checkpoint achieved return **0.5657**, survival **0.6719**, height failure **0.2188**, tilt failure **0.1250**, and first failure **399.6**; the paired baseline was **0.2753 / 0.6094 / 0.3594 / 0.0469 / 349.9**. The improvement persists over 500 steps, while the tilt/height trade-off is visible.
- A dose-response continuation with `--policy_tilt_penalty 0.002` (`logs/wmdyn_tilt002_s4000`) gave fresh-100-step **0.7493 / 0.9375 / 0.0547 / 0.0078** and long-500-step **0.6002 / 0.6719 / 0.2656 / 0.0781**. On clip-10 high-z starts it was **0.7053 / 0.8125 / 0.1250 / 0.0625**, versus `0.001` at **0.6916 / 0.8750 / 0.0313 / 0.0938**. Thus `0.002` trades lower tilt failures for substantially more height failures and lower hard-clip survival; it is retained as a diagnostic, not promoted over `0.001`.

## 2026-11-07 continuation: seed-0 replication

- Trained the promoted configuration from scratch with seed 0 (`logs/wmdyn_tilt001_s0_3000`, 3000 iterations, 64 envs), using `--wm_component_weights_file docs/wm_weights_dyn.txt --policy_tilt_penalty 0.001`.
- On the common 128 fixed starts and 100 steps, seed 0 dynamic weighting achieved return **0.6859**, survival **0.9063**, height failure **0.0859**, tilt failure **0.0078**, first failure **94.5**. The seed-0 baseline (`logs/schedsample05_s0_5000/last.pt`) achieved **0.2324 / 0.6484 / 0.3359 / 0.0234 / 76.8**.
- On the common 64 long starts and 500 steps, seed 0 dynamic weighting achieved return **0.5556**, survival **0.6875**, height failure **0.1563**, tilt failure **0.1563**, first failure **412.4**. The baseline achieved **0.1360 / 0.3906 / 0.5938 / 0.0469 / 274.4**. The improvement in return, survival, height robustness, and failure time transfers across seeds; the remaining long-horizon tilt trade-off is also seed-dependent and remains an open target.
