# SuperMimic concurrent WM-policy：下一阶段实验规格

## 已确立的边界

- 固定 environment interaction/iteration budget：`policy_updates=2` 在 subject4/5 的 500-step rollout 上优于 `policy_updates=1`。
- 固定 optimizer compute 或近似 wall-clock：单更新延长训练到 3000–4000 steps 反而更好；双更新不是 compute-efficient 方法。
- 固定 cutoff、raw gradient norm、batch-gradient cosine 和 held-out policy-loss delta 都没有形成可靠在线 gate。

## 实验 arm

### A. Sample-efficiency arm

保持当前 recipe：4-step teacher-forced WM、8-step free-running auxiliary (`.05`, ramp300)、`wm_batches=2`、独立 replay batch、`policy_updates=2`、BPTT truncation=4。固定 interaction budget 为 1000/2000 iterations，在 subject4/5 各跑两个 seeds，报告 100/200/500-step rollout。

### B. Compute-efficiency arm

使用 `policy_updates=1`，将总 iterations 调到与 A 的 policy/WM optimizer update 数相等（A@2000 对 B@4000），并额外记录 wall-clock、环境 transitions、GPU memory 和 optimizer-step 数。主指标必须按三种轴分别报告：interaction、optimizer compute、wall-clock。

### A2. Data-coverage arm（新增）

将 allmimic 风格的确定性/带噪声采样混合单独作为数据覆盖消融：`deterministic_collect_fraction=0.5`，其余参数完全沿用当前最佳 double-update recipe。先做 subject5 seed0 的 1000-step 筛选；若 matched 500-step real-sim return 不劣于 nominal，再扩展到 subject4 和第二个 seed。

预算审计后，subject4 严格 1000-iteration、三个训练 seed、12 个 paired 条件的 ramp/nominal 均值为 `0.22580/0.22024`（+2.52%，胜出 8/12）。因此 ramp 仍值得保留，但必须继续扩大 seed/subject split，先估计小幅收益的方差，再决定是否把它作为默认 recipe。

组合复核已完成：subject4 三个 seed、12 个 paired 条件下 `policy_updates=2 + aux-ramp` 12/12 胜出，但它是 interaction-matched、compute 约为单更新两倍。subject4 seed0 的 compute-matched 对照反而由单更新长训胜出约 8.0%，因此报告时必须同时保留 interaction/sample-efficiency 与 optimizer-compute 两条轴；预算审计工具显示此前对照仅匹配 policy steps，未匹配 WM steps；因此撤回 total-compute-matched 结论。已完成两个 subject 的 total-compute-matched 对照：subject4 seed0 单更新高约 13.3%，subject5 两 seed 汇总仅约 +0.6% 且方向翻转。nominal-prefix uncertainty gate 仅 +0.3%；recent-replay WM sampling 虽略升 sim reward，却使 err8 超过 1.25×；scheduled sampling p=.05 两 seed 合并也略低于 ramp。因此 scalar gate、recency coupling 和简单 scheduled sampling 暂停，后续 target construction 需要引入更明确的 real-transition / uncertainty target。

### C. 真正的 joint-coupling arm

不再只交换 WM/policy 更新顺序。每轮固定一个 policy update，并将 policy 的 imagined loss 对 WM 参数的梯度限制为 stop-gradient、低秩 adapter 或 trust-region correction 三种可控形式；WM 仍接受 replay teacher-forcing 和 free-running auxiliary。每个 variant 先做 300-step smoke，只有当 WM err8 不爆炸且 sim reward 不低于 matched single-update baseline 才进入 2000-step 正式 run。

## Gate 设计

在线 gate 不使用 raw gradient norm 或单 batch policy loss。候选 gate 必须在独立 replay batch 上聚合至少 8 个 windows，并同时满足：imagined-return delta 的 bootstrap 下界大于 0、WM err8 不超过 matched baseline 的 1.25 倍、以及 real-sim short rollout 不出现新的早期 failure mode。否则只保留为 offline diagnostic。

## 记录规范

每个 run 的 `args.json`、checkpoint iteration、总 optimizer steps、wall-clock、训练 replay split 和 evaluation seeds 必须一起归档。任何比较都必须明确标注是 interaction-matched、compute-matched 还是 wall-clock-matched；不允许用不同预算的 checkpoint 直接宣称算法优越性。

## D. Implicit target coupling (next principled arm)

The action-path experiments show that direct policy gradients into the nominal WM are short-run sensitive, long-run unstable, and non-transferable. The next coupling arm should separate the roles of the models instead of adding another scalar gate:

1. Keep `wm_nominal` replay-only and use it for all real-transition fitting and audit metrics.
2. Maintain a policy-sensitive target model `wm_policy` initialized from `wm_nominal`; only its action-conditioned adapter/output columns receive policy task gradients.
3. Update `wm_policy` with an EMA/implicit step toward the task-gradient proposal, while projecting the proposal onto a held-out replay-fidelity trust region. The trust region is measured on an independent audit split, not on the training minibatch.
4. Train the policy against `wm_policy` only after the audit split satisfies both one-step and 4-step error budgets; otherwise fall back to `wm_nominal` for that iteration.
5. Report nominal WM error, policy-model error, imagined-to-real gap, subject4 return, subject5 return, and policy fallback coverage. A variant is promoted only if it improves both subjects at 600 iterations with `policy_updates=2`.

The current code now has a separate frozen/EMA target wrapper and an opt-in independent replay audit gate (`--policy_target_audit_max_error`, `--policy_target_audit_interval`). The target remains disabled by default and falls back to nominal WM whenever the audit fails. What remains is the matched policy-updates=2 experiment and the policy-sensitive adapter/update itself; this is preferable to further scalar-weight or schedule scans because those variants already fail the 600-step and cross-subject gates.

## G2. Policy-aware data acquisition, not policy-gradient coupling

The current negative results point toward separating policy adaptation from
factual model fitting. The recent policy-aware simulator formulation makes the
same distinction: it casts model learning as a minimax game and uses a critic
error reward to find strategically bad regions, then acquires factual data
there rather than directly weighting the actor gradient ([Dann, Mansour & Mohri,
arXiv:2605.29032](https://arxiv.org/abs/2605.29032)).

For SuperMimic, the next implementation should therefore keep `wm_nominal`
replay-only, train a small error-critic on held-out strategic error, and use its
policy only to choose bounded simulator probes. Probe transitions go into an
isolated buffer and are accepted only when factual one-step/horizon-4 error and
the long-horizon simulator gate both pass. This is materially different from
the already-closed `wm_policy_aware_weight` and `wm_onpolicy_error_weight`
variants, which reweighted existing replay and produced model-exploitation
signals without improving control.

The first 32-env/300-step check is split-sensitive: subject4 improved `.1751→.1788`, while subject5 degraded `.1684→.1549`. At 600 steps on subject4, the target fell to `.1857` versus nominal `.2174`. A corrected horizon-matched subject5 replication of the pessimistic adapter recipe fell from `.04433` to `.03094` at 500 steps (−30.2%), with end survival `0.0078→0` and first failure 11.40 steps earlier. The bounded task-model adapter is therefore closed as a promoted coupling arm; further work must change target construction rather than tune its audit threshold.

## E. Horizon-calibrated active data collection (next arm)

The horizon-4/8 imagined-start gates were functional but did not improve 500-step survival at either 50% or 75% coverage. The next arm should therefore use calibrated uncertainty to **collect corrective real transitions**, rather than delete uncertain policy starts from the actor objective.

1. Use the horizon-specific disagreement thresholds in `docs/ensemble_coverage_s4_horizon.json` to define a per-step risk score.
2. During simulator collection, add a bounded action probe only when the predicted horizon risk is above the train-calibrated threshold; constrain the probe by action norm and a behavior-policy KL/proximity budget.
3. Store probe transitions in a separate buffer. Fit the nominal WM with uniform replay plus a small, isolated probe loss; never replace the nominal replay distribution.
4. Accept the probe update only when held-out one-step and horizon-4/8 error coverage do not regress and the real simulator short rollout has no new tilt-failure mode.
5. Compare against the same-seed no-probe control on interaction-matched and compute-matched axes. Promote only if both subject4 and subject5 show improved 500-step survival, not merely return_norm.

The first implementation uses `--adaptive_probe_onpolicy_only` with the existing isolated `onpolicy_buffer` and validates the probe window before launch. In a matched 300-iteration subject4 run, thresholded uncertainty noise in the probe buffer reduced 500-step return and first-failure time on both subject4 and subject5, with zero end survival throughout. This closes the current scalar-noise probe; any continuation of active collection must change the counterfactual/action target rather than increase probe frequency or noise scale.

This is deliberately different from the failed imagined-start gate: uncertainty determines where to obtain new factual data, while all policy starts remain available to the policy update.

## F. Vector-valued counterfactual supervision

The directional-probe audits show that scalar gain correction is insufficient: gain bias can improve while response-vector cosine and multi-scale Gram geometry worsen. If the intervention branch is revisited, train it with grouped counterfactuals at each state/action scale:

- fit the signed response vectors for `a+δd` and `a−δd`;
- add a paired action-effect loss on `(response_plus - response_minus)`;
- add a normalized Gram loss over at least four directions and δ∈{.1,.5,1.0};
- validate both vector cosine and Gram discrepancy on a held-out simulator split;
- deploy the correction only as a training-time WM auxiliary, never as an unconstrained inference-time residual;
- accept only when multi-step factual error and real short-rollout failure modes do not regress.

The current scalar intervention adapter and directional action probe should remain closed; this vector objective is the minimum redesign needed to test whether action geometry, rather than one-step state RMSE, is the useful concurrent signal.

The online implementation is now available as `--wm_counterfactual_gram_weight`. In the first matched 300-iteration run, adding the Gram term improved subject4 but degraded subject5 with zero end survival on both. Treat this as an instrumentation result only; the next attempt must add a held-out vector-cosine/Gram gate or motion-conditioned weighting before another online budget is spent.

The geometry gate is now implemented in `tools/multiscale_action_gram_audit.py`. It passed the Gram candidate on subject5 at all δ scales, despite the candidate losing in fixed-policy return. This establishes the gate's role as a model-geometry sanity check, not a standalone promotion criterion; every future candidate still requires simulator rollout and failure-mode acceptance.

An unweighted cross-motion follow-up using `counterfactual_train_not4_seed23.pt` slightly improved subject5 (`.03871→.03887`) but substantially reduced subject4 (`.05144→.04431`). Cross-motion supervision therefore needs calibrated weighting or a held-out acceptance rule; simply pooling all counterfactuals is closed.

The 1:1 mixture was also tested with `tools/merge_counterfactual_datasets.py`; it produced `.05101` on subject4 and `.03764` on subject5, including a new tilt-failure rate. Do not spend another budget on fixed mixture ratios. A future motion-conditioned scheme must use an explicit held-out gate and state/motion features to determine weights.

A source-separability check gives a constraint on that future design: `xi` has 5-fold AUC `0.50` for identifying the source collection, while the initial state has AUC `1.00`. Weighting from `xi` alone is therefore unidentifiable here; weighting from raw state is likely a domain shortcut and requires a held-out environment test before implementation.

`tools/counterfactual_source_probe.py` adds normalized response geometry to this audit; its AUC is `0.184` (equivalent to `0.816` after orientation inversion). Response-conditioned weighting is therefore potentially informative but also strongly domain-specific. A valid next experiment must use a third motion split and an explicit source-held-out gate.

The third-split pairwise check is now available: response-geometry AUC is `0.166` for subject4↔subject5 and `0.174` for subject5↔train-not4, while `xi` remains `0.50`. The next experiment should therefore use response geometry only to detect domain coverage and must reject any update that helps one pair while degrading the held-out third split.

The probe now emits pairwise metrics whenever three or more datasets are supplied, so future source-held-out gates should consume these pairwise reports instead of a single multiclass score.

A weak-regularization test (`gram_weight=.02`) improved subject5 return but introduced `18.75%` tilt failures and reduced subject4 return. Do not continue scanning Gram weights without an explicit failure-mode gate; the current `.1` and `.02` points already show the trade-off is not monotonic.

The strict gate is now implemented in `tools/fixed_policy_gate.py`; `fixed_policy_eval.py --json_out` provides clean inputs. Candidate acceptance requires non-degraded return, survival, failure rates, first-failure time, and each failure subtype. The `.02` arm is rejected by the tilt criterion.

The Gram-only arm (`effect_weight=0`, `gram_weight=.1`) also failed: subject5 return improved but tilt failure doubled, while subject4 degraded. This closes the simple decomposition scan; future work must change target construction or add an explicit failure-aware update rule.

Component-level counterfactual audits show no single pose component exploding across these arms; joint velocity remains the dominant one-step error for all of them. The next gate should therefore evaluate multi-step failure behavior directly, rather than adding more position/rotation component weights.

The required seed replication is now complete: the Gram `.1` arm failed on seed1 for both subjects and introduced tilt failures. The subject4 seed0 gain is classified as seed-specific, so no further Gram-only/effect+Gram weight scan is justified without a different target construction.

The one-step-only counterfactual baseline is substantially worse (`.03195/.02776` on subjects4/5, tilt failure `21.875%/25%`) than the seed0 vector arm. This keeps multi-direction supervision as a useful ingredient, but the next design must stabilize its target construction across seeds before online promotion.

The first stabilization attempt, cyclic deterministic sampling (`--wm_counterfactual_deterministic`), improved seed1 but still failed the dual-split gate and added tilt failures. Keep the option, but pair it with an explicit held-out multi-step acceptance rule before considering promotion.

The follow-up full-batch test (`--wm_counterfactual_batch=256`) failed more strongly, with subject5 tilt failure `15.625%`. Do not pursue larger counterfactual batches; stabilization must change the target or acceptance rule, not only sampling coverage.

### F0. First offline result and stop condition

`tools/train_intervention_adapter.py` now implements the grouped effect/Gram objective. On subject4→subject5 counterfactual transfer, ordinary MSE obtained holdout/cross-subject RMSE `0.687/2.810`, while effect+Gram obtained `0.679/3.135`. The small grouped holdout gain did not transfer, so no online integration is justified yet. Any follow-up must first improve cross-subject vector cosine/Gram metrics on a held-out simulator split before consuming an online training budget.

## New diagnostic result: policy-centered counterfactuals

The generator now supports checkpoint policy centers via `--policy_checkpoint`. This removes the previous Gaussian-action mismatch, but the matched 300-step candidate underperformed the random-action counterfactual branch on both held-out motions. Do not promote policy-centered one-step/Gram supervision alone. If revisited, generate counterfactuals along multi-step on-policy state distributions and gate against the fixed-policy return/survival criteria.

## Multi-step on-policy counterfactual result

The generator can now roll a checkpoint policy for a configurable burn-in and replicate the resulting physical state before applying symmetric action perturbations. Burn-in-4 effect+Gram supervision catastrophically failed the fixed-policy gate on both held-out motions. Do not continue increasing counterfactual coverage or Gram/effect weights. The next useful work is diagnostic: quantify whether local Jacobian targets conflict with multi-step value gradients, then design a consistency-preserving or auxiliary-only objective.

## Gradient/sensitivity audit update

Repeated gradient geometry does not show a stable negative task/replay cosine, so gradient projection alone is unlikely to solve the counterfactual branch. The stronger signal is reduced cumulative WM action sensitivity alongside worse fixed-policy return. Any next coupling objective should include an explicit sensitivity-collapse guard or Jacobian preservation diagnostic, while keeping the counterfactual loss auxiliary-only until that guard is validated.

## Sensitivity-preservation result

`--wm_counterfactual_norm_weight` was added as an anti-collapse term. Raw weight `.1` caused NaNs; clamped SmoothL1 at `.01` was stable but did not restore action sensitivity (100-step audit mean `0.93`). Do not increase this weight. The next candidate should compare against a frozen reference WM with a bounded relative-sensitivity ratio and an explicit rollback gate.

## Relative sensitivity gate

`tools/relative_sensitivity_audit.py` now compares candidates against a frozen reference on the exact same replay starts, policy actions, and probes. The policy-centered, burn-in-4, and norm-preservation candidates all failed the bounded relative sensitivity gate. Use this audit as a promotion prerequisite before any future WM/policy coupling run.

## Frozen-reference relative sensitivity result

The relative sensitivity loss is implemented, but a 100-step test at weight .01 still collapsed to ratio 0.429. The next experiment should control the counterfactual branch update scale (separate optimizer or strict per-step gradient cap) before testing the loss again; increasing the coefficient alone is not justified.

## Gate calibration correction

Sensitivity comparisons across training stages are invalid. The 300-step reference made 100-step candidates appear to collapse, while a matched 100-step no-counterfactual baseline yielded candidate ratios around .88. Use same-iteration, same-seed baselines and separate mean/tail thresholds before attributing failure to over-smoothing.

## Matched-stage control result

The low-weight counterfactual branch did not improve fixed-policy control at 100 steps: subject4 return dropped .03064 -> .02562 and subject5 .02880 -> .02756, with earlier failures. A mean sensitivity ratio near .88 is therefore insufficient for promotion; use tail-aware sensitivity plus fixed-policy gates.

## Lower-tail hinge result

A one-sided relative sensitivity floor (.8) worsened the matched-stage mean ratio to .699. Do not continue hinge/floor coefficient tuning. Any future tail protection must be implemented as a separate, tightly capped gradient branch rather than another scalar loss term.

## One-step-only ablation

Removing effect and Gram terms did not rescue direct counterfactual supervision. Even one-step simulator loss at .001 degraded both held-out fixed-policy returns at 100 steps. Close direct counterfactual injection for now; future use should be offline diagnostic or a separately optimized auxiliary model.

## Policy-side adapter result

The frozen intervention adapter is not safe as a direct imagined-transition correction: it produced infinite critic values and zero imagined survival at blend .1. Any future adapter use requires an explicit bounded residual, norm clamp, and validity gate before policy gradients are allowed through it.

## Bounded adapter result

A norm-ratio cap makes the intervention adapter numerically finite, but held-out control remains mixed and substantially worse on subject5. Adapter correction needs a state-validity gate and held-out calibration before any further use.

## Adapter validity gate result

A norm-ratio validity gate (1.2) prevented blow-up but degraded both held-out motions. Do not continue threshold tuning for direct imagined adapter correction; keep the adapter offline-only unless a learned state-validity classifier is added.

## Validity classifier transfer result

The existing residual validity classifier transfers only weakly: held-out selected states improve 62.2% of the time. It cannot be reused for the 170-D intervention adapter. Do not add an adapter gate without collecting adapter-specific labels and a separate held-out audit.

## Adapter domain-shift audit

The s4 adapter improves most one-step perturbations on s4 and s5, but worsens the broad train_not4 distribution on average. Future adapter work should train a source/domain validity model from these paired labels and evaluate held-out source transfer before imagined rollout use.

## Adapter-specific validity classifier result

The new audit tool shows no transferable validity signal from s4-trained adapter labels (AUC .529 on s5, .401 on train_not4). Do not add a classifier gate trained on one source domain; the next meaningful step would require multi-domain adapter training data and a held-out domain split.

## Multi-domain adapter result

A mixed s4/s5/train_not4 adapter improves offline fit but still degrades subject5 when used in policy imagination. Better local supervised RMSE is not sufficient; direct adapter correction remains closed for concurrent control.

## Policy-target adapter result

The matched policy-only action-conditioned target adapter improved neither dual-split control nor subject5 robustness. It remains closed under the current state/action input; further work would require explicit domain/context conditioning and a new held-out design.

## Policy-target audit mechanism

Reducing adapter learning rate and norm cap did not keep the target valid; audit errors remained far above threshold and the target was disabled. Further scalar LR/cap tuning is closed. Any redesign needs a different target construction or audit target, not smaller steps.

## EMA target isolation result

Removing the trainable target adapter did not rescue policy-target coupling. EMA target-only policy updates degraded both held-out motions at 100 steps. Close the current target-wrapper branch; a future target must change the policy objective or target horizon, not just adapter capacity.

## Full EMA correction result

After correcting the audit-threshold confound, all-audit-enabled EMA target coupling showed a transient 100-step gain but failed at 300 steps on both motions. Do not promote based on short smoke gains; future target coupling must include long-budget stability or scheduled fallback.

## Scheduled target fallback

`--policy_target_until_iter` enables a target only during an early phase, then returns to nominal WM. Seed0 at 300 steps improved both subjects, while seed1 improved subject4 but degraded subject5. Treat this as a promising but seed-sensitive lead; next work should replace the fixed cutoff with a held-out policy-value/fidelity fallback and replicate before promotion.

## Dynamic audit result

A strict relative replay-fidelity audit failed to detect policy-target degradation: all audits passed while control collapsed. Do not use replay error alone as dynamic fallback. Future fallback must include an independent policy-value or short real-sim signal.

## Cutoff-length result

Seed1 cutoff 50 improved subject4 but still slightly degraded subject5. Fixed cutoff tuning is therefore insufficient; a future schedule must use a subject/domain-aware real-sim or policy-value signal.

## Seed0 cutoff-50 replication

Seed0 at 300 iterations with `policy_target_until_iter=50` produced subject4/5 returns `.03386/.02995`, compared with matched baseline300 `.02376/.01419`; first-failure means were `51.59/46.41`. Together with seed1, this supports the early target window as the leading candidate, but not as a default: survival at 500 evaluation steps remains zero and the schedule has only two seeds. The next decisive experiment is a third-seed replication or a fallback signal tied to short real-simulation/policy value, not more cutoff-length scanning.

## Seed2 matched-baseline correction

The third-seed matched comparison rejects the fixed cutoff as a robust recipe. Candidate returns were `.03174/.02549` versus nominal `.03836/.03096` on subject4/5. Across seeds, the schedule is therefore seed-sensitive rather than a reproducible dual-split improvement. Do not promote `policy_target_until_iter=50`; replace the fixed schedule with a fallback based on policy value or a short real-simulation probe.

## Training-signal audit

The exported three-seed signal tables show that `sim_rew`, `imag_rew`, `imag_survival`, and `ep_ret_norm` cannot serve as standalone fallback criteria. Seed2 is the clearest counterexample: every listed training signal is comparable to or better than the nominal run while both fixed-policy test returns are worse. The next experiment should checkpoint at short intervals and run a small fixed-start simulator probe on the held-out motion split; the probe result, rather than replay metrics, should decide whether to disable the target.

## Checkpoint-probe result

The first implementation of this protocol is complete: `--ckpt_every 50` plus `fixed_policy_eval.py` evaluates all checkpoints on shared starts. Seed3 target returns oscillated from a mean `.0345` at iter100 to `.0182` at iter200 and back to `.0367` at iter250. Use repeated batches or a lower-cost confidence interval; do not gate on a single 32-env probe.

## Repeated-probe confidence result

Two batch128 start seeds confirm the signal: pooled means were `.03449` at iter100, `.01872` at iter200, and `.03547` at iter250. The iter200 drop is much larger than the observed `.0059–.0078` pooled standard deviation. A deployable fallback should run two short probes, compare against the last accepted checkpoint, and disable target usage only when the lower confidence bound crosses a fixed degradation margin.

## Executable probe gate

`tools/policy_probe_gate.py` now implements that rule. It takes paired probe JSONs, computes per-probe relative return changes, and applies a normal lower confidence bound. On the matched seed2 target versus baseline, it returned `passed=false` with mean relative change `-.1398` and lower bound `-.1400` under a 5% margin. The next integration step is to invoke this gate at checkpoint boundaries and switch policy imagination back to nominal WM when it rejects.

On the seed0 paired probes the same command returned `passed=true`: subject4/5 changes were `+.464/+ .785`, with lower bound `+.310`. The gate therefore has both a real rejection and a real acceptance case; integration can proceed without changing the decision rule.

## External control integration (completed)

Training accepts `--policy_target_control_file`. A probe process can write
`disable` to this file after a rejected paired probe, and the next imagined
rollout uses nominal WM. The complete orchestration is now available through
`run_probe_gate.py` and `watch_probe_gate.py`: checkpoints are paired by
iteration, probes can use repeated `--steps` horizons, and the optional
`--max-first-failure-drop` guard checks early collapse. A rejected shadow
candidate writes `shadow_discard` while leaving the live branch untouched; an
accepted one can write `shadow_promote`. Container-backed end-to-end replays
and the regression suite verify these transitions.

The gate can now perform the last step itself with `--control-file`, using an atomic replacement to avoid partial reads. The remaining experiment is to connect checkpoint saving, fixed-start probing, and this command in one periodic runner.

`tools/run_probe_gate.py` now provides that orchestration: it loops over requested splits and seeds, keeps reference/candidate starts paired, and writes the gate decision. A real smoke run produced `enable` for the seed0 subject4 pair, confirming the full path from simulator probe to control token.

`tools/watch_probe_gate.py` now adds the periodic layer: monitor a training log directory, probe each newly published checkpoint, and stop on rejection if requested. The next end-to-end experiment can launch this watcher alongside a training run with `--ckpt_every` set to the probe cadence.

The watcher smoke test is complete: it detected the iter250 degradation and wrote `disable` after processing earlier checkpoints. The next research run can use this exact orchestration with two motion splits and two seeds; the remaining design choice is probe cadence versus simulator cost.

The first live training run with the control file completed without interruption. Seed4 checkpoints 50/100/150/200 all passed a short subject5 probe and left the control at `enable`. The next decisive run should use both subject4 and subject5 plus two start seeds, then compare the final policy against a same-seed nominal baseline.

That decisive audit is now available: 500-step, two-split, two-seed evaluation rejected the final online target with mean relative change `-.335` and lower confidence bound `-.349`. The short 100-step watcher had falsely kept it enabled. Do not deploy the short probe configuration; the next online run must use 500-step probes or demonstrate a calibrated multi-horizon early warning rule.

The tooling now enforces this operationally: all probe runners default to and require a 500-step horizon through `--min-steps`. Any cheaper short probe must be explicitly redesigned and calibrated before it can override this safety floor.

The enforced long-horizon watcher has now triggered in a live run: seed5 checkpoint 50 fell `6.2%`, so the control changed to `disable` and training continued to iteration 200. This is the current reference implementation for safe online coupling; future performance comparisons should use same-seed nominal baselines and the final post-disable policy.

The same-seed final audit rejects the post-disable policy strongly (`-.373` mean relative change, lower bound `-.404`). This exposes the remaining limitation: a disable-only fallback is not a rollback. The next experiment should save the last accepted policy checkpoint and, on rejection, restore policy/critic state (or run target updates in a shadow branch) before continuing nominal training.

The rollback path is now implemented and smoke-tested. `policy_probe_gate.py --rollback-checkpoint` emits the rollback token on rejection; the trainer handles it outside compiled imagination and restores policy/critic/optimizer state before continuing with nominal WM. The next decisive experiment is to run this rollback mode online and compare its final policy against same-seed nominal training.

Because rollback cannot undo data-dependent optimizer history perfectly, the next architectural direction is specified in `docs/shadow_policy_experiment_spec.md`: train target coupling only in a shadow policy/critic branch and promote it only after the 500-step paired gate passes.

The watcher rollback source selection was corrected and verified in command traces: each candidate now receives the preceding checkpoint as its rollback source. A full rejection-triggered live run remains the next validation target.

That rejection-triggered path is now verified: the controlled watcher test accepted ckpt50, rejected ckpt100 by `32.3%`, and wrote the preceding ckpt50 as the rollback source. The remaining research question is performance under a full live training run with this rollback token, rather than control-path correctness.

Parameter-distance audit confirms the restore: the post-rollback policy is `0.52%` relative distance from the accepted source and `24.21%` from the rejected source. The next live experiment should measure whether this state restoration translates into recovered fixed-policy control return.
