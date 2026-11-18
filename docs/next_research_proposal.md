# 下一阶段：policy-aware transition game

## 证据边界

当前实验已经排除几类直接路线：让 policy-return 梯度改写 nominal WM、固定/EMA target WM、单步 strategic-error weighting、随机 counterfactual selector、WM-error critic 和未经校正的 action-gradient。它们要么在 500/600-step rollout 退化，要么只在 train split 短期有效。

真正尚未完成的是把 policy footprint 建模成 transition occupancy，而不是用 action-distance 或单步 WM error 近似。仓库已有 transition classifier odds 和 dual-Q audit，但还没有形成一个完整的 policy-aware minimax update。

## 算法草案

维护两个 world model：

- `wm_nominal`：只用 factual replay 更新，作为 deployment reference。
- `wm_task`：接收一个零初始化、范数受限的低秩 action-conditioned adapter，只用于 policy imagination。

从 replay transition `(s,a,s')` 构造 policy-occupancy discriminator，输入必须包含 `(s,a,s')`，并隐藏绝对 frame/age。用 discriminator odds 得到 transition weight，但限制在 `[w_min,w_max]`，并在独立 validation replay 上校准温度。

每次 task-model 更新分三步：

1. 用 occupancy odds 对 factual transition 做加权回归，保持 nominal fidelity。
2. 用 policy imagination 的 pessimistic return 更新 adapter：
   `min(J_task, stopgrad(J_nominal))`，只允许 task model 在 nominal 较差的方向上提供梯度。
3. 在独立 replay 上执行 horizon-1 和 horizon-4 audit；任一误差比超过预算就拒绝 adapter step，并回滚到 last-good state。

policy 本身只通过 `wm_task` imagination 更新；`wm_nominal` 不接收 policy-return 梯度。

## 最小实现顺序

1. ✅ 已将 `transition_occupancy_audit.py --save_model` 的 classifier 输入、温度和 odds clipping 封装为 `supermimic.transition_weight.TransitionOdds`，并加入维度、温度、detach、clipping 单元测试。
2. ✅ `PolicyTargetWorldModel` 已增加 task adapter 的 snapshot/restore rollback token，并由训练循环统一调用；独立 optimizer 已存在，下一步补 accepted/rejected step 记录。
3. ✅ 已保留 pessimistic return，并将独立 replay audit 的 horizon-1/horizon-4 task 与 nominal 误差写入 checkpoint；accepted/rejected 计数和 rollback 状态也随 checkpoint 保存。
   8-env、3-iteration smoke 已验证真实训练链路：audit 接受 `3` 次、拒绝 `0` 次；末次 task/nominal 误差分别为 h1 `0.3992/0.3976`、h4 `2.7164/2.6970`，adapter norm `1.1689`，CSV 与 `last.pt` 均成功写入。
   追加 50-iteration、阈值 `0.8` 的压力 smoke：audit 拒绝 `50/50` 次，adapter norm 保持 `1.1688`，证明拒绝路径确实回滚；末次 task/nominal h1 为 `2.4326/2.4325`、h4 为 `31.7151/31.7161`。该阈值会关闭 task adapter，后续正式实验需要用独立校准集设定审计预算，不能直接沿用 `0.8`。
   已加入独立的 `--policy_target_audit_h1_max_error` 与 `--policy_target_audit_h4_max_error`。在 h1=`3`、h4=`40`、legacy max-error=`0` 的 10-iteration smoke 中，audit 接受 `10/10` 次，末次 task/nominal h1 为 `0.3985/0.3955`、h4 为 `4.4306/4.3587`，证明拆分预算可在不关闭 adapter 的情况下工作。
   100-iteration 延长 smoke 显示审计会动态收紧：接受 `35` 次、拒绝 `65` 次，adapter norm 在约 `1.174` 后冻结；末次 task/nominal h1 为 `10.8425/10.9446`、h4 为 `4863.7/4864.4`。这验证了 rollback/拒绝机制能在模型漂移后自动停用 task 更新，但 simulator reward 同期由 `0.444` 降至 `0.340`，没有 baseline 对照，不能宣称收益。
   同配置 nominal baseline（seed 默认相同、无 adapter）100 iterations 的 simulator reward 为 `0.417 → 0.379`，而 task-adapter 组为 `0.444 → 0.340`。短期起点略高但末期更低，当前证据不支持 promotion；下一步应优先调查 adapter 在早期被接受后造成的分布漂移。
   为此新增 `--policy_target_adapter_l2` 零初始化锚定项；10-iteration smoke（系数 `0.01`）仍接受 `10/10` 次，但 adapter norm 从无锚定的 `1.1700` 降至 `1.1334`，训练链路正常。该正则尚未经过长程 baseline 对照，暂不宣称改进。
   同配置 100-iteration matched run（系数 `0.01`）的 simulator reward 为 `0.444 → 0.421`，相较无 L2 的 `0.444 → 0.340` 和 nominal baseline 的 `0.417 → 0.379`，末期表现更好；audit 接受 `36` 次、拒绝 `64` 次，adapter norm `1.0357`。这是当前最有希望的信号，但仍需第二个 seed 和 fixed-policy clean/perturb 评估。
   第二个 seed 的 matched 对照否定了跨 seed 稳定性：L2 组末期 simulator reward `0.339`，同 seed nominal baseline 为 `0.376`；L2 组 audit 接受 `41` 次、拒绝 `59` 次，adapter norm `0.9926`。因此 L2 只能作为稳定化工具，不能进入 promotion gate。
   新增 `tools/analyze_policy_target_seeds.py` 做日志对齐诊断。两 seed 的 audit h1 与 reward 相关性分别为 `-0.03/-0.18`，h4 相关性为 `+0.41/-0.64`，方向相反；当前 factual audit 指标不能作为跨 seed 的收益代理，后续必须依赖 fixed-policy rollout。
   seed0 的 fixed-policy 50-step 配对评估也没有发现 promotion 信号：clean return nominal/no-L2/L2 为 `0.2850/0.2872/0.2848`，perturb 为 `0.2857/0.2883/0.2897`；三组 survival 均为 `0`、height failure 均为 `1`。差异小于当前失败模式的波动量级，L2 不进入 promotion。
   针对 height failure 加入已有的 `policy_height_penalty=0.1` 后，训练末 simulator reward 为 `0.351`；fixed-policy clean return 为 `0.2929`，略高于 nominal/L2 的 `0.2850/0.2848`，但 survival 仍为 `0`、height failure 仍为 `1`、first-failure step 仍约 `33.5`。因此它改善了短期 tracking return，却没有解决真正的跌倒终点。
   对 L2 seed0 checkpoint 做 teacher-forcing audit（window=4）发现显著的 free-running compounding error：teacher-forced final-step WM error `0.482`，free-running final-step error `5.384`，all-step `2.848`。这比 height penalty 更直接地指向下一条路线：增加 multi-step factual consistency / longer-window training，再评估 first-failure，而不是继续调 policy return 权重。
   首次 window=8 smoke 没有验证这一假设：teacher-forced final-step error 降到 `0.338`，但 free-running final-step 仍为 `5.847`、all-step `2.531`；100-iteration simulator reward 末值为 `0.311`，低于 window=4 L2 组的 `0.421`。单纯拉长窗口会降低 teacher-forced 误差，却没有改善自由滚动或真实回报，暂不继续扩大该方向。
   打开已有的 `--wm_teacher_forcing --wm_consistency_weight 0.1` 后，free-running final-step error 从 `5.384` 降到 `0.956`，all-step 从 `2.848` 降到 `0.516`，说明显式 free-running consistency 确实修复了模型滚动误差；但 fixed-policy clean return 为 `0.2839`、first-failure `33.1`，略差于 L2 的 `0.2848/33.8`，survival 仍为 `0`。因此“模型更准”尚未转化为真实控制收益，下一步要校准 consistency loss 对 policy update 的影响。
   consistency-only 消融（关闭 policy-target adapter）fixed-policy clean return 为 `0.2891`、first-failure `32.6`，略高于带 adapter 的 `0.2839/33.1`，但 survival 仍为 `0`。这进一步说明当前 adapter 会引入额外偏差；后续应先保留 consistency-only 作为 WM 基线，再单独研究 policy 更新与真实 failure recovery。
   第二个 seed 的 consistency-only 结果给出较一致的方向：fixed-policy clean return `0.2888`、first-failure `33.9`，相对同 seed nominal baseline 的 `0.2618/32.4` 有改善，但 survival 仍为 `0`、height failure 仍为 `1`。因此 consistency-only 是目前唯一在两个 seed 都改善短期 fixed-policy 指标的候选，但尚未解决终止失败。
   在 seed0 consistency-only 上加入 `--wm_termination_weight 0.1` 后，fixed-policy clean 的 return `0.3090`、survival `0.125`、failure `0.875`、first-failure `34.25`；perturb return `0.3118`、survival `0.125`、failure `0.875`、first-failure `34.375`。相同评估下 consistency-only clean/perturb survival 都为 `0`。这是目前首个同时改善 clean 与 perturb survival 的候选，但只验证了一个 seed，尚不能 promotion。
   seed1 复现没有通过：termination-loss 组 clean/perturb survival 均为 `0`，failure `1.0`，虽然 height failure 降为 `0.875`，但出现 `0.125` tilt failure，且 return 只有 `0.2674/0.2653`，低于 consistency-only 的 `0.2888/0.2884`。因此 termination loss 的 seed0 survival 改善是 split-dependent，暂不进入 promotion。
   新的 schedule-sampling smoke（`wm_schedule_sampling_prob=0.5`、consistency=`0.1`）在 seed0 同时改善 clean/perturb：clean return `0.2882`、survival `0.125`；perturb return `0.2912`、survival `0.125`，而 consistency-only 两者 survival 都为 `0`。free-running final-step error 为 `4.339`，没有 termination-loss 那么低，但真实 survival 信号更好；需要 seed1 复现后再决定是否升级为主线。
   seed1 复现成功：clean return `0.3031`、survival `0.125`、first-failure `34.375`；perturb return `0.3101`、survival `0.125`、first-failure `34.5`，相对 consistency-only 的 clean/perturb return `0.2888/0.2884`、survival `0/0` 均改善。schedule sampling 是目前首个跨两个 seed、同时通过 clean/perturb survival 的候选，下一步进入 300-iteration matched promotion 前仍需扩大 batch/评估样本并保留 nominal baseline。
   300-iteration 延长实验给出分化结果：seed0 batch16 clean/perturb survival 都从 `0.1875`（100 iter）升到 `0.25`，first-failure `38.6`；但 seed1 survival 仍为 `0.1875`，return 从约 `0.33` 降至 `0.295/0.295`，first-failure 从 `37.1` 降至 `35.6`。因此 schedule sampling 的 100-step 信号尚未形成稳定的 300-step promotion，需保留为候选并继续调研训练稳定性。
   将 cosine policy 学习率在 100 iter 后降到 3%（`lr_decay_iters=100`）只改善 seed1：seed1 clean survival `0.1875→0.25`、return `0.295→0.324`，perturb survival `0.1875`、return `0.295→0.325`；seed0 则 clean/perturb survival `0.25→0.1875`、return `0.337→0.320/0.322`。学习率衰减同样是 split-dependent，不能作为统一修复。
   对两个 300-step checkpoint 做 policy-support audit 后发现明显的 WM 分布差异：seed0 one-step WM error 均值/P90 为 `0.657/1.269`，seed1 为 `0.270/0.608`；action-distance 与 WM error 相关性仅 `-0.105/0.058`。这说明 seed 差异主要来自 replay/world-model 误差分布，而不是简单的 policy action support，下一步应按 WM error/failure 状态做分层评估。
   作为直接干预，seed0 加入 prioritized replay（alpha=`0.6`）后，fixed-policy clean/perturb survival 从 schedule baseline 的 `0.125/0.125` 降为 `0/0`，return 从约 `0.288/0.291` 降至 `0.280/0.280`。单纯追踪高 WM-error 窗口会破坏策略行为，暂不采用。
   `fixed_policy_eval.py` 现已额外输出首次 failure 的 height/tilt 误差、速度和 motion frame，支持按 failure 状态分层；初步 batch8 评估显示 seed0/seed1 的 failure height error `0.269/0.265`、speed `2.34/2.26`，差异很小，后续需要扩大 starts 并结合 frame/clip 聚合，而不是凭单次均值判断。
   batch64 配对评估确认长程差异是真实的：seed0 clean/perturb survival `0.2656/0.2813`，seed1 `0.1406/0.1406`；seed1 的 tilt failure `0.0625/0.0469` 高于 seed0 的 `0.0156/0.0156`，且 first-failure 约早 3.2 步。height error 仍接近，说明下一步应针对 tilt failure 的状态/动作分层，而非继续调整 height 目标。
   evaluator 进一步记录首次 failure 的 angular speed/action norm：seed0 为 `3.04/3.36`，seed1 为 `3.49/2.74`。seed1 的 tilt failure 与更高角速度相关，而不是更大的 action norm，下一步应围绕 angular-velocity/tilt dynamics 做局部 consistency 或风险约束。
   直接加入 `policy_tilt_penalty=0.1` 的 seed1 smoke 反而恶化：clean/perturb survival `0.125/0.125`，return `0.256/0.260`，tilt failure `0.3125/0.3125`；首次 failure action norm 从 baseline `1.77` 飙到约 `6.99`。因此全局 tilt penalty 会诱发更激烈动作，不能作为局部 angular-risk 修复。
   新增 `policy_angular_velocity_penalty`，直接惩罚 imagined root angular velocity，避免把姿态误差放大成全局 tilt 梯度。seed1、100-iteration、系数 `0.001` 的 smoke 在 fixed-policy 300-step clean/perturb 上 survival 均为 `0`、return `0.0758/0.0768`、first-failure `51.1/52.5`；首次 failure angular speed `2.66/2.67`、action norm `3.01/3.02`。虽然 failure 延后，但 return 和最终 survival 明显恶化，说明未经归一化的角速度 penalty 改变了动作能量分配，暂不采用；后续若继续，只应做状态相关的短时风险门控并重新校准尺度。
   进一步加入 tilt-error gate（margin=`0.5`）后仍出现策略塌缩：系数 `0.01` 的 seed1 smoke fixed-policy clean/perturb return `0.0839/0.0840`、survival `0/0`，首次 failure `51.6/52.5`，failure action norm 仅约 `0.60`。门控延后了失败但没有保住步态，说明当前 imagined angular-velocity 梯度的尺度或方向本身不适合直接更新 policy；该路线关闭，转向只用于数据筛选/WM consistency 的风险信号。
   将角速度信号改为 factual WM replay 权重（`wm_angular_velocity_weight=0.5`）的早期 batch16 结果不能与历史 batch64 直接比较。当前 evaluator 的 batch64 匹配复核中，schedule baseline clean/perturb 为 `0.0691/0.0694`，schedule×action-recovery 为 `0.0708/0.0712`，两者 survival 都为 `0`；因此高角速度重采样尚无可靠收益，当前风险信号暂不进入训练权重，保留为 evaluator 分层指标。
4. 先做 50-iteration 8-env smoke，确认 adapter、odds、rollback 和 checkpoint 生命周期。
5. 再做 matched 300-iteration subject4/subject5 双 seed；只有在两个 split 的 500-step clean/perturb 都不退化后，才进入 600-iteration promotion gate。

## 必须记录的指标

- nominal/task imagined return
- factual WM error at horizons 1 and 4
- discriminator AUROC、温度、odds 的 p50/p95
- adapter norm、accepted/rejected ratio、fallback coverage
- interaction steps、policy/WM optimizer steps、wall-clock
- fixed-policy 500-step clean/perturb return、first failure、height/tilt failure、end survival

任何只提高 imagined return、但没有提高真实 fixed-policy rollout 的结果，都归类为 model exploitation，不进入主线。

### 300-step paired cross-seed checkpoint audit (2026-11-07)

Using the same 64 fixed starts for every checkpoint, the 300-step schedule checkpoints show seed-dependent behavior: seed0 clean/perturb return `0.06948/0.06866`, first failure `49.7/49.1`, while seed1 is `0.05821/0.05857`, first failure `44.0/44.3`. End survival is zero in all four evaluations; height failures are `93.8–96.9%` and tilt failures `3.1–9.4%`. This confirms that the long-horizon limit is still height collapse rather than a recoverable tilt-only mode. Schedule sampling stays in the candidate set for diagnostics, but does not pass a 300-step promotion gate. The next useful experiments should target state-conditioned height recovery or failure-aware data collection, with the paired-start evaluator retained as the acceptance test.

### Failure-focused policy starts (new candidate)

The current long-horizon failure is dominated by height collapse. I added an opt-in `--policy_failure_start_fraction` sampler: a fraction of imagined policy starts is selected from replay states with the largest normalized root-height/root-tilt termination proximity, while WM fitting and simulator collection remain unchanged. In a seed1 100-iteration smoke at fraction `0.25`, paired 64-start clean/perturb return improved `0.06908/0.06942 → 0.07298/0.07413`, first failure moved `47.1/47.3 → 49.7/50.8`, and tilt failure decreased `4.7% → 3.1%`; end survival stayed zero and height failure slightly increased. A matched 300-iteration run is required before interpreting this as recovery rather than short-horizon reshaping.

The 300-iteration mixed boundary sampler reproduced a modest return gain and reduced height failures, but traded them for more tilt failures. A height-only ranking ablation was neutral at 100 iterations, so the improvement is not explained by root-z exposure alone. The next refinement should use a staged or balanced sampler that allocates separate quotas to height-near and tilt-near states, rather than ranking their maximum in one pool.

The balanced height/tilt quota ablation was neutral at 100 iterations (return `0.06894/0.06901`, survival `0`), so simply splitting the boundary states does not recover the mixed sampler's short-term gain. A useful next step is to condition the quota on observed failure type or use a curriculum, rather than static replay ranking.

A gradual mixed-risk curriculum (`policy_failure_start_ramp_iters=100`) was also tested. It gave only a small return change (`0.07029/0.06982` versus `0.06908/0.06942`) and increased height failure to `98.4%`; survival stayed zero. Static or linearly ramped replay boundary exposure is insufficient, so further work should instrument actual terminal cause labels before adapting the curriculum.

### Failure-cause instrumentation

Replay transitions now carry separate `term_height` and `term_tilt` labels derived from the actual next simulator state. The fields are backward-compatible with old checkpoints, and `tools/analyze_replay_causes.py` summarizes them over the valid circular-buffer region. A 5-iteration Isaac smoke verified collection, checkpoint save/load, and the diagnostic path. The next experiment can therefore adapt failure-focused sampling from observed terminal causes instead of relying only on fixed-start evaluation aggregates.

The first instrumented 100-iteration baseline contained 22 terminal transitions, all height terminations and zero tilt terminations. This explains why cause-aware replay counts alone cannot learn a tilt quota: tilt failures appear mainly under long-horizon fixed evaluation, outside the short training buffer's terminal events. The next useful intervention is to retain nonterminal states ranked by tilt proximity (or inject a controlled tilt perturbation) while using the new labels to avoid oversampling already-dominant height failures.

A pure tilt-near nonterminal sampler (`policy_failure_start_tilt_only`, fraction `.25`) gave a seed1 100-iteration clean/perturb return of `0.07112/0.07211`, delayed failure to `48.8/49.5`, and lowered tilt failure to `3.1%/3.1%`, but height failure rose to `96.9%/96.9%` and survival stayed zero. The tilt arm is therefore a real learnable subproblem; the remaining challenge is a controller that alternates height recovery and tilt recovery without sacrificing either mode.

A 10-iteration alternating sampler between height-near and tilt-near starts improved short-horizon return and nearly removed tilt failures (`1.6%`), but height failure rose to `98.4%` and survival stayed zero. The two recovery modes compete for policy capacity; future work should use separate recovery heads or a failure-state gate rather than one shared policy objective.

The alternating sampler result strengthens the case for a gated multi-head recovery policy: height and tilt objectives each improve their own failure mode but compete in a shared policy. Future implementation should retain a nominal head and add separate height-recovery and tilt-recovery heads, with a detached gate from normalized height/tilt proximity; all heads should be evaluated on the same fixed starts and perturbations.

An inference-level gate prototype (`fixed_policy_eval.py --gate_mode max_error`) selected between height-focused and tilt-focused checkpoints using the larger current normalized error. It underperformed both specialized policies (clean/perturb return `0.06827/0.06852`, survival `0`), showing that instantaneous error magnitude is not a reliable mode selector. A future multi-head design needs a learned risk gate with temporal context or a predictive failure classifier, rather than a raw max-error rule.

A temporal gate based on one-step error growth (`gate_mode=error_trend`) was tested between height-focused and tilt-focused policies. It remained below the specialized policies (return `0.06825/0.06878`, survival `0`), so a useful gate needs longer history and a predictive failure label, rather than a hand-designed instantaneous or one-step heuristic.

### Offline predictive-gate audit

The evaluator now exports per-step traces, and `tools/train_recovery_gate.py` trains a causal selector from paired specialized-policy rollouts. On 64 seed1 starts over 300 steps, a small gate using height/tilt error, speed, and angular speed reached validation accuracy `0.5603`, exactly the majority baseline `0.5605`. This closes the simple state-feature gate; the next candidate should use a temporal encoder or imagined WM rollouts to predict future failure mode and head value.

An 8-step temporal gate trained from paired specialized-policy traces did not recover a selector signal (accuracy `0.5578` vs majority `0.5605`). This closes the simple kinematic-history gate. The remaining predictive-gate route should compare counterfactual WM rollouts for the two candidate actions/policies, because actual policy-trace state features are too weak and confounded by diverging trajectories.

### Counterfactual WM gate audit

`tools/policy_counterfactual_wm_gate.py` now rolls both specialized policies through one frozen WM from identical replay starts. The height-trained WM predicted the tilt head worse (win fraction `0.340`), while the tilt-trained WM predicted it better (`0.711`); real simulator evaluation slightly favored tilt. This confirms that counterfactual gating cannot trust one WM. The next design should aggregate multiple WMs with a pessimistic/uncertainty-aware criterion and validate ranking calibration before using it to select actions.

The two-WM pessimistic gate audit gave no usable selection signal. Each WM preferred a different policy, while the per-start minimum aggregation nearly tied and slightly preferred the height head (tilt win fraction `0.479`, margin `-9.99e-5`). Pessimism avoids trusting one model but collapses the ranking; future work needs calibrated model disagreement or a jointly trained policy target, not a raw minimum.

A small disagreement calibration sweep (`mean - λ·std`, λ=`0, .25, .5, 1, 2`) moved the tilt-head win fraction only from `0.469` to `0.521`; the largest margin was `3.33e-4`. This is a weak ranking shift, not a calibrated gate. The next required step is to validate λ on held-out starts against real simulator outcomes before integrating any uncertainty score into policy updates.

Held-out real-simulator validation on 64 new seed-202 starts preserved the tilt-head advantage (clean `0.06807 vs 0.06636`, perturb `0.06693 vs 0.06571` for tilt versus height), with both policies sharing `98.4%` height failure and `1.6%` survival. This gives a held-out target for calibrating WM counterfactual ranking, but the advantage is small and the dominant height failure remains unresolved.

### Cross-seed failure-focused sampler rejection

The mixed boundary sampler failed its seed0 300-iteration replication. Relative to the matched seed0 schedule baseline, clean/perturb return fell `0.06948/0.06866 → 0.06702/0.06649`, first failure moved earlier by about 4 steps, and height failure reached `100%`; survival was zero. The earlier seed1 height-failure improvement is split-dependent, so this sampler is rejected for promotion. Future recovery work must explicitly model split/seed state distribution rather than rely on replay boundary ranking.

The policy-support audit provides a mechanism for the seed split: failure-focused starts nearly doubled policy/replay action distance (`~.006 → .0115`). Seed0 also has much larger baseline one-step WM error (`.594` vs `.270` for seed1), so the same support shift is more dangerous there. The next recovery experiment should constrain recovery actions to replay support (proximity or trust-region objective) while preserving boundary-state exposure; otherwise lower factual WM error can coexist with worse control.

A support-constrained recovery smoke (boundary starts `.25` plus `policy_bc_weight=.5`) reduced height failures on seed0 but lowered return, accelerated first failure, and increased tilt failures. This confirms that a simple replay-action anchor suppresses off-support motion without learning the needed recovery action. Future trust-region designs need a state-dependent recovery allowance or a predictive action constraint rather than global behavior cloning.

The state-gated replay-action anchor (`bc_weight=.5`, risk margin `.5`) gave a small seed0 improvement but caused a large seed1 tilt-failure increase (`18.8–20.3%`). This confirms that even a risk-local trust region can distort the failure-mode balance across seeds. It remains a diagnostic clue for support constraints, not a candidate recipe.

A state-conditioned tilt penalty was tested as a failure-mode objective. It looked promising on seed1 but failed on seed0, with return dropping by about `7%` and first failure moving 4.5 steps earlier. Even localized tilt objectives remain split-dependent under the current WM/policy coupling. Future work should estimate the gate from an independent calibration split before applying any failure-mode penalty.

A lower boundary-start dose (`fraction=.10`) reduced the severity of the seed0 support failure but did not transfer: seed0 was nearly neutral, while seed1 return dropped by about 4%. This rules out simple dose tuning as the fix. Recovery exposure must be calibrated against WM reliability/support on each split, or moved into a separately regularized policy target.

### Exact-start counterfactual calibration

The evaluator now saves exact initial simulator states, allowing WM counterfactual rollouts and real policy rollouts to use identical starts. On 64 seed1 starts, the pessimistic two-WM ranking selected tilt on `65.6%` of starts, but real 300-step returns selected tilt on `42.2%`; margin correlation was `-0.060` and winner accuracy `0.484`. This is stronger evidence than replay-start audits: the current WM cannot serve as a policy-head selector. Any future gate must calibrate a learned value/error model against exact-start real outcomes before affecting control.

An exact-start affine calibration was fitted on seed1 WM-vs-real policy margins and tested on 64 held-out seed202 starts. It changed raw winner accuracy `0.438→0.563`, with learned slope `-10.92`, but failed on the seed1 held-out half (`0.344` accuracy) and the underlying seed1 correlation was `-0.060`. This is insufficient evidence for a deployable calibration; the next step is multi-seed calibration with confidence bounds and a fallback to the nominal policy.

A second held-out exact-start split (seed303) confirmed that affine WM calibration is not reliable: raw winner accuracy `0.375`, correlation `-0.075`, and affine-calibrated accuracy `0.516` (equal to the actual majority class). The seed202 gain does not replicate. Simple post-hoc ranking correction is closed; the remaining path requires a new training signal that improves multi-step WM fidelity itself.
