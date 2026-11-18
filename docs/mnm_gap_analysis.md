# MnM / policy-aware world-model gap audit (2026-11-07)

The literature audit changes the interpretation of the earlier online-realism experiments. In *Mismatched No More*, the discriminator is trained on paired real/model transitions sharing the same `(s,a)`, and the policy uses an augmented reward containing both the task reward and the discriminator log-odds. More importantly, the model objective is not simply `softplus(-logit)` or a GAN realism penalty: it contains a value term at the model next state plus the discriminator log-odds.

For a transition classifier `C(real)=1`, the paper's approximate augmented reward is

`r_tilde = log r(s_t,a_t) + log(C(s_t,a_t,s_{t+1}) / (1-C(...)))`.

The model-side objective is approximately

`E_q[ V(s_{t+1}) + logit_C(s_t,a_t,s_{t+1}) ]`.

The current SuperMimic options cover only fragments:

- `online_realism_weight` adds a policy-side softplus realism penalty;
- `online_realism_reward_weight` adds discriminator `logsigmoid` to imagined reward;
- `online_realism_model_weight` adds `softplus(-logit)` to the WM loss;
- the discriminator is trained on policy-transition fakes after the initialization bug was fixed.

None of these options jointly implements the paper's model-side value-plus-logit objective. Therefore the corrected online-realism failures in `current_frontier.md` close the current binary-discriminator fragments, but do not constitute a faithful MnM reproduction.

The paper also reports that the classifier term can hurt continuous-control performance and uses an MnM-approx variant that changes the model objective while retaining the ordinary policy reward. This is relevant to SuperMimic: adding raw discriminator penalties to policy imagination is especially likely to trade height and tilt failures, while a value-aware model update should be isolated behind a task-model adapter and a held-out multi-step audit.

## Next implementation gate

Before adding another training term, implement a diagnostic-only one-step lower-bound audit on identical `(s,a)` pairs:

1. train the real/model transition classifier on the exact paired tuple;
2. measure held-out classifier calibration and log-odds;
3. compare the detached value proxy and log-odds terms separately against factual WM error and fixed-policy failure subtype;
4. only if the value-plus-logit signal predicts held-out transition quality, add it to a zero-initialized task adapter with rollback and horizon-1/horizon-4 audits.

This keeps nominal WM fitting and deployment policy unchanged until the lower-bound signal is calibrated. The existing online-realism flags remain diagnostic and should not be interpreted as a full MnM result.
