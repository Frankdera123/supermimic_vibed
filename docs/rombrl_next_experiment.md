# ROMBRL / constrained Stackelberg experiment proposal

The supplied `2505.13709v3.pdf` describes a more precise concurrent objective than an unconstrained policy-to-WM gradient. It defines a maximin problem

`max_theta J(theta, phi*)`, with `phi* = argmin_phi J(theta, phi)`

subject to a replay-distribution KL trust region around a reference model `phi_bar`:

`E_D[ KL(P_phi_bar || P_phi) ] <= epsilon`.

The paper's practical update uses a model/follower step, a dual variable for the KL budget, and a policy/leader correction through the mixed Hessian (implicit Stackelberg gradient). It also recommends the timescale ordering `eta_phi >> eta_lambda >> eta_theta`.

This explains why the earlier `joint_adversarial_residual` experiment was not a valid ROMBRL test: it reversed the policy gradient without a probabilistic KL constraint, a dual budget, or the implicit policy correction. The observed collapse is therefore expected from an unconstrained minimax update.

## Minimal SuperMimic translation

1. Keep the promoted factual WM as `phi_bar` and add a zero-initialized low-rank/task adapter `delta_phi` used only in policy imagination.
2. Define a replay KL proxy using normalized one-step residuals, with an independent validation batch. The proxy must be measured on `(s,a)` pairs and never use policy-imagined states.
3. Update the adapter adversarially against the detached policy return, but project the step to the KL budget. Maintain a dual scalar `lambda >= 0`; increase it only when validation KL exceeds epsilon.
4. Keep policy updates on the adapted model, but initially omit the expensive mixed-Hessian correction. This is a constrained alternating baseline, not a full Stackelberg claim.
5. Add the implicit correction only after the constrained alternating baseline passes fixed-policy clean/perturb evaluation. Use Hessian-vector products, not a dense Hessian.

## Falsification gates

- 30-iteration smoke: adapter finite, validation KL bounded, dual lambda finite.
- 100-iteration paired test: adapted model must not increase h1/h4 factual error by more than 10% and must not reduce imagined survival below the nominal control.
- 300-iteration two-seed gate: fixed-policy clean and perturb return must both beat or match nominal, with no increase in either height or tilt failure rate.

Until these gates are implemented, no unconstrained policy-return gradient or adversarial residual should be called a ROMBRL reproduction.
