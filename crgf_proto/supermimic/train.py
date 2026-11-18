"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--policy_bptt_truncation", type=int, default=0,
                    help="detach imagined state/history every N steps (0 keeps full BPTT)")
parser.add_argument("--policy_bptt_state_only", action="store_true",
                    help="with truncation, detach only WM state (retain history/action graph)")
parser.add_argument("--joint_wm_policy_weight", type=float, default=0.0,
                    help="experimental extra policy-gradient update on WM parameters (0 disables)")
parser.add_argument("--joint_wm_replay_weight", type=float, default=0.0,
                    help="replay WM-loss anchor during the joint policy/WM update (0 disables)")
parser.add_argument("--joint_wm_replay_window", type=int, default=0,
                    help="window for the joint replay anchor (0 uses --wm_window)")
parser.add_argument("--joint_wm_policy_transition_ratio", type=float, default=0.0,
                    help="extra one-step real-transition anchor ratio inside the joint replay loss")
parser.add_argument("--joint_wm_policy_horizon", type=int, default=0,
                    help="prefix horizon used only for the residual policy task gradient (0 uses full horizon)")
parser.add_argument("--joint_wm_policy_last_layer", action="store_true",
                    help="route concurrent policy gradients only into the WM output layer")
parser.add_argument("--joint_wm_policy_root_outputs", action="store_true",
                    help="route concurrent policy gradients only into WM root p/R/v/w output rows")
parser.add_argument("--joint_wm_policy_action_path", action="store_true",
                    help="route concurrent policy gradients only through first-layer action columns")
parser.add_argument("--joint_wm_policy_action_norm_ratio", type=float, default=0.0,
                    help="if >0, normalize action-path task gradient to this fraction of replay gradient")
parser.add_argument("--joint_wm_replay_sensitivity", type=float, default=0.0,
                    help="reward-sensitivity weight for the residual joint replay anchor")
parser.add_argument("--joint_replace_wm_step", action="store_true",
                    help="replace one standalone WM step with the joint policy/WM step")
parser.add_argument("--joint_residual_hidden", type=int, default=0,
                    help="small policy-sensitive WM residual branch (0 disables)")
parser.add_argument("--joint_residual_max_norm", type=float, default=0.0,
                    help="trust-region norm for residual output in normalized WM space (0 disables)")
parser.add_argument("--joint_residual_l2", type=float, default=0.0,
                    help="L2 anchor toward the zero-initialized residual parameters")
parser.add_argument("--joint_residual_dual", action="store_true",
                    help="adapt replay-anchor weight from residual-vs-nominal fidelity violation")
parser.add_argument("--joint_residual_dual_lr", type=float, default=0.1,
                    help="dual ascent step for --joint_residual_dual")
parser.add_argument("--joint_residual_dual_target", type=float, default=0.0,
                    help="allowed residual-minus-nominal replay error before dual ascent")
parser.add_argument("--joint_residual_grad_clip", type=float, default=0.0,
                    help="global gradient norm cap applied only to residual parameters")
parser.add_argument("--ensemble_gate_member0", type=str, default=None,
                    help="frozen residual ensemble member checkpoint for policy imagination")
parser.add_argument("--ensemble_gate_member1", type=str, default=None,
                    help="second frozen residual ensemble member checkpoint for policy imagination")
parser.add_argument("--ensemble_gate_threshold", type=float, default=0.0,
                    help="rollout-calibrated disagreement threshold for ensemble gate")
parser.add_argument("--ensemble_gate_nominal_prefix", type=int, default=0,
                    help="imagined steps forced to nominal WM before ensemble gate can open (0 disables)")
parser.add_argument("--ensemble_uncertainty_weight", type=float, default=0.0,
                    help="penalize one-step bootstrap-ensemble disagreement in the policy loss")
parser.add_argument("--ensemble_policy_gate_fraction", type=float, default=0.0,
                    help="retain this low-disagreement fraction of imagined starts (0 disables)")
parser.add_argument("--ensemble_policy_gate_horizon", type=int, default=0,
                    help="use disagreement at this imagined step for the policy gate (0 averages all steps)")
parser.add_argument("--ensemble_policy_softmin_temperature", type=float, default=0.0,
                    help="soft-min temperature for nominal/ensemble policy returns (0 disables)")
parser.add_argument("--wm_ensemble_disagreement_weight", type=float, default=0.0,
                    help="weight replay WM loss toward transitions with frozen-ensemble disagreement")
parser.add_argument("--joint_residual_occupancy_sigma", type=float, default=0.0,
                    help="if >0, enable residual only near replay action occupancy")
parser.add_argument("--joint_residual_audit_threshold", type=float, default=0.0,
                    help="enable residual above a train-calibrated normalized disagreement threshold")
parser.add_argument("--joint_residual_validity_checkpoint", type=str, default=None,
                    help="frozen classifier checkpoint for state-dependent residual validity gating")
parser.add_argument("--joint_gradient_diagnostic", action="store_true",
                    help="log task/replay gradient cosine on the residual adapter")
parser.add_argument("--joint_gradient_projection", action="store_true",
                    help="project residual task gradient away from conflicting replay gradient")
parser.add_argument("--joint_gradient_conflict_gate", action="store_true",
                    help="drop residual policy gradient on batches whose task/replay dot product is negative")
parser.add_argument("--joint_gradient_norm_match", action="store_true",
                    help="normalize residual policy gradient to replay-gradient norm before applying coupling")
parser.add_argument("--joint_gradient_norm_ratio", type=float, default=1.0,
                    help="task/replay norm ratio used by --joint_gradient_norm_match")
parser.add_argument("--joint_wm_policy_period", type=int, default=1,
                    help="apply residual policy task gradients every N policy updates")
parser.add_argument("--joint_wm_policy_start_iter", type=int, default=0,
                    help="delay residual policy task gradients until this iteration")
parser.add_argument("--joint_gradient_task_ema", type=float, default=0.0,
                    help="EMA coefficient for residual task gradients (0 disables)")
parser.add_argument("--joint_adversarial_residual", action="store_true",
                    help="use the policy task gradient adversarially on the residual WM")
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--phase_conditioned_policy", action="store_true",
                    help="append sin/cos normalized motion-clip phase to policy observations")
parser.add_argument("--clip_conditioned_policy", action="store_true",
                    help="append a one-hot motion-clip identity to policy observations")
parser.add_argument("--contact_conditioned_policy", action="store_true",
                    help="append reference body-link contact pattern to policy observations")
parser.add_argument("--current_contact_policy", action="store_true",
                    help="append current FK body-link contact pattern to policy observations")
parser.add_argument("--reset_sampling_mode", choices=("return", "kinematic"), default="return",
                    help="reset-frame sampler; kinematic is an isolated coverage ablation")
parser.add_argument("--reset_kinematic_clusters", type=int, default=16,
                    help="number of kinematic clusters for --reset_sampling_mode kinematic")
parser.add_argument("--fall_grace_ref_height", type=float, default=0.0,
                    help="disable height termination while reference root height is below this value")
parser.add_argument("--fall_grace_ref_tilt", type=float, default=0.0,
                    help="disable tilt termination while reference uprightness is below this value")
parser.add_argument("--contact_safe_reward", action="store_true",
                    help="clamp reference body-link heights to the ground in the tracking reward")
parser.add_argument("--contact_masked_reward", action="store_true",
                    help="suppress body-link tracking terms on reference frames with below-ground links")
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_aux_window", type=int, default=0,
                    help="optional second autoregressive WM loss window")
parser.add_argument("--wm_aux_weight", type=float, default=0.0,
                    help="weight of the optional auxiliary WM window loss")
parser.add_argument("--wm_aux_ramp_iters", type=int, default=0,
                    help="linearly ramp auxiliary WM loss over this many iterations (0 disables ramp)")
parser.add_argument("--wm_multistep_weight", type=float, default=0.0,
                    help="weight of an explicit real multi-step transition target")
parser.add_argument("--wm_multistep_horizon", type=int, default=2,
                    help="number of replay actions in the explicit multi-step target")
parser.add_argument("--wm_multistep_detach_intermediate", action="store_true",
                    help="detach intermediate predicted states in the multi-step target")
parser.add_argument("--wm_aux_window2", type=int, default=0,
                    help="optional second auxiliary autoregressive WM window")
parser.add_argument("--wm_aux_weight2", type=float, default=0.0,
                    help="weight of the second auxiliary WM window loss")
parser.add_argument("--wm_aux_switch_iters", type=int, default=0,
                    help="if set, stage auxiliary windows: first ramps in, then decays as second ramps in")
parser.add_argument("--wm_loss_mean", action="store_true",
                    help="normalize WM loss by valid transitions instead of summing over window")
parser.add_argument("--wm_huber_delta", type=float, default=0.0,
                    help="optional smooth-L1 beta on normalized WM residuals (0 keeps MSE)")
parser.add_argument("--wm_action_effect_gain", type=float, default=1.0,
                    help="scale actions only inside imagined WM policy rollouts (1 keeps default)")
parser.add_argument("--wm_action_delta_calibration", type=str, default=None,
                    help="optional JSON 70-D local state-delta calibration for imagined WM rollouts")
parser.add_argument("--wm_intervention_adapter_checkpoint", type=str, default=None,
                    help="optional simulator-supervised local transition adapter checkpoint")
parser.add_argument("--wm_intervention_adapter_blend", type=float, default=0.1,
                    help="blend weight for simulator adapter correction (0 keeps nominal delta)")
parser.add_argument("--wm_intervention_adapter_max_ratio", type=float, default=0.0,
                    help="optional cap on corrected delta norm relative to nominal WM delta")
parser.add_argument("--wm_intervention_adapter_gate_ratio", type=float, default=0.0,
                    help="optional validity gate; reject adapter corrections above this norm ratio")
parser.add_argument("--wm_intervention_adapter_horizon", type=int, default=1,
                    help="horizon for endpoint adapter correction blocks (1 keeps the legacy one-step interface)")
parser.add_argument("--wm_intervention_adapter_endpoint_only", action="store_true",
                    help="apply a horizon adapter residual only on the final step of each block")
parser.add_argument("--wm_counterfactual_dataset", type=str, default=None,
                    help="optional simulator counterfactual dataset for WM auxiliary supervision")
parser.add_argument("--wm_counterfactual_weight", type=float, default=0.0,
                    help="weight of simulator counterfactual one-step WM loss")
parser.add_argument("--wm_counterfactual_effect_weight", type=float, default=0.0,
                    help="weight of simulator counterfactual action-effect loss")
parser.add_argument("--wm_counterfactual_norm_weight", type=float, default=0.0,
                    help="weight for preserving the magnitude of multi-direction action effects")
parser.add_argument("--wm_counterfactual_tilt_weight", type=float, default=0.0,
                    help="extra root-rotation endpoint loss for counterfactual supervision")
parser.add_argument("--wm_counterfactual_geometry_weight", type=float, default=0.0,
                    help="weight for simulator/WM action-effect direction cosine loss")
parser.add_argument("--wm_counterfactual_geometry_mag_weight", type=float, default=0.0,
                    help="weight for simulator/WM action-effect log-magnitude loss")
parser.add_argument("--wm_counterfactual_geometry_root_only", action="store_true",
                    help="compute counterfactual geometry loss only on 12-D root pose/velocity effects")
parser.add_argument("--wm_counterfactual_error_gate", type=float, default=0.0,
                    help="reliability weighting strength for simulator counterfactual targets; 0 disables")
parser.add_argument("--wm_counterfactual_reference_checkpoint", type=str, default=None,
                    help="frozen WM checkpoint used for relative counterfactual sensitivity")
parser.add_argument("--wm_counterfactual_relative_weight", type=float, default=0.0,
                    help="weight for bounded log-relative sensitivity to the reference WM")
parser.add_argument("--wm_counterfactual_relative_floor", type=float, default=0.0,
                    help="optional lower sensitivity ratio floor for one-sided relative loss")
parser.add_argument("--wm_counterfactual_gram_weight", type=float, default=0.0,
                    help="weight of normalized multi-direction counterfactual effect Gram loss")
parser.add_argument("--wm_counterfactual_risk_gradient_weight", type=float, default=0.0,
                    help="weight for simulator-vs-WM h4 height/tilt risk-gradient matching")
parser.add_argument("--wm_counterfactual_risk_gradient_balanced", action="store_true",
                    help="match height and tilt risk gradients as separate normalized objectives")
parser.add_argument("--wm_counterfactual_risk_gradient_tilt_scale", type=float, default=1.0,
                    help="relative tilt-gradient weight for balanced risk matching")
parser.add_argument("--wm_counterfactual_deterministic", action="store_true",
                    help="cycle counterfactual samples deterministically instead of random resampling")
parser.add_argument("--wm_counterfactual_batch", type=int, default=0,
                    help="counterfactual auxiliary batch size (0 uses --wm_batch)")
parser.add_argument("--wm_teacher_forcing", action="store_true",
                    help="fit each WM window step from the recorded state instead of autoregressive predictions")
parser.add_argument("--wm_schedule_sampling_prob", type=float, default=0.0,
                    help="probability of feeding a detached previous WM prediction into teacher-forced windows")
parser.add_argument("--wm_schedule_sampling_ramp_iters", type=int, default=0,
                    help="linearly ramp schedule-sampling probability after WM warmup (0 keeps it constant)")
parser.add_argument("--wm_consistency_weight", type=float, default=0.0,
                    help="weight for free-running vs detached teacher-forced WM consistency")
parser.add_argument("--wm_consistency_risk_weight", type=float, default=0.0,
                    help="extra free-running consistency weight near soft termination states")
parser.add_argument("--wm_consistency_alignment_min", type=float, default=-1.0,
                    help="minimum cosine between factual WM and consistency gradients; negative disables gating")
parser.add_argument("--wm_consistency_error_gate_ratio", type=float, default=0.0,
                    help="attenuate consistency when factual WM error exceeds this EMA ratio (0 disables)")
parser.add_argument("--wm_consistency_error_gate_beta", type=float, default=0.95,
                    help="EMA coefficient for --wm_consistency_error_gate_ratio")
parser.add_argument("--wm_norm_checkpoint", type=str, default=None,
                    help="load and freeze WM input/output running norms from a checkpoint")
parser.add_argument("--wm_occupancy_weight", type=float, default=0.0,
                    help="policy-action compatibility weight for WM windows (0 disables)")
parser.add_argument("--wm_occupancy_sigma", type=float, default=0.05,
                    help="normalized action-distance scale for occupancy weighting")
parser.add_argument("--wm_occupancy_floor", type=float, default=0.25,
                    help="uniform replay floor for occupancy weighting")
parser.add_argument("--wm_occupancy_error_ratio", type=float, default=0.0,
                    help="fallback to uniform WM loss when occupancy-weighted error exceeds this ratio")
parser.add_argument("--wm_occupancy_validation_beta", type=float, default=0.9,
                    help="EMA coefficient for independent replay validation used by occupancy fallback")
parser.add_argument("--wm_policy_aware_weight", type=float, default=0.0,
                    help="policy-compatible strategic-error replay weighting (0 disables)")
parser.add_argument("--wm_policy_aware_sigma", type=float, default=0.05,
                    help="normalized action-distance scale for policy-aware replay weighting")
parser.add_argument("--wm_policy_aware_floor", type=float, default=0.5,
                    help="uniform replay floor for policy-aware compatibility")
parser.add_argument("--wm_policy_aware_validation_ratio", type=float, default=0.0,
                    help="fallback to uniform policy-aware WM loss when validation error rises")
parser.add_argument("--wm_policy_aware_uncertainty", type=float, default=0.5,
                    help="ensemble-disagreement multiplier inside policy-aware weighting")
parser.add_argument("--strategic_probe_capacity", type=int, default=0,
                    help="separate factual buffer capacity for policy-selected strategic-error probes (0 disables)")
parser.add_argument("--strategic_probe_weight", type=float, default=0.0,
                    help="one-step WM loss weight for sampled strategic probes (0 disables)")
parser.add_argument("--strategic_probe_batch", type=int, default=0,
                    help="strategic probe batch size (0 uses --wm_batch)")
parser.add_argument("--strategic_probe_alpha", type=float, default=1.0,
                    help="score-prioritization exponent for strategic probe sampling")
parser.add_argument("--strategic_probe_dataset", type=str, default=None,
                    help="optional factual probe dataset to preload into the separate buffer")
parser.add_argument("--wm_reward_weight", type=float, default=0.0,
                    help="detached replay WM weighting toward high-reward transitions (0 disables)")
parser.add_argument("--wm_reward_mismatch_weight", type=float, default=0.0,
                    help="weight WM replay by predicted-vs-real reward/termination mismatch")
parser.add_argument("--wm_action_recovery_weight", type=float, default=0.0,
                    help="auxiliary inverse-dynamics loss on WM transitions (0 disables)")
parser.add_argument("--wm_action_recovery_fidelity_gate", type=float, default=0.0,
                    help="downweight action-recovery loss on high-error WM transitions")
parser.add_argument("--wm_action_recovery_alignment_min", type=float, default=-1.0,
                    help="if >=0, keep inverse-action loss only when its WM gradient cosine with factual loss exceeds this value")
parser.add_argument("--wm_action_recovery_project_grad", action="store_true",
                    help="project action-recovery WM gradients onto the nonnegative factual-replay gradient direction")
parser.add_argument("--wm_action_recovery_onpolicy_only", action="store_true",
                    help="apply action-recovery auxiliary only to recent on-policy WM batches")
parser.add_argument("--wm_action_recovery_hidden", type=int, default=128,
                    help="hidden width of the action-recovery auxiliary head")
parser.add_argument("--wm_action_contrastive_weight", type=float, default=0.0,
                    help="contrastive action-transition matching loss on WM predictions (0 disables)")
parser.add_argument("--wm_action_contrastive_hidden", type=int, default=128,
                    help="embedding width of the contrastive action-transition heads")
parser.add_argument("--wm_action_contrastive_temperature", type=float, default=0.1,
                    help="temperature for contrastive action-transition matching")
parser.add_argument("--wm_action_contrastive_normalize_actions", action="store_true",
                    help="divide actions by 20 before the contrastive action encoder")
parser.add_argument("--wm_action_contrastive_delta", action="store_true",
                    help="encode normalized frame-local WM state deltas instead of absolute state pairs")
parser.add_argument("--wm_action_contrastive_counterfactual", action="store_true",
                    help="use same-state counterfactual actions as contrastive negatives")
parser.add_argument("--wm_action_contrastive_corr_gate", type=float, default=0.0,
                    help="only apply contrastive WM loss when per-transition action/WM-error correlation exceeds this value")
parser.add_argument("--wm_transition_weight_checkpoint", type=str, default=None,
                    help="frozen offline transition classifier checkpoint for WM replay weighting")
parser.add_argument("--wm_transition_weight_scale", type=float, default=1.0,
                    help="odds weighting strength for the offline transition classifier")
parser.add_argument("--wm_dual_q_weight_checkpoint", type=str, default=None,
                    help="offline age-by-env dual-Q weights for WM replay windows")
parser.add_argument("--policy_realism_checkpoint", type=str, default=None,
                    help="frozen real-vs-WM transition discriminator for policy realism penalty")
parser.add_argument("--policy_realism_weight", type=float, default=0.0,
                    help="weight of the optional realism penalty on imagined policy transitions")
parser.add_argument("--online_realism_weight", type=float, default=0.0,
                    help="joint MnM-style realism penalty with an online real-vs-WM discriminator")
parser.add_argument("--online_realism_model_weight", type=float, default=0.0,
                    help="generator-side realism loss on WM replay transitions (requires online discriminator)")
parser.add_argument("--online_realism_reward_weight", type=float, default=0.0,
                    help="add online discriminator log-realism as per-step imagined reward shaping")
parser.add_argument("--online_realism_local_delta", action="store_true",
                    help="train online realism on frame-local transition delta plus action instead of absolute states")
parser.add_argument("--online_realism_steps", type=int, default=1,
                    help="discriminator optimizer steps per WM update")
parser.add_argument("--online_realism_policy_transitions", action="store_true",
                    help="also train the online realism discriminator on detached policy-imagined transitions")
parser.add_argument("--online_realism_gate_temperature", type=float, default=0.0,
                    help="if >0, use online realism logits as a normalized policy-gradient gate instead of only a penalty")
parser.add_argument("--online_realism_ess_gate", type=float, default=0.0,
                    help="if >0, skip imagined policy gradients when calibrated online realism odds ESS fraction is below this threshold")
parser.add_argument("--online_realism_ess_risk_floor", type=float, default=0.0,
                    help="normalized height-risk floor whose starts keep policy gradients when the ESS gate opens")
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_recent_fraction", type=float, default=0.0,
                    help="fraction of WM replay windows forced into the most recent buffer region")
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--adaptive_action_noise", type=float, default=0.0,
                    help="scale action exploration by frozen ensemble disagreement (0 disables)")
parser.add_argument("--adaptive_action_noise_threshold", type=float, default=0.0,
                    help="only apply adaptive exploration above normalized disagreement threshold (0 disables threshold)")
parser.add_argument("--adaptive_action_direction", type=float, default=0.0,
                    help="directional action perturbation toward frozen ensemble disagreement")
parser.add_argument("--adaptive_action_direction_symmetric", action="store_true",
                    help="sample random +/- signs for directional ensemble probes")
parser.add_argument("--adaptive_action_direction_fraction", type=float, default=1.0,
                    help="fraction of collected actions receiving directional perturbations")
parser.add_argument("--adaptive_noise_recoverability", type=float, default=0.0,
                    help="soft-termination threshold for allowing adaptive exploration (0 disables)")
parser.add_argument("--adaptive_probe_onpolicy_only", action="store_true",
                    help="apply adaptive exploration only to fresh on-policy probe-buffer collection")
parser.add_argument("--deterministic_collect_fraction", type=float, default=0.0,
                    help="fraction of training envs with action noise disabled (0 keeps uniform noise)")
parser.add_argument("--term_soft_height", type=float, default=0.02,
                    help="height-error sigmoid width for imagined soft termination")
parser.add_argument("--term_soft_tilt", type=float, default=0.04,
                    help="tilt-error sigmoid width for imagined soft termination")
parser.add_argument("--termination_diagnostic", action="store_true",
                    help="log calibration of analytic soft termination on WM replay rollouts")
parser.add_argument("--wm_termination_weight", type=float, default=0.0,
                    help="weight replay WM loss for matching real transition termination events")
parser.add_argument("--imag_action_noise", type=float, default=0.0,
                    help="Gaussian action noise added inside imagined policy rollouts")
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--dispersion_gate", type=float, default=0.0,
                    help="relative online/EMA imagined-return gate; 0 disables it")
parser.add_argument("--dispersion_gate_temperature", type=float, default=0.0,
                    help="soft gate temperature; 0 keeps the hard gate")
parser.add_argument("--dispersion_gate_fraction", type=float, default=0.0,
                    help="if >0, trust this batch quantile of starts instead of a fixed threshold")
parser.add_argument("--dispersion_gate_normalize", action="store_true",
                    help="renormalize policy gradients by the mean gate weight")
parser.add_argument("--dispersion_gate_warmup", type=int, default=0,
                    help="iterations before enabling the dispersion gate")
parser.add_argument("--wm_ema_beta", type=float, default=0.995,
                    help="EMA coefficient for the optional target world model")
parser.add_argument("--policy_ema_model", action="store_true",
                    help="use the EMA world model for imagined policy updates")
parser.add_argument("--policy_ema_start", type=int, default=0,
                    help="iteration at which --policy_ema_model becomes active")
parser.add_argument("--policy_target_audit_max_error", type=float, default=0.0,
                    help="enable a separate EMA policy target only when an independent replay audit error is below this threshold (0 disables)")
parser.add_argument("--policy_target_audit_interval", type=int, default=1,
                    help="iterations between independent replay audits for --policy_target_audit_max_error")
parser.add_argument("--policy_target_audit_horizon", type=int, default=1,
                    help="replay rollout horizon used by the policy-target audit (1 keeps the one-step audit)")
parser.add_argument("--policy_target_audit_h1_max_error", type=float, default=0.0,
                    help="optional mean normalized error budget at horizon 1 (0 uses the legacy max-error gate)")
parser.add_argument("--policy_target_audit_h4_max_error", type=float, default=0.0,
                    help="optional mean normalized error budget at horizon 4 (0 uses the legacy max-error gate)")
parser.add_argument("--policy_target_ema_interval", type=int, default=1,
                    help="WM updates between EMA syncs of the policy target (1 keeps continuous EMA)")
parser.add_argument("--policy_target_until_iter", type=int, default=0,
                    help="if >0, use the policy target only before this iteration")
parser.add_argument("--policy_target_control_file", type=str, default="",
                    help="optional external control file: write disable/off/0 to force nominal WM")
parser.add_argument("--policy_target_adapter_hidden", type=int, default=0,
                    help="hidden width of an opt-in zero-initialized policy-target adapter (0 disables)")
parser.add_argument("--policy_target_pessimistic", action="store_true",
                    help="use the minimum of nominal and policy-target imagined returns")
parser.add_argument("--policy_target_audit_ratio", type=float, default=0.0,
                    help="maximum relative target-vs-nominal held-out audit error increase (0 disables)")
parser.add_argument("--policy_target_adapter_max_norm", type=float, default=0.0,
                    help="absolute L2 cap on the policy-target adapter parameters (0 disables)")
parser.add_argument("--policy_target_adapter_l2", type=float, default=0.0,
                    help="L2 anchor toward the zero-initialized policy-target adapter (0 disables)")
parser.add_argument("--policy_target_adapter_lr", type=float, default=0.0,
                    help="adapter optimizer learning rate (0 uses --lr)")
parser.add_argument("--policy_target_adapter_max_delta", type=float, default=0.0,
                    help="per-step normalized RMS cap on task-adapter output correction (0 disables)")
parser.add_argument("--policy_target_kl_budget", type=float, default=0.0,
                    help="held-out replay KL proxy budget for the task adapter (0 disables)")
parser.add_argument("--policy_target_kl_dual_lr", type=float, default=0.0,
                    help="dual ascent learning rate for the task-adapter KL proxy (0 uses adapter lr)")
parser.add_argument("--shadow_policy", action="store_true",
                    help="train an isolated policy/critic branch through the policy target WM")
parser.add_argument("--shadow_policy_lr_scale", type=float, default=1.0,
                    help="learning-rate multiplier for the isolated shadow policy/critic")
parser.add_argument("--shadow_policy_updates", type=int, default=1,
                    help="number of isolated shadow policy updates per policy_due iteration")
parser.add_argument("--shadow_policy_max_step_norm", type=float, default=0.0,
                    help="optional L2 cap on each shadow policy parameter step (0 disables)")
parser.add_argument("--wm_prioritized", action="store_true",
                    help="sample world-model windows by recent model error")
parser.add_argument("--wm_priority_alpha", type=float, default=0.6,
                    help="exponent for world-model error priorities")
parser.add_argument("--wm_onpolicy_error_weight", type=float, default=0.0,
                    help="detached high-error weighting for isolated on-policy WM updates")
parser.add_argument("--wm_sensitivity_weight", type=float, default=0.0,
                    help="weight model loss by tracking-reward state sensitivity; 0 disables")
parser.add_argument("--wm_root_height_weight", type=float, default=0.0,
                    help="extra relative weight for the root-z WM error component; 0 disables")
parser.add_argument("--wm_angular_velocity_weight", type=float, default=0.0,
                    help="upweight factual WM windows with high root angular velocity; 0 disables")
parser.add_argument("--wm_component_weights", type=str, default="",
                    help="comma-separated 70-component WM loss weights (empty keeps uniform weights)")
parser.add_argument("--wm_component_weights_file", type=str, default="",
                    help="file containing comma-separated WM component weights")
parser.add_argument("--policy_height_penalty", type=float, default=0.0,
                    help="penalize imagined root-z tracking drift in the policy objective; 0 disables")
parser.add_argument("--policy_height_penalty_ramp_iters", type=int, default=0,
                    help="linearly ramp the policy height penalty after WM-only warmup")
parser.add_argument("--policy_height_penalty_start_margin", type=float, default=0.0,
                    help="normalized initial height error required before applying height penalty (0 keeps global penalty)")
parser.add_argument("--policy_height_hinge_margin", type=float, default=0.0,
                    help="normalized root-z error margin before height penalty activates; 0 keeps quadratic penalty")
parser.add_argument("--policy_tilt_penalty", type=float, default=0.0,
                    help="penalize imagined root tilt drift in the policy objective; 0 disables")
parser.add_argument("--policy_tilt_penalty_start_margin", type=float, default=0.0,
                    help="normalized start tilt error required before applying tilt penalty (0 keeps global penalty)")
parser.add_argument("--policy_angular_velocity_penalty", type=float, default=0.0,
                    help="penalize imagined root angular velocity in the policy objective; 0 disables")
parser.add_argument("--policy_angular_velocity_gate_margin", type=float, default=0.0,
                    help="normalized tilt error margin before angular-velocity penalty activates")
parser.add_argument("--shadow_policy_tilt_penalty", type=float, default=0.0,
                    help="additional imagined root tilt penalty for the isolated shadow policy only; 0 disables")
parser.add_argument("--wm_rollout_sensitivity_weight", type=float, default=0.0,
                    help="weight WM loss by multi-step imagined reward/termination sensitivity")
parser.add_argument("--wm_td_weight", type=float, default=0.0,
                    help="weight WM loss by absolute replay TD error from the critic")
parser.add_argument("--wm_td_target_beta", type=float, default=0.99,
                    help="EMA coefficient for the target critic used by TD-weighted WM")
parser.add_argument("--wm_disagreement_diagnostic", action="store_true",
                    help="log correlation between online/EMA disagreement and real one-step error")
parser.add_argument("--dispersion_gate_corr_min", type=float, default=0.0,
                    help="enable the gate only while diagnostic disagreement/error correlation exceeds this value")
parser.add_argument("--dispersion_gate_corr_beta", type=float, default=0.9,
                    help="EMA coefficient for the diagnostic disagreement/error correlation")
parser.add_argument("--dispersion_return_penalty", type=float, default=0.0,
                    help="penalize online imagined returns that disagree with the EMA model")
parser.add_argument("--dispersion_truncate_temperature", type=float, default=0.0,
                    help="if >0, exponentially downweight imagined steps by online/EMA state disagreement")
parser.add_argument("--dispersion_softmin_temperature", type=float, default=0.0,
                    help="if >0, optimize a soft-min of online and EMA imagined returns")
parser.add_argument("--dispersion_truncate_local", action="store_true",
                    help="use per-step disagreement weights instead of cumulative trust products")
parser.add_argument("--dispersion_truncate_normalize", action="store_true",
                    help="normalize local disagreement weights to preserve mean trust mass")
parser.add_argument("--policy_before_wm", action="store_true",
                    help="update policy before the world model within each iteration")
parser.add_argument("--onpolicy_refresh_steps", type=int, default=0,
                    help="after a policy update, collect fresh steps before WM fitting")
parser.add_argument("--onpolicy_replay_fraction", type=float, default=0.0,
                    help="fraction of each WM batch sampled from the fresh on-policy buffer")
parser.add_argument("--onpolicy_buffer_steps", type=int, default=0,
                    help="capacity of the fresh on-policy buffer (0 uses --buffer_steps)")
parser.add_argument("--onpolicy_separate_wm_update", action="store_true",
                    help="fit fresh on-policy data in a separate small-LR WM update")
parser.add_argument("--policy_updates", type=int, default=1,
                    help="number of independent policy/critic updates per iteration")
parser.add_argument("--policy_updates_warmup_iters", type=int, default=0,
                    help="use one policy update for this many iterations before --policy_updates")
parser.add_argument("--policy_updates_until_iter", type=int, default=0,
                    help="after this many post-WM iterations, reduce policy updates to one (0 disables)")
parser.add_argument("--policy_freeze_after_iter", type=int, default=0,
                    help="freeze policy/critic updates after this many post-WM iterations (0 disables)")
parser.add_argument("--policy_wm_error_gate_ratio", type=float, default=0.0,
                    help="if >0, skip the second policy update when current WM err8 exceeds this EMA ratio")
parser.add_argument("--policy_wm_error_gate_beta", type=float, default=0.95,
                    help="EMA coefficient for --policy_wm_error_gate_ratio")
parser.add_argument("--policy_freeze_wm_error_ratio", type=float, default=0.0,
                    help="skip the current policy update when WM err8 exceeds this EMA ratio (0 disables)")
parser.add_argument("--policy_freeze_wm_error_beta", type=float, default=0.95,
                    help="EMA coefficient for --policy_freeze_wm_error_ratio")
parser.add_argument("--policy_freeze_wm_error_min_iter", type=int, default=25,
                    help="do not trigger WM-error policy freezing before this post-WM iteration")
parser.add_argument("--policy_wm_reliability_quantile", type=float, default=0.0,
                    help="gate policy starts above this replay WM-error quantile (0 disables)")
parser.add_argument("--policy_wm_reliability_floor", type=float, default=0.25,
                    help="minimum policy-gradient weight for unreliable replay starts")
parser.add_argument("--policy_wm_horizon_decay", type=float, default=0.0,
                    help="detached replay-error trust decay across imagined horizon (0 disables)")
parser.add_argument("--policy_failure_start_fraction", type=float, default=0.0,
                    help="fraction of policy starts focused on replay states nearest height/tilt failure (0 disables)")
parser.add_argument("--policy_failure_start_height_only", action="store_true",
                    help="rank failure-focused policy starts by height error only")
parser.add_argument("--policy_failure_start_tilt_only", action="store_true",
                    help="rank failure-focused policy starts by tilt error only")
parser.add_argument("--policy_failure_start_alternate_iters", type=int, default=0,
                    help="alternate failure-focused ranking between height and tilt every N iterations (0 disables)")
parser.add_argument("--policy_failure_start_balanced", action="store_true",
                    help="split failure-focused starts between height-near and tilt-near replay states")
parser.add_argument("--policy_failure_start_ramp_iters", type=int, default=0,
                    help="ramp failure-focused start fraction over this many iterations (0 keeps it active immediately)")
parser.add_argument("--policy_strategic_error_weight", type=float, default=0.0,
                    help="penalize WM error on recent on-policy transitions")
parser.add_argument("--policy_strategic_error_sigma", type=float, default=0.05,
                    help="normalized action-distance scale for strategic-error proximity gate")
parser.add_argument("--policy_second_update_scale", type=float, default=1.0,
                    help="learning-rate multiplier for the second policy update (1 keeps equal LR)")
parser.add_argument("--policy_lr_scale", type=float, default=1.0,
                    help="global policy/critic learning-rate multiplier relative to the WM learning rate")
parser.add_argument("--policy_reuse_batch", action="store_true",
                    help="reuse one sampled policy batch for all updates in an iteration (ablation)")
parser.add_argument("--interleave_policy_wm", action="store_true",
                    help="interleave WM and policy updates as WM->policy1->WM->policy2 (ablation)")
parser.add_argument("--policy_accumulate_updates", action="store_true",
                    help="accumulate multiple independent policy batches into one optimizer step (ablation)")
parser.add_argument("--policy_period", type=int, default=1,
                    help="run policy updates every N iterations after WM warmup")
parser.add_argument("--policy_bc_weight", type=float, default=0.0,
                    help="behavior-cloning anchor on the replay action at each imagined start")
parser.add_argument("--policy_bc_risk_margin", type=float, default=0.0,
                    help="apply policy BC only when normalized height/tilt proximity exceeds this margin (0 disables gating)")
parser.add_argument("--policy_awbc_weight", type=float, default=0.0,
                    help="advantage-weighted replay-action anchor (uses normalized one-step reward)")
parser.add_argument("--policy_awbc_beta", type=float, default=1.0,
                    help="temperature for advantage-weighted behavior cloning")
parser.add_argument("--policy_proximal_weight", type=float, default=0.0,
                    help="proximal penalty against the pre-update policy action (0 disables)")
parser.add_argument("--policy_ramp_iters", type=int, default=0,
                    help="linearly ramp policy optimizer LR after wm_only_iters (0 disables)")
add_launcher_args(parser)
args = parser.parse_args()
if args.dispersion_gate < 0.0:
    parser.error("--dispersion_gate must be non-negative")
if not 0.0 <= args.wm_schedule_sampling_prob <= 1.0:
    parser.error("--wm_schedule_sampling_prob must be in [0, 1]")
if args.wm_schedule_sampling_ramp_iters < 0:
    parser.error("--wm_schedule_sampling_ramp_iters must be nonnegative")
if args.wm_consistency_weight < 0.0:
    parser.error("--wm_consistency_weight must be nonnegative")
if not 0.0 <= args.policy_failure_start_fraction <= 1.0:
    parser.error("--policy_failure_start_fraction must be in [0, 1]")
if args.policy_failure_start_ramp_iters < 0:
    parser.error("--policy_failure_start_ramp_iters must be nonnegative")
if args.policy_bc_risk_margin < 0.0:
    parser.error("--policy_bc_risk_margin must be nonnegative")
if args.policy_bc_risk_margin >= 1.0:
    parser.error("--policy_bc_risk_margin must be below 1")
if args.policy_tilt_penalty_start_margin < 0.0 or args.policy_tilt_penalty_start_margin >= 1.0:
    parser.error("--policy_tilt_penalty_start_margin must be in [0, 1)")
if args.policy_failure_start_alternate_iters < 0:
    parser.error("--policy_failure_start_alternate_iters must be nonnegative")
if args.policy_wm_horizon_decay < 0.0:
    parser.error("--policy_wm_horizon_decay must be nonnegative")
if args.policy_failure_start_height_only and args.policy_failure_start_tilt_only:
    parser.error("choose at most one of --policy_failure_start_height_only and --policy_failure_start_tilt_only")
if args.wm_consistency_risk_weight < 0.0:
    parser.error("--wm_consistency_risk_weight must be nonnegative")
if args.wm_consistency_alignment_min < -1.0 or args.wm_consistency_alignment_min > 1.0:
    parser.error("--wm_consistency_alignment_min must be in [-1, 1]")
if args.wm_consistency_error_gate_ratio < 0.0:
    parser.error("--wm_consistency_error_gate_ratio must be nonnegative")
if not 0.0 <= args.wm_consistency_error_gate_beta < 1.0:
    parser.error("--wm_consistency_error_gate_beta must satisfy 0 <= beta < 1")
if args.wm_termination_weight < 0.0:
    parser.error("--wm_termination_weight must be nonnegative")
if not 0.0 <= args.wm_recent_fraction <= 1.0:
    parser.error("--wm_recent_fraction must be in [0, 1]")
if args.wm_occupancy_error_ratio < 0.0:
    parser.error("--wm_occupancy_error_ratio must be nonnegative")
if args.wm_transition_weight_scale < 0.0:
    parser.error("--wm_transition_weight_scale must be nonnegative")
if args.policy_realism_weight < 0.0:
    parser.error("--policy_realism_weight must be nonnegative")
if (args.online_realism_weight < 0.0 or args.online_realism_model_weight < 0.0
        or args.online_realism_reward_weight < 0.0 or args.online_realism_steps < 1):
    parser.error("online realism weight must be nonnegative and steps positive")
if args.online_realism_gate_temperature < 0.0:
    parser.error("--online_realism_gate_temperature must be nonnegative")
if args.online_realism_ess_gate < 0.0 or args.online_realism_ess_gate > 1.0:
    parser.error("--online_realism_ess_gate must be in [0, 1]")
if args.online_realism_ess_risk_floor < 0.0:
    parser.error("--online_realism_ess_risk_floor must be nonnegative")
if (args.online_realism_weight > 0.0 or args.online_realism_model_weight > 0.0
        or args.online_realism_reward_weight > 0.0) and args.policy_realism_checkpoint is not None:
    parser.error("online and fixed realism discriminators are mutually exclusive")
if not 0.0 <= args.wm_occupancy_validation_beta < 1.0:
    parser.error("--wm_occupancy_validation_beta must satisfy 0 <= beta < 1")
if args.ensemble_gate_nominal_prefix < 0 or args.ensemble_gate_nominal_prefix > args.horizon:
    parser.error("--ensemble_gate_nominal_prefix must be in [0, horizon]")
if args.dispersion_gate_temperature < 0.0:
    parser.error("--dispersion_gate_temperature must be non-negative")
if not 0.0 <= args.dispersion_gate_fraction <= 1.0:
    parser.error("--dispersion_gate_fraction must be in [0, 1]")
if args.dispersion_gate_warmup < 0:
    parser.error("--dispersion_gate_warmup must be non-negative")
if not 0.0 <= args.wm_ema_beta < 1.0:
    parser.error("--wm_ema_beta must satisfy 0 <= beta < 1")
if args.wm_priority_alpha < 0.0:
    parser.error("--wm_priority_alpha must be non-negative")
if args.wm_onpolicy_error_weight < 0.0:
    parser.error("--wm_onpolicy_error_weight must be non-negative")
if args.wm_sensitivity_weight < 0.0:
    parser.error("--wm_sensitivity_weight must be non-negative")
if args.wm_root_height_weight < 0.0:
    parser.error("--wm_root_height_weight must be non-negative")
if args.wm_angular_velocity_weight < 0.0:
    parser.error("--wm_angular_velocity_weight must be non-negative")
if not 0.0 <= args.deterministic_collect_fraction <= 1.0:
    parser.error("--deterministic_collect_fraction must be in [0, 1]")
if args.policy_height_penalty < 0.0:
    parser.error("--policy_height_penalty must be non-negative")
if args.policy_height_penalty_ramp_iters < 0:
    parser.error("--policy_height_penalty_ramp_iters must be non-negative")
if not 0.0 <= args.policy_height_penalty_start_margin < 1.0:
    parser.error("--policy_height_penalty_start_margin must be in [0, 1)")
if not 0.0 <= args.policy_height_hinge_margin < 1.0:
    parser.error("--policy_height_hinge_margin must be in [0, 1)")
if args.policy_tilt_penalty < 0.0:
    parser.error("--policy_tilt_penalty must be non-negative")
if args.policy_angular_velocity_penalty < 0.0:
    parser.error("--policy_angular_velocity_penalty must be non-negative")
if args.policy_angular_velocity_gate_margin < 0.0:
    parser.error("--policy_angular_velocity_gate_margin must be non-negative")
if args.shadow_policy_tilt_penalty < 0.0:
    parser.error("--shadow_policy_tilt_penalty must be non-negative")
if args.wm_rollout_sensitivity_weight < 0.0:
    parser.error("--wm_rollout_sensitivity_weight must be non-negative")
if args.wm_td_weight < 0.0:
    parser.error("--wm_td_weight must be non-negative")
if not 0.0 <= args.wm_td_target_beta < 1.0:
    parser.error("--wm_td_target_beta must satisfy 0 <= beta < 1")
if args.wm_td_weight > 0.0 and args.no_critic:
    parser.error("--wm_td_weight requires the critic")
if not 0.0 <= args.dispersion_gate_corr_min <= 1.0:
    parser.error("--dispersion_gate_corr_min must be in [0, 1]")
if not 0.0 <= args.dispersion_gate_corr_beta < 1.0:
    parser.error("--dispersion_gate_corr_beta must satisfy 0 <= beta < 1")
if args.dispersion_gate_corr_min > 0.0 and not args.wm_disagreement_diagnostic:
    parser.error("--dispersion_gate_corr_min requires --wm_disagreement_diagnostic")
if args.dispersion_return_penalty < 0.0:
    parser.error("--dispersion_return_penalty must be non-negative")
if args.dispersion_truncate_temperature < 0.0:
    parser.error("--dispersion_truncate_temperature must be nonnegative")
if args.dispersion_softmin_temperature < 0.0:
    parser.error("--dispersion_softmin_temperature must be nonnegative")
if args.policy_updates < 1:
    parser.error("--policy_updates must be at least 1")
if args.policy_updates_warmup_iters < 0:
    parser.error("--policy_updates_warmup_iters must be nonnegative")
if args.policy_freeze_after_iter < 0:
    parser.error("--policy_freeze_after_iter must be nonnegative")
if args.policy_second_update_scale < 0.0:
    parser.error("--policy_second_update_scale must be nonnegative")
if args.policy_lr_scale <= 0.0:
    parser.error("--policy_lr_scale must be positive")
if args.policy_period < 1:
    parser.error("--policy_period must be at least 1")
if args.policy_wm_error_gate_ratio < 0.0:
    parser.error("--policy_wm_error_gate_ratio must be nonnegative")
if args.policy_freeze_wm_error_ratio < 0.0:
    parser.error("--policy_freeze_wm_error_ratio must be nonnegative")
if not 0.0 <= args.policy_wm_error_gate_beta < 1.0:
    parser.error("--policy_wm_error_gate_beta must satisfy 0 <= beta < 1")
if not 0.0 <= args.policy_freeze_wm_error_beta < 1.0:
    parser.error("--policy_freeze_wm_error_beta must satisfy 0 <= beta < 1")
if args.policy_freeze_wm_error_min_iter < 0:
    parser.error("--policy_freeze_wm_error_min_iter must be nonnegative")
if not 0.0 <= args.policy_wm_reliability_quantile <= 1.0:
    parser.error("--policy_wm_reliability_quantile must be in [0, 1]")
if not 0.0 <= args.policy_wm_reliability_floor <= 1.0:
    parser.error("--policy_wm_reliability_floor must be in [0, 1]")
if args.policy_strategic_error_weight < 0.0:
    parser.error("--policy_strategic_error_weight must be nonnegative")
if args.policy_strategic_error_sigma <= 0.0:
    parser.error("--policy_strategic_error_sigma must be positive")
if args.policy_bc_weight < 0.0:
    parser.error("--policy_bc_weight must be nonnegative")
if args.policy_awbc_weight < 0.0 or args.policy_awbc_beta < 0.0:
    parser.error("policy AWBC parameters must be nonnegative")
if args.policy_proximal_weight < 0.0:
    parser.error("--policy_proximal_weight must be nonnegative")
if args.policy_ramp_iters < 0:
    parser.error("--policy_ramp_iters must be nonnegative")
if args.policy_ema_start < 0:
    parser.error("--policy_ema_start must be nonnegative")
if args.policy_target_audit_max_error < 0.0:
    parser.error("--policy_target_audit_max_error must be nonnegative")
if args.policy_target_audit_h1_max_error < 0.0 or args.policy_target_audit_h4_max_error < 0.0:
    parser.error("horizon-specific policy-target audit budgets must be nonnegative")
if args.policy_target_audit_interval <= 0:
    parser.error("--policy_target_audit_interval must be positive")
if args.policy_target_audit_horizon <= 0:
    parser.error("--policy_target_audit_horizon must be positive")
if args.policy_target_audit_horizon > args.wm_window:
    parser.error("--policy_target_audit_horizon cannot exceed --wm_window")
if args.policy_target_ema_interval <= 0:
    parser.error("--policy_target_ema_interval must be positive")
if args.policy_target_until_iter < 0:
    parser.error("--policy_target_until_iter must be nonnegative")
if args.policy_target_adapter_hidden < 0:
    parser.error("--policy_target_adapter_hidden must be nonnegative")
if args.policy_target_audit_ratio < 0.0:
    parser.error("--policy_target_audit_ratio must be nonnegative")
if args.policy_target_adapter_max_norm < 0.0:
    parser.error("--policy_target_adapter_max_norm must be nonnegative")
if args.policy_target_adapter_l2 < 0.0:
    parser.error("--policy_target_adapter_l2 must be nonnegative")
if args.policy_target_adapter_lr < 0.0:
    parser.error("--policy_target_adapter_lr must be nonnegative")
if args.policy_target_adapter_max_delta < 0.0:
    parser.error("--policy_target_adapter_max_delta must be nonnegative")
if args.policy_target_kl_budget < 0.0:
    parser.error("--policy_target_kl_budget must be nonnegative")
if args.policy_target_kl_dual_lr < 0.0:
    parser.error("--policy_target_kl_dual_lr must be nonnegative")
if args.shadow_policy and args.policy_target_audit_max_error <= 0.0:
    parser.error("--shadow_policy requires --policy_target_audit_max_error > 0")
if args.shadow_policy_lr_scale <= 0.0:
    parser.error("--shadow_policy_lr_scale must be positive")
if args.shadow_policy_updates <= 0:
    parser.error("--shadow_policy_updates must be positive")
if args.shadow_policy_max_step_norm < 0.0:
    parser.error("--shadow_policy_max_step_norm must be nonnegative")
if args.imag_action_noise < 0.0:
    parser.error("--imag_action_noise must be nonnegative")
if args.adaptive_action_noise < 0.0:
    parser.error("--adaptive_action_noise must be nonnegative")
if args.adaptive_action_noise_threshold < 0.0:
    parser.error("--adaptive_action_noise_threshold must be nonnegative")
if args.adaptive_action_direction < 0.0:
    parser.error("--adaptive_action_direction must be nonnegative")
if not 0.0 <= args.adaptive_action_direction_fraction <= 1.0:
    parser.error("--adaptive_action_direction_fraction must be in [0, 1]")
if args.adaptive_action_noise > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--adaptive_action_noise requires both ensemble member checkpoints")
if args.adaptive_action_direction > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--adaptive_action_direction requires both ensemble member checkpoints")
if args.adaptive_noise_recoverability < 0.0:
    parser.error("--adaptive_noise_recoverability must be nonnegative")
if args.term_soft_height <= 0.0 or args.term_soft_tilt <= 0.0:
    parser.error("soft termination widths must be positive")
if (args.wm_aux_window < 0 or args.wm_aux_weight < 0.0 or args.wm_aux_ramp_iters < 0
        or args.wm_aux_window2 < 0 or args.wm_aux_weight2 < 0.0 or args.wm_aux_switch_iters < 0):
    parser.error("auxiliary WM window and weight must be nonnegative")
if args.wm_aux_weight > 0.0 and args.wm_aux_window < 1:
    parser.error("--wm_aux_weight requires --wm_aux_window")
if args.wm_aux_weight2 > 0.0 and args.wm_aux_window2 < 1:
    parser.error("--wm_aux_weight2 requires --wm_aux_window2")
if args.policy_bptt_truncation < 0:
    parser.error("--policy_bptt_truncation must be nonnegative")
if args.joint_wm_policy_weight < 0.0:
    parser.error("--joint_wm_policy_weight must be nonnegative")
if args.joint_wm_policy_action_norm_ratio < 0.0:
    parser.error("--joint_wm_policy_action_norm_ratio must be nonnegative")
if (args.joint_wm_policy_last_layer or args.joint_wm_policy_root_outputs) and args.joint_wm_policy_action_path:
    parser.error("output-row and action-path concurrent masks are mutually exclusive")
if args.joint_wm_policy_root_outputs and args.joint_wm_policy_last_layer:
    parser.error("--joint_wm_policy_root_outputs already selects the WM output layer")
if args.joint_wm_replay_weight < 0.0:
    parser.error("--joint_wm_replay_weight must be nonnegative")
if args.joint_replace_wm_step and args.joint_wm_replay_weight <= 0.0:
    parser.error("--joint_replace_wm_step requires --joint_wm_replay_weight")
if args.joint_replace_wm_step and args.wm_batches < 2:
    parser.error("--joint_replace_wm_step requires wm_batches >= 2")
if args.wm_occupancy_weight < 0.0 or args.wm_occupancy_sigma <= 0.0:
    parser.error("occupancy weighting parameters must be nonnegative with sigma positive")
if args.wm_multistep_weight < 0.0 or args.wm_multistep_horizon < 2:
    parser.error("multi-step target requires nonnegative weight and horizon >= 2")
if args.wm_multistep_weight > 0.0 and args.wm_multistep_horizon > args.wm_window:
    parser.error("multi-step horizon cannot exceed wm_window")
if args.wm_policy_aware_weight < 0.0 or args.wm_policy_aware_sigma <= 0.0:
    parser.error("policy-aware weighting must be nonnegative with positive sigma")
if args.wm_policy_aware_uncertainty < 0.0:
    parser.error("--wm_policy_aware_uncertainty must be nonnegative")
if not 0.0 <= args.wm_policy_aware_floor <= 1.0:
    parser.error("--wm_policy_aware_floor must be in [0, 1]")
if args.wm_policy_aware_validation_ratio < 0.0:
    parser.error("--wm_policy_aware_validation_ratio must be nonnegative")
if args.strategic_probe_capacity < 0 or args.strategic_probe_weight < 0.0:
    parser.error("strategic probe capacity and weight must be nonnegative")
if args.strategic_probe_batch < 0 or args.strategic_probe_alpha < 0.0:
    parser.error("strategic probe batch and alpha must be nonnegative")
if args.strategic_probe_weight > 0.0 and args.strategic_probe_capacity < 1:
    parser.error("--strategic_probe_weight requires --strategic_probe_capacity > 0")
if args.wm_reward_weight < 0.0:
    parser.error("--wm_reward_weight must be nonnegative")
if args.wm_reward_mismatch_weight < 0.0:
    parser.error("--wm_reward_mismatch_weight must be nonnegative")
if args.wm_huber_delta < 0.0:
    parser.error("--wm_huber_delta must be nonnegative")
if args.wm_action_effect_gain <= 0.0:
    parser.error("--wm_action_effect_gain must be positive")
if args.wm_counterfactual_weight < 0.0:
    parser.error("--wm_counterfactual_weight must be nonnegative")
if args.wm_counterfactual_effect_weight < 0.0:
    parser.error("--wm_counterfactual_effect_weight must be nonnegative")
if args.wm_counterfactual_norm_weight < 0.0:
    parser.error("--wm_counterfactual_norm_weight must be nonnegative")
if args.wm_counterfactual_tilt_weight < 0.0:
    parser.error("--wm_counterfactual_tilt_weight must be nonnegative")
if args.wm_counterfactual_relative_weight < 0.0:
    parser.error("--wm_counterfactual_relative_weight must be nonnegative")
if args.wm_counterfactual_relative_floor < 0.0 or args.wm_counterfactual_relative_floor > 1.0:
    parser.error("--wm_counterfactual_relative_floor must be in [0, 1]")
if args.wm_counterfactual_relative_weight > 0.0 and args.wm_counterfactual_reference_checkpoint is None:
    parser.error("relative sensitivity requires --wm_counterfactual_reference_checkpoint")
if args.wm_counterfactual_gram_weight < 0.0:
    parser.error("--wm_counterfactual_gram_weight must be nonnegative")
if args.wm_counterfactual_risk_gradient_weight < 0.0:
    parser.error("--wm_counterfactual_risk_gradient_weight must be nonnegative")
if args.wm_counterfactual_risk_gradient_tilt_scale < 0.0:
    parser.error("--wm_counterfactual_risk_gradient_tilt_scale must be nonnegative")
if args.wm_counterfactual_batch < 0:
    parser.error("--wm_counterfactual_batch must be nonnegative")
if (args.wm_counterfactual_weight > 0.0 or args.wm_counterfactual_effect_weight > 0.0
        or args.wm_counterfactual_norm_weight > 0.0
        or args.wm_counterfactual_tilt_weight > 0.0
        or args.wm_counterfactual_geometry_weight > 0.0
        or args.wm_counterfactual_geometry_mag_weight > 0.0
        or args.wm_counterfactual_relative_weight > 0.0
        or args.wm_counterfactual_gram_weight > 0.0
        or args.wm_counterfactual_risk_gradient_weight > 0.0) and args.wm_counterfactual_dataset is None:
    parser.error("counterfactual weights require --wm_counterfactual_dataset")
if args.wm_action_delta_calibration is not None and args.wm_intervention_adapter_checkpoint is not None:
    parser.error("delta calibration and intervention adapter are mutually exclusive")
if not 0.0 <= args.wm_intervention_adapter_blend <= 1.0:
    parser.error("--wm_intervention_adapter_blend must be in [0, 1]")
if args.wm_intervention_adapter_max_ratio < 0.0:
    parser.error("--wm_intervention_adapter_max_ratio must be nonnegative")
if args.wm_intervention_adapter_gate_ratio < 0.0:
    parser.error("--wm_intervention_adapter_gate_ratio must be nonnegative")
if args.wm_intervention_adapter_horizon < 1:
    parser.error("--wm_intervention_adapter_horizon must be positive")
if args.wm_counterfactual_error_gate < 0.0:
    parser.error("--wm_counterfactual_error_gate must be nonnegative")
if args.wm_counterfactual_geometry_weight < 0.0 or args.wm_counterfactual_geometry_mag_weight < 0.0:
    parser.error("counterfactual geometry weights must be nonnegative")
if args.wm_action_recovery_weight < 0.0 or args.wm_action_recovery_hidden < 1:
    parser.error("action-recovery weight must be nonnegative and hidden width positive")
if args.wm_action_recovery_fidelity_gate < 0.0:
    parser.error("--wm_action_recovery_fidelity_gate must be nonnegative")
if not -1.0 <= args.wm_action_recovery_alignment_min <= 1.0:
    parser.error("--wm_action_recovery_alignment_min must be in [-1, 1]")
if args.wm_action_recovery_project_grad and args.wm_action_recovery_weight <= 0.0:
    parser.error("--wm_action_recovery_project_grad requires --wm_action_recovery_weight > 0")
if (args.wm_action_contrastive_weight < 0.0 or args.wm_action_contrastive_hidden < 1
        or args.wm_action_contrastive_temperature <= 0.0):
    parser.error("contrastive action parameters are invalid")
if not -1.0 <= args.wm_action_contrastive_corr_gate <= 1.0:
    parser.error("--wm_action_contrastive_corr_gate must be in [-1, 1]")
if not 0.0 <= args.wm_occupancy_floor <= 1.0:
    parser.error("wm_occupancy_floor must be in [0, 1]")
if args.wm_occupancy_weight > 0.0 and any((args.wm_sensitivity_weight > 0.0,
                                          args.wm_rollout_sensitivity_weight > 0.0,
                                          args.wm_td_weight > 0.0,
                                          args.wm_reward_weight > 0.0,
                                          args.wm_prioritized)):
    parser.error("occupancy weighting is mutually exclusive with other WM weighting modes")
if args.wm_policy_aware_weight > 0.0 and any((args.wm_occupancy_weight > 0.0,
                                              args.wm_sensitivity_weight > 0.0,
                                              args.wm_rollout_sensitivity_weight > 0.0,
                                              args.wm_td_weight > 0.0,
                                              args.wm_reward_weight > 0.0,
                                              args.wm_prioritized,
                                              args.wm_ensemble_disagreement_weight > 0.0,
                                              args.wm_transition_weight_checkpoint is not None,
                                              args.wm_dual_q_weight_checkpoint is not None)):
    parser.error("policy-aware weighting is mutually exclusive with other WM weighting modes")
if args.wm_reward_mismatch_weight > 0.0 and any((args.wm_occupancy_weight > 0.0,
                                                 args.wm_policy_aware_weight > 0.0,
                                                 args.wm_sensitivity_weight > 0.0,
                                                 args.wm_rollout_sensitivity_weight > 0.0,
                                                 args.wm_td_weight > 0.0,
                                                 args.wm_reward_weight > 0.0,
                                                 args.wm_prioritized,
                                                 args.wm_ensemble_disagreement_weight > 0.0,
                                                 args.wm_transition_weight_checkpoint is not None,
                                                 args.wm_dual_q_weight_checkpoint is not None)):
    parser.error("reward-mismatch weighting is mutually exclusive with other WM weighting modes")

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, NUM_J, OBS_DIM, flatten_state, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.dispersion import ema_update_, gate_weight, quantile_gate_weight
from supermimic.policy_target import PolicyTargetWorldModel
from supermimic.transition_weight import TransitionOdds
from supermimic.shadow_policy import ShadowPolicyBranch, cap_parameter_step
from supermimic.strategic_probe import StrategicProbeBuffer
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import so3_exp, yaw_mat

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
# Separate task/replay gradient diagnostics need a retained compiled graph;
# disable AOT donated buffers only for this opt-in diagnostic mode.
if (args.joint_gradient_diagnostic or args.joint_gradient_projection or args.joint_gradient_conflict_gate
        or args.joint_gradient_norm_match or args.joint_adversarial_residual or args.joint_residual_dual
        or args.wm_action_recovery_alignment_min >= 0.0 or args.wm_action_recovery_project_grad
        or args.wm_consistency_alignment_min >= 0.0):
    import torch._functorch.config as _functorch_config
    _functorch_config.donated_buffer = False
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
          phase_features=args.phase_conditioned_policy,
          clip_features=args.clip_conditioned_policy,
          contact_safe_reference=args.contact_safe_reward,
          contact_masked_reward=args.contact_masked_reward,
          contact_features=args.contact_conditioned_policy,
          current_contact_features=args.current_contact_policy)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

if args.joint_residual_hidden < 0:
    parser.error("--joint_residual_hidden must be nonnegative")
if args.joint_residual_occupancy_sigma < 0.0:
    parser.error("--joint_residual_occupancy_sigma must be nonnegative")
if args.joint_residual_max_norm < 0.0:
    parser.error("--joint_residual_max_norm must be nonnegative")
if args.joint_residual_l2 < 0.0:
    parser.error("--joint_residual_l2 must be nonnegative")
if args.joint_residual_dual_lr < 0.0:
    parser.error("--joint_residual_dual_lr must be nonnegative")
if args.joint_residual_dual_target < 0.0:
    parser.error("--joint_residual_dual_target must be nonnegative")
if args.joint_residual_grad_clip < 0.0:
    parser.error("--joint_residual_grad_clip must be nonnegative")
if (args.ensemble_gate_member0 is None) != (args.ensemble_gate_member1 is None):
    parser.error("ensemble gate requires both member checkpoints")
if args.ensemble_gate_threshold < 0.0:
    parser.error("--ensemble_gate_threshold must be nonnegative")
if args.ensemble_uncertainty_weight < 0.0:
    parser.error("--ensemble_uncertainty_weight must be nonnegative")
if args.ensemble_uncertainty_weight > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--ensemble_uncertainty_weight requires both ensemble member checkpoints")
if not 0.0 <= args.ensemble_policy_gate_fraction <= 1.0:
    parser.error("--ensemble_policy_gate_fraction must be in [0, 1]")
if args.ensemble_policy_gate_horizon < 0 or args.ensemble_policy_gate_horizon > args.horizon:
    parser.error("--ensemble_policy_gate_horizon must be in [0, horizon]")
if args.ensemble_policy_gate_fraction > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--ensemble_policy_gate_fraction requires both ensemble member checkpoints")
if args.ensemble_policy_softmin_temperature < 0.0:
    parser.error("--ensemble_policy_softmin_temperature must be nonnegative")
if args.ensemble_policy_softmin_temperature > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--ensemble_policy_softmin_temperature requires both ensemble member checkpoints")
if args.wm_ensemble_disagreement_weight < 0.0:
    parser.error("--wm_ensemble_disagreement_weight must be nonnegative")
if args.wm_ensemble_disagreement_weight > 0.0 and args.ensemble_gate_member0 is None:
    parser.error("--wm_ensemble_disagreement_weight requires both ensemble member checkpoints")
if args.joint_residual_audit_threshold < 0.0:
    parser.error("--joint_residual_audit_threshold must be nonnegative")
if args.joint_wm_replay_window < 0:
    parser.error("--joint_wm_replay_window must be nonnegative")
if args.joint_wm_policy_transition_ratio < 0.0:
    parser.error("--joint_wm_policy_transition_ratio must be nonnegative")
if args.joint_wm_policy_horizon < 0:
    parser.error("--joint_wm_policy_horizon must be nonnegative")
if args.joint_wm_replay_sensitivity < 0.0:
    parser.error("--joint_wm_replay_sensitivity must be nonnegative")
if args.joint_gradient_norm_ratio < 0.0:
    parser.error("--joint_gradient_norm_ratio must be nonnegative")
if args.joint_wm_policy_period <= 0:
    parser.error("--joint_wm_policy_period must be positive")
if args.joint_wm_policy_start_iter < 0:
    parser.error("--joint_wm_policy_start_iter must be nonnegative")
if not 0.0 <= args.joint_gradient_task_ema < 1.0:
    parser.error("--joint_gradient_task_ema must be in [0, 1)")
if args.onpolicy_refresh_steps < 0:
    parser.error("--onpolicy_refresh_steps must be nonnegative")
if args.onpolicy_refresh_steps > 0 and not args.policy_before_wm:
    parser.error("--onpolicy_refresh_steps requires --policy_before_wm")
if not 0.0 <= args.onpolicy_replay_fraction <= 1.0:
    parser.error("--onpolicy_replay_fraction must be in [0, 1]")
if args.onpolicy_replay_fraction > 0.0 and args.onpolicy_refresh_steps <= 0:
    parser.error("--onpolicy_replay_fraction requires --onpolicy_refresh_steps > 0")
if args.onpolicy_buffer_steps < 0:
    parser.error("--onpolicy_buffer_steps must be nonnegative")
if args.onpolicy_separate_wm_update and args.onpolicy_replay_fraction <= 0.0:
    parser.error("--onpolicy_separate_wm_update requires --onpolicy_replay_fraction > 0")
if args.onpolicy_separate_wm_update:
    required_probe_window = max(args.wm_window, args.wm_aux_window, args.wm_aux_window2)
    if args.onpolicy_refresh_steps < required_probe_window:
        parser.error(
            "--onpolicy_separate_wm_update requires --onpolicy_refresh_steps >= "
            f"max(wm_window, wm_aux_window, wm_aux_window2) ({required_probe_window})")
if (args.wm_policy_aware_weight > 0.0
        and (args.onpolicy_replay_fraction > 0.0 or args.onpolicy_separate_wm_update)):
    parser.error("policy-aware weighting currently requires historical replay only")
if (args.joint_residual_hidden > 0
        and args.joint_wm_policy_weight <= 0.0
        and args.joint_wm_replay_weight <= 0.0):
    parser.error("--joint_residual_hidden requires a positive joint WM policy or replay weight")
if args.joint_adversarial_residual and args.joint_residual_hidden <= 0:
    parser.error("--joint_adversarial_residual requires --joint_residual_hidden > 0")
if args.joint_residual_occupancy_sigma > 0.0 and args.joint_residual_hidden <= 0:
    parser.error("--joint_residual_occupancy_sigma requires --joint_residual_hidden > 0")
if args.joint_wm_policy_horizon > 0 and args.joint_wm_policy_horizon > args.horizon:
    parser.error("--joint_wm_policy_horizon cannot exceed --horizon")
if args.joint_residual_dual and args.joint_residual_hidden <= 0:
    parser.error("--joint_residual_dual requires --joint_residual_hidden > 0")
wm = WorldModel(residual_hidden=args.joint_residual_hidden,
                residual_max_norm=args.joint_residual_max_norm).to(dev)
online_realism_input_dim = 70 + 29 if args.online_realism_local_delta else 2 * 71 + 29

def _online_realism_features(s0_flat, action, s1_flat):
    """Build discriminator features without exposing absolute state when requested."""
    if args.online_realism_local_delta:
        delta = wm.state_error(unflatten_state(s1_flat), unflatten_state(s0_flat))
        return torch.cat((delta, action), -1)
    return torch.cat((s0_flat, action, s1_flat), -1)
wm_counterfactual_reference = None
if args.wm_counterfactual_reference_checkpoint is not None:
    rck = torch.load(args.wm_counterfactual_reference_checkpoint, map_location=dev, weights_only=False)
    rst = rck.get("wm")
    if rst is None:
        parser.error("counterfactual reference checkpoint must contain wm")
    rh = int(rst["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in rst else 0
    wm_counterfactual_reference = WorldModel(residual_hidden=rh).to(dev)
    wm_counterfactual_reference.load_state_dict(rst); wm_counterfactual_reference.eval()
    for p in wm_counterfactual_reference.parameters(): p.requires_grad_(False)
wm_action_delta_gain = None
if args.wm_action_delta_calibration is not None:
    try:
        cal = json.load(open(args.wm_action_delta_calibration))
        gain = cal.get("component_gain")
        if gain is None or len(gain) != 70:
            raise ValueError("component_gain must contain 70 values")
        wm_action_delta_gain = torch.tensor(gain, device=dev, dtype=torch.float32).clamp(0.25, 4.0)
    except Exception as exc:
        parser.error(f"invalid --wm_action_delta_calibration: {exc}")
wm_intervention_adapter = None
wm_intervention_mean = wm_intervention_std = None
if args.wm_intervention_adapter_checkpoint is not None:
    ia = torch.load(args.wm_intervention_adapter_checkpoint, map_location=dev, weights_only=False)
    if "model" not in ia or "input_mean" not in ia or "input_std" not in ia or int(ia.get("input_dim", -1)) != 170:
        parser.error("intervention adapter checkpoint must contain a 170-D model and input statistics")
    wm_intervention_adapter = nn.Sequential(
        nn.Linear(170, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 70)
).to(dev)
    wm_intervention_adapter.load_state_dict(ia["model"])
    wm_intervention_adapter.eval()
    for p in wm_intervention_adapter.parameters():
        p.requires_grad_(False)
    wm_intervention_mean = ia["input_mean"].to(dev)
    wm_intervention_std = ia["input_std"].to(dev).clamp_min(1e-4)
counterfactual_data = None
counterfactual_motion = None
if args.wm_counterfactual_dataset is not None:
    if not os.path.exists(args.wm_counterfactual_dataset):
        parser.error(f"counterfactual dataset not found: {args.wm_counterfactual_dataset}")
    cf = torch.load(args.wm_counterfactual_dataset, map_location=dev, weights_only=False)
    if not all(k in cf for k in ("s0", "xi", "action", "directions", "pairs")):
        # The generated format stores s0 inside each pair; retain a compact
        # normalized view here for the training update.
        if not all(k in cf for k in ("xi", "action", "directions", "pairs")):
            parser.error("counterfactual dataset is missing required fields")
    counterfactual_data = cf
    if args.wm_counterfactual_risk_gradient_weight > 0.0:
        eval_dir = cf.get("eval_motion_dir")
        if not eval_dir:
            parser.error("risk-gradient counterfactual training requires eval_motion_dir metadata")
        counterfactual_motion = MotionLib.from_dir(eval_dir, kin, dev)
wm_action_recovery = (nn.Sequential(
    nn.Linear(2 * 71, args.wm_action_recovery_hidden), nn.SiLU(), nn.Linear(args.wm_action_recovery_hidden, 29)
).to(dev) if args.wm_action_recovery_weight > 0.0 else None)
wm_action_transition_encoder = (nn.Sequential(
    nn.Linear(70 if args.wm_action_contrastive_delta else 2 * 71, args.wm_action_contrastive_hidden), nn.SiLU(), nn.Linear(args.wm_action_contrastive_hidden, args.wm_action_contrastive_hidden)
).to(dev) if args.wm_action_contrastive_weight > 0.0 else None)
wm_action_encoder = (nn.Sequential(
    nn.Linear(29, args.wm_action_contrastive_hidden), nn.SiLU(), nn.Linear(args.wm_action_contrastive_hidden, args.wm_action_contrastive_hidden)
).to(dev) if args.wm_action_contrastive_weight > 0.0 else None)
online_realism_model = None
online_realism_opt = None
online_realism_mean = online_realism_std = None
if (args.online_realism_weight > 0.0 or args.online_realism_model_weight > 0.0
        or args.online_realism_reward_weight > 0.0 or args.online_realism_ess_gate > 0.0):
    # The optional discriminator must not change the initialization stream of
    # the policy/WM. Otherwise a gate-vs-control comparison confounds the
    # intervention with different random initial weights.
    _rng_cpu = torch.get_rng_state()
    _rng_cuda = torch.cuda.get_rng_state(dev) if str(dev).startswith("cuda") else None
    online_realism_model = nn.Sequential(
        nn.Linear(online_realism_input_dim, 128), nn.SiLU(), nn.Linear(128, 1)
    ).to(dev)
    online_realism_opt = torch.optim.AdamW(
        online_realism_model.parameters(), lr=3e-4, weight_decay=1e-4)
    online_realism_mean = torch.zeros(online_realism_input_dim, device=dev)
    online_realism_std = torch.ones(online_realism_input_dim, device=dev)
    torch.set_rng_state(_rng_cpu)
    if _rng_cuda is not None:
        torch.cuda.set_rng_state(_rng_cuda, dev)
transition_weight_model = None
dual_q_weight_table = None
policy_realism_model = None
policy_realism_mean = policy_realism_std = None
joint_residual_validity_model = None
joint_residual_validity_threshold = 0.0
if args.wm_transition_weight_checkpoint is not None:
    tw_ck = torch.load(args.wm_transition_weight_checkpoint, map_location=dev, weights_only=False)
    try:
        transition_weight_model = TransitionOdds(tw_ck, device=dev)
    except ValueError as exc:
        parser.error(str(exc))
if args.wm_dual_q_weight_checkpoint is not None:
    dq_ck = torch.load(args.wm_dual_q_weight_checkpoint, map_location=dev, weights_only=False)
    dual_q_weight_table = dq_ck.get("weights")
    if dual_q_weight_table is None or dual_q_weight_table.ndim != 2:
        parser.error("dual-Q weighting checkpoint is missing a 2D weights table")
    dual_q_weight_table = dual_q_weight_table.to(dev).float()
if args.policy_realism_checkpoint is not None:
    real_ck = torch.load(args.policy_realism_checkpoint, map_location=dev, weights_only=False)
    real_in = int(real_ck.get("input_dim", -1)); real_hidden = int(real_ck.get("hidden", 128))
    if real_in <= 0 or "model" not in real_ck or "mean" not in real_ck or "std" not in real_ck:
        parser.error("realism checkpoint is missing model, input_dim, mean, or std")
    policy_realism_model = nn.Sequential(
        nn.Linear(real_in, real_hidden), nn.SiLU(), nn.Linear(real_hidden, 1)
    ).to(dev)
    policy_realism_model.load_state_dict(real_ck["model"])
    for p in policy_realism_model.parameters():
        p.requires_grad_(False)
    policy_realism_model.eval()
    policy_realism_mean = real_ck["mean"].to(dev)
    policy_realism_std = real_ck["std"].to(dev).clamp_min(1e-4)
if args.joint_residual_validity_checkpoint is not None:
    valid_ck = torch.load(args.joint_residual_validity_checkpoint, map_location=dev, weights_only=False)
    valid_in = int(valid_ck.get("input_dim", -1))
    if valid_in <= 0 or "model" not in valid_ck:
        parser.error("validity checkpoint is missing model or input_dim")
    joint_residual_validity_model = nn.Sequential(
        nn.Linear(valid_in, 64), nn.SiLU(), nn.Linear(64, 1)
    ).to(dev)
    joint_residual_validity_model.load_state_dict(valid_ck["model"])
    for p in joint_residual_validity_model.parameters():
        p.requires_grad_(False)
    joint_residual_validity_model.eval()
    joint_residual_validity_threshold = float(valid_ck.get("threshold", 0.5))
    if not 0.0 < joint_residual_validity_threshold < 1.0:
        parser.error("validity checkpoint threshold must be in (0,1)")
ensemble_gate_wms = None
ensemble_gate_residual = False
if args.ensemble_gate_member0 is not None:
    gate_models = []
    gate_has_residual = []
    for path in (args.ensemble_gate_member0, args.ensemble_gate_member1):
        gate_ck = torch.load(path, map_location=dev, weights_only=False)
        gate_state = gate_ck["wm"]
        key = "residual_net.0.weight"
        has_residual = key in gate_state
        gate_wm = WorldModel(residual_hidden=(int(gate_state[key].shape[0]) if has_residual else 0)).to(dev)
        gate_wm.load_state_dict(gate_state)
        for p in gate_wm.parameters():
            p.requires_grad_(False)
        gate_wm.eval()
        gate_models.append(gate_wm)
        gate_has_residual.append(has_residual)
    if len(set(gate_has_residual)) != 1:
        parser.error("ensemble members must either both have or both lack residual branches")
    ensemble_gate_residual = gate_has_residual[0]
    ensemble_gate_wms = tuple(gate_models)
wm_norm_frozen = args.wm_norm_checkpoint is not None
if wm_norm_frozen:
    norm_ck = torch.load(args.wm_norm_checkpoint, map_location=dev)
    norm_state = norm_ck.get("wm", norm_ck)
    for name in ("in_norm", "out_norm"):
        module = getattr(wm, name)
        for field in ("mean", "var", "count"):
            key = f"{name}.{field}"
            if key not in norm_state:
                parser.error(f"normalization checkpoint is missing {key}")
            getattr(module, field).copy_(norm_state[key])
policy_target_wrapper = (PolicyTargetWorldModel(
    wm, ema_beta=args.wm_ema_beta, adapter_hidden=args.policy_target_adapter_hidden,
    adapter_max_delta=args.policy_target_adapter_max_delta)
                         if (args.policy_target_audit_max_error > 0.0
                             or args.policy_target_adapter_hidden > 0) else None)
wm_target = (policy_target_wrapper.model if policy_target_wrapper is not None else
             WorldModel(residual_hidden=args.joint_residual_hidden,
                        residual_max_norm=args.joint_residual_max_norm).to(dev)) if (args.policy_ema_model or policy_target_wrapper is not None or args.dispersion_gate > 0.0 or args.wm_disagreement_diagnostic or args.dispersion_return_penalty > 0.0 or args.dispersion_truncate_temperature > 0.0 or args.dispersion_softmin_temperature > 0.0) else None
if wm_target is not None:
    wm_target.load_state_dict(wm.state_dict())
policy_obs_dim = OBS_DIM + (2 if args.phase_conditioned_policy else 0)
policy_obs_dim += len(motion.names) if args.clip_conditioned_policy else 0
policy_obs_dim += len(motion.body_p[0]) if args.contact_conditioned_policy else 0
policy_obs_dim += len(motion.body_p[0]) if args.current_contact_policy else 0
pi = Policy(policy_obs_dim, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
critic_target = Critic(CRITIC_IN).to(dev) if args.wm_td_weight > 0.0 else None
if critic_target is not None:
    critic_target.load_state_dict(critic.state_dict())
wm_opt_params = list(wm.parameters())
for _m in (wm_action_recovery, wm_action_transition_encoder, wm_action_encoder):
    if _m is not None:
        wm_opt_params += list(_m.parameters())
opt_wm = torch.optim.Adam(wm_opt_params, lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr * args.policy_lr_scale, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr * args.policy_lr_scale)
shadow_branch = ShadowPolicyBranch(pi, critic) if args.shadow_policy else None
shadow_opt_pi = (torch.optim.Adam(shadow_branch.shadow_policy.parameters(), lr=args.lr * args.shadow_policy_lr_scale, betas=(0.9, 0.95))
                 if shadow_branch is not None else None)
shadow_opt_v = (torch.optim.Adam(shadow_branch.shadow_critic.parameters(), lr=args.lr * args.shadow_policy_lr_scale)
                if shadow_branch is not None else None)
shadow_critic_target = (Critic(CRITIC_IN).to(dev) if shadow_branch is not None and args.wm_td_weight > 0.0 else None)
if shadow_critic_target is not None:
    shadow_critic_target.load_state_dict(shadow_branch.shadow_critic.state_dict())
opt_target = (torch.optim.Adam(policy_target_wrapper.adapter_parameters(),
                               lr=(args.policy_target_adapter_lr or args.lr))
              if policy_target_wrapper is not None and policy_target_wrapper.adapter_parameters()
              else None)
start_it, wall_offset, sim_steps = 0, 0.0, 0
policy_target_audit_passes = 0
policy_target_audit_failures = 0
policy_target_last_good = None
policy_target_audit_h1 = float("inf")
policy_target_audit_h4 = float("inf")
policy_target_nominal_h1 = float("inf")
policy_target_nominal_h4 = float("inf")
policy_target_kl_dual = 0.0
policy_target_control_token = None
policy_target_external_allowed = True
policy_shadow_active = False
if policy_target_wrapper is not None and policy_target_wrapper.adapter is not None:
    policy_target_last_good = policy_target_wrapper.snapshot_adapter()
resume_buffer = None
resume_probe = None
diag_corr_ema = 0.0
resume_dual = resume_dual_reference = None

def _restore_optimizer_state(optimizer, state, name):
    """Restore optimizer state when its parameter groups still match.

    Continuations may intentionally add an auxiliary module. In that case the
    model weights are reusable but the old optimizer groups are not; keep the
    freshly initialized optimizer instead of aborting the whole continuation.
    """
    if state is None:
        return False
    try:
        optimizer.load_state_dict(state)
    except ValueError as exc:
        print(f"warning: resetting {name} optimizer state ({exc})")
        return False
    return True

if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"])
    if wm_action_recovery is not None and ck.get("wm_action_recovery") is not None:
        wm_action_recovery.load_state_dict(ck["wm_action_recovery"])
    if wm_action_transition_encoder is not None and ck.get("wm_action_transition_encoder") is not None:
        wm_action_transition_encoder.load_state_dict(ck["wm_action_transition_encoder"])
    if wm_action_encoder is not None and ck.get("wm_action_encoder") is not None:
        wm_action_encoder.load_state_dict(ck["wm_action_encoder"])
    pi.load_state_dict(ck["pi"])
    # Older no-critic checkpoints contain only WM/policy weights.  Keep the
    # freshly initialized critic and optimizer state when those optional keys
    # are absent so long-budget continuation remains backward compatible.
    if "critic" in ck:
        critic.load_state_dict(ck["critic"])
    if shadow_branch is not None and ck.get("shadow_policy") is not None:
        shadow_branch.load_state_dict(ck["shadow_policy"])
    if wm_target is not None:
        # Older checkpoints predate the optional target model.
        wm_target.load_state_dict(ck.get("wm_target") or ck["wm"])
    if policy_target_wrapper is not None:
        policy_target_wrapper.enabled = bool(ck.get("policy_target_enabled", False))
        if "policy_target_kl_proxy" in ck:
            policy_target_wrapper.replay_kl_proxy.copy_(torch.as_tensor(
                ck["policy_target_kl_proxy"], device=dev))
        if "policy_target_kl_dual" in ck:
            policy_target_kl_dual = float(ck["policy_target_kl_dual"])
        if "policy_target_audit_error" in ck:
            policy_target_wrapper.audit_error.copy_(torch.as_tensor(
                ck["policy_target_audit_error"], device=dev))
        if (policy_target_wrapper.adapter is not None
                and ck.get("policy_target_adapter") is not None):
            policy_target_wrapper.adapter.load_state_dict(ck["policy_target_adapter"])
            saved_good = ck.get("policy_target_last_good")
            policy_target_last_good = {
                k: v.detach().clone() for k, v in (saved_good or policy_target_wrapper.adapter.state_dict()).items()
            }
        policy_target_audit_passes = int(ck.get("policy_target_audit_passes", 0))
        policy_target_audit_failures = int(ck.get("policy_target_audit_failures", 0))
        policy_target_audit_h1 = float(ck.get("policy_target_audit_h1", float("inf")))
        policy_target_audit_h4 = float(ck.get("policy_target_audit_h4", float("inf")))
        policy_target_nominal_h1 = float(ck.get("policy_target_nominal_h1", float("inf")))
        policy_target_nominal_h4 = float(ck.get("policy_target_nominal_h4", float("inf")))
    if critic_target is not None:
        critic_target.load_state_dict(ck.get("critic_target") or ck.get("critic", critic.state_dict()))
    if "opt_wm" in ck:
        _restore_optimizer_state(opt_wm, ck["opt_wm"], "world-model")
    if "opt_pi" in ck:
        _restore_optimizer_state(opt_pi, ck["opt_pi"], "policy")
    if "opt_v" in ck:
        _restore_optimizer_state(opt_v, ck["opt_v"], "value")
    if shadow_opt_pi is not None and ck.get("shadow_opt_pi") is not None:
        _restore_optimizer_state(shadow_opt_pi, ck["shadow_opt_pi"], "shadow-policy")
    if shadow_opt_v is not None and ck.get("shadow_opt_v") is not None:
        _restore_optimizer_state(shadow_opt_v, ck["shadow_opt_v"], "shadow-value")
    if opt_target is not None and ck.get("opt_target") is not None:
        _restore_optimizer_state(opt_target, ck["opt_target"], "target")
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    resume_buffer = ck.get("buffer")
    resume_probe = ck.get("strategic_probe")
    resume_dual = ck.get("residual_dual")
    resume_dual_reference = ck.get("residual_dual_reference")


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip
WM_COMPONENT_WEIGHT = torch.ones_like(ERR_FLOOR)
if args.wm_component_weights:
    if args.wm_component_weights_file:
        parser.error("--wm_component_weights and --wm_component_weights_file are mutually exclusive")
if args.wm_component_weights_file:
    try:
        with open(args.wm_component_weights_file, encoding="utf-8") as handle:
            args.wm_component_weights = handle.read().strip()
    except OSError as exc:
        parser.error(f"cannot read --wm_component_weights_file: {exc}")
if args.wm_component_weights:
    try:
        component_values = [float(x.strip()) for x in args.wm_component_weights.split(",") if x.strip()]
    except ValueError as exc:
        parser.error(f"--wm_component_weights must be comma-separated numbers: {exc}")
    if len(component_values) != ERR_FLOOR.numel():
        parser.error(f"--wm_component_weights requires {ERR_FLOOR.numel()} values, got {len(component_values)}")
    if not all(math.isfinite(x) and x > 0.0 for x in component_values):
        parser.error("--wm_component_weights values must be finite and positive")
    WM_COMPONENT_WEIGHT.copy_(torch.tensor(component_values, device=dev))
WM_COMPONENT_WEIGHT[2] *= 1.0 + args.wm_root_height_weight


def _wm_rollout(s, a, xi, use_residual: bool = False):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), use_residual=use_residual)
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_rollout_window(s, a, xi, k, use_residual: bool = False):
    """Uncompiled auxiliary rollout used only for multi-scale WM loss."""
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(k):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), use_residual=use_residual)
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    z = wm.state_error(S_hat, unflatten_state(s1)) / std
    e = (torch.nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=args.wm_huber_delta,
                                            reduction="none")
         if args.wm_huber_delta > 0.0 else z.square())  # (B, k, D)
    e = (e * WM_COMPONENT_WEIGHT).sum(-1) / WM_COMPONENT_WEIGHT.sum()
    denom = mask.sum().clamp_min(1.0) if args.wm_loss_mean else s1.shape[0]
    return (e * mask).sum() / denom, e[:, -1]


def _wm_err_weighted(S_hat, s1, mask, weight):
    """Same normalized WM error, with a per-transition policy-sensitivity weight."""
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    z = wm.state_error(S_hat, unflatten_state(s1)) / std
    e = (torch.nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=args.wm_huber_delta,
                                            reduction="none")
         if args.wm_huber_delta > 0.0 else z.square())
    e = (e * WM_COMPONENT_WEIGHT).sum(-1) / WM_COMPONENT_WEIGHT.sum()
    # Keep the overall loss scale comparable to the unweighted objective.
    w = weight * mask
    if args.wm_loss_mean:
        return (e * w).sum() / w.sum().clamp_min(1.0), e[:, -1]
    return (e * w).sum() / w.sum().clamp_min(1.0) * mask.sum() / s1.shape[0], e[:, -1]


def _wm_consistency_err(S_free, S_teacher, mask):
    """Free-running prediction error against a detached teacher-forced target."""
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    target = tuple(x.detach() for x in S_teacher)
    e = (wm.state_error(S_free, target) / std) ** 2
    e = (e * WM_COMPONENT_WEIGHT).sum(-1) / WM_COMPONENT_WEIGHT.sum()
    return (e * mask).sum() / mask.sum().clamp_min(1.0)


def _wm_consistency_risk_err(S_free, S_teacher, f, mask):
    """Consistency loss with detached extra weight near teacher-state termination."""
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    target = tuple(x.detach() for x in S_teacher)
    e = (wm.state_error(S_free, target) / std) ** 2
    e = (e * WM_COMPONENT_WEIGHT).sum(-1) / WM_COMPONENT_WEIGHT.sum()
    with torch.no_grad():
        term = torch.stack([
            mdp.soft_termination(tuple(x[:, j] for x in target), f[:, j],
                                  tz=args.term_soft_height, tt=args.term_soft_tilt)
            for j in range(f.shape[1])
        ], 1)
        weight = 1.0 + args.wm_consistency_risk_weight * term
    weighted_mask = weight * mask
    return (e * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def _wm_termination_err(S_hat, w):
    """Binary termination calibration against recorded replay events."""
    pred = []
    for j in range(w["f"].shape[1]):
        pred.append(mdp.soft_termination(tuple(x[:, j] for x in S_hat), w["f"][:, j],
                                         tz=args.term_soft_height, tt=args.term_soft_tilt))
    pred = torch.stack(pred, 1).clamp(1e-5, 1.0 - 1e-5)
    target = w["done"].float()
    per = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
    return (per * w["mask"]).sum() / w["mask"].sum().clamp_min(1.0)


def _wm_teacher_forced_rollout(s, a, xi):
    """Predict recorded transitions with optional detached scheduled sampling."""
    preds = []
    previous = None
    schedule_prob = args.wm_schedule_sampling_prob
    if args.wm_schedule_sampling_ramp_iters > 0:
        progress = max(0.0, min(1.0, (it - args.wm_only_iters + 1)
                                / args.wm_schedule_sampling_ramp_iters))
        schedule_prob *= progress
    for j in range(s.shape[1]):
        observed = unflatten_state(s[:, j])
        if previous is not None and schedule_prob > 0.0:
            use_prediction = (torch.rand(s.shape[0], device=s.device)
                              < schedule_prob)
            observed = tuple(torch.where(
                use_prediction.view(-1, *([1] * (x.ndim - 1))),
                x.detach(), y) for x, y in zip(previous, observed))
        current = wm.step(observed, a[:, j], normalize_xi(xi[:, j]))
        preds.append(current)
        previous = current
    return tuple(torch.stack([p[i] for p in preds], 1) for i in range(6))


def _rollout_sensitivity(S_hat, w):
    """Discounted reward/termination sensitivity along a WM prediction window."""
    B, k = w["f"].shape
    # Sensitivity is a detached diagnostic weight.  Keeping the model graph
    # here would turn every WM update into a second-order computation.
    S_flat = tuple(x.detach().reshape(-1, *x.shape[2:]).requires_grad_(True) for x in S_hat)
    r_hat, d_hat = reward_term(
        S_flat, w["f"].reshape(-1), w["a"].reshape(-1, w["a"].shape[-1]),
        w["a_prev"].reshape(-1, w["a_prev"].shape[-1]),
    )
    grads_r = torch.autograd.grad(r_hat.sum(), S_flat, retain_graph=True,
                                  allow_unused=True)
    grads_d = torch.autograd.grad(d_hat.sum(), S_flat, retain_graph=True,
                                  allow_unused=True)
    sens = torch.zeros(B * k, device=r_hat.device)
    for gr, gd in zip(grads_r, grads_d):
        if gr is not None:
            sens = sens + gr.reshape(B * k, -1).square().sum(-1)
        if gd is not None:
            sens = sens + 0.25 * gd.reshape(B * k, -1).square().sum(-1)
    sens = sens.sqrt().reshape(B, k)
    sens = sens / sens.detach().mean().clamp_min(1e-6)
    return sens.clamp(max=5.0).detach()


def _apply_local_delta(S, delta):
    """Apply a frame-local 70-D WM delta while preserving rotation validity."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
    p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
    R1 = R @ so3_exp(dth)
    v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
    w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
    return (p1, R1, v1, w1, q + dq, qd + dqd)


def _calibrate_policy_transition(S, S1, action=None, block_residual=None, apply_block_residual=True):
    if block_residual is not None:
        return (_apply_local_delta(S, wm.state_error(S1, S) + block_residual)
                if apply_block_residual else S1)
    if wm_intervention_adapter is not None:
        delta = wm.state_error(S1, S)
        feat = torch.cat((flatten_state(S), action, delta), -1)
        proposal = wm_intervention_adapter((feat - wm_intervention_mean) / wm_intervention_std)
        corrected = delta + args.wm_intervention_adapter_blend * (proposal - delta)
        if args.wm_intervention_adapter_gate_ratio > 0.0:
            dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cn = corrected.norm(dim=-1, keepdim=True)
            valid = (cn / dn <= args.wm_intervention_adapter_gate_ratio).to(corrected.dtype)
            corrected = valid * corrected + (1.0 - valid) * delta
        if args.wm_intervention_adapter_max_ratio > 0.0:
            dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cn = corrected.norm(dim=-1, keepdim=True)
            corrected = corrected * (args.wm_intervention_adapter_max_ratio * dn / cn).clamp_max(1.0)
        return _apply_local_delta(S, corrected)
    if wm_action_delta_gain is None:
        return S1
    delta = wm.state_error(S1, S)
    return _apply_local_delta(S, delta * wm_action_delta_gain)


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, residual_gate=None,
               ensemble_enable=None, block_residual=None, block_start=False,
               block_end=False):
    global policy_target_external_allowed, policy_shadow_active
    a = pi(o_hist, g)
    if args.imag_action_noise > 0.0:
        a = torch.clamp(a + args.imag_action_noise * torch.randn_like(a), -20, 20)
    if args.shadow_policy:
        target_requested = policy_shadow_active
    else:
        target_requested = ((args.policy_ema_model and it >= args.policy_ema_start)
                            or (policy_target_wrapper is not None
                                and (args.policy_target_until_iter <= 0
                                     or it < args.policy_target_until_iter)))
    target_allowed = (policy_shadow_active or policy_target_wrapper is None
                      or policy_target_wrapper.enabled)
    # A probe runner can switch the target off without interrupting training.
    # Missing/empty files preserve the default behavior; only an explicit
    # disable token changes the decision, so a partially written file cannot
    # accidentally enable an unsafe target.
    target_allowed = target_allowed and policy_target_external_allowed
    if (target_requested and target_allowed and wm_target is not None):
        model = (policy_target_wrapper
                 if policy_target_wrapper is not None and policy_target_wrapper.adapter is not None
                 else wm_target)
    else:
        model = wm
    # The simulator-grounded action audit shows a systematic WM under-response
    # to local action perturbations. This optional diagnostic rescales only the
    # imagined action passed to the WM; replay fitting and real collection keep
    # the original action, so the default (gain=1) is exactly unchanged.
    a_model = torch.clamp(a * args.wm_action_effect_gain, -20.0, 20.0)
    next_block_residual = block_residual
    use_residual = args.joint_residual_hidden > 0 and model is wm
    if ensemble_gate_wms is not None and model is wm:
        # Frozen bootstrap members provide a policy-distribution uncertainty
        # gate. Their parameters never receive gradients; action/state
        # derivatives remain available for policy learning when the gate opens.
        S_nom = _calibrate_policy_transition(S, model.step(S, a_model, xi_n, use_residual=False), a_model)
        S_m0 = ensemble_gate_wms[0].step(S, a, xi_n, use_residual=ensemble_gate_residual)
        S_m1 = ensemble_gate_wms[1].step(S, a, xi_n, use_residual=ensemble_gate_residual)
        std = torch.maximum(model.out_norm.std(), ERR_FLOOR)
        disagreement = (model.state_error(S_m0, S_m1) / std).square().mean(-1)
        gate = (disagreement <= args.ensemble_gate_threshold).detach()
        if ensemble_enable is not None:
            gate = gate & ensemble_enable
        S1 = tuple(torch.where(gate.view(-1, *([1] * (x0.ndim - 1))),
                               (x0 + x1) * 0.5, xn)
                   for x0, x1, xn in zip(S_m0, S_m1, S_nom))
    elif use_residual and (residual_gate is not None or args.joint_residual_audit_threshold > 0.0 or
                           joint_residual_validity_model is not None):
        S_base = model.step(S, a, xi_n, use_residual=False)
        S_res = model.step(S, a, xi_n, use_residual=True)
        gate = residual_gate
        if joint_residual_validity_model is not None:
            with torch.no_grad():
                x_valid = model.in_norm(model.inputs(S, a_model, xi_n))
                step_valid = f.float() / 1000.0
                feat_valid = torch.cat((x_valid, torch.stack((f.float() / 1000.0,
                                                               step_valid), -1)), -1)
                score_valid = joint_residual_validity_model(feat_valid).squeeze(-1).sigmoid()
                validity_gate = (score_valid >= joint_residual_validity_threshold).float()
            gate = validity_gate if gate is None else gate * validity_gate
        if args.joint_residual_audit_threshold > 0.0:
            std = torch.maximum(model.out_norm.std(), ERR_FLOOR)
            disagreement = (model.state_error(S_res, S_base) / std).square().mean(-1).sqrt()
            audit_gate = (disagreement >= args.joint_residual_audit_threshold).float().detach()
            gate = audit_gate if gate is None else gate * audit_gate
        # Gate is detached and broadcast over state components. This keeps the
        # residual task-sensitive without giving the policy a gate gradient.
        gate = gate.detach()
        S1 = tuple(xb + gate.view(-1, *([1] * (xb.ndim - 1))) * (xr - xb)
                   for xb, xr in zip(S_base, S_res))
    else:
        nominal = model.step(S, a_model, xi_n, use_residual=use_residual)
        next_block_residual = block_residual
        if (block_start and wm_intervention_adapter is not None
                and args.wm_intervention_adapter_horizon > 1 and model is wm):
            S_probe = S
            for _ in range(args.wm_intervention_adapter_horizon):
                S_probe = model.step(S_probe, a_model, xi_n, use_residual=False)
            one_delta = wm.state_error(nominal, S)
            endpoint_delta = wm.state_error(S_probe, S)
            feat = torch.cat((flatten_state(S), a_model, one_delta), -1)
            proposal = wm_intervention_adapter(
                (feat - wm_intervention_mean) / wm_intervention_std)
            residual = proposal - endpoint_delta
            if not args.wm_intervention_adapter_endpoint_only:
                residual = residual / float(args.wm_intervention_adapter_horizon)
            next_block_residual = residual
            if args.wm_intervention_adapter_max_ratio > 0.0:
                dn = one_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                rn = next_block_residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                next_block_residual = next_block_residual * (
                    args.wm_intervention_adapter_max_ratio * dn / rn).clamp_max(1.0)
        block_apply = not (args.wm_intervention_adapter_endpoint_only and not block_end)
        S1 = _calibrate_policy_transition(S, nominal, a_model, next_block_residual, block_apply)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), next_block_residual


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(
        S, f, tz=args.term_soft_height, tt=args.term_soft_tilt)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


structured_compile_off = (args.joint_wm_policy_last_layer or args.joint_wm_policy_root_outputs or args.joint_wm_policy_action_path
                          or args.wm_consistency_alignment_min >= 0.0)
wm_rollout = _wm_rollout if structured_compile_off else torch.compile(_wm_rollout)
wm_err = _wm_err if structured_compile_off else torch.compile(_wm_err)
# Structured WM-gradient routing needs two retained autograd queries on the
# imagined graph; eager mode avoids donated-buffer restrictions in AOTAutograd.
imag_step = _imag_step if structured_compile_off else torch.compile(_imag_step)
reward_term = _reward_term if (args.joint_wm_policy_last_layer or args.joint_wm_policy_root_outputs
                                or args.joint_wm_policy_action_path) else torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)
if args.joint_gradient_diagnostic or args.joint_gradient_projection or args.joint_gradient_conflict_gate or args.joint_gradient_norm_match or args.joint_adversarial_residual:
    # AOT-compiled imagination may donate intermediate buffers, which is
    # incompatible with two retained autograd.grad calls below.
    imag_step = _imag_step


def refresh_policy_target_control() -> None:
    """Apply external target, rollback, or shadow-branch commands."""
    global policy_target_control_token, policy_target_external_allowed
    if not args.policy_target_control_file:
        return
    try:
        with open(args.policy_target_control_file, encoding="utf-8") as handle:
            control_raw = handle.read().strip()
    except FileNotFoundError:
        control_raw = ""
    control = control_raw.lower()
    if control == policy_target_control_token:
        return
    policy_target_control_token = control
    if control in {"shadow_promote", "shadow-promote", "promote_shadow"}:
        if shadow_branch is None:
            print("[shadow-policy] promote ignored: --shadow_policy is disabled", flush=True)
        else:
            shadow_branch.promote()
            if shadow_opt_pi is not None:
                opt_pi.load_state_dict(shadow_opt_pi.state_dict())
            if shadow_opt_v is not None:
                opt_v.load_state_dict(shadow_opt_v.state_dict())
            if critic_target is not None and shadow_critic_target is not None:
                critic_target.load_state_dict(shadow_critic_target.state_dict())
            shadow_branch.reset_from_live()
            if shadow_opt_pi is not None:
                shadow_opt_pi.load_state_dict(opt_pi.state_dict())
            if shadow_opt_v is not None:
                shadow_opt_v.load_state_dict(opt_v.state_dict())
            if shadow_critic_target is not None and critic_target is not None:
                shadow_critic_target.load_state_dict(critic_target.state_dict())
            print("[shadow-policy] promoted shadow policy/critic into live branch", flush=True)
        policy_target_external_allowed = False
    elif control in {"shadow_discard", "shadow-discard", "discard_shadow"}:
        if shadow_branch is None:
            print("[shadow-policy] discard ignored: --shadow_policy is disabled", flush=True)
        else:
            shadow_branch.discard()
            if shadow_opt_pi is not None:
                shadow_opt_pi.load_state_dict(opt_pi.state_dict())
            if shadow_opt_v is not None:
                shadow_opt_v.load_state_dict(opt_v.state_dict())
            if shadow_critic_target is not None and critic_target is not None:
                shadow_critic_target.load_state_dict(critic_target.state_dict())
            print("[shadow-policy] discarded shadow policy/critic updates", flush=True)
        policy_target_external_allowed = False
    elif control.startswith("rollback:"):
        rollback_path = control_raw.split(":", 1)[1].strip()
        try:
            rollback = torch.load(rollback_path, map_location=dev, weights_only=False)
            if "pi" not in rollback:
                raise KeyError("checkpoint has no pi state")
            pi.load_state_dict(rollback["pi"])
            if "critic" in rollback:
                critic.load_state_dict(rollback["critic"])
            if critic_target is not None and rollback.get("critic_target") is not None:
                critic_target.load_state_dict(rollback["critic_target"])
            if rollback.get("opt_pi") is not None:
                opt_pi.load_state_dict(rollback["opt_pi"])
            if rollback.get("opt_v") is not None:
                opt_v.load_state_dict(rollback["opt_v"])
            print(f"[policy-target] rolled back policy/critic from {rollback_path}", flush=True)
        except (OSError, KeyError, RuntimeError) as exc:
            print(f"[policy-target] rollback failed for {rollback_path}: {exc}", flush=True)
        policy_target_external_allowed = False
    elif control in {"0", "false", "off", "disable", "disabled"}:
        policy_target_external_allowed = False
    elif control in {"", "1", "true", "on", "enable", "enabled"}:
        policy_target_external_allowed = True


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


if wm_target is not None:
    set_requires_grad(wm_target, False)
if critic_target is not None:
    set_requires_grad(critic_target, False)


# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def _occupancy_window_weight(w, sigma=None, floor=None):
    """Approximate current-policy compatibility at each sampled window start.

    The action distance is computed from the stored observation history at the
    first transition.  It is a detached sampling weight, not a policy-learning
    objective; a floor preserves coverage of replay windows outside the current
    policy's local action distribution.
    """
    age, env = w["window_age"], w["window_env"]
    src = wm_buffer
    row = src._row(age)
    ep_step = src.data["ep_step"][row, env].long()
    back = torch.arange(Hh - 1, -1, -1, device=dev)
    back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
    back = torch.minimum(back, age.unsqueeze(-1))
    hrows = src._row(age.unsqueeze(-1) - back)
    henv = env.unsqueeze(-1).expand_as(hrows)
    hs = src.data["s"][hrows, henv]
    hap = src.data["a_prev"][hrows, henv]
    hxi = src.data["xi"][hrows, henv]
    hf = src.data["f"][hrows, henv]
    oh = mdp.obs(unflatten_state(hs.reshape(-1, hs.shape[-1])),
                 hap.reshape(-1, hap.shape[-1]), hxi.reshape(-1, hxi.shape[-1]),
                 hf.reshape(-1)).reshape(age.shape[0], Hh, -1)
    pred = pi(oh, motion.goal(w["f"][:, 0], K))
    dist = ((pred - w["a"][:, 0]) / 20.0).square().mean(-1)
    scale = args.wm_occupancy_sigma if sigma is None else sigma
    compat = torch.exp(-dist / (2.0 * scale ** 2))
    compat = compat / compat.mean().clamp_min(1e-6)
    floor = args.wm_occupancy_floor if floor is None else floor
    return (floor + (1.0 - floor) * compat.clamp(max=5.0)).detach()


def wm_update():
    global diag_corr_ema, last_fresh_wm_err, occupancy_val_ema
    global policy_target_kl_dual
    global policy_target_audit_passes, policy_target_audit_failures, policy_target_last_good
    global policy_target_audit_h1, policy_target_audit_h4
    global policy_target_nominal_h1, policy_target_nominal_h4
    src = wm_buffer
    projected_recovery_grads = None
    fresh_n = 0
    if (args.onpolicy_replay_fraction > 0.0
            and not args.onpolicy_separate_wm_update
            and onpolicy_buffer.size >= args.wm_window):
        fresh_n = min(args.wm_batch - 1, max(1, int(round(args.wm_batch * args.onpolicy_replay_fraction))))
    elif (args.onpolicy_separate_wm_update and src is onpolicy_buffer
          and onpolicy_buffer.size >= args.wm_window):
        # The isolated update is entirely fresh data. Mark all rows so the
        # diagnostics and optional fresh-only auxiliaries do not report a
        # misleading zero fresh-error value.
        fresh_n = args.wm_batch
    if fresh_n > 0:
        main_n = args.wm_batch - fresh_n
        w_main = buffer.sample_windows(main_n, args.wm_window,
                                       prioritized=args.wm_prioritized,
                                       priority_alpha=args.wm_priority_alpha,
                                       return_indices=False,
                                       recent_fraction=args.wm_recent_fraction)
        w_fresh = onpolicy_buffer.sample_windows(fresh_n, args.wm_window,
                                                 return_indices=False)
        w = {k: torch.cat((w_main[k], w_fresh[k]), 0) for k in w_main}
    else:
        w = src.sample_windows(
            args.wm_batch, args.wm_window,
            prioritized=args.wm_prioritized,
            priority_alpha=args.wm_priority_alpha,
            return_indices=(args.wm_prioritized or args.wm_occupancy_weight > 0.0
                            or args.wm_policy_aware_weight > 0.0
                            or dual_q_weight_table is not None
                            or args.strategic_probe_capacity > 0),
            recent_fraction=args.wm_recent_fraction,
        )
    S_hat = (_wm_teacher_forced_rollout(w["s"], w["a"], w["xi"])
             if args.wm_teacher_forcing else
             wm_rollout(w["s"], w["a"], w["xi"]))
    if online_realism_model is not None:
        # Online MnM-style discriminator: real replay transition vs the current
        # nominal WM transition, both detached from this discriminator update.
        with torch.no_grad():
            real_feat = _online_realism_features(w["s"][:, 0], w["a"][:, 0], w["s1"][:, 0])
            fake_feat = _online_realism_features(
                w["s"][:, 0], w["a"][:, 0], flatten_state(tuple(x[:, 0] for x in S_hat)))
            all_feat = torch.cat((real_feat, fake_feat), 0).float()
            batch_mean, batch_std = all_feat.mean(0), all_feat.std(0).clamp_min(1e-3)
            online_realism_mean.mul_(0.99).add_(batch_mean, alpha=0.01)
            online_realism_std.mul_(0.99).add_(batch_std, alpha=0.01)
            disc_x = (all_feat - online_realism_mean) / online_realism_std
            disc_y = torch.cat((torch.ones(len(real_feat), device=dev),
                                torch.zeros(len(fake_feat), device=dev)))
        for _ in range(args.online_realism_steps):
            logits = online_realism_model(disc_x).squeeze(-1)
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, disc_y)
            online_realism_opt.zero_grad(set_to_none=True)
            disc_loss.backward(); online_realism_opt.step()
    last_fresh_wm_err = 0.0
    if fresh_n > 0:
        main_n = args.wm_batch - fresh_n
        with torch.no_grad():
            std_diag = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            fresh_err = wm.state_error(
                tuple(x[main_n:] for x in S_hat),
                unflatten_state(w["s1"][main_n:]),
            ) / std_diag
            last_fresh_wm_err = float(fresh_err.square().mean().sqrt().cpu())
    term_brier = torch.zeros((), device=dev)
    term_prob = torch.zeros((), device=dev)
    term_rate = torch.zeros((), device=dev)
    if args.termination_diagnostic:
        with torch.no_grad():
            pred_term = []
            for j in range(args.wm_window):
                pred_term.append(mdp.soft_termination(
                    tuple(x[:, j] for x in S_hat), w["f"][:, j],
                    tz=args.term_soft_height, tt=args.term_soft_tilt))
            pred_term = torch.stack(pred_term, 1)
            mask = w["mask"]
            denom = mask.sum().clamp_min(1.0)
            term_brier = ((pred_term - w["done"].float()).square() * mask).sum() / denom
            term_prob = (pred_term * mask).sum() / denom
            term_rate = (w["done"].float() * mask).sum() / denom
    occ_weight = None
    if args.wm_reward_mismatch_weight > 0.0:
        # Task-aware active fitting: prioritize transitions whose predicted
        # reward/termination differs from the recorded simulator transition.
        # The signal is detached and only chooses factual WM gradient weight.
        with torch.no_grad():
            pred_flat = tuple(x.reshape(-1, *x.shape[2:]) for x in S_hat)
            real_flat = unflatten_state(w["s1"].reshape(-1, w["s1"].shape[-1]))
            f_flat = w["f"].reshape(-1)
            a_flat = w["a"].reshape(-1, w["a"].shape[-1])
            ap_flat = w["a_prev"].reshape(-1, w["a_prev"].shape[-1])
            rp, dp = reward_term(pred_flat, f_flat, a_flat, ap_flat)
            rr, dr = reward_term(real_flat, f_flat, a_flat, ap_flat)
            mismatch = (rp - rr).abs() + 0.5 * (dp - dr).abs()
            mismatch = mismatch.reshape(w["mask"].shape)
            mismatch = mismatch / mismatch.detach().mean().clamp_min(1e-6)
            mismatch_weight = 1.0 + args.wm_reward_mismatch_weight * mismatch.clamp(max=5.0)
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], mismatch_weight)
    elif args.wm_policy_aware_weight > 0.0:
        # Policy-aware active selection: fit replay transitions that are both
        # close to the current policy action and currently hard for the WM.
        # The score is detached; this changes only which factual targets are
        # fitted and never sends a policy-return gradient into the WM.
        with torch.no_grad():
            compat = _occupancy_window_weight(
                w, args.wm_policy_aware_sigma, args.wm_policy_aware_floor)
            std_pa = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            strategic = (wm.state_error(S_hat, unflatten_state(w["s1"])) / std_pa).square().mean(-1)
            strategic = strategic / strategic.detach().mean().clamp_min(1e-6)
            weight = compat[:, None] * (1.0 + args.wm_policy_aware_weight * strategic.clamp(max=5.0))
            if (args.wm_policy_aware_uncertainty > 0.0
                    and ensemble_gate_wms is not None):
                S0 = unflatten_state(w["s"][:, 0])
                xi0 = normalize_xi(w["xi"][:, 0])
                sm0 = ensemble_gate_wms[0].step(S0, w["a"][:, 0], xi0, use_residual=ensemble_gate_residual)
                sm1 = ensemble_gate_wms[1].step(S0, w["a"][:, 0], xi0, use_residual=ensemble_gate_residual)
                dis = (wm.state_error(sm0, sm1) / std_pa).square().mean(-1)
                dis = dis / dis.detach().mean().clamp_min(1e-6)
                weight = weight * (1.0 + args.wm_policy_aware_uncertainty * dis[:, None].clamp(max=5.0))
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], weight)
        if args.wm_policy_aware_validation_ratio > 0.0:
            uniform_loss, _ = _wm_err_weighted(
                S_hat, w["s1"], w["mask"], torch.ones_like(w["mask"]))
            w_val = src.sample_windows(args.wm_batch, args.wm_window,
                                       recent_fraction=args.wm_recent_fraction)
            with torch.no_grad():
                S_val = (_wm_teacher_forced_rollout(w_val["s"], w_val["a"], w_val["xi"])
                         if args.wm_teacher_forcing else
                         wm_rollout(w_val["s"], w_val["a"], w_val["xi"]))
                val_loss, _ = _wm_err(S_val, w_val["s1"], w_val["mask"])
                val_now = float(val_loss.detach().cpu())
            previous_val = occupancy_val_ema
            occupancy_val_ema = val_now if previous_val is None else (
                args.wm_occupancy_validation_beta * previous_val
                + (1.0 - args.wm_occupancy_validation_beta) * val_now)
            if previous_val is not None and val_now > args.wm_policy_aware_validation_ratio * previous_val:
                loss = uniform_loss
    elif args.wm_onpolicy_error_weight > 0.0 and src is onpolicy_buffer:
        # Error-MDP-inspired active selection: on isolated recent on-policy
        # data, emphasize transitions where the current WM is most wrong.
        # The weights are detached and only affect WM replay fitting; policy
        # gradients and real transition targets remain unchanged.
        with torch.no_grad():
            std_err = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            strategic_err = (wm.state_error(S_hat, unflatten_state(w["s1"]))
                             / std_err).square().mean(-1)
            strategic_err = strategic_err / strategic_err.detach().mean().clamp_min(1e-6)
            err_weight = 1.0 + args.wm_onpolicy_error_weight * strategic_err.clamp(max=5.0)
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], err_weight)
    elif dual_q_weight_table is not None:
        if "window_age" not in w or "window_env" not in w:
            raise RuntimeError("dual-Q WM weighting requires replay window indices")
        with torch.no_grad():
            age = w["window_age"].long().clamp(0, dual_q_weight_table.shape[0] - 1)
            env = w["window_env"].long().clamp(0, dual_q_weight_table.shape[1] - 1)
            dq_weight = dual_q_weight_table[age, env].clamp(0.25, 4.0)
            dq_weight = dq_weight[:, None].expand_as(w["mask"])
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], dq_weight)
    elif transition_weight_model is not None:
        # Offline classifier odds are a detached diagnostic weighting signal.
        # The matched audit intentionally excludes absolute age/frame fields;
        # use the same (s, a, xi, s1) transition tuple here.
        with torch.no_grad():
            tw_feat = torch.cat((w["s"][:, 0].float(), w["a"][:, 0].float(),
                                 w["xi"][:, 0].float(), w["s1"][:, 0].float()), -1)
            if tw_feat.shape[-1] != transition_weight_mean.numel():
                raise RuntimeError(
                    f"transition classifier expects {transition_weight_mean.numel()} features, "
                    f"got {tw_feat.shape[-1]}")
            tw = transition_weight_model.weights(
                tw_feat, scale=args.wm_transition_weight_scale,
                minimum=0.25, maximum=4.0)[:, None].expand_as(w["mask"])
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], tw)
    elif args.wm_ensemble_disagreement_weight > 0.0:
        # Use frozen, matched bootstrap members only as a detached replay
        # weighting signal. The nominal WM remains the model being optimized;
        # no ensemble gradient or policy-return gradient enters this update.
        with torch.no_grad():
            S0 = unflatten_state(w["s"][:, 0])
            xi0 = normalize_xi(w["xi"][:, 0])
            sm0 = ensemble_gate_wms[0].step(S0, w["a"][:, 0], xi0, use_residual=ensemble_gate_residual)
            sm1 = ensemble_gate_wms[1].step(S0, w["a"][:, 0], xi0, use_residual=ensemble_gate_residual)
            std_ens = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            dis = (wm.state_error(sm0, sm1) / std_ens).square().mean(-1)
            dis = dis / dis.detach().mean().clamp_min(1e-6)
            ens_weight = 1.0 + args.wm_ensemble_disagreement_weight * dis.clamp(max=5.0)
            if src is onpolicy_buffer:
                # The on-policy buffer is intentionally isolated from the
                # historical calibration distribution; use uniform fitting in
                # its dedicated update instead of applying stale member spread.
                ens_weight.fill_(1.0)
            elif fresh_n > 0:
                # Members are calibrated on the historical replay distribution;
                # applying their spread to newly collected on-policy samples can
                # amplify pure distribution shift. Keep fresh data uniform.
                ens_weight[main_n:] = 1.0
            ens_weight = ens_weight[:, None].expand_as(w["mask"])
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], ens_weight)
    elif args.wm_occupancy_weight > 0.0:
        occ_weight = 1.0 + args.wm_occupancy_weight * (_occupancy_window_weight(w) - 1.0)
        occ_weight = occ_weight[:, None].expand_as(w["mask"])
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], occ_weight)
        if args.wm_occupancy_error_ratio > 0.0:
            # Compare against an independent replay validation batch. The gate
            # only chooses the sampling objective; it never introduces a
            # policy-return gradient into the nominal WM.
            uniform_loss, _ = _wm_err_weighted(
                S_hat, w["s1"], w["mask"], torch.ones_like(w["mask"]))
            w_val = src.sample_windows(args.wm_batch, args.wm_window,
                                       recent_fraction=args.wm_recent_fraction)
            with torch.no_grad():
                S_val = (_wm_teacher_forced_rollout(w_val["s"], w_val["a"], w_val["xi"])
                         if args.wm_teacher_forcing else
                         wm_rollout(w_val["s"], w_val["a"], w_val["xi"]))
                val_loss, _ = _wm_err(S_val, w_val["s1"], w_val["mask"])
                val_now = float(val_loss.detach().cpu())
            previous_val = occupancy_val_ema
            occupancy_val_ema = val_now if previous_val is None else (
                args.wm_occupancy_validation_beta * previous_val
                + (1.0 - args.wm_occupancy_validation_beta) * val_now)
            if previous_val is not None and val_now > args.wm_occupancy_error_ratio * previous_val:
                loss = uniform_loss
    elif args.wm_reward_weight > 0.0:
        # WAM-RL-style successful-trajectory adaptation in a minimal
        # state-space form: emphasize replay transitions with above-batch
        # one-step tracking reward, while keeping the true next-state target
        # and the policy objective unchanged.  The weight is detached and
        # clipped so this remains a data-selection diagnostic rather than a
        # reward-to-dynamics gradient.
        with torch.no_grad():
            reward_z = (w["r"] - w["r"].mean()) / w["r"].std().clamp_min(1e-6)
            reward_weight = (1.0 + args.wm_reward_weight * reward_z).clamp(0.25, 4.0)
            reward_weight = reward_weight * w["mask"]
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], reward_weight)
    elif args.wm_td_weight > 0.0:
        with torch.no_grad():
            S0, S1 = unflatten_state(w["s"]), unflatten_state(w["s1"])
            xi_n = normalize_xi(w["xi"])
            f0, f1 = w["f"], w["f"] + 1
            B, k = f0.shape
            flat = lambda S: tuple(x.reshape(-1, *x.shape[2:]) for x in S)  # noqa: E731
            f0f, f1f = f0.reshape(-1), f1.reshape(-1)
            xif = xi_n.reshape(-1, xi_n.shape[-1])
            td_critic = critic_target if critic_target is not None else critic
            v0 = td_critic(_critic_input(flat(S0), f0f, xif, motion.goal(f0f, K))).reshape(B, k)
            v1 = td_critic(_critic_input(flat(S1), f1f, xif, motion.goal(f1f, K))).reshape(B, k)
            td = (w["r"] + args.gamma * (1 - w["done"].float()) * v1 - v0).abs()
            td = td / td.detach().mean().clamp_min(1e-6)
            weight = 1.0 + args.wm_td_weight * td.clamp(max=5.0)
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], weight)
    elif args.wm_rollout_sensitivity_weight > 0.0:
        sens = _rollout_sensitivity(S_hat, w)
        weight = 1.0 + args.wm_rollout_sensitivity_weight * sens
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], weight)
    elif args.wm_sensitivity_weight > 0.0:
        # The gradient of the analytic tracking reward with respect to the
        # observed next state identifies state errors that matter directly to
        # control.  It is a detached sampling weight, never a second-order
        # gradient through the world-model parameters.
        s1_sens = w["s1"].detach().requires_grad_(True)
        sens_state = unflatten_state(s1_sens)
        sens_reward = mdp.reward(sens_state, w["f"], w["a"], w["a_prev"])
        sens_grad = torch.autograd.grad(sens_reward.sum(), s1_sens, retain_graph=False)[0]
        sens = sens_grad.square().sum(-1).sqrt()
        sens = sens / sens.detach().mean().clamp_min(1e-6)
        weight = 1.0 + args.wm_sensitivity_weight * sens.clamp(max=5.0)
        weight = weight.detach()
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], weight)
    elif args.wm_angular_velocity_weight > 0.0:
        # Fit transitions near the observed high-angular-velocity failure
        # regime more often, while keeping the policy objective unchanged.
        with torch.no_grad():
            next_state = unflatten_state(w["s1"])
            ang = next_state[3].square().mean(-1).sqrt()
            ang = ang / ang.detach().mean().clamp_min(1e-6)
            weight = 1.0 + args.wm_angular_velocity_weight * ang.clamp(max=5.0)
            weight = weight.detach()
        loss, e_last = _wm_err_weighted(S_hat, w["s1"], w["mask"], weight)
    else:
        loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    if strategic_probe_buffer is not None:
        # Store factual transitions selected by the current policy footprint
        # and the current model's strategic error.  The score is detached:
        # probes affect future factual fitting only, never the policy loss or
        # the transition target itself.
        with torch.no_grad():
            std_probe = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            strategic = (wm.state_error(S_hat, unflatten_state(w["s1"])) / std_probe).square().mean(-1)[:, 0]
            if "window_age" in w and "window_env" in w:
                compat = _occupancy_window_weight(w).clamp_min(1e-6)
            else:
                compat = torch.ones_like(strategic)
            score = (strategic * compat).clamp_min(1e-6)
            source = w.get("window_env", torch.zeros_like(score, dtype=torch.long))
            strategic_probe_buffer.add(
                {"s": w["s"][:, 0], "a": w["a"][:, 0], "xi": w["xi"][:, 0],
                 "s1": w["s1"][:, 0]}, score, source)
        if args.strategic_probe_weight > 0.0 and strategic_probe_buffer.size > 0:
            probe_batch = args.strategic_probe_batch or args.wm_batch
            probe = strategic_probe_buffer.sample(
                probe_batch, prioritized=True, alpha=args.strategic_probe_alpha)
            probe_state = unflatten_state(probe["s"])
            probe_next = wm.step(probe_state, probe["a"], normalize_xi(probe["xi"]))
            probe_loss, _ = _wm_err(
                tuple(x.unsqueeze(1) for x in probe_next),
                probe["s1"].unsqueeze(1),
                torch.ones((probe["s"].shape[0], 1), device=dev))
            loss = loss + args.strategic_probe_weight * probe_loss
    if args.wm_multistep_weight > 0.0:
        # Explicit simulator-grounded multi-step target.  The recursive
        # prediction starts from the recorded state and uses the recorded
        # action sequence, while the final target is the actual state after
        # the complete replay horizon.  This couples WM recursion to real
        # transitions without sending policy-return gradients into WM.
        h = args.wm_multistep_horizon
        if args.wm_multistep_detach_intermediate:
            S_cur = unflatten_state(w["s"][:, 0])
            S_last = None
            for j in range(h):
                S_last = wm.step(S_cur, w["a"][:, j], normalize_xi(w["xi"][:, j]))
                if j + 1 < h:
                    S_cur = tuple(x.detach() for x in S_last)
            S_multi = tuple(x.unsqueeze(1) for x in S_last)
        else:
            S_multi = _wm_rollout_window(w["s"], w["a"], w["xi"], h)
        target_multi = w["s1"][:, h - 1:h]
        multi_mask = torch.ones_like(w["mask"][:, h - 1:h])
        multi_loss, _ = _wm_err(
            tuple(x[:, -1:] for x in S_multi), target_multi, multi_mask)
        loss = loss + args.wm_multistep_weight * multi_loss
    if wm_action_recovery is not None:
        # Action-discriminative auxiliary target: infer the executed action
        # from the current state and the WM-predicted next state.  Gradients
        # through the predicted next state encourage the dynamics model to
        # preserve action-dependent differences, while the inverse head is
        # discarded at deployment.
        cur_flat = flatten_state(unflatten_state(w["s"].reshape(-1, w["s"].shape[-1])))
        nxt_flat = flatten_state(tuple(x.reshape(-1, *x.shape[2:]) for x in S_hat))
        rec_in = torch.cat((cur_flat, nxt_flat), -1)
        rec_pred = wm_action_recovery(rec_in).reshape_as(w["a"])
        rec_err = ((rec_pred - w["a"]) / 20.0).square().mean(-1)
        rec_weight = w["mask"]
        if args.wm_action_recovery_onpolicy_only:
            # In a mixed historical/fresh batch, the fresh rows are appended
            # after the historical rows. In a dedicated on-policy update all
            # rows are fresh. Keep the mask detached from the optimizer.
            if src is onpolicy_buffer:
                rec_weight = rec_weight
            elif fresh_n > 0:
                fresh_mask = torch.zeros_like(rec_weight)
                # ``fresh_n`` rows are appended by construction below; the
                # historical count is the leading dimension of the batch.
                fresh_mask[-fresh_n:] = w["mask"][-fresh_n:]
                rec_weight = fresh_mask
            else:
                rec_weight = rec_weight * 0.0
        if args.wm_action_recovery_fidelity_gate > 0.0:
            with torch.no_grad():
                std_fid = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
                fid_err = (wm.state_error(S_hat, unflatten_state(w["s1"])) / std_fid).square().mean(-1)
                fid_scale = fid_err.detach().mean().clamp_min(1e-6)
                rec_weight = rec_weight * torch.exp(
                    -args.wm_action_recovery_fidelity_gate * (fid_err / fid_scale).clamp(max=5.0))
        rec_loss = (rec_err * rec_weight).sum() / rec_weight.sum().clamp_min(1.0)
        if args.wm_action_recovery_project_grad:
            wm_params = tuple(p for p in wm.parameters() if p.requires_grad)
            factual_grads = torch.autograd.grad(
                loss, wm_params, retain_graph=True, allow_unused=True)
            recovery_grads = torch.autograd.grad(
                rec_loss, wm_params, retain_graph=True, allow_unused=True)
            vf = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, factual_grads)])
            vr = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, recovery_grads)])
            coeff = ((vf * vr).sum() / vf.square().sum().clamp_min(1e-8)).clamp_min(0.0)
            projected_recovery_grads = tuple(
                ((torch.zeros_like(p) if g_f is None else g_f)
                 + args.wm_action_recovery_weight * coeff
                 * (torch.zeros_like(p) if g_f is None else vf_i.reshape_as(p))).detach()
                for p, g_f, vf_i in zip(wm_params, factual_grads, factual_grads))
        if args.wm_action_recovery_alignment_min >= 0.0:
            # The auxiliary is useful only when its induced WM update agrees
            # with factual replay fitting.  Compute the cosine on WM
            # parameters only; the inverse head still receives gradients when
            # the gate opens, but cannot hide a conflicting dynamics update.
            wm_params = tuple(p for p in wm.parameters() if p.requires_grad)
            factual_grads = torch.autograd.grad(
                loss, wm_params, retain_graph=True, allow_unused=True)
            recovery_grads = torch.autograd.grad(
                rec_loss, wm_params, retain_graph=True, allow_unused=True)
            vf = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, factual_grads)])
            vr = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, recovery_grads)])
            alignment = (vf * vr).sum() / (vf.square().sum().sqrt()
                                           * vr.square().sum().sqrt()).clamp_min(1e-8)
            if alignment < args.wm_action_recovery_alignment_min:
                rec_loss = rec_loss * 0.0
        loss = loss + args.wm_action_recovery_weight * rec_loss
    if wm_action_transition_encoder is not None:
        # Contrastive variant: transitions from the same mini-batch must
        # identify their executed actions, preventing a low-error WM from
        # collapsing distinct action effects into the same prediction.
        cur_flat = flatten_state(unflatten_state(w["s"].reshape(-1, w["s"].shape[-1])))
        nxt_flat = flatten_state(tuple(x.reshape(-1, *x.shape[2:]) for x in S_hat))
        valid = w["mask"].reshape(-1) > 0.5
        if args.wm_action_contrastive_delta:
            cur_state = unflatten_state(w["s"].reshape(-1, w["s"].shape[-1]))
            pred_state = tuple(x.reshape(-1, *x.shape[2:]) for x in S_hat)
            trans_feat = wm.state_error(pred_state, cur_state)
            trans_feat = trans_feat / torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        else:
            trans_feat = torch.cat((cur_flat, nxt_flat), -1)
        zt = wm_action_transition_encoder(trans_feat[valid])
        action_in = w["a"].reshape(-1, w["a"].shape[-1])[valid]
        if args.wm_action_contrastive_normalize_actions:
            action_in = action_in / 20.0
        za = wm_action_encoder(action_in)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        za = torch.nn.functional.normalize(za, dim=-1)
        if args.wm_action_contrastive_counterfactual:
            # Same-state negative: generate a second next state from a
            # shuffled action at the identical current state. This blocks the
            # batch-negative shortcut where the encoder can identify an action
            # from the state/episode identity alone.
            cur_valid_state = unflatten_state(w["s"].reshape(-1, w["s"].shape[-1])[valid])
            xi_valid = normalize_xi(w["xi"].reshape(-1, w["xi"].shape[-1])[valid])
            action_valid = w["a"].reshape(-1, w["a"].shape[-1])[valid]
            neg_action = action_valid[torch.randperm(action_valid.shape[0], device=dev)]
            neg_state = wm.step(cur_valid_state, neg_action, xi_valid)
            pos_state = tuple(x.reshape(-1, *x.shape[2:])[valid] for x in S_hat)
            if args.wm_action_contrastive_delta:
                std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
                pos_feat = wm.state_error(pos_state, cur_valid_state) / std
                neg_feat = wm.state_error(neg_state, cur_valid_state) / std
            else:
                cur_flat_valid = flatten_state(cur_valid_state)
                pos_feat = torch.cat((cur_flat_valid, flatten_state(pos_state)), -1)
                neg_feat = torch.cat((cur_flat_valid, flatten_state(neg_state)), -1)
            zpos = torch.nn.functional.normalize(wm_action_transition_encoder(pos_feat), dim=-1)
            zneg = torch.nn.functional.normalize(wm_action_transition_encoder(neg_feat), dim=-1)
            action_valid_enc = action_valid
            neg_action_enc = neg_action
            if args.wm_action_contrastive_normalize_actions:
                action_valid_enc = action_valid_enc / 20.0
                neg_action_enc = neg_action_enc / 20.0
            ztrue = torch.nn.functional.normalize(wm_action_encoder(action_valid_enc), dim=-1)
            zfalse = torch.nn.functional.normalize(wm_action_encoder(neg_action_enc), dim=-1)
            pos_logits = torch.stack(((zpos * ztrue).sum(-1), (zpos * zfalse).sum(-1)), -1)
            neg_logits = torch.stack(((zneg * ztrue).sum(-1), (zneg * zfalse).sum(-1)), -1)
            logits = torch.cat((pos_logits, neg_logits), 0) / args.wm_action_contrastive_temperature
            labels = torch.cat((torch.zeros(pos_logits.shape[0], device=dev, dtype=torch.long),
                                torch.ones(neg_logits.shape[0], device=dev, dtype=torch.long)), 0)
            per_transition = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            contrastive_loss = per_transition.mean()
        else:
            logits = (zt @ za.t()) / args.wm_action_contrastive_temperature
            labels = torch.arange(logits.shape[0], device=dev)
            per_transition = -torch.log_softmax(logits, dim=-1).diagonal()
            contrastive_loss = per_transition.mean()
        if args.wm_action_contrastive_corr_gate > 0.0 and per_transition.numel() > 2:
            pred_valid = tuple(x.reshape(-1, *x.shape[2:])[valid] for x in S_hat)
            target_valid = unflatten_state(w["s1"].reshape(-1, w["s1"].shape[-1])[valid])
            wm_err_valid = wm.state_error(pred_valid, target_valid).square().mean(-1).detach()
            # The counterfactual branch has two losses per transition (positive
            # and shuffled-action negative), while the factual WM error has one.
            # Gate on the positive half so the optional diagnostic remains
            # dimensionally well-defined and measures the intended pair.
            gate_loss = per_transition[:wm_err_valid.numel()] if args.wm_action_contrastive_counterfactual else per_transition
            pc, wc = gate_loss.detach() - gate_loss.detach().mean(), wm_err_valid - wm_err_valid.mean()
            corr = (pc * wc).mean() / (pc.square().mean().sqrt() * wc.square().mean().sqrt()).clamp_min(1e-8)
            if corr < args.wm_action_contrastive_corr_gate:
                contrastive_loss = contrastive_loss * 0.0
        loss = loss + args.wm_action_contrastive_weight * contrastive_loss
    if (args.wm_consistency_weight > 0.0 and args.wm_teacher_forcing
            and args.wm_window > 1):
        S_free = wm_rollout(w["s"], w["a"], w["xi"])
        consistency_err = (_wm_consistency_risk_err(S_free, S_hat, w["f"], w["mask"])
                           if args.wm_consistency_risk_weight > 0.0 else
                           _wm_consistency_err(S_free, S_hat, w["mask"]))
        if args.wm_consistency_alignment_min >= 0.0:
            # A free-running consistency target is useful only when its WM
            # update agrees with factual replay fitting on this batch.  The
            # gate is detached from both objectives: it decides whether the
            # auxiliary contributes, but never changes either gradient.
            wm_params = tuple(p for p in wm.parameters() if p.requires_grad)
            factual_grads = torch.autograd.grad(
                loss, wm_params, retain_graph=True, allow_unused=True)
            consistency_grads = torch.autograd.grad(
                consistency_err, wm_params, retain_graph=True, allow_unused=True)
            vf = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, factual_grads)])
            vc = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                            for p, g in zip(wm_params, consistency_grads)])
            alignment = (vf * vc).sum() / (vf.square().sum().sqrt()
                                           * vc.square().sum().sqrt()).clamp_min(1e-8)
            if alignment < args.wm_consistency_alignment_min:
                consistency_err = consistency_err * 0.0
        consistency_scale = 1.0
        if args.wm_consistency_error_gate_ratio > 0.0:
            factual_err = e_last.detach().mean()
            global consistency_error_ema
            if consistency_error_ema is None:
                consistency_error_ema = factual_err
            else:
                threshold = args.wm_consistency_error_gate_ratio * consistency_error_ema
                consistency_scale = torch.minimum(
                    torch.ones_like(factual_err),
                    threshold / factual_err.clamp_min(1e-6)).clamp_min(0.1)
                consistency_error_ema = (
                    args.wm_consistency_error_gate_beta * consistency_error_ema
                    + (1.0 - args.wm_consistency_error_gate_beta) * factual_err)
        loss = loss + args.wm_consistency_weight * consistency_scale * consistency_err
    if args.wm_termination_weight > 0.0:
        loss = loss + args.wm_termination_weight * _wm_termination_err(S_hat, w)
    # Start from the stable primary objective and introduce deployment
    # horizons gradually. With --wm_aux_switch_iters, the first auxiliary
    # ramps in and then ramps out while the second ramps in, avoiding a long
    # interval where conflicting horizons are both fully weighted.
    aux_step = max(0.0, float(it - args.wm_only_iters + 1))
    if args.wm_aux_switch_iters > 0:
        sw = float(args.wm_aux_switch_iters)
        aux_weight = args.wm_aux_weight * max(0.0, min(1.0, 1.0 - max(0.0, aux_step - sw) / sw))
        aux_weight2 = args.wm_aux_weight2 * max(0.0, min(1.0, (aux_step - sw) / sw))
    else:
        progress = (max(0.0, min(1.0, aux_step / args.wm_aux_ramp_iters))
                    if args.wm_aux_ramp_iters > 0 else 1.0)
        aux_weight = args.wm_aux_weight * progress
        aux_weight2 = args.wm_aux_weight2 * progress
    if aux_weight > 0.0 and args.wm_aux_window != args.wm_window:
        if args.wm_aux_window > src.size:
            raise RuntimeError(f"auxiliary WM window {args.wm_aux_window} exceeds replay size {src.size}")
        wa = src.sample_windows(args.wm_batch, args.wm_aux_window,
                                recent_fraction=args.wm_recent_fraction)
        Sa = _wm_rollout_window(wa["s"], wa["a"], wa["xi"], args.wm_aux_window)
        aux_loss, _ = _wm_err(Sa, wa["s1"], wa["mask"])
        loss = loss + aux_weight * aux_loss
    if aux_weight2 > 0.0:
        if args.wm_aux_window2 == args.wm_window:
            raise RuntimeError("second auxiliary WM window must differ from --wm_window")
        if args.wm_aux_window2 > src.size:
            raise RuntimeError(f"second auxiliary WM window {args.wm_aux_window2} exceeds replay size {src.size}")
        wb = src.sample_windows(args.wm_batch, args.wm_aux_window2,
                                recent_fraction=args.wm_recent_fraction)
        Sb = _wm_rollout_window(wb["s"], wb["a"], wb["xi"], args.wm_aux_window2)
        aux_loss2, _ = _wm_err(Sb, wb["s1"], wb["mask"])
        loss = loss + aux_weight2 * aux_loss2
    cf_gate_mean = 1.0
    cf_gate_min = 1.0
    cf_gate_max = 1.0
    if (args.wm_counterfactual_weight > 0.0 or args.wm_counterfactual_effect_weight > 0.0
            or args.wm_counterfactual_norm_weight > 0.0
            or args.wm_counterfactual_tilt_weight > 0.0
            or args.wm_counterfactual_geometry_weight > 0.0
            or args.wm_counterfactual_geometry_mag_weight > 0.0
            or args.wm_counterfactual_relative_weight > 0.0
            or args.wm_counterfactual_gram_weight > 0.0
            or args.wm_counterfactual_risk_gradient_weight > 0.0):
        # Simulator-supervised one-step intervention loss. The targets are
        # detached real counterfactual states; this updates only the nominal
        # WM and never changes the policy objective directly.
        cf_pairs = counterfactual_data["pairs"]
        n_cf = counterfactual_data["action"].shape[0]
        k_cf = counterfactual_data["directions"].shape[1]
        cf_batch = args.wm_counterfactual_batch or args.wm_batch
        if args.wm_counterfactual_deterministic:
            # Cycle through states/directions/pair scales as WM updates advance.
            # This removes per-update sampling noise without freezing one
            # narrow counterfactual subset.
            offset = (int(it) * cf_batch) % n_cf
            ix = (torch.arange(cf_batch, device=dev) + offset) % n_cf
            kk = (torch.arange(cf_batch, device=dev) + int(it)) % k_cf
            pp = (torch.arange(cf_batch, device=dev) + int(it)) % len(cf_pairs)
        else:
            ix = torch.randint(n_cf, (cf_batch,), device=dev)
            kk = torch.randint(k_cf, (cf_batch,), device=dev)
            pp = torch.randint(len(cf_pairs), (cf_batch,), device=dev)
        s0_cf = torch.stack([cf_pairs[int(j)]["s0"][int(i)] for j, i in zip(pp.tolist(), ix.tolist())]).to(dev)
        xi_cf = counterfactual_data["xi"][ix].to(dev)
        a_cf = counterfactual_data["action"][ix].to(dev)
        dirs_cf = counterfactual_data["directions"][ix, kk].to(dev)
        d_cf = torch.tensor([float(cf_pairs[int(j)]["delta"]) for j in pp.tolist()], device=dev)
        s_plus = torch.stack([cf_pairs[int(j)]["plus"][int(i), int(k)] for j, i, k in zip(pp.tolist(), ix.tolist(), kk.tolist())]).to(dev)
        s_minus = torch.stack([cf_pairs[int(j)]["minus"][int(i), int(k)] for j, i, k in zip(pp.tolist(), ix.tolist(), kk.tolist())]).to(dev)
        S_cf = unflatten_state(s0_cf); xi_cf = normalize_xi(xi_cf)
        cf_horizon = int(counterfactual_data.get("counterfactual_horizon", 1))
        def _counterfactual_rollout(S, action):
            for _ in range(cf_horizon):
                S = wm.step(S, action, xi_cf, use_residual=False)
            return S
        pred_p = _counterfactual_rollout(S_cf, a_cf + d_cf[:, None] * dirs_cf)
        pred_m = _counterfactual_rollout(S_cf, a_cf - d_cf[:, None] * dirs_cf)
        std_cf = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        true_p, true_m = unflatten_state(s_plus), unflatten_state(s_minus)
        pred_eff = wm.targets(S_cf, pred_p) - wm.targets(S_cf, pred_m)
        true_eff = wm.targets(S_cf, true_p) - wm.targets(S_cf, true_m)
        cf_per = ((wm.state_error(pred_p, true_p) / std_cf).square().mean(-1)
                  + (wm.state_error(pred_m, true_m) / std_cf).square().mean(-1))
        cf_weight = None
        if args.wm_counterfactual_error_gate > 0.0:
            # Detached reliability weight: high-error endpoints contribute less
            # until the nominal WM can represent their sequence dynamics.
            with torch.no_grad():
                cf_weight = torch.exp(-args.wm_counterfactual_error_gate
                                      * cf_per.detach().clamp(max=10.0))
                cf_weight = cf_weight / cf_weight.mean().clamp_min(1e-6)
                cf_gate_mean = float(cf_weight.mean().item())
                cf_gate_min = float(cf_weight.min().item())
                cf_gate_max = float(cf_weight.max().item())
                if int(it) % 50 == 0:
                    print(f"cf gate stats | mean {cf_gate_mean:.3f} min {cf_gate_min:.3f} max {cf_gate_max:.3f}")
            cf_loss = (cf_per * cf_weight).mean()
        else:
            cf_loss = cf_per.mean()
        loss = loss + args.wm_counterfactual_weight * cf_loss
        if args.wm_counterfactual_tilt_weight > 0.0:
            # Root rotation occupies components 3:6 in frame-local
            # state_error. This auxiliary is intentionally endpoint-only: it
            # preserves tilt fidelity while the existing h4 terms address the
            # broader action effect and height response.
            err_p = wm.state_error(pred_p, true_p) / std_cf
            err_m = wm.state_error(pred_m, true_m) / std_cf
            tilt_loss = 0.5 * (err_p[..., 3:6].square().mean(-1)
                               + err_m[..., 3:6].square().mean(-1))
            tilt_loss = ((tilt_loss * cf_weight).mean()
                         if cf_weight is not None else tilt_loss.mean())
            loss = loss + args.wm_counterfactual_tilt_weight * tilt_loss
        if args.wm_counterfactual_effect_weight > 0.0:
            effect_per = ((pred_eff - true_eff) / std_cf).square().mean(-1)
            effect_loss = ((effect_per * cf_weight).mean()
                           if cf_weight is not None else effect_per.mean())
            loss = loss + args.wm_counterfactual_effect_weight * effect_loss
        if args.wm_counterfactual_norm_weight > 0.0:
            # The Gram term is scale-free and can be minimized by shrinking
            # all action effects. Preserve the simulator effect magnitude in
            # normalized state space as a separate anti-collapse constraint.
            pred_norm = (pred_eff / std_cf).square().mean(-1).add(1e-8).sqrt().clamp_max(10.0)
            true_norm = (true_eff / std_cf).square().mean(-1).add(1e-8).sqrt().clamp_max(10.0)
            norm_per = torch.nn.functional.smooth_l1_loss(pred_norm, true_norm.detach(), reduction="none")
            norm_loss = ((norm_per * cf_weight).mean()
                         if cf_weight is not None else norm_per.mean())
            loss = loss + args.wm_counterfactual_norm_weight * norm_loss
        if args.wm_counterfactual_geometry_weight > 0.0 or args.wm_counterfactual_geometry_mag_weight > 0.0:
            pred_g = pred_eff / std_cf
            true_g = (true_eff / std_cf).detach()
            if args.wm_counterfactual_geometry_root_only:
                pred_g, true_g = pred_g[..., :12], true_g[..., :12]
            pred_unit = pred_g / pred_g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            true_unit = true_g / true_g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            geom_per = 1.0 - (pred_unit * true_unit).sum(-1)
            pred_logmag = pred_g.norm(dim=-1).clamp_min(1e-6).log()
            true_logmag = true_g.norm(dim=-1).clamp_min(1e-6).log()
            geom_mag_per = torch.nn.functional.smooth_l1_loss(pred_logmag, true_logmag, reduction="none")
            geom_loss = ((geom_per * cf_weight).mean()
                         if cf_weight is not None else geom_per.mean())
            geom_mag_loss = ((geom_mag_per * cf_weight).mean()
                             if cf_weight is not None else geom_mag_per.mean())
            loss = (loss + args.wm_counterfactual_geometry_weight * geom_loss
                    + args.wm_counterfactual_geometry_mag_weight * geom_mag_loss)
        if args.wm_counterfactual_relative_weight > 0.0:
            with torch.no_grad():
                ref_p = wm_counterfactual_reference.step(
                    S_cf, a_cf + d_cf[:, None] * dirs_cf, xi_cf, use_residual=False)
                ref_m = wm_counterfactual_reference.step(
                    S_cf, a_cf - d_cf[:, None] * dirs_cf, xi_cf, use_residual=False)
                ref_eff = wm_counterfactual_reference.targets(S_cf, ref_p) - wm_counterfactual_reference.targets(S_cf, ref_m)
                ref_norm = (ref_eff / torch.maximum(wm_counterfactual_reference.out_norm.std(), ERR_FLOOR)).square().mean(-1).add(1e-8).sqrt()
            cur_norm = (pred_eff / std_cf).square().mean(-1).add(1e-8).sqrt()
            log_ratio = torch.log(cur_norm.clamp_min(1e-4)) - torch.log(ref_norm.clamp_min(1e-4))
            if args.wm_counterfactual_relative_floor > 0.0:
                log_floor = math.log(args.wm_counterfactual_relative_floor)
                rel_loss = torch.relu(log_floor - log_ratio).square().clamp_max(4.0).mean()
            else:
                rel_loss = log_ratio.square().clamp_max(4.0).mean()
            loss = loss + args.wm_counterfactual_relative_weight * rel_loss
        if args.wm_counterfactual_gram_weight > 0.0:
            # Reuse each sampled state with all available directions so the
            # Gram target measures relative action geometry, rather than a
            # randomly selected one-direction effect.
            g_n = min(cf_batch, 64)
            gix = ix[:g_n]
            s0g = torch.stack([cf_pairs[int(j)] ["s0"][int(i)]
                               for j, i in zip(pp[:g_n].tolist(), gix.tolist())]).to(dev)
            xig = counterfactual_data["xi"][gix].to(dev)
            ag = counterfactual_data["action"][gix].to(dev)
            Sg = unflatten_state(s0g)
            pred_effects, true_effects = [], []
            for d_idx in range(k_cf):
                dg = torch.tensor([float(cf_pairs[int(j)]["delta"])
                                   for j in pp[:g_n].tolist()], device=dev)
                dirg = counterfactual_data["directions"][gix, d_idx].to(dev)
                pred_gp = wm.step(Sg, ag + dg[:, None] * dirg, normalize_xi(xig), use_residual=False)
                pred_gm = wm.step(Sg, ag - dg[:, None] * dirg, normalize_xi(xig), use_residual=False)
                true_gp = unflatten_state(torch.stack([
                    cf_pairs[int(j)]["plus"][int(i), d_idx]
                    for j, i in zip(pp[:g_n].tolist(), gix.tolist())]).to(dev))
                true_gm = unflatten_state(torch.stack([
                    cf_pairs[int(j)]["minus"][int(i), d_idx]
                    for j, i in zip(pp[:g_n].tolist(), gix.tolist())]).to(dev))
                pred_effects.append(wm.targets(Sg, pred_gp) - wm.targets(Sg, pred_gm))
                true_effects.append(wm.targets(Sg, true_gp) - wm.targets(Sg, true_gm))
            pred_e = torch.stack(pred_effects, 1) / std_cf
            true_e = torch.stack(true_effects, 1) / std_cf
            pred_n = pred_e / pred_e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            true_n = true_e / true_e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            gram_loss = ((pred_n @ pred_n.transpose(-1, -2)
                          - true_n @ true_n.transpose(-1, -2)) ** 2).mean()
            loss = loss + args.wm_counterfactual_gram_weight * gram_loss
        if args.wm_counterfactual_risk_gradient_weight > 0.0:
            # Model-gradient target: compare the finite-difference change in
            # endpoint height/tilt tracking risk for simulator and WM.  The
            # simulator endpoints and reference frame are detached labels;
            # only the predicted risk gradient updates the WM.
            frames_cf = counterfactual_data["frames"][ix].to(dev).long()
            target_f = (frames_cf + cf_horizon).clamp(
                max=counterfactual_motion.root_p.shape[0] - 1)
            target_p = counterfactual_motion.root_p[target_f]
            target_R = counterfactual_motion.root_R[target_f]
            def _risk(S):
                height = ((S[0][:, 2] - target_p[:, 2]) / 0.10).square()
                tilt = ((S[1][:, 2, 2] - target_R[:, 2, 2]) / 0.25).square()
                return height + tilt
            true_grad = (_risk(true_p) - _risk(true_m)) / (2.0 * d_cf)
            pred_grad = (_risk(pred_p) - _risk(pred_m)) / (2.0 * d_cf)
            grad_scale = true_grad.detach().std().clamp_min(1e-3)
            if args.wm_counterfactual_risk_gradient_balanced:
                def _component_risk(S):
                    h = ((S[0][:, 2] - target_p[:, 2]) / 0.10).square()
                    t = ((S[1][:, 2, 2] - target_R[:, 2, 2]) / 0.25).square()
                    return h, t
                true_h, true_t = _component_risk(true_p), _component_risk(true_m)
                pred_h, pred_t = _component_risk(pred_p), _component_risk(pred_m)
                true_hg = (true_h[0] - true_h[1]) / (2.0 * d_cf)
                true_tg = (true_t[0] - true_t[1]) / (2.0 * d_cf)
                pred_hg = (pred_h[0] - pred_h[1]) / (2.0 * d_cf)
                pred_tg = (pred_t[0] - pred_t[1]) / (2.0 * d_cf)
                hs = true_hg.detach().std().clamp_min(1e-3)
                ts = true_tg.detach().std().clamp_min(1e-3)
                risk_grad_loss = 0.5 * (
                    ((pred_hg - true_hg.detach()) / hs).square().mean()
                    + args.wm_counterfactual_risk_gradient_tilt_scale
                    * ((pred_tg - true_tg.detach()) / ts).square().mean())
            else:
                risk_grad_loss = ((pred_grad - true_grad.detach()) / grad_scale).square().mean()
            loss = loss + args.wm_counterfactual_risk_gradient_weight * risk_grad_loss
    if online_realism_model is not None and args.online_realism_model_weight > 0.0:
        # Generator side of the MnM game: after the discriminator update above,
        # make the current WM replay transition look real. The discriminator
        # parameters are not part of opt_wm, so only the fake transition graph
        # contributes to the WM gradient.
        gen_fake = _online_realism_features(
            w["s"][:, 0], w["a"][:, 0], flatten_state(tuple(x[:, 0] for x in S_hat)))
        gen_logit = online_realism_model(
            (gen_fake - online_realism_mean) / online_realism_std).squeeze(-1)
        loss = loss + args.online_realism_model_weight * torch.nn.functional.softplus(-gen_logit).mean()
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    if projected_recovery_grads is not None:
        # Replace the WM portion of the combined backward pass. The inverse
        # head keeps its ordinary gradient; only its dynamics-model component
        # is projected onto factual replay fitting.
        for p, g in zip(tuple(p for p in wm.parameters() if p.requires_grad), projected_recovery_grads):
            p.grad = g
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    if wm_target is not None:
        if policy_target_wrapper is not None and it % args.policy_target_ema_interval == 0:
            policy_target_wrapper.ema_from(wm)
        elif policy_target_wrapper is None:
            ema_update_(wm_target, wm, args.wm_ema_beta)
    if (policy_target_wrapper is not None
            and it % args.policy_target_audit_interval == 0
            and buffer.size >= args.wm_window):
        # Draw a fresh replay batch after the fitting update.  It is not the
        # minibatch used above, so target activation is based on an independent
        # factual rollout rather than the optimizer's training loss.
        with torch.no_grad():
            audit = buffer.sample_windows(args.wm_batch, args.wm_window,
                                          return_indices=False)
            Sa0 = unflatten_state(audit["s"][:, 0])
            std_audit = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            S_roll = Sa0
            S_nom = Sa0
            rollout_error = None
            nominal_error = None
            rollout_h = []
            nominal_h = []
            audit_model = (policy_target_wrapper
                           if policy_target_wrapper is not None and policy_target_wrapper.adapter is not None
                           else wm_target)
            for h in range(args.policy_target_audit_horizon):
                S_roll = audit_model.step(
                    S_roll, audit["a"][:, h], normalize_xi(audit["xi"][:, h]))
                S_nom = wm_target.step(
                    S_nom, audit["a"][:, h], normalize_xi(audit["xi"][:, h]))
                S_true = unflatten_state(audit["s1"][:, h])
                err_h = (wm.state_error(S_roll, S_true) / std_audit).square().mean(-1)
                nom_h = (wm.state_error(S_nom, S_true) / std_audit).square().mean(-1)
                rollout_h.append(float(err_h.mean().detach().cpu()))
                nominal_h.append(float(nom_h.mean().detach().cpu()))
                rollout_error = err_h if rollout_error is None else torch.maximum(rollout_error, err_h)
                nominal_error = nom_h if nominal_error is None else torch.maximum(nominal_error, nom_h)
            policy_target_audit_h1 = rollout_h[0]
            policy_target_nominal_h1 = nominal_h[0]
            policy_target_audit_h4 = rollout_h[min(3, len(rollout_h) - 1)]
            policy_target_nominal_h4 = nominal_h[min(3, len(nominal_h) - 1)]
            kl_proxy = wm.state_error(S_roll, S_nom).square().mean(-1).mean()
            policy_target_wrapper.replay_kl_proxy.copy_(kl_proxy.detach().float())
            if args.policy_target_kl_budget > 0.0:
                dual_lr = (args.policy_target_kl_dual_lr or
                           args.policy_target_adapter_lr or args.lr)
                policy_target_kl_dual = max(
                    0.0, policy_target_kl_dual
                    + dual_lr * (float(kl_proxy.detach().cpu()) - args.policy_target_kl_budget))
            audit_error = rollout_error
            policy_target_wrapper.audit_error.copy_(audit_error.detach().float().mean())
            audit_ok = (True if args.policy_target_audit_max_error <= 0.0 else
                        policy_target_wrapper.audit_gate(audit_error, args.policy_target_audit_max_error))
            if args.policy_target_audit_h1_max_error > 0.0:
                audit_ok = audit_ok and bool(
                    policy_target_audit_h1 <= args.policy_target_audit_h1_max_error)
            if args.policy_target_audit_h4_max_error > 0.0 and len(rollout_h) >= 4:
                audit_ok = audit_ok and bool(
                    policy_target_audit_h4 <= args.policy_target_audit_h4_max_error)
            if args.policy_target_audit_ratio > 0.0:
                nominal_mean = nominal_error.mean().clamp_min(1e-6)
                target_mean = audit_error.mean()
                audit_ok = audit_ok and bool(
                    target_mean <= nominal_mean * (1.0 + args.policy_target_audit_ratio))
            if args.policy_target_kl_budget > 0.0:
                audit_ok = audit_ok and bool(
                    float(kl_proxy.detach().cpu()) <= args.policy_target_kl_budget)
            policy_target_wrapper.enabled = bool(audit_ok)
            if audit_ok:
                if policy_target_wrapper.adapter is not None:
                    policy_target_last_good = policy_target_wrapper.snapshot_adapter()
                policy_target_audit_passes += 1
            else:
                # Reject the complete adapter proposal, rather than merely
                # disabling it for the next rollout. This makes the audit a
                # genuine trust-region acceptance test.
                if (policy_target_wrapper.adapter is not None
                        and policy_target_last_good is not None):
                    policy_target_wrapper.restore_adapter(policy_target_last_good)
                policy_target_audit_failures += 1
    if args.wm_prioritized and fresh_n == 0:
        # e_last is the detached normalized squared error of each sampled
        # window's final transition. It is only a sampling signal; it never
        # enters the WM or policy gradient.
        src.update_priorities(w["window_age"], e_last.detach())
    m = w["mask"][:, -1]
    diag_corr = torch.zeros((), device=dev)
    diag_err = torch.zeros((), device=dev)
    if args.wm_disagreement_diagnostic and wm_target is not None:
        with torch.no_grad():
            S0, S1 = unflatten_state(w["s"][:, 0]), unflatten_state(w["s1"][:, 0])
            a0, xi0 = w["a"][:, 0], normalize_xi(w["xi"][:, 0])
            pred_online = wm.step(S0, a0, xi0)
            pred_target = wm_target.step(S0, a0, xi0)
            std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            d = (wm.state_error(pred_online, pred_target) / std).square().mean(-1)
            e = (wm.state_error(pred_online, S1) / std).square().mean(-1)
            dc, ec = d - d.mean(), e - e.mean()
            diag_corr = (dc * ec).mean() / (dc.square().mean().sqrt() * ec.square().mean().sqrt()).clamp_min(1e-6)
            diag_err = e.mean()
            diag_corr_ema = args.dispersion_gate_corr_beta * diag_corr_ema + (1.0 - args.dispersion_gate_corr_beta) * float(diag_corr)
    return (loss.detach() / args.wm_window,
            (e_last.detach() * m).sum() / m.sum().clamp(min=1),
            diag_corr.detach(), diag_err.detach(), term_brier.detach(), term_prob.detach(), term_rate.detach())


def sample_valid_starts(B: int):
    # Keep the default sampler unchanged.  The optional failure-focused arm
    # deliberately samples replay states that are already close to the
    # simulator's height/tilt termination boundary, so the imagined policy
    # objective can learn recovery instead of seeing only nominal gait states.
    ramp = (min(1.0, max(0.0, (it + 1) / args.policy_failure_start_ramp_iters))
            if args.policy_failure_start_ramp_iters > 0 else 1.0)
    risk_n = int(round(B * args.policy_failure_start_fraction * ramp))
    normal_n = B - risk_n
    starts, hists, n = [], [], 0
    while n < normal_n:
        s, h = buffer.sample_starts(2 * max(normal_n, 1), Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: normal_n - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    if risk_n > 0:
        s, h = buffer.sample_starts(max(8 * risk_n, 2 * B), Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        valid = ok.nonzero(as_tuple=False).squeeze(-1)
        # Rank by the larger normalized height or tilt error.  Ranking is
        # detached and only chooses starts; it does not alter the policy loss.
        Sv = unflatten_state(s["s"])
        height_err = ((motion.root_p[f, 2] - Sv[0][:, 2]).abs()
                      / max(mdp.cfg.term_height, 1e-6))
        tilt_err = ((motion.root_R[f, 2, 2] - Sv[1][:, 2, 2]).abs()
                    / max(mdp.cfg.term_tilt, 1e-6))
        alternate_height = (args.policy_failure_start_alternate_iters > 0 and
                            ((it // args.policy_failure_start_alternate_iters) % 2 == 0))
        alternate_tilt = (args.policy_failure_start_alternate_iters > 0 and not alternate_height)
        if args.policy_failure_start_height_only or alternate_height:
            score = height_err.masked_fill(~ok, -torch.inf)
            take = torch.topk(score, k=min(risk_n, len(s["f"]))).indices
        elif args.policy_failure_start_tilt_only or alternate_tilt:
            score = tilt_err.masked_fill(~ok, -torch.inf)
            take = torch.topk(score, k=min(risk_n, len(s["f"]))).indices
        elif args.policy_failure_start_balanced:
            valid_k = min(risk_n, len(s["f"]))
            kh = valid_k // 2
            kt = valid_k - kh
            hscore = height_err.masked_fill(~ok, -torch.inf)
            tscore = tilt_err.masked_fill(~ok, -torch.inf)
            take = torch.cat((torch.topk(hscore, k=kh).indices,
                              torch.topk(tscore, k=kt).indices))
        else:
            score = torch.maximum(height_err, tilt_err).masked_fill(~ok, -torch.inf)
            take = torch.topk(score, k=min(risk_n, len(s["f"]))).indices
        starts.append({k: v[take] for k, v in s.items()}), hists.append({k: v[take] for k, v in h.items()})
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))
task_grad_ema = None
residual_dual = resume_dual
residual_dual_reference = resume_dual_reference
last_nominal_wm_loss = 0.0
policy_wm_error_ema = None
consistency_error_ema = None
occupancy_val_ema = None


def policy_update(reuse_batch=None, step_optimizer=True, clear_grad=True):
    global task_grad_ema, residual_dual, residual_dual_reference, last_nominal_wm_loss
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = reuse_batch if reuse_batch is not None else sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    # The default freezes the WM during imagined policy optimization.  The
    # optional joint mode deliberately lets a scaled copy of the policy
    # gradient update WM as an explicit concurrent-training diagnostic.
    joint_wm = (args.joint_wm_policy_weight > 0.0 or
                args.joint_wm_replay_weight > 0.0)
    set_requires_grad(wm, joint_wm)
    residual_joint = args.joint_residual_hidden > 0 and joint_wm
    if residual_joint:
        # The nominal branch and its normalizers remain replay-only.  Only the
        # zero-initialized adapter receives the concurrent policy gradient.
        for p in list(wm.net.parameters()) + list(wm.in_norm.parameters()) + list(wm.out_norm.parameters()):
            p.requires_grad_(False)
        for p in wm.residual_net.parameters():
            p.requires_grad_(True)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    reference_action = (pi(o_hist, g0).detach()
                        if args.policy_proximal_weight > 0.0 else None)
    g = g0
    S0 = S
    bc_action = s["a"]
    reliability_scale = torch.ones(B, device=dev)
    horizon_trust = None
    if args.policy_wm_reliability_quantile > 0.0:
        # Conservative concurrent update: keep WM fitting uniform, but reduce
        # policy gradients from starts whose factual replay transition is in a
        # high-error tail.  The gate is detached and has a floor for coverage.
        with torch.no_grad():
            replay_pred = wm.step(S, s["a"], xi_n)
            std_rel = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            rel_err = (wm.state_error(replay_pred, unflatten_state(s["s1"])) / std_rel).square().mean(-1)
            threshold = torch.quantile(rel_err, args.policy_wm_reliability_quantile)
            reliable = (rel_err <= threshold).float()
            reliability_scale = args.policy_wm_reliability_floor + (
                1.0 - args.policy_wm_reliability_floor) * reliable
    if args.policy_wm_horizon_decay > 0.0:
        # MBPO-style short-model trust without changing factual WM fitting:
        # estimate one-step replay error at each imagined start, then attenuate
        # only the later policy-return terms for starts whose model is already
        # unreliable.  The error and trust curve are detached, so this cannot
        # turn the policy objective into an auxiliary WM loss.
        with torch.no_grad():
            replay_pred = wm.step(S, s["a"], xi_n)
            std_h = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            err_h = (wm.state_error(replay_pred, unflatten_state(s["s1"])) / std_h).square().mean(-1)
            progress = torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
            horizon_trust = torch.exp(
                -args.policy_wm_horizon_decay * err_h.unsqueeze(0)
                * progress / max(HZ - 1, 1)
            ).detach()
    residual_gate = None
    if (args.joint_residual_hidden > 0 and
            args.joint_residual_occupancy_sigma > 0.0):
        # Estimate occupancy from the first policy action and detach it. The
        # residual can affect states only where the current policy is near data.
        a_probe = pi(o_hist, g0)
        dist = ((a_probe - bc_action) / 20.0).square().mean(-1)
        sigma = args.joint_residual_occupancy_sigma
        residual_gate = torch.exp(-dist / (2.0 * sigma * sigma)).detach()
    # Delay the extra target rollout until the world model has seen enough
    # real transitions to make the disagreement signal meaningful.
    target_rollout_active = ((args.dispersion_gate > 0.0 or args.dispersion_return_penalty > 0.0 or args.dispersion_truncate_temperature > 0.0 or args.dispersion_softmin_temperature > 0.0)
                             and wm_target is not None and it >= args.dispersion_gate_warmup)
    gate_active = (args.dispersion_gate > 0.0 and target_rollout_active
                   and (args.dispersion_gate_corr_min <= 0.0 or diag_corr_ema >= args.dispersion_gate_corr_min))
    S_target = S if target_rollout_active else None
    S_targets = []
    robust_member_seq = [[], []]
    robust_members_active = (args.ensemble_policy_softmin_temperature > 0.0
                             and ensemble_gate_wms is not None)
    robust_member_states = [S0, S0] if robust_members_active else None
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    block_residual = None
    for t in range(HZ):
        Aps.append(a_prev)
        ensemble_enable = None
        if args.ensemble_gate_nominal_prefix > 0:
            ensemble_enable = torch.full(
                (B,), t >= args.ensemble_gate_nominal_prefix,
                device=dev, dtype=torch.bool)
        S, a_prev, o_hist, f, g, block_residual = imag_step(
            S, o_hist, a_prev, f, xi, xi_n, g, residual_gate, ensemble_enable,
            block_residual, t % max(args.wm_intervention_adapter_horizon, 1) == 0,
            (t + 1) % max(args.wm_intervention_adapter_horizon, 1) == 0)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        if (t + 1) % max(args.wm_intervention_adapter_horizon, 1) == 0:
            block_residual = None
        if robust_members_active:
            robust_member_states = [
                ensemble_gate_wms[j].step(
                    robust_member_states[j], As[-1], xi_n,
                    use_residual=ensemble_gate_residual)
                for j in range(2)
            ]
            robust_member_seq[0].append(robust_member_states[0])
            robust_member_seq[1].append(robust_member_states[1])
        if (args.policy_bptt_truncation > 0 and (t + 1) % args.policy_bptt_truncation == 0
                and t + 1 < HZ):
            # Keep the forward state, but prevent long-horizon model Jacobians
            # from multiplying across segments.
            S = tuple(x.detach() for x in S)
            if not args.policy_bptt_state_only:
                o_hist = o_hist.detach()
                a_prev = a_prev.detach()
        if S_target is not None:
            # The target rollout is used only to compute a detached gate.
            with torch.no_grad():
                # ``xi_n`` is the fixed 33-D disturbance/context vector for
                # this imagined start, matching the online rollout above.
                S_target = wm_target.step(
                    S_target, torch.clamp(As[-1] * args.wm_action_effect_gain, -20.0, 20.0), xi_n)
            S_targets.append(S_target)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    S_before = tuple(flat([S0[i]] + [St[i] for St in Ss[:-1]]) for i in range(6))
    if (online_realism_model is not None and args.online_realism_policy_transitions
            and online_realism_opt is not None and buffer.size >= 1):
        # MnM-style calibration on the transitions actually consumed by the
        # policy objective.  The discriminator sees detached policy-imagined
        # multi-step transitions as fake and replay transitions as real; its
        # updated logits are then used immediately by the realism penalty.
        with torch.no_grad():
            real_n = min(int(B), int(buffer.size))
            rw = buffer.sample_windows(real_n, 1, return_indices=False)
            real_feat = _online_realism_features(rw["s"][:, 0], rw["a"][:, 0], rw["s1"][:, 0])
            fake_before = flatten_state(tuple(x.detach() for x in S_before))
            fake_actions = flat([a.detach() for a in As])
            fake_after = flatten_state(tuple(x.detach() for x in S_all))
            fake_feat = _online_realism_features(fake_before, fake_actions, fake_after)
            all_feat = torch.cat((real_feat, fake_feat), 0).float()
            batch_mean = all_feat.mean(0)
            batch_std = all_feat.std(0).clamp_min(1e-3)
            online_realism_mean.mul_(0.99).add_(batch_mean, alpha=0.01)
            online_realism_std.mul_(0.99).add_(batch_std, alpha=0.01)
            disc_x = (all_feat - online_realism_mean) / online_realism_std
            disc_y = torch.cat((torch.ones(len(real_feat), device=dev),
                                torch.zeros(len(fake_feat), device=dev)))
        for _ in range(args.online_realism_steps):
            logits = online_realism_model(disc_x).squeeze(-1)
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, disc_y)
            online_realism_opt.zero_grad(set_to_none=True)
            disc_loss.backward(); online_realism_opt.step()
    realism_penalty = torch.zeros((), device=dev)
    realism_logit = None
    realism_gate_scale = torch.ones(B, device=dev)
    realism_ess_fraction = 1.0
    realism_model = policy_realism_model if policy_realism_model is not None else online_realism_model
    realism_weight = args.policy_realism_weight if policy_realism_model is not None else args.online_realism_weight
    if (realism_model is not None and
        (realism_weight > 0.0 or args.online_realism_reward_weight > 0.0
             or args.online_realism_gate_temperature > 0.0
             or args.online_realism_ess_gate > 0.0)):
        realism_feat = _online_realism_features(flatten_state(S_before), flat(As), flatten_state(S_all))
        real_mean = policy_realism_mean if policy_realism_model is not None else online_realism_mean
        real_std = policy_realism_std if policy_realism_model is not None else online_realism_std
        if realism_feat.shape[-1] != real_mean.numel():
            raise RuntimeError(
                f"realism discriminator expects {real_mean.numel()} features, "
                f"got {realism_feat.shape[-1]}")
        realism_logit = realism_model((realism_feat - real_mean) / real_std).squeeze(-1)
        # Real transitions are label 1. Softplus(-logit) encourages imagined
        # transitions to look real while retaining the normal policy-return
        # gradient through the online WM and policy.
        if realism_weight > 0.0:
            realism_penalty = torch.nn.functional.softplus(-realism_logit).mean()
        if args.online_realism_gate_temperature > 0.0 and policy_realism_model is None:
            # Gate only the policy gradient; keep the forward model and reward
            # unchanged. Normalization preserves the mean update magnitude.
            realism_conf = torch.sigmoid(realism_logit / args.online_realism_gate_temperature)
            realism_conf = realism_conf.view(HZ, B).mean(0)
            realism_gate_scale = realism_conf / realism_conf.mean().clamp_min(1e-3)
        if args.online_realism_ess_gate > 0.0 and policy_realism_model is None:
            # Use calibrated odds only as a trust signal. Concentrated odds
            # mean the imagined batch is outside replay support, so this
            # update keeps collection and factual WM fitting but drops the
            # imagined policy gradient. The gate is opt-in.
            ratio = torch.exp((realism_logit.detach() /
                               max(args.online_realism_gate_temperature, 1.0)).clamp(-8.0, 8.0))
            ratio = ratio.view(HZ, B).mean(0)
            realism_ess_fraction = float(
                ratio.sum().square().div(ratio.square().sum().clamp_min(1e-6)).div(max(B, 1)).item())
            if realism_ess_fraction < args.online_realism_ess_gate:
                if args.online_realism_ess_risk_floor > 0.0:
                    # Preserve gradients for starts whose imagined trajectory
                    # is already close to a height termination boundary. This
                    # prevents a support gate from simply converting tilt
                    # failures into untrained height collapse.
                    height_risk = (motion.root_p[f_all, 2] - S_all[0][:, 2]).abs()
                    height_risk = (height_risk / max(mdp.cfg.term_height, 1e-6)).view(HZ, B).amax(0)
                    realism_gate_scale = (height_risk >= args.online_realism_ess_risk_floor).float()
                else:
                    realism_gate_scale = torch.zeros_like(realism_gate_scale)
                if int(it) % 50 == 0:
                    kept = float((realism_gate_scale > 0).float().mean().item())
                    print(f"realism ESS gate | fraction {realism_ess_fraction:.3f} < threshold {args.online_realism_ess_gate:.3f} | kept {kept:.3f}")
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    if realism_logit is not None and args.online_realism_reward_weight > 0.0:
        r = r + args.online_realism_reward_weight * torch.nn.functional.logsigmoid(realism_logit)
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    if horizon_trust is None:
        J = (disc * c_t * r).sum(0)
    else:
        J = (disc * c_t * horizon_trust * r).sum(0)
    if robust_members_active:
        # Evaluate the same policy action sequence under both frozen members;
        # the soft-min is differentiable through actions and model states but
        # never updates the ensemble parameters.
        robust_returns = [J]
        for seq in robust_member_seq:
            m_all = tuple(torch.stack([st[i] for st in seq], 0).flatten(0, 1)
                          for i in range(6))
            mr, md = reward_term(m_all, f_all, flat(As), flat(Aps))
            mr, md = mr.view(HZ, B), md.view(HZ, B)
            malive = torch.cumprod(1 - md, 0)
            mc = torch.cat((torch.ones(1, B, device=dev), malive[:-1]), 0)
            robust_returns.append((disc * mc * mr).sum(0))
        tau = args.ensemble_policy_softmin_temperature
        J = -tau * torch.logsumexp(
            -torch.stack(robust_returns, 0) / tau, dim=0)
    if args.policy_height_penalty > 0.0:
        height_progress = (max(0.0, min(1.0,
            (it - args.wm_only_iters + 1) / args.policy_height_penalty_ramp_iters))
            if args.policy_height_penalty_ramp_iters > 0 else 1.0)
        root_z_err = (motion.root_p[f_all, 2] - S_all[0][:, 2]).view(HZ, B)
        root_z_err = root_z_err / max(mdp.cfg.term_height, 1e-6)
        if args.policy_height_hinge_margin > 0.0:
            root_z_err = torch.relu(root_z_err.abs() - args.policy_height_hinge_margin)
        height_weight = 1.0
        if args.policy_height_penalty_start_margin > 0.0:
            start_height = ((motion.root_p[s["f"], 2] - S0[0][:, 2]).abs()
                            / max(mdp.cfg.term_height, 1e-6))
            height_weight = (start_height >= args.policy_height_penalty_start_margin).float().detach()
        J = J - args.policy_height_penalty * height_progress * height_weight * (
            disc * c_t * root_z_err.square()).sum(0)
    tilt_penalty = args.policy_tilt_penalty + (args.shadow_policy_tilt_penalty if policy_shadow_active else 0.0)
    if tilt_penalty > 0.0:
        root_tilt_err = (motion.root_R[f_all, 2, 2] - S_all[1][:, 2, 2]).view(HZ, B)
        root_tilt_err = root_tilt_err / max(mdp.cfg.term_tilt, 1e-6)
        tilt_weight = 1.0
        if args.policy_tilt_penalty_start_margin > 0.0:
            start_tilt = ((motion.root_R[s["f"], 2, 2] - S0[1][:, 2, 2]).abs()
                          / max(mdp.cfg.term_tilt, 1e-6))
            tilt_weight = (start_tilt >= args.policy_tilt_penalty_start_margin).float().detach()
        J = J - tilt_penalty * tilt_weight * (disc * c_t * root_tilt_err.square()).sum(0)
    if args.policy_angular_velocity_penalty > 0.0:
        # Penalize angular velocity only near a predicted tilt failure. The
        # gate keeps this auxiliary objective from reshaping normal gait.
        angular_velocity = S_all[3].view(HZ, B, -1)
        tilt_gate = (motion.root_R[f_all, 2, 2] - S_all[1][:, 2, 2]).view(HZ, B)
        tilt_gate = (tilt_gate / max(mdp.cfg.term_tilt, 1e-6)).abs()
        tilt_gate = torch.relu(tilt_gate - args.policy_angular_velocity_gate_margin)
        J = J - args.policy_angular_velocity_penalty * (
            disc * c_t * tilt_gate * angular_velocity.square().mean(-1)).sum(0)
    gate_fraction = torch.ones(B, device=dev)
    gate_scale = gate_fraction
    if S_target is not None:
        # Compare full-horizon returns. The target rollout is detached and only
        # determines whether this start receives an imagined policy gradient.
        with torch.no_grad():
            target_r, target_d = [], []
            for St, ft, at, apt in zip(S_targets, Fs, As, Aps):
                rt, dt = reward_term(St, ft, at, apt)
                target_r.append(rt), target_d.append(dt)
            target_r = torch.stack(target_r)
            target_d = torch.stack(target_d)
            target_alive = torch.cumprod(1 - target_d, 0)
            target_c = torch.cat((torch.ones(1, B, device=dev), target_alive[:-1]), 0)
            target_proxy = (disc * target_c * target_r).sum(0)
            if args.policy_target_pessimistic:
                # The target rollout is a detached nominal/EMA reference.  A
                # minimum keeps the adapter from gaining policy gradient by
                # exploiting optimistic task-model dynamics.
                J = torch.minimum(J, target_proxy)
            if args.dispersion_truncate_temperature > 0.0:
                # Per-step uncertainty attenuation: preserve early reliable
                # gradients while suppressing the tail after online/EMA state
                # predictions diverge. The scale is detached by construction.
                std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
                step_dis = []
                for So, St in zip(Ss, S_targets):
                    So_det = tuple(x.detach() for x in So)
                    step_dis.append((wm.state_error(So_det, St) / std).square().mean(-1))
                step_dis = torch.stack(step_dis)
                trust = torch.exp(-step_dis / args.dispersion_truncate_temperature).detach()
                if args.dispersion_truncate_normalize:
                    trust = trust / trust.mean().clamp_min(1e-3)
                if args.dispersion_truncate_local:
                    # Local attenuation avoids multiplying 64 uncertain-step
                    # trusts into an almost-zero return on long horizons.
                    c_t = c_t * torch.cat((torch.ones(1, B, device=dev), trust[:-1]), 0)
                else:
                    trust_c = torch.cumprod(trust, dim=0)
                    c_t = c_t * torch.cat((torch.ones(1, B, device=dev), trust_c[:-1]), 0)
        if args.dispersion_softmin_temperature > 0.0:
            # EMA return is detached; only the online model carries the
            # policy gradient. This is a robust-model diagnostic, not a joint
            # update of WM parameters.
            tau = args.dispersion_softmin_temperature
            J = -tau * torch.logsumexp(
                torch.stack((-J / tau, -target_proxy / tau), dim=0), dim=0)
        if args.dispersion_truncate_temperature > 0.0:
            # Recompute outside the no-grad target rollout context so the
            # online WM/policy graph remains intact.
            J = (disc * c_t * r).sum(0)
        if args.dispersion_return_penalty > 0.0:
            # target_proxy is detached; this keeps the EMA model a reference
            # while discouraging optimistic online-model returns.
            J = J - args.dispersion_return_penalty * (J - target_proxy).abs()
        if args.dispersion_gate > 0.0:
            if args.dispersion_gate_fraction > 0.0:
                gate_fraction = quantile_gate_weight(
                    J.detach(), target_proxy, args.dispersion_gate_fraction,
                    args.dispersion_gate_temperature,
                )
            else:
                gate_fraction = gate_weight(
                    J.detach(), target_proxy, args.dispersion_gate,
                    args.dispersion_gate_temperature,
                )
            if args.dispersion_gate_normalize:
                gate_scale = gate_fraction / gate_fraction.mean().clamp_min(1e-3)
            else:
                gate_scale = gate_fraction
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
    # Apply the gate after the critic term so high-disagreement starts cannot
    # leak a policy gradient through the value bootstrap either.
    gate_scale = gate_scale * reliability_scale * realism_gate_scale
    J = J * gate_scale
    loss = -J.mean()
    if (args.policy_target_adapter_l2 > 0.0
            and policy_target_wrapper is not None
            and policy_target_wrapper.adapter is not None):
        # This term touches only the task adapter.  It anchors policy-induced
        # dynamics changes without altering the nominal WM or policy gradient.
        loss = loss + args.policy_target_adapter_l2 * policy_target_wrapper.adapter_norm().square()
    if (policy_target_kl_dual > 0.0
            and policy_target_wrapper is not None
            and policy_target_wrapper.adapter is not None):
        # The independent replay audit is the actual constraint. This local
        # adapter-norm penalty only discourages rapid growth between audits.
        loss = loss + policy_target_kl_dual * policy_target_wrapper.adapter_norm().square()
    if realism_model is not None and realism_weight > 0.0:
        loss = loss + realism_weight * realism_penalty
    if args.policy_proximal_weight > 0.0:
        # Proximal trust region around the policy before this optimizer step.
        # The reference is detached, so this only limits policy motion and
        # cannot alter the WM or introduce a replay-action imitation target.
        proximal_loss = ((As[0] - reference_action) / 20.0).square().mean()
        loss = loss + args.policy_proximal_weight * proximal_loss
    ensemble_disagreement = None
    if args.ensemble_uncertainty_weight > 0.0 or args.ensemble_policy_gate_fraction > 0.0:
        # Evaluate a one-step epistemic-risk proxy at each nominal imagined
        # state. The bootstrap members are frozen, so this adds a policy-only
        # regularizer while preserving nominal dynamics and reward gradients.
        std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        disagreement_terms = []
        for St, at in zip(Ss, As):
            Sm0 = ensemble_gate_wms[0].step(St, at, xi_n, use_residual=ensemble_gate_residual)
            Sm1 = ensemble_gate_wms[1].step(St, at, xi_n, use_residual=ensemble_gate_residual)
            dis = (wm.state_error(Sm0, Sm1) / std).square().mean(-1)
            disagreement_terms.append(dis)
        disagreement_stack = torch.stack(disagreement_terms)
        if args.ensemble_policy_gate_horizon > 0:
            ensemble_disagreement = disagreement_stack[args.ensemble_policy_gate_horizon - 1]
        else:
            ensemble_disagreement = disagreement_stack.mean(0)
        if args.ensemble_policy_gate_fraction > 0.0:
            # Quantile is computed from detached risk; gradients are retained
            # only for nominal return and never optimize the gate itself.
            q = torch.quantile(ensemble_disagreement.detach(), args.ensemble_policy_gate_fraction)
            gate_fraction = (ensemble_disagreement.detach() <= q).float()
            gate_scale = gate_fraction / gate_fraction.mean().clamp_min(1e-3)
            J = J * gate_scale
            loss = -J.mean()
        if args.ensemble_uncertainty_weight > 0.0:
            loss = loss + args.ensemble_uncertainty_weight * ensemble_disagreement.mean()
    bc_pred = None
    if args.policy_bc_weight > 0.0 or args.policy_awbc_weight > 0.0:
        # The replay action is only an anchor against immediate off-distribution
        # exploitation; normalize by the action scale so the coefficient is
        # comparable across environments.
        bc_pred = pi(o_hist, g0)
    if args.policy_bc_weight > 0.0:
        bc_per_start = ((bc_pred - bc_action) / 20.0).square().mean(-1)
        if args.policy_bc_risk_margin > 0.0:
            risk_height = ((motion.root_p[s["f"], 2] - S0[0][:, 2]).abs()
                           / max(mdp.cfg.term_height, 1e-6))
            risk_tilt = ((motion.root_R[s["f"], 2, 2] - S0[1][:, 2, 2]).abs()
                         / max(mdp.cfg.term_tilt, 1e-6))
            risk_weight = (torch.maximum(risk_height, risk_tilt)
                           >= args.policy_bc_risk_margin).float().detach()
            bc_loss = (bc_per_start * risk_weight).sum() / risk_weight.sum().clamp_min(1.0)
        else:
            bc_loss = bc_per_start.mean()
        loss = loss + args.policy_bc_weight * bc_loss
    if args.policy_awbc_weight > 0.0:
        aw_r = s["r"]
        aw_r = (aw_r - aw_r.mean()) / aw_r.std().clamp_min(1e-6)
        aw = torch.exp((args.policy_awbc_beta * aw_r).clamp(-4.0, 4.0)).detach()
        awbc_loss = (((bc_pred - bc_action) / 20.0).square().mean(-1) * aw).mean()
        loss = loss + args.policy_awbc_weight * awbc_loss
    if args.policy_strategic_error_weight > 0.0 and onpolicy_buffer.size >= 1:
        # The real next state is paired with the recent on-policy action.  Use
        # the current policy's action at the same state and a detached
        # proximity gate so stale behavior transitions cannot be interpreted
        # as counterfactual labels.  This is a policy-only Error-MDP proxy:
        # minimize model error in regions the current policy actually visits.
        se, he = onpolicy_buffer.sample_starts(B, hist)
        with torch.no_grad():
            se_hist = mdp.obs(unflatten_state(he["s"]), he["a_prev"], he["xi"], he["f"])
        se_hist = mdp.add_noise(se_hist)
        se_goal = motion.goal(se["f"], K)
        a_se = pi(se_hist, se_goal)
        proximity = (((a_se - se["a"]) / 20.0).square().mean(-1)
                     / (2.0 * args.policy_strategic_error_sigma ** 2)).neg().exp().detach()
        S_se = unflatten_state(se["s"])
        pred_se = wm.step(S_se, a_se, normalize_xi(se["xi"]), use_residual=False)
        std_se = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        se_err = (wm.state_error(pred_se, unflatten_state(se["s1"])) / std_se).square().mean(-1)
        se_loss = (se_err * proximity).sum() / proximity.sum().clamp_min(1e-4)
        loss = loss + args.policy_strategic_error_weight * se_loss
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    # Optional replay anchor for the WM part of the same optimizer step.  This
    # keeps concurrent coupling tied to real transitions instead of letting a
    # policy return alone reshape the dynamics model.
    policy_loss = loss
    # The policy can use the full imagined horizon while the concurrent WM
    # adapter receives only a short-prefix task gradient. This isolates the
    # long-horizon policy objective from model co-adaptation.
    joint_task_loss = policy_loss
    if args.joint_wm_policy_horizon > 0:
        joint_task_loss = -J[:args.joint_wm_policy_horizon].mean()
    replay_anchor = None
    transition_anchor = None
    replay_scale = 0.0
    if args.joint_wm_replay_weight > 0.0:
        joint_window = args.joint_wm_replay_window or args.wm_window
        wj = buffer.sample_windows(args.wm_batch, joint_window)
        if joint_window == args.wm_window:
            Sj = wm_rollout(wj["s"], wj["a"], wj["xi"], use_residual=residual_joint)
        else:
            Sj = _wm_rollout_window(wj["s"], wj["a"], wj["xi"], joint_window,
                                    use_residual=residual_joint)
        if args.joint_wm_replay_sensitivity > 0.0:
            s1_sens = wj["s1"].detach().requires_grad_(True)
            sens_state = unflatten_state(s1_sens)
            sens_reward = mdp.reward(sens_state, wj["f"], wj["a"], wj["a_prev"])
            sens_grad = torch.autograd.grad(sens_reward.sum(), s1_sens, retain_graph=False)[0]
            sens = sens_grad.square().sum(-1).sqrt()
            sens = sens / sens.detach().mean().clamp_min(1e-6)
            sens_weight = 1.0 + args.joint_wm_replay_sensitivity * sens.clamp(max=5.0)
            replay_anchor, _ = _wm_err_weighted(Sj, wj["s1"], wj["mask"], sens_weight)
        else:
            replay_anchor, _ = wm_err(Sj, wj["s1"], wj["mask"])
        if args.joint_wm_policy_transition_ratio > 0.0 and residual_joint:
            # Supervise the residual at the *current policy action* while the
            # state and next-state target remain a real replay transition.
            # The action is detached so this term updates only the residual WM.
            a_pi = pi(o_hist, g0).detach()
            S_pi = wm.step(S, a_pi, xi_n, use_residual=True)
            std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
            e_pi = (wm.state_error(S_pi, unflatten_state(s["s1"])) / std).square().mean(-1)
            # When an occupancy gate is available, discard mismatched
            # behavior-policy transitions instead of forcing their next state
            # onto the current policy action.
            if residual_gate is not None:
                transition_anchor = (e_pi * residual_gate).sum() / residual_gate.sum().clamp_min(1e-4)
            else:
                transition_anchor = e_pi.mean()
            replay_anchor = replay_anchor + args.joint_wm_policy_transition_ratio * transition_anchor
        if args.joint_residual_l2 > 0.0 and residual_joint:
            # Keep the policy-sensitive adapter close to its zero-initialized
            # nominal model. This is an explicit parameter anchor, separate
            # from replay prediction error and easy to ablate.
            n = sum(p.numel() for p in wm.residual_net.parameters())
            residual_l2 = torch.stack([p.square().sum() for p in wm.residual_net.parameters()]).sum() / max(n, 1)
            replay_anchor = replay_anchor + args.joint_residual_l2 * residual_l2
        if args.joint_residual_dual and residual_joint:
            # Adapt the replay multiplier from a paired nominal-vs-residual
            # fidelity violation on the same real window.
            with torch.no_grad():
                # replay_anchor already is the scalar residual replay error
                # (plus any explicitly requested anchor terms); reusing its
                # detached value avoids a second compiled WM traversal.
                residual_err = replay_anchor.detach() / max(joint_window, 1)
                # Prefer the current standalone nominal WM loss. The fallback
                # is only used before the first replay update is available.
                reference = last_nominal_wm_loss if last_nominal_wm_loss > 0.0 else residual_dual_reference
                if reference is None:
                    reference = float(residual_err.detach().cpu())
                    residual_dual_reference = reference
                if residual_dual is None:
                    residual_dual = args.joint_wm_replay_weight
                violation = float((residual_err - reference
                                   - args.joint_residual_dual_target).cpu())
                residual_dual = max(0.0, min(10.0, residual_dual + args.joint_residual_dual_lr * violation))
        # A WM gradient hook below scales the complete WM gradient by the
        # policy coupling coefficient.  Compensate the replay term here so its
        # effective coefficient remains exactly joint_wm_replay_weight.
        if args.joint_residual_dual and residual_joint:
            # The WM hook below multiplies the whole residual gradient by the
            # policy weight, so convert the desired effective replay weight
            # back to the loss coefficient here.
            replay_scale = residual_dual / max(args.joint_wm_policy_weight, 1e-8)
        elif args.joint_wm_policy_weight > 0.0 and not (args.joint_wm_policy_last_layer
                                                        or args.joint_wm_policy_root_outputs
                                                        or args.joint_wm_policy_action_path):
            replay_scale = args.joint_wm_replay_weight / args.joint_wm_policy_weight
        else:
            replay_scale = args.joint_wm_replay_weight
        loss = policy_loss + replay_scale * replay_anchor
    joint_grad_cos = torch.zeros((), device=dev)
    task_grads = replay_grads = None
    structured_params = None
    structured_pi_grads = None
    structured_wm_params = None
    structured_wm_replay_grads = None
    structured_mode = (args.joint_wm_policy_last_layer or args.joint_wm_policy_root_outputs
                       or args.joint_wm_policy_action_path)
    structured_task_masks = None
    structured_task_coeff = args.joint_wm_policy_weight
    if (structured_mode and joint_wm and replay_anchor is not None
            and not residual_joint and args.joint_wm_policy_weight > 0.0):
        if args.joint_wm_policy_last_layer or args.joint_wm_policy_root_outputs:
            structured_params = [wm.net[-1].weight, wm.net[-1].bias]
            if args.joint_wm_policy_root_outputs:
                row_mask = torch.zeros_like(wm.net[-1].weight)
                row_mask[:12] = 1.0
                bias_mask = torch.zeros_like(wm.net[-1].bias)
                bias_mask[:12] = 1.0
                structured_task_masks = [row_mask, bias_mask]
            else:
                structured_task_masks = [None, None]
        else:
            structured_params = [wm.net[0].weight]
            mask = torch.zeros_like(wm.net[0].weight)
            mask[:, INV_DIM:INV_DIM + NUM_J] = 1.0
            structured_task_masks = [mask]
        structured_pi_params = tuple(pi.parameters())
        task_all = torch.autograd.grad(
            joint_task_loss, tuple(structured_params) + structured_pi_params,
            retain_graph=False, allow_unused=True)
        task_grads = task_all[:len(structured_params)]
        structured_pi_grads = task_all[len(structured_params):]
        if args.joint_gradient_task_ema > 0.0:
            beta = args.joint_gradient_task_ema
            current = tuple((g if g is not None else torch.zeros_like(p)).detach()
                            for p, g in zip(structured_params, task_grads))
            if task_grad_ema is None or len(task_grad_ema) != len(current):
                task_grad_ema = tuple(g.clone() for g in current)
            else:
                task_grad_ema = tuple(beta * old + (1.0 - beta) * new
                                      for old, new in zip(task_grad_ema, current))
            task_grads = task_grad_ema
        structured_wm_params = tuple(p for p in wm.parameters() if p.requires_grad)
        structured_wm_replay_grads = torch.autograd.grad(
            replay_anchor, structured_wm_params, retain_graph=False, allow_unused=True)
        replay_by_id = {id(p): g for p, g in zip(structured_wm_params,
                                                 structured_wm_replay_grads)}
        replay_grads = tuple(replay_by_id[id(p)] for p in structured_params)
        if (args.joint_wm_policy_action_path and
                args.joint_wm_policy_action_norm_ratio > 0.0):
            gt_norm = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                 for p, g in zip(structured_params, task_grads)]).norm().clamp_min(1e-8)
            gr_norm = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                 for p, g in zip(structured_params, replay_grads)]).norm().clamp_min(1e-8)
            structured_task_coeff = (args.joint_wm_policy_action_norm_ratio
                                     * gr_norm / gt_norm).clamp(max=10.0)
        structured_conflict = None
        if args.joint_gradient_conflict_gate or args.joint_gradient_projection:
            gt_vec = []
            gr_vec = []
            for p, gt_i, gr_i, mask in zip(structured_params, task_grads,
                                           replay_grads, structured_task_masks):
                gt_i = torch.zeros_like(p) if gt_i is None else gt_i
                gr_i = torch.zeros_like(p) if gr_i is None else gr_i
                if mask is not None:
                    gt_i = gt_i * mask
                gt_vec.append(gt_i.reshape(-1)); gr_vec.append(gr_i.reshape(-1))
            gt_vec = torch.cat(gt_vec); gr_vec = torch.cat(gr_vec)
            structured_conflict = (gt_vec * gr_vec).sum()
            if args.joint_gradient_conflict_gate and structured_conflict < 0:
                structured_task_coeff = 0.0
            if args.joint_gradient_projection:
                proj = (structured_conflict / gr_vec.square().sum().clamp_min(1e-8)).clamp_max(0.0)
            else:
                proj = None
        else:
            proj = None
    if ((args.joint_gradient_diagnostic or args.joint_gradient_projection or args.joint_gradient_conflict_gate or args.joint_gradient_norm_match or args.joint_adversarial_residual)
            and residual_joint and replay_anchor is not None):
        residual_params = [p for p in wm.residual_net.parameters() if p.requires_grad]
        gt = torch.autograd.grad(joint_task_loss, residual_params, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(replay_anchor, residual_params, retain_graph=True, allow_unused=True)
        if args.joint_gradient_task_ema > 0.0:
            beta = args.joint_gradient_task_ema
            current = tuple((g if g is not None else torch.zeros_like(p)).detach()
                            for p, g in zip(residual_params, gt))
            if task_grad_ema is None:
                task_grad_ema = tuple(g.clone() for g in current)
            else:
                task_grad_ema = tuple(beta * old + (1.0 - beta) * new
                                      for old, new in zip(task_grad_ema, current))
            gt = task_grad_ema
        task_grads, replay_grads = gt, gr
        vt = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for p, g in zip(residual_params, gt)])
        vr = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for p, g in zip(residual_params, gr)])
        joint_grad_cos = (vt * vr).sum() / (vt.square().sum().sqrt() * vr.square().sum().sqrt()).clamp_min(1e-8)
    wm_hooks = []
    if args.joint_wm_policy_weight > 0.0 and not structured_mode:
        scale = args.joint_wm_policy_weight
        for p in wm.parameters():
            if p.requires_grad:
                wm_hooks.append(p.register_hook(lambda grad, scale=scale: grad * scale))
    if clear_grad:
        opt_pi.zero_grad(set_to_none=True)
        if opt_target is not None:
            opt_target.zero_grad(set_to_none=True)
    if joint_wm:
        opt_wm.zero_grad(set_to_none=True)
    if structured_wm_params is None:
        loss.backward()
    else:
        for p, g_i in zip(structured_wm_params, structured_wm_replay_grads):
            p.grad = torch.zeros_like(p) if g_i is None else args.joint_wm_replay_weight * g_i.detach()
    if structured_pi_grads is not None:
        for p, g_i in zip(structured_pi_params, structured_pi_grads):
            p.grad = torch.zeros_like(p) if g_i is None else g_i.detach()
    if structured_params is not None:
        task_active = (it >= args.joint_wm_policy_start_iter and
                       (args.joint_wm_policy_period == 1 or it % args.joint_wm_policy_period == 0))
        for p, gt_i, gr_i, mask in zip(structured_params, task_grads, replay_grads,
                                       structured_task_masks):
            gt_i = torch.zeros_like(p) if gt_i is None else gt_i.detach()
            gr_i = torch.zeros_like(p) if gr_i is None else gr_i.detach()
            if mask is not None:
                gt_i = gt_i * mask
            if proj is not None:
                gt_i = gt_i - proj * gr_i
            p.grad = (structured_task_coeff * gt_i if task_active else 0.0) + args.joint_wm_replay_weight * gr_i
    if (args.joint_gradient_projection or args.joint_gradient_conflict_gate or args.joint_gradient_norm_match or args.joint_adversarial_residual) and task_grads is not None and not structured_mode:
        # Replace only the residual gradients. Policy/critic gradients remain
        # the ordinary backward result; nominal WM is frozen in residual mode.
        vt = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                        for p, g in zip(residual_params, task_grads)])
        vr = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                        for p, g in zip(residual_params, replay_grads)])
        coeff = (vt * vr).sum() / vr.square().sum().clamp_min(1e-8)
        coeff = coeff.clamp_max(0.0)
        adversarial_coeff = args.joint_wm_policy_weight
        if args.joint_adversarial_residual:
            # Bound the adversarial residual update relative to the replay
            # fidelity gradient.  A raw sign-flipped task gradient can be
            # orders of magnitude larger and immediately destroy the model.
            task_norm = vt.norm().clamp_min(1e-8)
            replay_norm = vr.norm().clamp_min(1e-8)
            adversarial_coeff = (args.joint_gradient_norm_ratio
                                 * replay_norm / task_norm).clamp(max=10.0)
        for p, gt_i, gr_i in zip(residual_params, task_grads, replay_grads):
            gt_i = torch.zeros_like(p) if gt_i is None else gt_i.detach()
            gr_i = torch.zeros_like(p) if gr_i is None else gr_i.detach()
            task_active = (it >= args.joint_wm_policy_start_iter and
                           (args.joint_wm_policy_period == 1 or it % args.joint_wm_policy_period == 0))
            if not task_active:
                p.grad = args.joint_wm_replay_weight * gr_i
            elif args.joint_adversarial_residual:
                p.grad = -adversarial_coeff * gt_i + replay_scale * gr_i
            elif args.joint_gradient_conflict_gate and (vt * vr).sum() < 0.0:
                # On a conflicting batch, keep the replay-fidelity update and
                # suppress only the residual policy gradient. This is stricter
                # than projection and makes the gate's policy consequence
                # explicit for multi-seed diagnostics.
                p.grad = replay_scale * gr_i
            elif args.joint_gradient_norm_match:
                # Match the task/replay gradient norm ratio to the user
                # coefficients. The raw task norm grows strongly with BPTT
                # horizon, so fixed scalar coupling otherwise changes meaning
                # across horizon/seed. Keep the task direction intact.
                task_norm = vt.norm().clamp_min(1e-8)
                replay_norm = vr.norm().clamp_min(1e-8)
                task_coeff = (args.joint_wm_policy_weight * args.joint_gradient_norm_ratio
                              * replay_norm / task_norm)
                task_coeff = task_coeff.clamp(max=10.0)
                p.grad = task_coeff * gt_i + args.joint_wm_replay_weight * gr_i
            elif args.joint_residual_dual:
                p.grad = args.joint_wm_policy_weight * gt_i + replay_scale * gr_i
            else:
                p.grad = args.joint_wm_policy_weight * (gt_i - coeff * gr_i) + replay_scale * gr_i
    for hook in wm_hooks:
        hook.remove()
    if residual_joint and args.joint_residual_grad_clip > 0.0:
        rg = torch.stack([p.grad.detach().square().sum()
                          for p in wm.residual_net.parameters()
                          if p.grad is not None]).sum().sqrt()
        rscale = (args.joint_residual_grad_clip / rg.clamp_min(1e-8)).clamp_max(1.0)
        for p in wm.residual_net.parameters():
            if p.grad is not None:
                p.grad.mul_(rscale)
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    if step_optimizer:
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    else:
        gn = torch.sqrt(torch.stack([
            p.grad.detach().square().sum() for p in pi.parameters() if p.grad is not None
        ]).sum().clamp_min(1e-12))
    if step_optimizer and torch.isfinite(gn):
        opt_pi.step()
        if opt_target is not None:
            nn.utils.clip_grad_norm_(policy_target_wrapper.adapter_parameters(), 1.0)
            opt_target.step()
            if args.policy_target_adapter_max_norm > 0.0:
                # Project the task adapter back into a bounded parameter ball
                # after every optimizer step.  The nominal WM is untouched;
                # this only limits cumulative policy-induced model drift.
                with torch.no_grad():
                    params = policy_target_wrapper.adapter_parameters()
                    norm = torch.sqrt(torch.stack([p.square().sum() for p in params]).sum())
                    scale = (args.policy_target_adapter_max_norm / norm.clamp_min(1e-8)).clamp_max(1.0)
                    for p in params:
                        p.mul_(scale)
        if joint_wm:
            # Keep the policy update unchanged while applying only a scaled
            # copy of its gradient to the world model.  This isolates the
            # concurrent coupling from a hidden policy learning-rate change.
            nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt_wm.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        if critic_target is not None:
            ema_update_(critic_target, critic, args.wm_td_target_beta)
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    # Report only the policy objective; the replay anchor is an auxiliary WM
    # term and can be rescaled internally to preserve its effective gradient.
    return (policy_loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(),
            gn.detach(), c.mean().detach(), gate_fraction.mean().detach(), joint_grad_cos.detach())


def shadow_policy_update(reuse_batch=None):
    """Run one policy update on the isolated branch through the target WM."""
    global pi, critic, opt_pi, opt_v, critic_target, policy_shadow_active
    if shadow_branch is None:
        return None
    saved = (pi, critic, opt_pi, opt_v, critic_target, policy_shadow_active)
    pi = shadow_branch.shadow_policy
    critic = shadow_branch.shadow_critic
    opt_pi = shadow_opt_pi
    opt_v = shadow_opt_v
    critic_target = shadow_critic_target
    shadow_before = ([p.detach().clone() for p in pi.parameters()]
                     if args.shadow_policy_max_step_norm > 0.0 else None)
    policy_shadow_active = True
    try:
        result = policy_update(reuse_batch)
        if shadow_before is not None:
            cap_parameter_step(pi, shadow_before, args.shadow_policy_max_step_norm)
        return result
    finally:
        pi, critic, opt_pi, opt_v, critic_target, policy_shadow_active = saved


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma,
                      reset_sampling_mode=args.reset_sampling_mode,
                      kinematic_clusters=args.reset_kinematic_clusters,
                      fall_grace_ref_height=args.fall_grace_ref_height,
                      fall_grace_ref_tilt=args.fall_grace_ref_tilt)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
    onpolicy_buffer = ReplayBuffer(args.onpolicy_buffer_steps or args.buffer_steps, args.num_envs, dev)
    strategic_probe_buffer = (StrategicProbeBuffer(args.strategic_probe_capacity, dev)
                              if args.strategic_probe_capacity > 0 else None)
    wm_buffer = buffer
    last_fresh_wm_err = 0.0
    if resume_buffer is not None:
        buffer.load_state_dict(resume_buffer)
    if strategic_probe_buffer is not None and resume_probe is not None:
        # Older checkpoints have no strategic probe state; an empty buffer is
        # the correct backward-compatible continuation in that case.
        strategic_probe_buffer.load_state_dict(resume_probe)
    if strategic_probe_buffer is not None and args.strategic_probe_dataset is not None:
        probe_data = torch.load(args.strategic_probe_dataset, map_location=dev, weights_only=False)
        required = {"s", "a", "xi", "s1"}
        if not required.issubset(probe_data):
            parser.error("strategic probe dataset must contain s, a, xi, and s1")
        probe_score = probe_data.get("probe_score", torch.ones(probe_data["s"].shape[0], device=dev))
        probe_source = probe_data.get("f", torch.zeros(probe_data["s"].shape[0], dtype=torch.long, device=dev))
        strategic_probe_buffer.add(
            {key: probe_data[key].to(dev) for key in ("s", "a", "xi", "s1")},
            probe_score.to(dev), probe_source.to(dev))
    obs_hist = torch.zeros(args.num_envs, Hh, policy_obs_dim, device=dev)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict, target_buffer=None):
        global obs_hist, sim_steps
        dst = buffer if target_buffer is None else target_buffer
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = motion.goal(env.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            probe_active = (not args.adaptive_probe_onpolicy_only or target_buffer is not None)
            if args.adaptive_action_direction > 0.0 and probe_active:
                # Error-MDP-inspired directional exploration: differentiate a
                # detached ensemble disagreement proxy with respect to the
                # action, then query the real simulator in that direction.
                # The policy and WM parameters remain untouched; only the
                # collected action is perturbed.
                with torch.enable_grad():
                    a_probe = a.detach().requires_grad_(True)
                    xi_n_probe = normalize_xi(env.xi)
                    S0_probe = env.state()
                    p0 = ensemble_gate_wms[0].step(
                        S0_probe, a_probe, xi_n_probe, use_residual=ensemble_gate_residual)
                    p1 = ensemble_gate_wms[1].step(
                        S0_probe, a_probe, xi_n_probe, use_residual=ensemble_gate_residual)
                    std_probe = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
                    probe_error = (wm.state_error(p0, p1) / std_probe).square().mean(-1)
                    direction = torch.autograd.grad(
                        probe_error.sum(), a_probe, retain_graph=False,
                        create_graph=False, allow_unused=False)[0]
                    direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                active = (torch.rand(args.num_envs, device=dev)
                          < args.adaptive_action_direction_fraction).to(a.dtype)
                sign = torch.ones(args.num_envs, device=dev, dtype=a.dtype)
                if args.adaptive_action_direction_symmetric:
                    sign = torch.where(torch.rand(args.num_envs, device=dev) < 0.5, -sign, sign)
                a = a + args.adaptive_action_direction * active[:, None] * sign[:, None] * direction.detach()
            noise_scale = torch.ones(args.num_envs, device=dev)
            if args.adaptive_action_noise > 0.0 and probe_active:
                # Active data collection: use frozen ensemble spread only to
                # increase real-simulator exploration where the current model
                # is uncertain. WM fitting itself remains uniform factual replay.
                xi_n_collect = normalize_xi(env.xi)
                S0_collect = env.state()
                m0 = ensemble_gate_wms[0].step(S0_collect, a, xi_n_collect, use_residual=ensemble_gate_residual)
                m1 = ensemble_gate_wms[1].step(S0_collect, a, xi_n_collect, use_residual=ensemble_gate_residual)
                std_collect = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
                dis_collect = (wm.state_error(m0, m1) / std_collect).square().mean(-1)
                dis_collect = dis_collect / dis_collect.mean().clamp_min(1e-6)
                if args.adaptive_action_noise_threshold > 0.0:
                    risk = (dis_collect - args.adaptive_action_noise_threshold).relu()
                    adaptive_component = args.adaptive_action_noise * risk.clamp(max=5.0)
                else:
                    adaptive_component = args.adaptive_action_noise * dis_collect.clamp(max=5.0)
                if args.adaptive_noise_recoverability > 0.0:
                    predicted_term = mdp.soft_termination(
                        wm.step(S0_collect, a, xi_n_collect), env.f + 1,
                        tz=args.term_soft_height, tt=args.term_soft_tilt)
                    recoverable = (predicted_term <= args.adaptive_noise_recoverability).to(a.dtype)
                    adaptive_component = adaptive_component * recoverable
                noise_scale = (1.0 + adaptive_component).detach()
            if args.deterministic_collect_fraction > 0.0 and args.action_noise > 0.0:
                noisy = (torch.rand(args.num_envs, device=dev)
                         >= args.deterministic_collect_fraction).to(a.dtype).unsqueeze(-1)
                a = a + args.action_noise * noise_scale[:, None] * torch.randn_like(a) * noisy
            else:
                a = a + args.action_noise * noise_scale[:, None] * torch.randn_like(a)
            a = torch.clamp(a, -20, 20)
            out = env.step(a)
            dst.add(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            if not wm_norm_frozen:
                wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
    required_window = max(args.wm_window, args.wm_aux_window, args.wm_aux_window2)
    if buffer.size < required_window:
        collect(max(args.warmup_steps, required_window), stats)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    existing_header = ""
    if start_it > 0 and os.path.exists(log_path):
        with open(log_path) as prev_log:
            existing_header = prev_log.readline()
    write_residual_log = start_it == 0 or "residual_param_norm" in existing_header
    write_residual_head_log = start_it == 0 or "residual_head_norm" in existing_header
    write_jcos_log = start_it == 0 or "joint_grad_cos" in existing_header
    write_policy_norm_log = args.shadow_policy or "policy_param_norm" in existing_header
    write_target_log = start_it == 0 or "policy_target_audit_h1" in existing_header
    log_f = open(log_path, "a")
    if start_it == 0:
        header = "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,gate_fraction,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,wm_dis_corr,wm_diag_err1,term_brier,term_prob,term_rate,residual_param_norm,residual_head_norm,joint_grad_cos"
        if write_policy_norm_log:
            header += ",policy_param_norm,shadow_policy_param_norm"
        if write_target_log:
            header += ",policy_target_audit_h1,policy_target_audit_h4,policy_target_nominal_h1,policy_target_nominal_h4,policy_target_kl_proxy,policy_target_kl_dual,policy_target_accepts,policy_target_rejects,policy_target_adapter_norm"
        log_f.write(header + "\n")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "gate": [], "dcorr": [], "derr": [], "tb": [], "tp": [], "tr": [], "jcos": []}
    for it in range(start_it, args.iterations):
        refresh_policy_target_control()
        lr = lr_at(it)
        for o in (opt_wm, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        for gparam in opt_v.param_groups:
            gparam["lr"] = lr * args.policy_lr_scale
        # After a long WM-only phase, ramp the policy LR to avoid a sudden
        # distribution shift in the first imagined update. Default is unchanged.
        if args.policy_ramp_iters > 0 and it >= args.wm_only_iters:
            pi_scale = min(1.0, (it - args.wm_only_iters + 1) / args.policy_ramp_iters)
        elif args.policy_ramp_iters > 0:
            pi_scale = 0.0
        else:
            pi_scale = 1.0
        for gparam in opt_pi.param_groups:
            gparam["lr"] = lr * args.policy_lr_scale * pi_scale * (args.shadow_policy_lr_scale if policy_shadow_active else 1.0)
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        # The default is the paper's collect -> WM -> policy ordering.  The
        # optional reversed ordering is a controlled concurrent-training
        # ablation: policy uses the previous model, then the WM fits the newly
        # collected replay in the same iteration.
        policy_due = (it >= args.wm_only_iters and
                      (it - args.wm_only_iters) % args.policy_period == 0)
        if (args.policy_freeze_after_iter > 0
                and it >= args.wm_only_iters + args.policy_freeze_after_iter):
            policy_due = False
        n_policy_updates = (1 if args.policy_updates_warmup_iters > 0
                            and it < args.wm_only_iters + args.policy_updates_warmup_iters
                            else args.policy_updates)
        if (args.policy_updates_until_iter > 0
                and it >= args.wm_only_iters + args.policy_updates_until_iter):
            n_policy_updates = 1
        shared_policy_batch = (sample_valid_starts(args.pi_batch)
                               if args.policy_reuse_batch and policy_due and n_policy_updates > 1
                               else None)
        shadow_saved_external_allowed = policy_target_external_allowed
        if args.shadow_policy and policy_due:
            # Keep the live branch on the nominal WM; the target WM is reserved
            # for the isolated shadow update below.
            policy_target_external_allowed = False
        if args.policy_before_wm and policy_due:
            accumulate = args.policy_accumulate_updates and n_policy_updates > 1
            if accumulate:
                opt_pi.zero_grad(set_to_none=True)
            for policy_idx in range(n_policy_updates):
                if policy_idx == 1 and args.policy_second_update_scale != 1.0:
                    for gparam in opt_pi.param_groups:
                        gparam["lr"] = lr * args.policy_lr_scale * pi_scale * args.policy_second_update_scale * (args.shadow_policy_lr_scale if policy_shadow_active else 1.0)
                pl, vl, ir, gn, surv, gate_frac, jcos = policy_update(
                    shared_policy_batch, step_optimizer=not accumulate, clear_grad=not accumulate)
                if prof:
                    torch.cuda.synchronize(); tp2 = time.time()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv), agg["gate"].append(gate_frac), agg["jcos"].append(jcos)
            if accumulate:
                gn_acc = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
                nn.utils.clip_grad_value_(pi.parameters(), 0.1)
                if torch.isfinite(gn_acc):
                    opt_pi.step()
            for gparam in opt_pi.param_groups:
                gparam["lr"] = lr * args.policy_lr_scale * pi_scale * (args.shadow_policy_lr_scale if policy_shadow_active else 1.0)
            if args.onpolicy_refresh_steps > 0:
                collect(args.onpolicy_refresh_steps, stats, target_buffer=onpolicy_buffer)
        replace_joint_step = (args.joint_replace_wm_step and policy_due and
                              not args.policy_before_wm)
        interleave = (args.interleave_policy_wm and not args.policy_before_wm and policy_due
                      and n_policy_updates > 1 and args.wm_batches > 1 and not replace_joint_step)
        n_wm_batches = (1 if interleave else args.wm_batches) - (1 if replace_joint_step else 0)
        wm_buffer = (onpolicy_buffer if args.onpolicy_replay_fraction >= 1.0
                     and onpolicy_buffer.size >= args.wm_window else buffer)
        for _ in range(n_wm_batches):
            l, e8, dcorr, derr, tb, tp, tr = wm_update()
        last_nominal_wm_loss = float(l.detach().cpu())
        if (args.onpolicy_separate_wm_update and onpolicy_buffer.size >= args.wm_window
                and args.onpolicy_replay_fraction < 1.0):
            # Inject fresh data through an isolated optimizer step so its
            # distribution cannot distort the historical-replay gradient.
            wm_buffer = onpolicy_buffer
            old_lrs = [g["lr"] for g in opt_wm.param_groups]
            for g in opt_wm.param_groups:
                g["lr"] *= args.onpolicy_replay_fraction
            l, e8, dcorr, derr, tb, tp, tr = wm_update()
            for g, old_lr in zip(opt_wm.param_groups, old_lrs):
                g["lr"] = old_lr
        if prof:
            torch.cuda.synchronize(); tp3 = time.time()
        agg["wm"].append(l), agg["wm8"].append(e8), agg["dcorr"].append(dcorr), agg["derr"].append(derr)
        agg["tb"].append(tb), agg["tp"].append(tp), agg["tr"].append(tr)
        # Model-health gate: the leading recipe benefits from a second
        # sequential policy update, but that update can amplify a transiently
        # inaccurate WM.  Compare the current final-step replay error with an
        # EMA and skip only the extra update when the model has degraded.
        n_policy_updates_run = n_policy_updates
        if (args.policy_wm_error_gate_ratio > 0.0 and policy_due
                and not args.policy_before_wm and n_policy_updates > 1):
            current_wm_err = float(e8.detach().cpu())
            if policy_wm_error_ema is None:
                policy_wm_error_ema = current_wm_err
            else:
                if current_wm_err > args.policy_wm_error_gate_ratio * policy_wm_error_ema:
                    n_policy_updates_run = 1
                policy_wm_error_ema = (args.policy_wm_error_gate_beta * policy_wm_error_ema
                                       + (1.0 - args.policy_wm_error_gate_beta) * current_wm_err)
        if (args.policy_freeze_wm_error_ratio > 0.0 and policy_due
                and (it - args.wm_only_iters) >= args.policy_freeze_wm_error_min_iter
                and not args.policy_before_wm):
            current_wm_err = float(e8.detach().cpu())
            if policy_wm_error_ema is None:
                policy_wm_error_ema = current_wm_err
            else:
                if current_wm_err > args.policy_freeze_wm_error_ratio * policy_wm_error_ema:
                    n_policy_updates_run = 0
                policy_wm_error_ema = (args.policy_freeze_wm_error_beta * policy_wm_error_ema
                                       + (1.0 - args.policy_freeze_wm_error_beta) * current_wm_err)
        interleave = interleave and n_policy_updates_run > 1
        if not args.policy_before_wm and policy_due:
            accumulate = args.policy_accumulate_updates and n_policy_updates_run > 1 and not interleave
            if accumulate:
                opt_pi.zero_grad(set_to_none=True)
            for policy_idx in range(n_policy_updates_run):
                if interleave and policy_idx > 0:
                    # The default runs all WM batches before policy adaptation.
                    # This controlled ablation refreshes the WM between policy
                    # steps, exposing the second policy update to a newer model.
                    l, e8, dcorr, derr, tb, tp, tr = wm_update()
                    last_nominal_wm_loss = float(l.detach().cpu())
                    agg["wm"].append(l), agg["wm8"].append(e8), agg["dcorr"].append(dcorr), agg["derr"].append(derr)
                    agg["tb"].append(tb), agg["tp"].append(tp), agg["tr"].append(tr)
                if policy_idx == 1 and args.policy_second_update_scale != 1.0:
                    for gparam in opt_pi.param_groups:
                        gparam["lr"] = lr * args.policy_lr_scale * pi_scale * args.policy_second_update_scale * (args.shadow_policy_lr_scale if policy_shadow_active else 1.0)
                pl, vl, ir, gn, surv, gate_frac, jcos = policy_update(
                    shared_policy_batch, step_optimizer=not accumulate, clear_grad=not accumulate)
                if prof:
                    torch.cuda.synchronize(); tp3 = time.time()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv), agg["gate"].append(gate_frac), agg["jcos"].append(jcos)
            if accumulate:
                gn_acc = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
                nn.utils.clip_grad_value_(pi.parameters(), 0.1)
                if torch.isfinite(gn_acc):
                    opt_pi.step()
            for gparam in opt_pi.param_groups:
                gparam["lr"] = lr * args.policy_lr_scale * pi_scale * (args.shadow_policy_lr_scale if policy_shadow_active else 1.0)
        if args.shadow_policy and policy_due:
            for _ in range(args.shadow_policy_updates):
                shadow_result = shadow_policy_update(shared_policy_batch)
                if shadow_result is not None:
                    spl, svl, sir, sgn, ssurv, sgate, sjcos = shadow_result
                    agg["pi"].append(spl), agg["v"].append(svl), agg["ir"].append(sir)
                    agg["gn"].append(sgn), agg["surv"].append(ssurv), agg["gate"].append(sgate), agg["jcos"].append(sjcos)
            policy_target_external_allowed = shadow_saved_external_allowed
        if prof:
            if args.policy_before_wm:
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | policy+critic {1e3*(tp2-tp1):.0f} ms | wm {1e3*(tp3-tp2):.0f} ms", flush=True)
            else:
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            residual_param_norm = 0.0
            residual_head_norm = 0.0
            if wm.residual_net is not None:
                residual_param_norm = float(torch.sqrt(torch.stack([
                    p.detach().square().sum() for p in wm.residual_net.parameters()
                ]).sum()).cpu())
                residual_head = wm.residual_net[-1]
                residual_head_norm = float(torch.sqrt(torch.stack([
                    p.detach().square().sum() for p in residual_head.parameters()
                ]).sum()).cpu())
            policy_param_norm = float(torch.sqrt(torch.stack([
                p.detach().square().sum() for p in pi.parameters()
            ]).sum()).cpu())
            shadow_policy_param_norm = 0.0
            if shadow_branch is not None:
                shadow_policy_param_norm = float(torch.sqrt(torch.stack([
                    p.detach().square().sum() for p in shadow_branch.shadow_policy.parameters()
                ]).sum()).cpu())
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(agg["gate"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                   mean(agg["dcorr"]), mean(agg["derr"]), mean(agg["tb"]), mean(agg["tp"]), mean(agg["tr"])]  # fmt: skip
            if write_residual_log:
                row.append(residual_param_norm)
            if write_residual_head_log:
                row.append(residual_head_norm)
            if write_jcos_log:
                row.append(mean(agg["jcos"]))
            if write_policy_norm_log:
                row.extend((policy_param_norm, shadow_policy_param_norm))
            if write_target_log:
                target_norm = (float(policy_target_wrapper.adapter_norm().detach().cpu())
                               if policy_target_wrapper is not None else 0.0)
                row.extend((policy_target_audit_h1, policy_target_audit_h4,
                            policy_target_nominal_h1, policy_target_nominal_h4,
                            float(policy_target_wrapper.replay_kl_proxy.detach().cpu())
                            if policy_target_wrapper is not None else float("inf"),
                            policy_target_kl_dual,
                            policy_target_audit_passes, policy_target_audit_failures,
                            target_norm))
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} gate {row[10]:.3f} "
                  f"| sim_r {row[11]:.3f} ep_len {row[12]:.1f} ep_ret {row[13]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
            if args.onpolicy_replay_fraction > 0.0:
                print(f"fresh_wm_err {last_fresh_wm_err:.3f}", flush=True)
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            ck = {"wm": wm.state_dict(), "wm_action_recovery": wm_action_recovery.state_dict() if wm_action_recovery is not None else None,
                  "wm_action_transition_encoder": wm_action_transition_encoder.state_dict() if wm_action_transition_encoder is not None else None,
                  "wm_action_encoder": wm_action_encoder.state_dict() if wm_action_encoder is not None else None,
                  "wm_target": wm_target.state_dict() if wm_target is not None else None,
                  "policy_target_adapter": (policy_target_wrapper.adapter.state_dict()
                                             if policy_target_wrapper is not None
                                             and policy_target_wrapper.adapter is not None else None),
                  "pi": pi.state_dict(), "critic": critic.state_dict(),
                  "shadow_policy": (shadow_branch.state_dict() if shadow_branch is not None else None),
                  "critic_target": critic_target.state_dict() if critic_target is not None else None,
                  "opt_wm": opt_wm.state_dict(),
                  "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                  "shadow_opt_pi": (shadow_opt_pi.state_dict() if shadow_opt_pi is not None else None),
                  "shadow_opt_v": (shadow_opt_v.state_dict() if shadow_opt_v is not None else None),
                  "opt_target": opt_target.state_dict() if opt_target is not None else None,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "residual_dual": residual_dual,
                  "residual_dual_reference": residual_dual_reference,
                  "policy_target_enabled": (policy_target_wrapper.enabled
                                             if policy_target_wrapper is not None else False),
                  "policy_target_kl_proxy": (float(policy_target_wrapper.replay_kl_proxy.detach().cpu())
                                             if policy_target_wrapper is not None else float("inf")),
                  "policy_target_kl_dual": policy_target_kl_dual,
                  "policy_target_audit_error": (float(policy_target_wrapper.audit_error.detach().cpu())
                                                if policy_target_wrapper is not None else float("inf")),
                  "policy_target_audit_passes": policy_target_audit_passes,
                  "policy_target_audit_failures": policy_target_audit_failures,
                  "policy_target_audit_h1": policy_target_audit_h1,
                  "policy_target_audit_h4": policy_target_audit_h4,
                  "policy_target_nominal_h1": policy_target_nominal_h1,
                  "policy_target_nominal_h4": policy_target_nominal_h4,
                  "policy_target_last_good": policy_target_last_good,
                  "strategic_probe": (strategic_probe_buffer.state_dict()
                                       if strategic_probe_buffer is not None else None),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "buffer": buffer.state_dict()}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("pi", "wm", "wm_target", "critic_target", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
            if shadow_branch is not None:
                shadow_small = {"pi": shadow_branch.shadow_policy.state_dict(),
                                "wm": wm.state_dict(),
                                "wm_target": wm_target.state_dict() if wm_target is not None else None,
                                "iteration": it + 1, "wall_time": time.time() - t_start,
                                "sim_steps": sim_steps, "args": vars(args),
                                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "source_checkpoint": path}
                shadow_path = os.path.join(args.log_dir, f"shadow_ckpt_{it + 1:06d}.pt")
                torch.save(shadow_small, shadow_path + ".tmp")
                os.replace(shadow_path + ".tmp", shadow_path)
