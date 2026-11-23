# CVF-0_PHASE1A_SMOKE_REPORT（Phase 1a GPU 烟测报告）

日期：2026-11-19。设备：单卡 RTX 5090（`crgf_smoke` 容器）。只读评估，未修改任何模型权重。
实测耗时：首轮（`T_max=200` 触发 18 min 熔断，1,115.8 s ≈ 0.31 GPU-h）；补采轮（`T_max=160` 完整落盘，897.0 s ≈ 0.25 GPU-h）。

## 1. 基础设施与速率标定（全部通过）

1. **`env.restore_state()` 连续注入逐位确定性**：在 `deterministic_mode="run_to_run" + disable_sensors=True + 4 envs` 下，连续 3 次注入 `S_star_sim` 并闭环滚动 200 步 / 160 步，**`repeat_range = 0.0`**（零漂移）。
2. **真实物理状态边界**：`S_star_sim` 上 `sim_slack_min = +0.0611 rad`（全部处于关节硬限位内），`sim_clear_min = -0.0138 m`（真实支撑足底存在约 1.4 cm 柔性接触压缩，证实必须坚持 `Tier-V` 与 `Tier-P-safe` 隔离）。
3. **闭环仿真速率**：含闭环 `pi_old` 推理与特征计算，4-env 实测 **`0.371 s / batch-step`**。

## 2. 门禁通过率、接触分叉与 $\epsilon$/$T$ 校准（[phase1a_rows.csv](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/phase1a/phase1a_rows.csv)）

在 `T=160`（3.2 s）下，两个子空间在两组 $\epsilon$ 对上均呈现 **`passed = 1/4`**：
- **原因归因（非仿真噪声，而是真实接触切换/步态分叉）**：
  - `dance2_subject2 (vel_critic)` 在 $T=32/64$ 时 `sim = (+0.319, +0.657)` 平滑且与 `critic (+0.391)` 同号，但到 $T=128/160$ 发生步态分叉（末端分离 `sep = 2.19–3.83`），导致 `(J+ - J-)/(2ε) ∝ 1/ε`（`-14.22 / -6.65 / -2.76`）。
  - `eps = 0.005` 在接触丰富帧（如 `fight1_subject5`）易被单步微接触跳变放大；而 `eps ∈ {0.015, 0.03}` 在平滑样本上（如 `fallAndGetUp1_subject1` pose `1.215 vs 1.183`，`jumps1_subject5` vel `1.337 vs 1.202`）跨 $\epsilon$ 一致性极佳。
  - 在 **$T=32, 64$** 档，绝大部分轨迹尚未发生长程步态分叉，且已展现出与 WM 高度一致或可分辨的信号（例如 `fallAndGetUp1_subject1 pose_safe`: `T=32/64/128` `sim=(+0.423/+0.770/+1.025)` vs `wm=(+0.428/+0.774/+0.953)`）。

## 3. 对 Phase 1b 协议的精确校准

基于 Phase 1a 实测，Phase 1b 做如下最小必要参数锁定（均在预注册框架内）：
1. **将直接扰动 $\epsilon$ 网格锁定为 `{0.015, 0.03}`**（移除极易受接触微跳变放大的 `0.005`）；
2. **将续推上限调整为 $T_{\max} = 128$**（覆盖 72.4% 折扣回报质量，显著减少 128→160 步区间高发的步态分叉，且单分支耗时从 59 s 降至 ~47 s），并离线同时导出 **`T ∈ {32, 64, 96, 128}`** 四档的门禁与保真度曲线；
3. **16 起点（8 个不同 LAFAN1 clip × 2 帧）精确预算**：
   - **Session A（Oracle 1）**：4 个 4-env 批 × 5 方向（`vel_critic`, `vel_reachable`, `vel_random`, `pose_safe_critic`, `pose_safe_random`）× 2 $\epsilon$ × ±分支 + 基线×2 = 88 分支 × 128 步 ≈ **1.16 GPU-h**。
   - **Session B（Oracle 2）**：4 个 4-env 批 × 4 方向（`delta_2100`, `delta_2300`, `ascent`, `random`）× 2 $\epsilon$ × ±分支 + 基线×2 = 72 分支 × 128 步（$H=32$ 开环 + $T=96$ 闭环续推）≈ **0.95 GPU-h**。
   - **总计 ≈ 2.11 GPU-h**（单卡 RTX 5090，可分两场独立运行，各自落盘）。
