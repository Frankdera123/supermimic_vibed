# CVF1B_CRITIC_RESULTS（CVF-1b Critic 微调对照实验结果）

日期：2026-11-20。数据：44 个全新起点（Train 12 clips/24 起点、Val 4/8、Test 6/12，
零 CVF-0 重叠），真实 Simulator 采集 78,848 env transitions（== simulator_batch_steps，
1.83 GPU-h）。训练：C1/C2/C0 从同一 C-init（B0@2100_post）微调 500 步（匹配步数/批/流/
lr=2e-4 Adam，RunningNorm 冻结 eval 模式），主时域 T=96（冻结依据见 OVERNIGHT_PROGRESS）。
评价：Val 做模型选择（n=48 pairs；跨 ε strict-gated n=8），Test 一次性盲测确认
（n=72/gated 6）。裁定：**INCONCLUSIVE（负面倾向）**。

## 1. 主结果（pair 差分响应 ΔV vs ΔG⁹⁶，非 circular 口径，无 learned 尾项）

| Critic | Val ALL sign/sp/ratio/wss | Val gated sign/sp | Test ALL sign/sp/ratio/wss |
|---|---|---|---|
| **C-init** | **0.729 / +0.504 / 1.152 / 0.564** | **1.000 / +0.929** | 0.625 / +0.379 / 0.456 / 0.286 |
| C0 (WM 目标) | 0.708 / +0.458 / 0.930 / 0.557 | 1.000 / +0.905 | 0.611 / +0.326 / 0.467 / 0.248 |
| C1 (Sim 标量) | 0.688 / +0.491 / 1.004 / 0.593 | 1.000 / +0.881 | 0.611 / +0.409 / 0.403 / 0.395 |
| C2 (Sim 标量+pair) | 0.688 / +0.444 / 1.239 / 0.407 | 0.875 / +0.333 | 0.639 / +0.501 / 0.697 / 0.290 |

- **没有任何微调组在 Val 上超过 C-init**；C1 vs C-init 的 gated 配对检验方向为
  C-init 显著更好（mean |残差| 差 −0.0026，cluster CI [0.00005, 0.010] 不跨 0）。
- **C2 在 Val gated 层明确退化**（sign 1.000→0.875，Spearman 0.881→0.333）。
- Test 上 C2 各项略优（sign 0.639 / sp +0.501 / rmse 最低 / ratio 0.697）但与 Val 方向
  相反——跨 split flip-flop，属样本量不足 + 标签噪声的典型表现，**不构成 PASS 证据**
  （协议禁止挑选 Test 有利面）。
- 分族：delta_2100/2300（policy-reachable 主族）所有组 sign≈0.75（Val）/0.54–0.67（Test），
  微调未拉开差距；random 族全组 ≈0.5–0.6（标签本身近噪声）。
- **幅度普遍欠响应**：所有组（含 C-init）Test ratio 0.40–0.70，与 CVF-0 "幅度失败是
  WM+Critic 共有" 一致。

## 2. D6 预注册门限核对（C2 vs C1，Val 判定 split）

| 门限 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 1. Value RMSE 退化 ≤10% | ≥ −10% | −17.1%（改善） | ✓ |
| 2. Reachable sign ≥+10pp | ≥ +10pp | ALL 0pp；gated −12.5pp | ✗ |
| 3. Reachable RMSE ≥10% 改善 | ≥ +10% | ALL −24.6%（恶化）；gated −61.0% | ✗ |
| 4. 无严重幅度膨胀 | ratio 合理 | 1.24（Val ALL）/ 0.99（gated） | ✓ |
| 5. Total-gradient fidelity 优于 C-init | 改善 | sign/sp 全低于 C-init（D4 代用口径） | ✗ |
| 6. 不依赖单一 clip | LOCO span 小 | 0.031（ALL 层） | ✓ |

→ **C2 未通过（3/6 核心门限失败）；C1 亦无正收益。Phase E 门禁条件 ②⑤ 失败，
CVF-2a 不启动**（预注册纪律，无 Condition E 之外的补救训练）。

## 3. 遗忘检查（C5 探针，Val）

标量 value RMSE（vs 含尾项 sim 目标）：C-init 0.448 → C1 0.792 / C2 0.657 / C0 0.464。
C0（同 V_ref 尾合同的 WM continuation 监督）几乎无标量退化；C1/C2 的标量退化与其
pair-强化权重成正比。Pose/random 探针见 stats JSON per-family。

## 4. 裁定

`INCONCLUSIVE`（负面倾向）：pilot 规模（144 train pairs @h96 / 24 起点 / 500 步）下，
真实 Simulator 锚定的标量与差分监督未产生超出 C-init 的 control-relevant 改善；
唯一统计上明确的效应是 **C2 的 pair 约束在 strict-gated 层的退化**。根因诊断
（标签构造缺陷，可修复）见 `CVF1B_MECHANISM_ANALYSIS.md`。
