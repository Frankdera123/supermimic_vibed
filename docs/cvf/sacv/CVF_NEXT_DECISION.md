# CVF_NEXT_DECISION（下一阶段决策建议）

日期：2026-11-20。依据：`CVF_OVERNIGHT_RESEARCH_REPORT.md` Q8 与
`CVF1B_MECHANISM_ANALYSIS.md` §4。**本文件仅为建议，不自动执行任何后续实验。**

## 推荐的唯一下一任务：CVF-1b′（修正标签合同的 Sim-anchor 重训）

### 设计要点（每条都有本轮量化依据）

| 修正 | 依据 |
|---|---|
| pair 标签用 ΔG⁹⁶（无尾项）；标量目标保留 Y_T | Train 标签方差 99.5% 为尾项（delta 族 163–203×） |
| 训练族收紧到 delta_2100/2300；random/ascent 仅评价 | Sim ±噪声族（94.9% 不对称）污染差分；C2 gated 退化集中在噪声族权重 |
| ΔG 跨 ε 平均（CRN）或小 ε 网格 | 压 Sim 分岔噪声；CVF-0 gated 率 3–4/16 的教训 |
| 起点 ×3（≥72 起点，Train ≥48）+ Val gated ≥24 对 | 本轮 gated Val n=8 检验功效不足（10pp 门限） |
| 训练步数上限 1000 步 + 早停（Val gated sp 平台即停） | 500 步时 loss 已收敛两个数量级 |
| 可选 2×2：+V_MAX clip 口径统一报告 grad_raw/grad_eff | CVF-0 遗留口径差（当前零影响） |

### 预算估计

- 采集：72 起点 × R2 口径（每起点 14 rollouts × 128）≈ 3.0 GPU-h（可再降档：
  新 clips 24 + 复用本轮 44 起点 → 只采 28 新起点 ≈ 1.2 GPU-h）。
- C0 目标 + 3 组 × 1000 步训练 ≈ 1.2 GPU-h；评价 CPU。
- 总计 **≈2.5 GPU-h**（单卡 RTX 5090）+ 4–5 h wall。

### 预注册判据建议（沿用 D6 并加严）

- 主判据：修复标签后，最优微调组在 Val gated（≥24 对）上 pair sign ≥ C-init +10pp
  且 Spearman 改善 ≥0.10，clip-clustered bootstrap CI 不跨 0；
- 次判据：Test 盲测方向一致；幅度 ratio 向 1 移动且不越上界；
- 传导判据（条件性 Phase E 门禁不变）：D3 WM 域保持 + 单步 ΔJ_Sim 配对检验。

### 若失败的决策树

- 若修复标签后仍无改善 → "Sim 标量/差分监督在该 Critic 容量与该状态分布下不足以
  改善可达方向响应"获得强证据，转向 Route D（速度误差特征 + normalizer 重整，
  CVF-0 的 2×2 建议与本轮工具链兼容）或 Route C（o_hist 条件化）；
- 若 Val 改善但 Phase E 的 ΔJ_Sim 无改善 → 情况 F（WM Jacobian/参数-动作转换），
  高价值负结果路径，工具已就绪。

## 不推荐的方向（本轮证据）

- 继续加大 α 或 pair 权重（在旧标签下被证明放大噪声）；
- 用长程 WM continuation 当教师（CVF-0 CF B 已证伪）；
- 在 n_gated<10 的评价上宣称任何 PASS；
- 直接进入完整在线协同训练（Gate §8 全部禁止项仍然有效）。
