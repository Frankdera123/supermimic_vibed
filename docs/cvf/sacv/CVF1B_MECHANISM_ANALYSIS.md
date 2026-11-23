# CVF1B_MECHANISM_ANALYSIS（机制分析：为什么 sim-anchor 微调没有改善 Critic）

日期：2026-11-20。本报告回答 Goal Q2/Q3/Q4/Q7 的机制层问题，并给出根因分解与可修复设计。

## 1. 决定性根因：pair 标签被 V_ref 尾项淹没（标签构造缺陷）

Train split 的 pair 标签方差分解（ΔY = ΔG⁹⁶ + γ⁹⁶·c₉₆·ΔV_ref(x_T^±)，γ⁹⁶≈0.386）：

| 族 | n | std(ΔG) | std(尾项) | **尾项方差占比** |
|---|---:|---:|---:|---:|
| delta_2100 | 48 | 0.054 | 0.691 | **162.7×** |
| delta_2300 | 48 | 0.068 | 0.970 | **202.8×** |
| random | 24 | 0.043 | 0.506 | **139.7×** |
| ascent | 24 | 0.087 | 0.419 | **23.0×** |

（`metrics/label_snr_decomposition.json`）

即 **ΔY 方差的 99.5%+ 来自 C-init 自己对两个分支末端状态的价值差**。微调的
pair/标量监督实际上是"向 C-init 的末端价值差回归"——**自我回归，几乎不含新信息**
（ΔG 的真实 continuation 信号方差占比 <1%）。这精确预测并解释了实验结果：
- C1/C2 训练后 ≈ C-init（学习目标 ≈ C-init 自身）；
- C2 无增益（pair loss 强化的是尾项差分 = C-init 差分本身）；
- C2 的 gated 退化（Val sp 0.881→0.333）：α_eff = 0.3 × 1/σ²(ΔY) = 74.4 的权重
  放大了噪声族（random/ascent，Sim ±不对称 94.9% 的族）的差分噪声。

**CVF-0 教训的适用边界被本轮划清**："有限时域 return 必须拼 γ^T c_T V_ref 尾项"
对标量校准成立（避免 +2.1 假偏差）；但同一尾项进入 **pair 差分标签**时，因
ΔV_ref ≫ ΔG（V_ref 输出量级 ~7，ΔG std ~0.05），信号被淹没 ~200×。

## 2. 次级因素（情况 E 检查清单逐项）

| 假设 | 证据 | 判定 |
|---|---|---|
| Horizon target mismatch | 跨时域秩稳定满足（G64~G96 sp 0.977） | 排除 |
| 数据覆盖不足 | 144 pairs@h96 / 24 起点 vs 938 维 Critic；gated n=8 | **contributing**（规模不足，且与 1 相互放大） |
| Critic 历史信息不足 | 本轮未改输入（合同）；无法归因本轮失败 | 未测（Route C 保持低优先） |
| 真实 return 局部不可辨识 | random 族 sign≈0.5-0.6（全组）：Sim ±不对称族标签近噪声 | **contributing**（方向族选择需收紧到 delta 族） |
| 优化器未适配 | C1 loss 0.0034→0.0003 正常收敛，学习确实发生 | 排除 |
| 灾难性遗忘 | C0 标量 RMSE 几乎不退化；C1/C2 退化与 pair 权重成正比 | 部分（C2 特有） |
| 评价 circularity | 已用非 circular ΔG⁹⁶ 口径复评，结论不变 | 排除 |

## 3. 跨 split 不一致的教训

Val（起点稳定，c₉₆≈1，尾项方差占比 0.18–0.94）与 Train（22–203×）的**标签构成
根本不同**；Test（0.03–2.5）又不同。C2 在 Test 的"改善"（sign 0.639/sp +0.501，
random 族 0.75）发生在尾项占比最低的 split——进一步支持"尾项是主导变量"。
**任何单 split 的结论都不可靠；本轮不做 split 挑选。**

## 4. 对下一轮的直接设计修正（已验证的量化依据，非猜测）

1. **pair 监督改用 ΔG（无尾项 continuation 差分）**，或"标量用 Y_T、pair 用 ΔG"的
   混合合同——ΔG 的 std(0.03–0.09) 与其物理意义（96 步真实 continuation 价值差）
   匹配 pair loss 的语义。
2. **方向族收紧**：训练 pair 只用 delta_2100/2300（标签 std 最大、CVF-0 中 WM 有
   0.69–0.81 sign 的方向）；random/ascent 仅作评价探针（其 Sim ±噪声会污染训练）。
3. **扩大 ΔG 信号**：降低 ε 到响应线性区间的更小值或提高 continuation 长度一致性
   （gated 率），或对 ΔG 做跨 ε 平均（CRN 式期望响应）以压 Sim 分岔噪声。
4. **规模**：起点数与 pair 数至少 ×3（本轮 gated Val n=8 不足以支撑 10pp 门限的
   检验功效——功效分析见 NEXT_DECISION）。

## 5. 与 CVF-0 反事实上限的关系

CF D（Sim oracle continuation）sign 0.958 的口径是**无尾项** 96 步 continuation。
本轮 ΔG⁹⁶ 口径下 C-init 在 Val gated sign=1.000 / sp=+0.929（起点数与方向分布不同，
不可直接对标），但方向一致：**Critic 的 pair 响应在干净标签上有可学信号**——
本轮失败不在"Critic 学不到"，而在"标签没有把可学信号放进来"。
