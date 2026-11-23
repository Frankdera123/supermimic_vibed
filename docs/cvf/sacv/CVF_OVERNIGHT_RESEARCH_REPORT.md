# CVF_OVERNIGHT_RESEARCH_REPORT（CVF-1a/1b/2a 夜间研究总结报告）

日期：2026-11-20。执行跨度：Phase A（CPU）→ 授权 → Phase B（GPU 采集）→ Phase C
（Critic 微调对照）→ Phase D（评价与统计）→ Phase E（门禁失败，未启动）。
GPU 总消耗 **≈2.96 / 6.0 GPU-h**；wall ≈5h。

## 0. 一句话结论

**在新采集的 44 起点/78,848 真实 transitions 上，Simulator 锚定的 Critic 微调
（C1 标量 / C2 标量+pair / C0 WM 对照）在预注册判据下均未改善 control-relevant
价值响应（裁定 INCONCLUSIVE，负面倾向）；根因已定位并量化：pair 标签的方差
99.5%+ 被 frozen V_ref 尾项占据（delta 族 163–203×），训练退化为"向 C-init
自我回归"——这是一个已定位、可修复、下一轮可直接验证的标签构造缺陷。**

## 1. 执行链与资源

| 阶段 | 内容 | 结果 | GPU-h |
|---|---|---|---:|
| A | 母本核验 15/15、合同/协议/划分冻结、16/16 测试 | READY | 0 |
| B smoke | 确定性/一致性/吞吐（B=8 MJWarp int32 溢出→B=4） | 全绿 | 0.25 |
| B collect | R2 阶梯 11 批 44 起点，rep_rng=0.0，reward 重建 1.5e-8 | 完整 | 1.83 |
| C | C0 WM 目标生成 + C1/C2/C0 × 500 步（σ_ΔY=0.0635 冻结） | 完成 | 0.68 |
| D | Val 选择 + Test 盲测 + 非 circular 复评 + SNR 诊断 | INCONCLUSIVE | ~0（CPU） |
| E | 门禁 ②⑤ 失败 | NOT RUN | 0 |
| 合计 | | | **2.96** |

失败与修复记录（不影响数据完整性）：B=8 尝试（MJWarp deterministic workspace
int32 溢出，预注册回退）、容器 NVML 掉线（重启恢复）、crgf import/xi_n 口径、
setdefault 缓存低效、批游标 off-by-logic、builder 两处——全部在产出数据前修复，
已入 OVERNIGHT_PROGRESS。

## 2. Q1–Q8 正式回答

**Q1 真实价值数据**：是。44 个全新起点（零 CVF-0 重叠）的合法可训练数据集建成；
位级确定性（rep_rng=0.0）、reward 重建 1.5e-8、G 重算交叉核对 3.8e-6。有限尺度
响应信号**存在但被尾项淹没**（delta 族 std(ΔG)=0.054 vs 标签总方差 99.5% 为尾项）。

**Q2 Value Learning（C1 vs C0）**：无差异证据。Val 上 C1（0.688/+0.491）与 C0
（0.708/+0.458）统计不可分（配对 CI 跨 0）。"真实动力学锚定优势"未显现——因为
两者的标签在 pair 层都被同一 V_ref 尾项主导（C0 的 WM 目标也拼同一尾项合同）。
本轮无法支持"锚定来自真实动力学"的归因。

**Q3 Pair Learning（C2 vs C1）**：**否定**。Val 上 C2 相对 C1：reachable sign
0pp（门限 +10pp）、reachable RMSE −24.6%（门限 +10%）、gated sp 0.881→0.333。
显式差分约束在当前标签下放大噪声族差分（α_eff=74.4 权重 × random/ascent 噪声）
并强化尾项（=C-init 自身）差分。Pair Loss 没有创造新信息（数学事实成立），
且在信噪比 ~1/200 的标签上是净伤害。

**Q4 Critic Geometry**：未改善。Within-start spearman：C1 0.593 ≈ C-init 0.564，
C2 0.407（退化）；gated 内 C-init sp +0.929 最高。幅度校准：全组（含 C-init）
Test ratio 0.40–0.70 欠响应——本轮未改变"幅度失败是 WM+Critic 共有"的 CVF-0 结论。
有限差分与 autograd 的差异本轮未测出新分离（D2 的 1/2 口径在 pair 层一致）。

**Q5 Control-Relevant Fidelity（双域）**：未达成比较条件（无改善的候选 Critic）。
真实状态域评价已完成（pair 响应表）；WM 想象状态域（D3）因 Phase E 门禁失败未执行。
Total-gradient 口径（D4 代用：prefix 差分 + continuation 差分）下各组排序与 pair
口径一致，C-init 仍最优。

**Q6 Policy Improvement**：NOT RUN（门禁失败）。无 ΔJ_Sim 数据。预实现工具
（sacv_policy_update.py）已就绪。

**Q7 机制判别**：证据最支持——**"价值监督不足"的一个具体子形态：监督标签的构造
缺陷（尾项淹没）+ 规模不足**，而非"Critic 历史信息不足"（未测）或"WM 梯度传播
问题"（CVF-0 已示 WM prefix 方向大体正确）。Sim 锚噪声（±不对称族）是第二贡献
因素。允许的综合判定：多种机制共同作用，其中**标签构造是本轮新识别的主导可修复项**。

**Q8 下一阶段决策**：若只能选一个新任务——
**"CVF-1b'：修正标签合同的 Sim-anchor 重训"**（标量用 Y_T、pair 用 ΔG⁹⁶、
训练族收紧 delta_2100/2300、CRN 跨 ε 平均压噪、起点 ×3 ≥72、Val gated ≥24 对）。
理由：①本轮已把失败归因到单一可修复变量并量化（不是泛泛的"数据不够"）；
②CF D 上限（0.958/+0.954）证明"32 步 WM prefix + 准确 continuation 梯度"足以
近理想——ΔG 监督正是直接逼近该上限的最短路径；③复用本轮全部 44 起点数据与
工具链（边际成本≈一次采集+训练，~2.5 GPU-h 即可完成 A/B/C/D 全链）。
尚缺证据：修复标签后 ΔG 监督是否真的能把 gated sign 从 C-init 的基线显著抬升
（需要 ≥24 gated pairs 的检验功效）；以及改善是否传导到 WM 域（D3）与 Policy（E）。

## 3. 诚实边界

1. pilot 规模：gated Val n=8 / Test n=6——D6 十 pp 级门限的检验功效不足；
   除"C2 gated 退化"外，多数组间比较是"无法区分"而非"证明相等"。
2. 本轮所有结论限于 opcl_proto 体系（H=32、γ=0.99、nominal ξ、noise-free）。
3. Test 的 C2 优势是事后观察，已按纪律不作为任何 PASS 依据。
4. 未 commit/push（ALLOW_GIT_PUSH=false）；所有原始数据/中间产物落盘可追溯
   （`experiment_registry.json` 的 sha256 清单）。
