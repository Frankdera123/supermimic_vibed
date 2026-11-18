# GRADIENT_FIDELITY_EXPERIMENT_PROTOCOL

日期：2026-11-18。评估体系设计：复用既有工具，补齐缺口，全部可追溯。

## 1. 目标量的定义

对同一初始状态、参考运动、观测历史与**开环动作序列** A=(a_0..a_{H-1})：

- WM 侧（精确）：D_WM(path, d) = grad_A J_WM^(path) · d，autograd，路径分解见 SOURCE_AUDIT §3。
- Sim 侧（中心差分）：D_sim(path, d) = [J_sim^(path)(A+εd) − J_sim^(path)(A−εd)] / (2ε)，分支并行执行（同一 reset 三个并行 env，不做求解器状态恢复）。
- J_sim 的 critic 路径真值：γ^H·c_{H-1}·V(s_H^sim)，V 用指定 critic 在**真实仿真器终态**上求值（分支持有各自终态 → critic 路径的 FD 真值可得）。

三者严格区分（用户要求 6）：action-space 方向导数（本体系）、policy-space 梯度（g_θ = ∇_θ J，BPTT 隐式计算；sim 端需策略参数扰动，**未授权，仅设计**）、真实 Policy Improvement（历史 ΔJ 配对审计，两线已有数据）。

## 2. 方向集（决定结论的相关性）

| 集合 | 构造 | 检验什么 |
|---|---|---|
| random (per_step) | 每步 iid 单位方向，展平归一 | 平均意义上的响应结构 |
| repeated | 单方向保持全 horizon | 连贯推动（与单步审计可比） |
| **delta** | 真实策略更新 δa = π_new(o) − π_old(o)（`tools/crgf_delta_action_directions.py`，teacher-forced 共享 obs 序列），按 horizon 截断重归一 | **policy 实际会走的方向**（Level-2 与 Level-3 的桥） |
| ascent | 单位化的 D_WM^total | 模型自认为最优的方向——sim 是否同意 |

所有方向按展平 (H·29) 单位范数归一，原始 ‖δa‖ 单独记录。

## 3. 可靠性门禁（fail-closed，采纳 OPCL 修复协议）

`supermimic/crgf.py::reliability_gate`：基线回报重复极差 ≤1e-5；方向导数跨 ε/重复/顺序极差 ≤ max(5e-4, 20%·|mean|)；|mean| > 5e-4 死区。不满足者保留原因、不进入 sign 统计。阈值是探索性的，不是可微性证明。

**acos-clamp 诊断**：`ori_cosine_stats` 逐样本标记近对齐扭结（cos > 1−1e-5 或 < −1+1e-5）；扭结活跃样本的方向导数单独报告。

## 4. 统计规范

- 以 motion clip（或独立起点）为分析单位；clip 内配对差 + cluster bootstrap CI；不把帧×方向当独立样本。
- 指标：sign agreement（gated only）、MAE/RMSE、|D_WM|/|D_sim| 中位比值、Spearman、per-component 增益（70 维逐分量）。
- 预注册判定规则：B2 判为"控制相关改善"当且仅当 (a) D_pure/D_state 相对 B1 的 sign agreement 提升、CI 不含 0；(b) 改善在 H=32/64 存续；(c) 幅值校准比值回正。单条不成立即 NO。
- **指标有效性检验（H3）**：per-clip gradient fidelity 必须预测 per-clip ΔJ 一致性（对接两线已有的真实单步 update pair 数据），否则停止优化该指标。

## 5. 关键控制变量

- 共享 policy（501 checkpoint π）生成 base 动作序列；所有分支 WM 在**同一动作序列**上评估（`--save_base_actions` 固化，sim 侧复用）。
- 共享 critic（501 ψ）与分支自有 critic 双轮交叉（2×3），分离 WM 误差与 critic 误差。
- 训练附加机制开关（reliability gate、horizon decay、dispersion、realism、BPTT truncation）在评估协议中**全部关闭**并记录；否则跨 checkpoint 不可比。
- ε 扫描 {0.01, 0.03}（可扩）；γ、tz、tt 与训练一致（0.99 / 0.02 / 0.04）。

## 6. 执行边界

- CPU 已完成：合成验证（18/18，见 EVALUATION_REPORT §1）+ B0/B1/B2 WM 侧真实剖面（无仿真器）。
- GPU 未授权：sim 侧 FD（`tools/crgf_sim_fd_audit.py` 已实现未运行）与任何训练/策略扰动。
- 所有产物在 `/home/sustech/Desktop/crgf_proto/`（隔离副本，原仓库仅追加 `docs/crgf/` 报告副本）；原始 checkpoint 只读，SHA256 已记录。
