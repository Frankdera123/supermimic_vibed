# CVF1A_DATA_CONTRACT（真实价值数据合同，预注册）

日期：2026-11-20。本合同在任何 GPU 采集之前落盘锁定；与 `CVF1A_EXPERIMENT_PROTOCOL.md`、
`CVF1A_SPLIT_MANIFEST.json`、`CVF1A_BUDGET_PLAN.md`、`CVF1A_READINESS_REPORT.md` 共同构成 CVF-1a
准备阶段冻结文档。实现代码：`cvf_proto/sacv/`（targets/losses 纯函数 + CPU 单元测试
`cvf_proto/sacv/tests/test_sacv.py`）。

母本与身份链核验结果见 `CVF1A_READINESS_REPORT.md` §1（全部 15 项 fail-closed 检查通过，
manifest：`results/cvf/sacv/manifests/mother_asset_manifest.json`）。

---

## 1. 冻结母本（已核验，禁止修改）

| 角色 | 载体 | 组件 SHA256（训练时 manifest 自记录，已复现核对） |
|---|---|---|
| World Model `WM@2100` | `server_artifacts/online_B0/ckpt_002100.pt["wm"]` == `pair_it_002100.pt["wm"]` | `43cd7f5a550e03645bdbedc5599d260d5bb99fbdc5850509ed35bdb65cb160f5` |
| 策略 `pi_old@2100` | 仅存在于 `pair_it_002100.pt["pi_old"]`（ckpt 内 `pi` 为更新后 pi_new） | `c4810c4db2b49ca55d20d758c718cf04976260e3f9cfb3ec04275eead372f6b0` |
| Critic `C-init = B0@2100_post` | `pair_it_002100.pt["critic_post"]` == `ckpt_002100.pt["critic"]` | `65118115d917958430279afc5ffc37a8666c52104bcea3f10d4de1ae8bd73f54` |
| 参考价值 `V_ref` | == C-init（冻结，所有对照组共享，禁止任何组更新） | 同上 |

训练配置（`args.json`）：H=32，γ=0.99，λ=0.95，hist=10，OBS_DIM=93，goal=GOAL_FRAMES×73，
Critic 输入 938 维、[256,128] LayerNorm+GELU、RunningNorm、V_max=10（`critic_input` clamp），
lr=2e-4，Adam，bf16 AMP（训练），seed=0，motion=lafan1_g1，nominal ξ=[1,0₂₉,0₃]。
WM：35-D acceleration-level proto 变体。Critic 输入**不含** `o_hist`/`a_prev`（已知结构限制，
今晚不改）。仿真：`TrackingSim(eval_mode=True, reset_noise=False)`，`deterministic_mode="run_to_run"`，
dt=0.02，substeps=4，MJWarp njmax=500/nconmax=128。

**纪律**：本轮冻结 WM 与 pi_old；只有 CVF-2a 允许从冻结母本复制 Policy 做有限更新；严禁混用
主仓库 70-D WM 体系、no_critic+H64 体系、以及任何其他 checkpoint（`online_B1/B2/B3` 仅可作
离线对照分析，不得进入本轮训练）。

---

## 2. 扩展状态与价值状态定义（A2）

实际闭环价值依赖扩展状态

$$x=(s,\;h,\;a_{\mathrm{prev}},\;f,\;\xi)$$

- `s`：物理状态五元组 `(p, R, v, w, q, qd)`（落盘为 `flatten_state` 70+? 维扁平张量，
  恢复用 `TrackingSim` state-restoration 通道；烟测必须验证 restore→step 的 run_to_run 一致性）。
- `h`：观测历史 `o_hist ∈ R^{10×93}`（policy 消费；Critic 不消费，但必须落盘以便后续
  历史条件化分析与价值混叠诊断）。
- `a_prev`：上一动作 `R^29`。
- `f`：reference motion frame（整数张量）。
- `ξ`：物理环境参数（本轮恒为 nominal `ξ_one[:,0]=1.0`，全零扰动）。

**采集合同**：每个正负分支必须完整保存 `(s_H, o_hist_H, a_{H-1}, f_H, ξ)` 以及前缀 H 步与
续推 T 步的逐时间步记录 `(r_state_t, r_direct_t, r_t=r_state+r_direct, d_soft_t, d_hard_t)`。
诊断要求：同时保存前缀末端与前缀中间（t=8）状态，用于复现 CVF-0 Oracle 2 的 H∈{8,32} 对照。

**价值混叠诊断（只读，不改输入结构）**：用已保存的 `o_hist` 检查同一 `s`（±扰动）在不同
历史下的 Critic 输出差异是否存在（即使 Critic 输入不含 h，其特征 `critic_features(s,f,ξ,goal)`
仍可能因 s 不足而混叠；该诊断量化"同一标称状态、不同历史"的目标值方差），结果仅进入
机制分析，不影响本轮训练。

---

## 3. 数据集定义（A3）

### 3.1 标量数据集

$$\mathcal D_v=\{(x_i,\;G_i)\},\qquad x_i \in \{x_H^{\mathrm{sim},+},\,x_H^{\mathrm{sim},-},\,x_H^{\mathrm{sim},\mathrm{base}}\}$$

`G_i` 为**续推段**（closed-loop pi_old continuation）的折现软终止回报（见 §4），不含前缀段回报
（前缀回报属于"已实现价值"，不属于 x_H 的状态价值；CVF-0 Oracle 2 的 `D_sim,cont` 同口径）。

### 3.2 成对数据集

$$\mathcal D_p=\{(x_i^+,x_i^-,G_i^+,G_i^-)\},\qquad
A^\pm = A_0 \pm \epsilon\, d_A$$

- 正负分支共享同一 `x_0=(s_0,h_0,a_{prev,0}=0,f_0,ξ)` 与同一 `d_A`、同一 `ε`；仅动作前缀符号相反。
- `d_A` 为 29 维动作序列方向 `(H,29)`，按 `crgf.normalize_dir_seqs` 扁平单位化；
  执行前 `clamp(±20)`（与训练一致）。记录原始方向范数、扰动前后动作差范数、clip 后有效扰动。
- 前缀在真实 Simulator 中开环执行 H=32 步得到物理可达末端 `x_H^±`；**禁止**任何噪声合成状态
  或把动作差分称为状态梯度。
- 续推为同一冻结 `pi_old` 的闭环确定性 rollout，T 步；正负分支续推段相互独立（各自从
  `x_H^±` 出发）。
- `ΔG_i = G_i^+ − G_i^-` 仅在**同一 i（同起点同方向同 ε）**内定义，禁止跨样本差分。

### 3.3 数据类别标签（不得混称）

| 类别 | 定义 | 用途 |
|---|---|---|
| ALL 有限尺度响应 | 全部通过物理可行性筛查（前缀未在 H 内 hard-terminate 于步 0）的样本 | 训练 Dv/Dp 主体 |
| Strict-gated 局部方向响应 | 再通过 CVF-0 门禁（基线重复 ≤1e-5、跨 ε 相对差 ≤max(5e-4, 0.2|D|)、死区过滤、排除单侧跌倒分叉） | 局部导数保真评价 |

不因局部 gate 失败丢弃合法有限尺度价值差分；但 Strict-gated 之外的数据**不得**称为严格局部导数。
确定性控制：分支基线重复 2 次，`repeat_range = max|ΔG_base|`；若烟测发现残余物理随机性，
按 Goal §A7 升级为期望价值响应估计（CRN 重复取均值），本轮默认 `deterministic_mode="run_to_run"`
+ `reset_noise=False` 下单次 rollout 即确定性。

---

## 4. 统一价值目标（A4）

### 4.1 逐时间步记录与多时域导出

一次长续推 rollout（T_max）同时导出多个较短时域目标，禁止重复采集：

$$G_T^\pi(x)=\sum_{k=0}^{T-1}\gamma^k c_k r_k,\qquad
c_0=1,\;c_k=\prod_{j=0}^{k-1}(1-d_j)$$

- 主参考 `d = soft_termination`（tz=0.02, tt=0.04）；辅参考 `d = hard_termination` 同时落盘。
- `c_t` 在乘 `r_t` **之后**才更新（训练源码对齐口径，`rollout_wm_continuation`/Oracle2 已验证）。
- Pilot 采集 T_max=96（H=32 前缀 + 96 续推），导出 `T ∈ {32, 64, 96}`；
  T=128/256 仅当剩余预算与 clip 边界允许（决策规则见 §4.4）。
- clip 边界合同：`f_0 + H + T_max ≤ clip_end − 1`；起点选择阶段强制筛查，禁止越界截断。

### 4.2 独立评价目标（禁止 learned bootstrap 自证）

$$G_T^\pi(x)\ \text{（§4.1，无任何 Critic 尾项）}$$

用于 D1/D2 的主要真值与 horizon 稳定性诊断。

### 4.3 匹配训练目标（共享冻结 V_ref）

$$Y_T^\pi(x)=G_T^\pi(x)+\gamma^T c_T\,V_{\mathrm{ref}}(x_T)$$

- `V_ref` = 冻结 C-init（B0@2100_post），`x_T` 为该分支续推 T 步后的**真实 Simulator 状态**
  （每分支在每个导出时域保存 `(s_T, o_hist_T, a_{T-1}, f_T)`，V_ref 输入按 proto `critic_input`
  构造）。禁止把真实状态 return 标签赋予 WM 预测状态。
- 所有对照组（C0/C1/C2）使用同一 `V_ref` 权重与同一 target 合同；C0 的 WM 目标同样拼接
  `γ^T c_T V_ref(ŝ_T^WM)`（在 WM 预测末端状态上求值）——评价时域与 bootstrap 合同一致。
- **tail bias 风险报告义务**：Validation 上计算 `bias_tail = E[γ^T c_T V_ref(x_T)]` 及其相对
  `std(G_T)` 的占比；CVF-0 Level 1 已知 C-init 标量 r≈0.99，预期尾项占比 ~γ^96·V̄≈0.38·V̄，
  必须量化报告，不得忽略。
- 禁止把有限时域 Y_T 直接当作原始无限时域 bootstrap Critic；两者语义区别写入训练代码注释。

### 4.4 训练目标时域冻结规则（训练前冻结）

在 **Train+Validation** 数据上计算逐时域诊断（对 Test 只读不动）：

1. 跨时域秩稳定：`Spearman(G_32, G_96)` 与 `Spearman(G_64, G_96)`（start 级）≥ 0.9 的时域。
2. 尾项占比：`|E[γ^T c_T V_ref]| / std(G_T)` 随 T 单调下降且 T=96 时 < 0.5。
3. 选择满足 1 的最短 T 为主训练/评价时域（预期 T=96，与 CVF-0 Oracle 2 的 T_cont=96 保持
   可比）；若 1 在所有 T 上失败，裁定 horizon-target mismatch 信号，触发 Goal §9 情况 E 诊断。
4. T=128 扩展决策：仅当 Train+Val 采集完成后实测吞吐推算全量 ≤ 2.2 GPU-h，才对 **Test split
   补采 T=96→128 延长段**（独立 rollout 记录，不得与已落盘数据混合命名）；否则本轮 T_max=96
   并如实记录偏差。

### 4.5 目标数值精度

所有目标以 float64 累加后转 float32 落盘；`r`、`c`、`γ^t` 逐项保存（(T,) 向量），
使任意时域 G 可离线重算并允许独立复核。

---

## 5. 数据划分（A5）

- 池：`data/lafan1_g1/` 全部 40 clips，**剔除 CVF-0 已用 10 clips**（`dance2_subject2,
  fallAndGetUp1_subject1, fight1_subject5, fightAndSports1_subject4, jumps1_subject5,
  run2_subject4, sprint1_subject2, walk1_subject2, walk3_subject3, walk4_subject1`），
  CVF-0 数据仅作历史诊断回归集。
- Pilot 目标规模（预算内最低可行）：**Train 12 clips / Val 4 clips / Test 6 clips**，
  每 clip 2 个独立起点（frame 间隔 ≥ 64），覆盖 ≥ 5 种运动类别（walk/run/sprint/dance/
  fight/jumps/fallAndGetUp/hibernate→ fightAndSports 类）。
- 分层约束：每 split 内 clip 类别尽量均衡；每 clip 的正负分支、ε、prefix、continuation
  与重叠窗口**绝不跨 split**；同一轨迹窗口不得跨 split。
- 随机性：split 生成用固定 seed=2701 的 `torch.Generator`，clip→split 分配与起点 frame 选择
  落盘 `CVF1A_SPLIT_MANIFEST.json`（含每 clip 的 sha256、clip 长度、起点 frame、运动类别标签）。
- Test 不参与超参/α/时域选择；不允许依据 Test 返回修改模型后再称盲测。
- 最终规模若受吞吐限制小于上述规模，裁定 pilot 并如实标注（不得宣称正式泛化）。

---

## 6. 方向设计（A6）

每起点 4 个方向族（与 CVF-0 Oracle 2 同构，保证跨轮可比；生成代码复用
`crgf_proto/supermimic/crgf.py` 的 `delta_dir_seqs / random_dir_seqs /
path_directional_derivatives`）：

| 族 | 构造 | 说明 |
|---|---|---|
| `delta_2100` | 闭环 WM rollout（pi_old, H=32）逐步 `clamp(π_new^2100(o,g))−clamp(π_old(o,g))`，扁平单位化 | 真实策略更新 action delta |
| `delta_2300` | 同上，用 pair@2300 | 第二个真实更新方向（跨迭代） |
| `ascent` | 开环 WM path（prefix + critic）total 方向导数，单位化 | WM ascent direction |
| `random` | `random_dir_seqs(mode="per_step", seed=起点相关)` | 普通随机方向 |

structured-random（策略时间结构）族作为 **预算允许时的可选第 5 族**（`random_dir_seqs
(mode="repeated")` 为其粗近似），本轮默认不采（预算优先保障 4 主族 × 2 ε 的独立样本量）。

- ε 网格：`{0.015, 0.03}`（与 CVF-0 Oracle 2 相同，保证跨 ε 稳定性可检验）。
- 每分支记录：方向来源、生成时实际状态、原始动作差范数、单位化方式、扰动后
  `‖A± − A0‖`、clamp 后有效扰动、ε、末端物理状态变化 `‖flatten(s_H^±) − flatten(s_H^base)‖`。
- 尺度可比性：4 族均为扁平单位化（`‖d_A‖_F = 1`），动作空间扰动尺度严格可比；
  末端状态变化幅度按族分层报告。

---

## 7. 参考可靠性检查（A7）

烟测与正式采集每 batch 强制执行：

1. 重复确定性：每起点 base 分支跑 2 次，`repeat_range = max|ΔG|` 要求 ≤ 1e-5（软）。
2. 跨 ε 稳定性：`|D(ε1) − D(ε2)| ≤ max(5e-4, 0.20|D(ε1)|)`（仅进入 Strict-gated 层）。
3. 正负分支不对称性：记录 `|G^+| vs |G^-|` 分布与 base 漂移；asymmetry 指标入 ledger。
4. 接触切换与步态分叉：以 `min_clear(s_H^±)`（8 球 FK，复用 `cvf_core.foot_ball_clearances`）
   与末端姿态距离诊断；单侧跌倒分叉标记进 gate。
5. Soft/hard termination、timeout、horizon truncation：逐分支计数并分类落盘
   （soft_d / hard_d 时间索引；clip 边界截断计数应为 0）。
6. 参考值大小与数值精度：`G` 分布（mean/std/min/max）、float64 累加复核误差 < 1e-6。
7. 观测噪声：本轮 noise-free（`reset_noise=False`）；CRN 噪声敏感性留待 Phase 1c 类扩展，
   不在今晚预算内。

产出：`results/cvf/sacv/metrics/reliability_report.json`（烟测后与采集后各一份）。

---

## 8. 落盘与完整性

- 原始轨迹（逐 batch 增量）：`results/cvf/sacv/datasets/raw/{split}/batch_*.pt`
- 训练/评价数据（由 raw 派生）：`results/cvf/sacv/datasets/{train,val,test}_pairs.pt`
- manifest：`results/cvf/sacv/manifests/`（split manifest、collection manifest、
  每文件 sha256 清单）
- ledger：`results/cvf/sacv/budget/budget_ledger.jsonl`（每 batch 一行：时间戳、
  env_transitions、simulator_batch_steps、wall time、显存峰值、batch sha256）
- 统计口径**强制区分**：`env_transitions` = B_env × 每环境步数（采样步数）；
  `simulator_batch_steps` = env.step 调用次数 × B_env（仿真推进步数）。历史 4 倍口径
  错误不得重现：一个 rollout 步 = 1 次 `env.step` = 每环境 1 个 transition；
  两组数字必须分开累计并交叉核对 `env_transitions == simulator_batch_steps`（单环境同步
  step 模型下两者相等，若实现中出现不等必须查明原因）。
- 所有 `.pt` 文件写入后立即补 sha256 入 manifest；原始文件一经落盘不得修改，
  违反即判 BLOCKED。
