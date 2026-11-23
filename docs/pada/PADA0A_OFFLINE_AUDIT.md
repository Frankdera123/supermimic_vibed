# PADA0A_OFFLINE_AUDIT

日期：2026-11-18。对象：PADA-0 提交 `6e9b4e7` 的梯度保真度评估器数学审计与离线修正。**无训练、无仿真采集、无 GPU；全部复算基于已归档数据（CPU，446 s）。原始 `results/pada/eval/` 与协议零改动（SHA256 前后一致，`git diff --check` 干净）。**

## 0. 裁定（先说结论）

**原结论需要降级。**

- PADA-0 的 INCONCLUSIVE **维持**（且更稳健）；
- 但原报告的解读——"M2 在 control-relevant fidelity 上方向一致地优于 M1"——**在修正口径下不成立为 dynamics fidelity 证据**：修正测量显示该优势（a）主要由 **critic 路径**贡献而非 dynamics/state 路径（seen 集上 state 路径配对差 **−0.333**，CI 不含 0，即 M2 的 dynamics 路径反而更差）；（b）**clip 脆弱**（seen 集优势全部来自 walk3_subject5 单一 clip，leave-one-clip-out 后 M2≈M1）；（c）伴随 **2–10× 的梯度幅值畸变**（M2 的 median magnitude ratio），符号一致率单独不足以支撑"fidelity 提升"；
- PADA-Data（正式训练）**不获得支持**。

## 1. 资产与数据完整性（第一阶段）

- HEAD = `6e9b4e790be427b4b9d494088072b0dfdd05207e`，工作区干净（仅本地 setup symlink `crgf_proto/data` 未跟踪）。
- 11 个已归档工件与 `provenance.json` SHA256 **全部一致**。
- **GitHub 缺失声明**：`*.pt` 被忽略，以下仅存在于本地（SHA256 已记录在 provenance 或本次 `data_integrity.json` 的 `ckpt_sha256_supplement`）：12 个模型快照（35 MB each）、`collect/probes.pt`、`collect/eval_data.pt`、smoke 工件。**provenance.json 当时漏记了 ckpt 的 SHA256**，本次已补记（原始 provenance.json 未改动）。
- 原始 `eval_rows.csv` / `eval_summary.json` / `level1.json` 与 HEAD 逐字节一致（0 diff），审计前后 SHA256 不变。

## 2. 发现的数学问题（第二阶段）

### 问题 1（主要）：WM 梯度评价点与 Simulator 不一致

原 `pada_eval.py` 的 `D_wm` 是 **A+εd 处**的 autograd 方向导数，而 `D_sim` 是围绕 **A** 的中心差分。二者不是同一数学量：Legacy(+点) − Center 的差在分路径上很小（H32 total 平均 0.004），但足以在零点附近翻转符号——**eval_pol_C H8 的符号结论在修正后翻转**（legacy: M2 0.833 > M1 0.667；center: M0 1.000 ≥ M1 0.833 > M2 0.667）。

### 问题 2（主要）：direct/critic 路径口径不匹配（survival 梯度）

Simulator 的 `branch_J` 每条路径的 FD 都**包含** survival factor 随动作变化的梯度；CRGF 的 `L_direct_iso`/`L_critic_iso` 是 survival **detached** 的。两者不能同名比较。修正：用与 `branch_J` 逐项一致的 attached 定义重建 WM 路径返回（`matched_path_returns`），并验证恒等式。**实测影响**：中心点 survival 项极小（direct 均值 6.2e-8，critic 均值 1.1e-4）——因为参考起点附近 soft_termination 饱和于 0。这是一个真实但在本数据上数值不重要的口径缺陷。

### 问题 3：total 路径本身的口径是对的

`crgf.L_total = L_state + L_direct_attached + L_critic_attached`，与 Simulator `J_total` 定义一致（γ、soft_term、critic 权重 1.0、goal refresh、action clamp、a_prev=0、S0、H、末步 critic bootstrap 均逐项核对一致）。**T3 验证 matched-J_total 梯度与 crgf L_total 梯度精确相等（rel=0）**。原评估的 total 路径只带问题 1（评价点）。

### 问题 4（次要）：pooled-ε 统计

原 summary 把 2 个 ε 的 FD 取均值后再算 fidelity——混合线性化尺度。本次按 ε 分别报告（`per_eps_fidelity`）并保留 pooled 视图供对比。

### 问题 5（本次审计自身发现并修复的 bug）

`matched_path_returns` 首版中 critic 项误写 `(c_prev * V).sum(-1)`——对 (B,) 张量坍缩 batch，使 B>1 时每行梯度混入整批（B=1 的单元测试无法暴露）。表现为 matched 恒等式残差 max 6.09；修复后 **max 4.2e-7**。此 bug 只存在于审计工具开发过程中，未影响原始 PADA-0 数据。

## 3. 修正方案（第三阶段）

新工具 `pada_proto/tools/pada_eval_matched_fd.py`（不改动原 `pada_eval.py` 与 `crgf_proto/`）：

- 指标 A `D_wm_legacy_plusgrad`：复现原定义（用于回归对照）；
- 指标 B `D_wm_center`：∇_A J_wm(A)·d 于未扰动 A（matched 主指标）；
- 指标 C `D_wm_fd`：与 Simulator 完全相同 ε 的 WM 中心差分；
- matched 路径返回 = Simulator `branch_J` 的 WM 版本（attached survival），恒等式 `D_total = D_state+D_direct+D_critic` 验证至 4.2e-7；
- 门禁确定性重算并与原始 gated 样本集**逐一断言相等**（identity_ok=true）；
- 全部统计沿用原始 gated 样本；新增 per-ε、clip 聚类 bootstrap、leave-one-clip-out、per-path 分解。
- 6 项 CPU 测试（`test_pada_audit.py`，12/12 通过）：线性/二次函数的 autograd-FD 关系、路径分解恒等式、survival 梯度存在性（surrogate 解析匹配 + 真实栈精确恒等式）、样本同一性、原始结果保护。

## 4. 结果变化（第四阶段）

复算覆盖：7 模型 × 4 方向集 × {8,32} × {0.03,0.1}，245 样本，980 行。

### 4.1 主指标对照（gated sign agreement，D_total，pooled-ε）

| 方向集 H32 | 原始 M1 | 原始 M2 | 修正 M1(center) | 修正 M2(center) | M0(center) | n |
|---|---|---|---|---|---|---|
| pol_seen | 0.222 | 0.667 | 0.333 | 0.667 | 0.778 | 9 |
| pol_B | 0.250 | 0.750 | 0.250 | 0.750 | 0.500 | 4 |
| pol_C | 0.500 | 1.000 | 0.500 | 1.000 | 1.000 | 4 |
| eval_random | 1.000 | 0.333 | 1.000 | 0.333 | 0.333 | 3 |

配对差 M2−M1（center，H32）：seen +0.333 CI[+0.108,+0.667]（原始 +0.444）；clip 聚类 bootstrap 后 **seen CI [0, +0.43]、B/C CI [0, +0.67]——全部含 0**。

### 4.2 路径分解（center，H32，M2−M1 配对差）

| 方向集 | state | direct | critic | total |
|---|---|---|---|---|
| pol_seen | **−0.333** [−0.67,−0.11] | 0 | +0.333 [−0.11,+0.78] | +0.333 [+0.11,+0.67] |
| pol_B | 0 [−0.75,+0.75] | 0 | **+0.750** [+0.25,+1.0] | +0.500 [0,+1.0] |
| pol_C | +0.250 [0,+0.75] | 0 | **+0.500** [0,+1.0] | +0.500 [0,+1.0] |

→ **总梯度"改善"由 critic 路径主导；dynamics/state 路径无改善，seen 集上显著变差。**

### 4.3 幅值畸变（median |D_wm|/|D_sim|，H32，total）

M0：0.35–1.30；M1：0.31–3.21；**M2：1.99–9.99**。微调后梯度幅值严重偏离 sim 真值，符号一致率的证据力被高估。

### 4.4 clip 集中度与 leave-one-clip-out

gated 样本 54.3% 来自 walk3_subject5；H32 上 pol_B/pol_C 仅 2 个 clip、各 clip 1–3 个样本。seen H32：全样本 M2 0.667 > M1 0.333，但**去掉 walk3 后 M2 = M1 = 0.50（n=2），均低于 M0 = 1.00**——seen 集优势完全由单一 clip 携带。pol_B 的优势来自 fight1（n=3）。**不存在跨 clip 稳定的 M2 优势。**

### 4.5 一致性检查（gradient_consistency）

- WM-FD ≈ WM-center autograd（|FD−center| ≤ 7e-5，所有路径/时域）：这些 ε 下非线性可忽略，评价点修正的实质影响是符号翻转而非量级；
- legacy 复现 vs 原始：mean rel dev 1.5%（CPU vs GPU），max 281%（近零分母）；gated 样本集逐一相同；
- snapshot 曲线（center）：policy 集 H32 上 M2 ≥ M1 在 3 个快照仍成立，但 M0 在 seen/pol_C 上更高。

## 5. 对核心研究假设的影响（第五阶段）

1. **Policy probing 是否仍优于 random probing？** 在"total-sign @H32"这个特定读数上方向仍为正，但修正后：(a) 统计显著性在 clip 聚类下消失；(b) 优势来自 critic 路径与单一 clip；(c) 幅值畸变 2–10×。**不支持"更高效学习 control-relevant dynamics"的解读。**
2. **优势是否真正来自 dynamics learning？** **否**——state 路径配对差在 seen 集显著为负，B/C 集为零或 CI 含 0。
3. **是否存在 critic 主导的假象？** **是**。且 M2 的 magnitude ratio 高达 10×，提示微调让 critic 路径梯度整体膨胀。
4. **unseen 优势是否仍存在？** 形式上存在（B/C 的 critic 路径），但 n=4、2 clips、CI 含 0，不构成证据。
5. **当前证据是否足以支持 PADA-Data？** **否**。修正后的证据方向反而提示：probe-only 微调损害已校准的 WM（M0 全面最优），且"收益"是 critic 路径的 clip 特异现象。

## 6. 最终裁定

**原结论需要降级**（选项二）：

- PADA-0 = INCONCLUSIVE：**维持**；
- "policy-shift probes 在 control-relevant fidelity 上方向一致优于 random probes"：**降级为不成立**（critic 主导 + clip 脆弱 + 幅值畸变 + LOCO 消失）；
- 对 PADA 后续：PADA-Data 的立项依据不存在。若未来重启，必须先解决：(i) blind eval 的 clip/样本量（每集 ≥10 gated、≥3 clips）；(ii) dynamics 路径与 critic 路径分开预注册判定；(iii) magnitude-ratio 约束进判定；(iv) 微调防遗忘设计（anchor 数据）。

## 7. 执行与保护声明

- 未启动训练/仿真/轨迹再生成；未使用 GPU（全部 CPU，累计 ≈13 min：复算 2×~7.5 min + 测试）；
- 未修改 `crgf_proto/`、协议、原始判定阈值、原始实验数据；
- 原始评估文件 SHA256 审计前后一致；未执行任何 git commit/push；
- 新产物：`results/pada/offline_audit/{matched_eval_rows.csv, matched_eval_summary.json, path_decomposition.json, gradient_consistency.json, data_integrity.json, per_sample_consistency.csv}`、`pada_proto/tools/{pada_eval_matched_fd.py, test_pada_audit.py}`（12/12 测试通过）。
