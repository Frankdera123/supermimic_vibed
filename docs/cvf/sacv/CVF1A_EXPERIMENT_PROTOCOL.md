# CVF1A_EXPERIMENT_PROTOCOL（CVF-1a/1b/2a 预注册实验协议）

日期：2026-11-20。本协议在 Critic 训练启动前冻结；数据合同见 `CVF1A_DATA_CONTRACT.md`，
划分见 `CVF1A_SPLIT_MANIFEST.json`，预算见 `CVF1A_BUDGET_PLAN.md`。
实现：`cvf_proto/sacv/`。核心研究问题继承 CVF-0：**真实 Simulator 的状态价值及有限尺度
局部价值差分监督 Critic，能否改善其 control-relevant value response，并最终让 Policy 从
相同 World Model 中获得更加可靠的真实优化收益？** 不预设答案为 YES。

## 0. 继承事实（本轮设计的直接依据）

来自 CVF-0 及 deep_analysis（源数据已核验，见 READINESS_REPORT）：

1. C-init 标量校准已达标（vs Sim128 boot：r=0.977–0.990，slope≈1.09，bias+0.14 含尾项）；
   **主失败面是 within-start 方向响应**：`vel_reachable` sign 0.31–0.38（p≈0.45），Spearman≈0，
   1000→2300 从未改善；`pose/vel_critic` 靠纯 TD 已学到（sign 0.62–0.75 平台）。
2. `vel_reachable` 的 MSE 中 **e_B（WM 长程项）占 97–98%**，但**符号失败是 Critic 特有**
   （WM 在该方向 sign 0.69–0.81）；幅度失败两者共有（ratio 0.08–0.35×）。
3. 反事实上限（同一 WM prefix + Sim 真实 continuation，CF D）：sign 0.958 / Sp +0.954 /
   r +0.995 / ratio 0.98；C-init（A）0.562/+0.078；prefix-only（C）0.604/+0.133 但幅度
   仅 0.08×。**成功判据 = 显著超过 A 与 C，向 D 靠拢；且必须保留 Critic 提供幅度。**
4. WM 96 步 continuation 不能当一阶教师（CF B：0.583/+0.124，与 A 无显著差）→ C0 仅作
   matched-target 对照，不是主路线。
5. Sim 锚本身在 vel_reachable 有噪声（±不对称 94.9%、n_gated 3–4/16）→ 训练标签必须
   依赖 ±ε 成对差分与门禁分层，评测沿用 CVF-0 gate 口径并逐 clip 分层（clip 间 sign 0.0–1.0）。
6. 任何有限步 return 目标**必须**拼 γ^T c_T V_ref 尾项（T=32 丢 71%、T=128 丢 26% 尾部梯度）。
7. 合同继承：survival 用 c_t·r_t（c₀=1, c_t=∏_{k<t}(1−d_k)）；B=1 vs B=N 无 reduction bug
   （已独立复验）；grad_raw vs clamp(max=V_MAX) 口径统一为**训练目标比较用 v_eff，梯度
   保真比较同时报告两者**（CVF-0 已证实当前无状态达 clamp，风险为零）。
8. 不动 Critic 输入结构（无 o_hist/a_prev，Route C 低优先）；不做特征增广（Route D 的
   normalizer/MLP 双瓶颈留待下轮，避免本轮多变量混淆）。

## 1. 对照组定义（Phase C，全部从同一 C-init 初始化）

冻结：WM、pi_old、Critic Normalizer（初始化自 C-init 且**停止统计更新**，eval 模式）、
网络架构、V_ref（=C-init 权重）、lr、优化器配置、采样流。唯一自由变量 = 监督信号。

| 组 | Loss | 目标来源 |
|---|---|---|
| **C-init** | 无更新 | 原始 B0@2100_post（退化判据基线） |
| **C0** | $L_{C0}=\mathbb E[(V_\psi(x)-Y_{\mathrm{WM}}(x))^2]$ | WM continuation：同真实起点、同 o_hist/a_prev 合同，WM 开环前缀? **否——与 C1/C2 相同的闭环合同**：从同一 $x_0$ 以同一 $\pi_{old}$ 在 WM 中闭环续推，$Y_{\mathrm{WM}}=G^{\mathrm{WM}}_T(x)+\gamma^T c_T V_{\mathrm{ref}}(\hat x_T^{\mathrm{WM}})$ |
| **C1** | $L_{C1}=\frac12\mathbb E[(V_\psi(x^+)-Y^+)^2+(V_\psi(x^-)-Y^-)^2]$ | 真实 Simulator：$\hat Y^\pm$ 见数据合同 §4.3 |
| **C2** | $L_{C2}=L_{\mathrm{value}}+\alpha L_{\mathrm{pair}}$，$L_{\mathrm{pair}}=\mathbb E[((V_\psi(x^+)-V_\psi(x^-))-(Y^+-Y^-))^2]$ | 同 C1 的**同一批**正负分支、标量标签、sampling stream |

**数学事实（必须随结果报告）**：$L_{\mathrm{pair}}$ 不在两个标量标签之外创造任何新信息——
$\Delta Y=Y^+-Y^-$ 是 $Y^+,Y^-$ 的线性泛函；C2 的作用是把局部差分误差在优化泛函中的权重
从（由标量项隐式决定）提升为显式 $\alpha$ 加权。这正是本轮要验证的机制假设，不是新算法。

- C0/C1/C2 使用完全相同的起点集合、批大小、步数与随机数流；C0 的 WM 目标用**同一冻结 WM**
  生成（对 C0 而言 WM 是数据生成器，不是被训对象）。
- C1 与 C2 的唯一差异是 $\alpha L_{\mathrm{pair}}$ 项；两者不得有额外数据或额外计算路径差异。
- 主 pilot α=0.3（中等）；α∈{0.1, 1.0} 为预批准备选，仅当 Validation 显示 0.3 的 pair 项
  相对 value 项权重失配（<0.1× 或 >10× 梯度范数比）时按预冻结规则换档，禁止读 Test。
- **α 尺度定义冻结**：训练启动前在 Train 首批上测量 $|L_{\mathrm{value}}|$、$|L_{\mathrm{pair}}|$
  及两组梯度范数；若 $|L_{\mathrm{pair}}| \gg |L_{\mathrm{value}}|$（>10×），pair 项按
  $L_{\mathrm{pair}}/\hat\sigma^2_{\Delta Y}$ 归一（$\hat\sigma_{\Delta Y}$=Train 上 ΔY 标准差，
  常数、不随训练更新），该定义一次冻结、全组共用并在结果中报告。
- 训练长度：250 步 smoke snapshot → 500 步早期 checkpoint → 1000 步 pilot 主比较 →
  至多 2000 步（仅当 Validation 趋势健康且预算允许）。组间比较使用**匹配 optimizer steps、
  匹配样本暴露、同初始化**。优先单种子机制筛选；仅当出现明确正向信号且剩余预算 ≥0.4 GPU-h
  时对预选定配置补 seed 复验（最多 2 个额外 seed）。禁止无边界超参扫描。

## 2. 防灾难性遗忘（C5）

所有微调组与 C-init 在**同一套冻结探针**上比较：

- Nominal states 标量值（Val 全部起点 + CVF-0 历史 16 起点）；
- Pose response（CVF-0 Oracle 1 方向族的 pose_safe_critic/pose_safe_random）；
- Random directions；Policy-reachable directions（本轮新数据 Val split）；
- 普通 LAFAN 片段标量（Test split 的 base 分支）。

本轮**默认不添加 WM 标量 anchor**（第一轮保持单变量）；若遗忘显著（任一探针退化 >10% 且
排除噪声），在预批准的补充预算内允许对 C1/C2 同时加同一固定 anchor 项重跑一次，anchor
数据与权重全组一致——不得只给 C2 加 anchor。

## 3. 评价协议（Phase D，统一数据、统一真实目标、CPU 优先）

### D1 标量价值质量
MAE / RMSE / Bias / Pearson / Spearman / Calibration slope / Within-start value correlation，
分别对 (a) 独立 Simulator 有限时域 $G_T$（主真值，无 learned bootstrap）与 (b) 带冻结尾项
$Y_T$。禁止用任何 learned Critic bootstrap 自证。

### D2 局部价值响应
对每方向族分层（`delta_2100 / delta_2300 / ascent / random`）与全部合并：
Sign agreement / Directional RMSE / Magnitude ratio / Spearman / Within-start ranking /
跨 ε 敏感性；三类响应分别报告：
1. 有限尺度 value difference：$\frac{V(x^+)-V(x^-)}{2\epsilon}$ vs $\frac{Y^+-Y^-}{2\epsilon}$（含 γ^H c_H^± 权重口径，逐分支记录）；
2. 中心点 autograd：`cvf_core.critic_tangent_grad` 在 $x^{base}$（及 $x^\pm$）的方向导数；
3. WM/Sim matched finite difference 对照（CVF-0 Oracle 2 口径）。
**若 pair loss 改善 1 但不改善 2，必须显式报告**（Policy 消费的是 autograd 梯度而非差分）。

### D3 真实状态域 vs WM 状态域
对同一 prefix 分支：评价 $V_\psi(s_H^{\mathrm{Sim}})$ 与 $V_\psi(\hat s_H^{\mathrm{WM}})$
（WM 用同一动作序列从同一 $x_0$ 闭环续推至 H；禁止把真实 return 标签赋予 WM 预测状态）。
报告 $\hat s_H^{\mathrm{WM}}-s_H^{\mathrm{Sim}}$ 沿 Critic 高敏感方向的投影分布、
Critic 在两域上的响应差。核心问题：**Critic 在真实状态上学到的价值响应，能否在 WM 实际
查询的位置保持？**

### D4 完整 Total Gradient
$D_{\mathrm{WM,total}}=D_{\mathrm{WM,prefix}}+D_{\mathrm{WM,critic}}$（各组 Critic）；
$D_{\mathrm{Sim,total}}=D_{\mathrm{Sim,prefix}}+D_{\mathrm{Sim,cont}}$（同一冻结 Simulator
total-return 参考，γ^H·c_H^± 权重逐分支核算，防漏乘/重乘）。
报告 C-init / C0 / C1 / C2 / Prefix-only / Oracle-D 诊断参考（Oracle D 不参与学习模型排名）。
主指标：对 $D_{\mathrm{Sim,total}}$ 的 sign / Spearman / magnitude ratio / RMSE
（ALL 与 Strict-gated 两层 + 逐 clip 分层 + LOCO）。

### D5 统计方法
同一样本配对比较（同起点同方向跨组）；start 级 cluster；clip 级 cluster bootstrap
（B=10000, seed=2700，复用 `cvf_core.clip_clustered_bootstrap`）；exact paired sign test；
leave-one-clip-out；方向族分层。同一起点的多个 ε、正负分支、多个 checkpoint **不是**独立样本。

### D6 通过标准（训练前冻结）
C2 相对 C1（Validation，全部满足）：
1. Value RMSE 退化 ≤ 10%；
2. Reachable-direction sign 提升 ≥ 10 个百分点；
3. Reachable directional RMSE 改善 ≥ 10%；
4. 无新的严重 magnitude inflation（median ratio 不越出 [0.5, 2.0]×1.5 倍原区间）；
5. Total-gradient fidelity（D4 主指标）优于 C-init；
6. 不依赖单一 motion clip（LOCO 极差 < 主效应的 50%）。
正式 PASS 还需合理独立样本量与关键配对差异的统计支持（clip 级 bootstrap CI 不跨 0）。
样本量不足但方向一致且效应有实际意义时标记 `PILOT_PROMISING`，**不得**标 PASS。
裁定标签全集：`SUPPORTED / PILOT_PROMISING / INCONCLUSIVE / CONTRADICTED / BLOCKED`。
C1 有效而 C2 无额外收益时，不得为保留 pair loss 强行推进 C2（如实报告机制证据）。

## 4. Phase E：CVF-2a 条件性 Policy Improvement（单次更新）

**准入（全部满足）**：①数据/实现合同通过；②≥1 个新 Critic 在 Validation 有可信改善；
③无明显标量遗忘；④WM 查询状态无严重梯度失真（D3）；⑤D2/D4 有正向机制信号；
⑥剩余 GPU ≥0.6 GPU-h 且 wall-clock 充足。评价用独立 policy-evaluation split
（Val+Test starts 的并集，不重复使用 Critic Test 做模型选择——模型选择只读 Val）。

### E1 单次 Policy Update
- 组：P-init（不更新）/ P0（C-init）/ P1（C1）/ P2（C2）/ P-noV（H32 prefix-only）。
- 全组共享完全相同：Policy 权重（=pi_old@2100）、Adam optimizer state、WM、imagination
  starts（Train WM rollout 起点）、观测噪声（同训练分布）、随机数状态、H=32、lr。
  仅替换 Critic。执行一次真实优化器更新（复用 proto 训练代码路径，冻结 WM 不更新）。
- **Adam state 可恢复性**：若 `ckpt_002100.pt`/`last.pt` 不含 pi optimizer state（预检中，
  68MB ckpt 无 opt 键，倾向不可恢复），则改用所有组共享的显式 cold-optimizer 初始化并全程
  标注 `cold-optimizer experiment`，不得宣称忠实恢复原训练更新。
- 预定义 matched-update-norm 敏感性对照（若预算允许）：把各组更新归一到 P0 的参数范数后
  重新评估，作为尺度混淆的次要证据；正常 Adam 更新结果为主结论。

### E2 真实闭环评价
Simulator（非 WM）为最终评价环境：$\Delta J_{\mathrm{Sim}}=J_{\mathrm{after}}-J_{\mathrm{before}}$
（每 policy 变体在同一批起点确定性闭环 rollout，T_eval=96，γ^t c_t 折现，paired per-start）。
记录：mean/normalized return、tracking error、episode length、fall rate、termination 分布、
逐 clip paired Δreturn、实际动作变化幅度、参数更新范数、gradient norm、优化器实际更新方向。

### E3 短期重复更新
仅当单次更新出现正向控制信号且剩余预算充足：最多 10 次重复更新，每组从同一母本独立开始，
冻结 WM 与 Critic；报告 policy-value staleness（Critic 对应旧 Policy 的退化轨迹）。若单步
收益消失或 fall rate 恶化立即停止。禁止自动扩展到 50/100/1000 轮，禁止完整在线协同训练。

## 5. 机制判别树（结果 → Goal §9 情况映射）

| 观察形态 | 裁定路径 |
|---|---|
| C1、C2 均优于 C0 | 情况 A：真实价值锚定有效 → 拆分改善来源（标量 vs 差分权重）、检查是否集中 velocity-reachable、是否保持于 WM 域、进 Phase E |
| C1 有效、C2 无效 | 情况 B：不默认加 α；检验 pair 冗余/非线性尾部/与标量冲突/局部表达力；预批准 α 在 Val 换档一次；否则停止训练，报告"简单标量锚定可能足够" |
| C2 改善真实状态响应但未改善 WM total gradient | 情况 C：分析 $s^{\mathrm{Sim}}$ vs $\hat s^{\mathrm{WM}}$ 分布差与投影；不动 WM 参数 |
| C2 有限差分改善但 autograd 未改善 | 情况 D：检查扰动尺度/Critic 曲率/插值几何/参考非光滑；只提出下阶段实验，不自行实现新 loss |
| C0、C1、C2 均无改善 | 情况 E：检查 horizon mismatch / 覆盖不足 / 历史缺失 / 局部不可辨识 / 优化器 / 遗忘；只给机制证据边界，不扩网络 |
| 局部梯度改善但 Policy 不改善 | 情况 F：高价值负结果；检查 WM Jacobian/参数-动作转换/Adam 实际更新/scale/distribution shift/reward 冲突/staleness；不进在线训练 |

## 6. 执行纪律

- GPU 阶段（B/C/D/E）仅在 APPROVED=true 后启动；每阶段结束停机重算预算（见 BUDGET_PLAN）。
- NaN / OOM / 持续无效梯度 / 原始文件哈希变化 / 评价合同不一致 → 暂停该阶段，有限 CPU 诊断。
- 所有新增 checkpoint 保存：参数、Normalizer、optimizer state、parent hash、split hash、
  配置、seed、steps、loss 定义、Val metrics。
- 失败实验照常计入预算并如实记录；不利证据不得删除。
- 本轮不 commit / 不 push（ALLOW_GIT_PUSH=false），产物全部落在 cvf_proto/sacv、
  docs/cvf/sacv、results/cvf/sacv，不覆盖任何既有资产。
