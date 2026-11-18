# B0_B1_B2_CONTROL_FIDELITY_COMPARISON

日期：2026-11-18。回答用户要求："确认三个模型 checkpoint 的来源、初始化和配置是否真正可比；不在当前仓库必须明确说明。"

## 1. 声明：B0/B1/B2 不在本仓库

三个分支 checkpoint 属于 **OPCL 研究线**（`~/Documents/Codex/2026-11-14/new-chat-2/work/`），supermimic_vibed 仓库内没有任何 B0/B1/B2 训练产物，也没有 process/outcome loss 实现。本审计**只读加载** OPCL checkpoint，用 **proto 代码**（`--proto_root`）评估，不使用其他研究线的模型冒充。两线 WM 结构不同（70 维全增量 vs 35 维加速度级，OBS 99 vs 93），见 SOURCE_AUDIT §5。

## 2. 来源与可比性核对（全部机器可验证）

文件：`results/b0b1b2_provenance.json`、`results/b0b1b2_sha256.txt`。

| 项 | online_B0 | online_B1 | online_B2 | 共享原点 ckpt_000501 |
|---|---|---|---|---|
| iteration | 2300 | 2300 | 2300 | 501 |
| pi_updates / wm_updates | 1800 | 1800 | 1800 | 1 |
| 含 critic | 是 | 是 | 是 | 是 |
| seed / γ / H / hist / batches | 0 / 0.99 / 32 / 10 / pi 4096, wm 2048 | 同 | 同 | 同 |
| motion_dir | lafan1_g1（容器路径） | 同 | 同 | 同 |

args 全量 diff 的差异键（唯一分支定义）：

| 键 | B0 | B1 | B2 |
|---|---|---|---|
| b1_match_opcl | false | **true** | false |
| opcl_enabled | false | false | **true** |
| opcl_lambda_p | 0.0 | 0.0 | **0.01** |
| opcl_freeze_policy_pair | false | false | **true** |

**可比性结论**：
1. 同源（501）、同初始化链、同 seed、同预算、同评估上下文 → B0/B1/B2 互相可比。
2. **B2 vs B1 非单变量对照**（与 Codex 的 [FACT] 一致）：B2 开启 opcl_enabled（含其自身采集模式）+ λ_p=0.01 + freeze_policy_pair，且未开 b1_match_opcl；B1 是"预算匹配但无 OPCL loss"。B2 相对 B1 的差异不能唯一归因 Process loss。
3. 一个诚实性提醒：交接文档 4.3 的 B0/B1/B2 数字（0.615/0.635/0.625）对应 300 轮实验；本地可得分支 checkpoint 是 **iter 2300**（继续训练过的）。二者机制定义相同但训练量不同——本报告的梯度剖面属于 iter-2300 模型，引用历史回报数字时必须注明迭代差异。

## 3. 当前对照状态

| 维度 | B0 | B1 | B2 | 状态 |
|---|---|---|---|---|
| WM 侧三路径梯度剖面（共享 critic） | ✓ | ✓ | ✓ | 本轮 CPU 完成（EVALUATION_REPORT §2） |
| WM 侧梯度剖面（own critic） | ✓ | ✓ | ✓ | 本轮 CPU 完成 |
| 单步动作响应 ratio（s4/s5 协议） | 历史 checkpoint 有 | — | — | OPCL 线 ckpt 未测，可补 |
| Return 级方向导数 vs sim FD | 未测 | 未测 | 未测 | **GPU 授权后执行**（crgf_sim_fd_audit.py） |
| 完整任务回报 | 0.6153* | 0.6350* | 0.6250* | *300 轮历史数字，单 seed，40/40 训练内 clip |
| 真实 Policy Improvement 预测 | 501 单步 pair 存在 | — | — | OPCL 线已有 |

主要对照为 **B2 vs B1**（用户要求 + Codex FACT）；B0 仅作参考点。

## 4. 已有对照数据的小结（含边界）

- 固定历史 pair 8 步归一化状态误差：B0 1.459 / B1 1.471 / **B2 1.275**（单 pair；B2 短时预测更好）。
- 同 pair 策略改进预测 MAE：B0 0.01751 / **B1 0.01236** / B2 0.01372（B1 更好——"预测更准 ≠ 改进预测更准"的直接例证）。
- 本轮梯度剖面：B2 状态路径梯度更大（H=32）但想象回报更差；B2−B1 配对差 CI 含 0。
- **综合**：三条独立证据（状态误差、改进预测、梯度剖面）在 B0/B1/B2 上互不共线——预测精度、改进方向质量、梯度可信度是三个可分离的量，正是 Level-1/2/3 不等价的实证形态。
