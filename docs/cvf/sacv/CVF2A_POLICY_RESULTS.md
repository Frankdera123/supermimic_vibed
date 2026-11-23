# CVF2A_POLICY_RESULTS（CVF-2a Policy Improvement 验证）

日期：2026-11-20。

## 裁定：NOT RUN（预注册门禁失败，未消耗任何 GPU）

## 1. 门禁核对（协议 §4，全部 6 条须满足）

| 条件 | 要求 | 实测 | 判定 |
|---|---|---|---|
| ① 数据/实现合同通过 | 通过 | 通过（16/16 测试 + smoke 全绿） | ✓ |
| ② ≥1 个新 Critic 在 Validation 有可信改善 | 有 | C0/C1/C2 均未超过 C-init（D6 三门限失败） | **✗** |
| ③ 无明显标量遗忘 | 无 | C1 标量 RMSE 0.448→0.792（+77%） | ✗（C1/C2） |
| ④ WM 查询状态无严重梯度失真 | 无 | D3 未执行（前置条件 ② 已失败） | — |
| ⑤ D2/D4 有正向机制信号 | 有 | 无（pair 响应全组 ≤ C-init） | **✗** |
| ⑥ 剩余预算充足 | ≥0.6 GPU-h | 3.0/6.0（形式上满足） | ✓ |

## 2. 决策依据

按 Goal §8/§15 与协议 D6："如果某个假设被数据否证，应当及时停止无价值的后续实验"。
用一个未改善的 Critic 替换单次 policy update 的 bootstrap，期望产生可归因的正向
ΔJ_Sim 的先验概率极低，且负结果无法与"更新噪声"区分（cold-optimizer 单步更新的
per-start ΔJ 噪声 >> 本轮任何 Critic 间差异）。将该预算转投机制诊断（标签 SNR 分解）
产出了可执行的下一轮设计修正。

## 3. 预实现资产（本轮已就绪，下一轮可直接使用）

`sacv/tools/sacv_policy_update.py` 已完整实现并通过编译/评审：
- 忠实复刻原训练 update 路径（imag_step 无动作噪声、opt_step norm1.0+value0.1、
  γ^32·c₃₂·V.clamp(10) bootstrap、gw 末端刷新）；
- cold-optimizer 预案标注、组间共享想象轨迹（manual_seed 重置 → 强配对）、
  MJWarp B≤4 分批（想象 6 批/评价 5 批）、cos(update, P0-update) 诊断、
  ΔJ_Sim per-start 配对评价（T_eval=96，soft survival）。

**下一轮 Phase E 的启动条件**：修复后的标签合同（标量用 Y、pair 用 ΔG、族收紧到
delta_2100/2300）产出一个在 Val gated 上稳定优于 C-init 的 Critic（bootstrap CI
不跨 0），即可直接运行本工具。
