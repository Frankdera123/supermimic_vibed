# PADA_RESEARCH_DECISION

日期：2026-11-18。决定：**PADA-0 = INCONCLUSIVE，不进入正式训练（PADA-Data 不启动）。**

## 1. 回答任务的唯一研究问题

> 在相同 simulator budget 下，Policy-driven probing 是否比 random probing 更高效地学习 control-relevant dynamics？

**当前答案：无法判定（INCONCLUSIVE），但方向性证据一致偏 YES。**

- 等预算（每臂 6144 sim batch-steps）下，policy-shift probes 相对 random probes 在主指标（D_total sign agreement @ H=32，CRGF 三路径体系、fail-closed 门禁）上于全部 3 个 policy 方向集为正（+0.44 / +0.50 / +0.50），在 500/1000/1500 三个微调快照上全部稳定，且迁移到 **unseen** policy directions（同 run 后续单步 update 与跨 run 的真实单步 update）。
- 训练方向集的配对 95% CI 不含 0（+0.444, [0.111, 0.778], n=9）。
- 但两个 unseen 集 gated n=4，CI 含 0；H32 门禁通过率 20.8% < 50%。按预注册树，这阻断 PASS、也使 NO-GO 不成立 → INCONCLUSIVE。

## 2. 三个重要观察（如实报告，不做过度解读）

1. **方向特异性，不是全局效应**：M2（policy probes）的收益只出现在 policy-relevant 方向族（含 unseen），控制组（fresh random directions）上 M2 ≤ M1；反过来 M1 的收益也只出现在 random 方向族。每个臂都在改善自己数据分布上的 fidelity——PADA 关心的是"哪一族对 control 才重要"，在这一族上 M2 胜出。这与 CRGF 的 H4（random 方向随 H 恶化、policy 方向可靠）互相印证。
2. **灾难性遗忘背景**：replay buffer 未随 checkpoint 保存（审计 §6.1 预声明），M1/M2 只能用 6144 条 probe 转移微调，两臂的普通 prediction（Level-1）与对 M0 已有保真度均受损。因此测得的是"退化背景下的净差"，不能直接外推为正式训练场景收益。M2 在 Level-1 上还略差于 M1（free-H32 0.91 vs 0.78），而 Level-3 上 M2 更好——普通 MSE 与 control-relevant fidelity 的可分离性（PASS 条件 5）在数据中真实出现。
3. **H=8（critic 主导时域）无分离**，差异集中在 H=32（dynamics 主导时域）——与 CRGF 的路径分解结论一致：dynamics 保真度的价值要在 H≥32 上谈。

## 3. 决定与理由

| 选项 | 决定 | 理由 |
|---|---|---|
| 进入正式训练（PADA-Data / directional loss / co-adaptation） | **否** | PASS 未达成；在 INCONCLUSIVE 上堆训练管线违反任务纪律（"不预设 PADA 有效"） |
| 宣告 NO-GO、停止整线 | **否** | 方向一致 + 跨快照稳定 + seen 集 CI 排除 0 + unseen 迁移的形态不符合 NO-GO 的预注册要求；此时收线会丢掉一个尚未判定的真实信号 |
| **追加一次有界审计（PADA-0b）后再判** | **推荐，待授权** | 见 §4；成本 ≈0.6–0.9 GPU-h，不改变判定标准 |

## 4. PADA-0b 提案（未授权不执行）

1. **评估加密**（≈0.6 GPU-h）：eval 方向 4→10/集、starts 6→12、ε 加 {0.2}，把 H32 每集 gated n 提到 ≥10。判定标准不变（协议 §7 原文适用）。
2. **锚定微调**（≈0.3 GPU-h，需正式修订协议 §5 并记录理由）：两臂加入同量 M0-自生成 anchor 数据，消除遗忘背景。
3. 判定分支：若加密后 M2>M1 且 unseen CI 不含 0 → PASS → 才进入 PADA-Data 最小可行验证；若优势消失或翻转 → NO-GO 收线。

## 5. 研究纪律自检

- 未把 shared-obs delta 说成 closed-loop（方向生成口径在协议/审计/结果三处一致标注）；
- 未把 train-direction 改善说成 generalization（Test A/B/C 分离，unseen 迁移单独报告且不凑显著）；
- 未把更多样本的收益说成 direction 收益（两臂严格等预算）；
- 未把 Level-1 改善说成 control 改善（判定只用 Level-3/2）；
- 结果不足 → 如实 INCONCLUSIVE，未凑 PASS；
- 全程 GPU 1.0/4.0 GPU-h，未超预算，未自行扩权。

## 6. 下一步是否值得进入正式训练

**不值得（当前证据下）。** 先执行 PADA-0b（评估加密 + 锚定微调，需授权）；其结果为 YES 才谈 PADA-Data，为 NO 则整线收束到论文写作。若不授权 0b，则 PADA 线以 INCONCLUSIVE 封存，与 OPCL/CRGF 的结论链（欠响应→方向隔离→critic 主导→NO-GO）合并成完整的机制叙事。
