# Batch 3 确认实验预注册（2026-09-15 冻结）

> 计划 v0.5 §12.1 要求:descriptor、margin、风险阈值与统计模型必须在
> confirmation data 生成前冻结。本文件即冻结记录,后续不得因看到结果
> 而修改。

## 1. 数据集

- 部署:simplified 后端 × 27B 采集批次
  (`check_point_traces/appworld_simplified_qwen3_8-27b_batch1`),
  baseline=Qwen3.8-27B@8101(vLLM,thinking 开启,服务端默认),
  SLM=Qwen3.5-9B@8000,engine 2026-09-13.1+
- Screening(已完成):50 span 抽样 × I/B/T_span × 5 轮;去重后
  **48 span 完整** = 720 轮(结果:results/main_comparison_27b_final.json)
- 确认集:**18 span**(12 任务),选择脚本
  `scripts/select_confirmation_spans.py`(确定性:组内按 T−B 效应排序
  取两端+均匀取中,同任务≤2;配额 ts 6 / fs 6 / tag 4 / other 2)
- 本轮扩跑:**I/B/T_span 各 15 轮**(累计 20),block_seed=0,
  md5 稳定计划(引擎修复版)

## 2. 冻结的 descriptor 与统计模型

- Descriptor 集(运行时可得):node_pattern、control_role、
  state_effect(span_descriptor.mechanism_tags)、verifier_type、
  position(invocation_index / 总长三分位:early/middle/late)、
  task_id
- 指标:全部走 `span_offload/controlflow_metrics.py` 规范实现
- Binary(strict_success、collateral_damage):报告率 + risk difference;
  CI = task/span cluster bootstrap(2000);建模 = mixed-effects
  logistic(随机截距: task、span)
- Count(tools/llm/fail/repair):mean/median/P90/P95;配对 cluster
  bootstrap;zero-inflation 明显时用 two-part model
- 连续(token/latency):配对均值差 + bootstrap CI;P50/P95/tail 放大
  分开报告
- 轨迹:Traxr 式 d_norm、t★、within-B 配对距离 vs T_span-to-B 距离
  (energy distance/MMD)

## 3. 冻结的判定阈值(§14)

| 判据 | 冻结值 |
|---|---|
| 质量非劣效 margin ε(strict success) | **ε = 0.05**(LCB(ΔSuccess_{T−B}) ≥ −0.05) |
| 风险阈值 δ(collateral UCB) | **δ = 0.05**(UCB(P(Collateral_T)) ≤ 0.05) |
| 效用 | LCB(E[NetSaving]) > 0;**NetSaving = per-span 总 token 配对差**(target+downstream 分项,B−T),token 层主口径;价格换算 = 2026-09-13 公开价目版(cache-miss 主口径,cache-aware 敏感性)。**token_delta 字段废弃**(混合口径,仅目标层) |
| 显著性 | primary 单独报告;secondary 用 Benjamini-Hochberg FDR;报告效应量,不以 p<0.05 代替实际意义 |

## 4. 主假设(可证伪表述)

- H2':T_span−B 效应在预定义 descriptor(node_pattern/position/
  state_effect)间存在稳定异质性
- H4':至少存在满足 quality/risk 约束且 NetSaving LCB>0 的 span 子集
- 失败降级:退回 aggregate measurement(计划 §19 Batch 3 后的允许结论)

## 5. 执行

```bash
# spans 文件: out_span_offload/spans_simplified_qwen27b_confirm18.jsonl
# 服务器命令(累计 15 轮,block_seed=0,RESUME=1 可断点续跑)
```
