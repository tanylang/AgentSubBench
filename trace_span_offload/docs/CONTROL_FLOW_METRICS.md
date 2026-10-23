# 控制流指标总表(Control-Flow Metrics Canonical Table)

> 数据基础:53 span × 10 轮(SLM×5 + identity×5)全干净基线;
> 值 = span 级均值(SLM 组 vs identity 组);Δ = 配对差;95%CI 为 2000 次
> bootstrap。显著 = CI 不跨 0。
> **规范实现(单一代码来源)**:[span_offload/controlflow_metrics.py](span_offload/controlflow_metrics.py)
> (L0-L5 全部指标 + 注册表 + 配对总表生成),入口脚本
> [scripts/analysis/controlflow_table.py](scripts/analysis/controlflow_table.py);
> 历史实现:[analyze_offloadability.py](scripts/analysis/analyze_offloadability.py)(22 变量)、
> [controlflow_analysis.py](scripts/analysis/controlflow_analysis.py)(12 控制流专项)、
> [dynamics_analysis.py](scripts/analysis/dynamics_analysis.py)(L6 动力学)、
> [trace_edit_distance.py](scripts/analysis/trace_edit_distance.py)(L7 编辑距离)。

## L0 事件编码(一切指标的基础)

下游轨迹逐事件压缩为符号串:

| 符号 | 判定条件 |
|---|---|
| D | llm.call 且带 tool_calls(决策) |
| R | llm.call 仅文本(空转) |
| O | tool.execution ok 且只读(观察) |
| A | tool.execution ok 且写状态(动作,含 complete_task) |
| F | tool.execution ok=False(失败执行) |
| Rp | source=repair(harness 协议修复) |

## L1 路径与分支结构(走了哪些路)

| 指标 | 定义/公式 | SLM | identity | Δ [95%CI] | 判定 |
|---|---|---|---|---|---|
| KeyChainCoverage | 基线关键链(每步首工具名)在反事实序列的 LCS 覆盖率 | 0.860 | 0.933 | −0.072 [−0.123,−0.027] | **显著** |
| ToolSeqJaccard | 工具名多重集 Jaccard(反事实 vs 基线剩余) | 0.482 | 0.510 | −0.028 | n.s. |
| BranchWidth | \|不同成功工具名\|(扇出) | 4.40 | 4.69 | −0.298 | n.s.(方向负) |
| **MissingBranchRate** | 基线剩余工具名中未被成功调用覆盖的比例 | 0.157 | 0.098 | **+0.059 [+0.025,+0.099]** | **显著** |
| NewBranchRate | 成功调用中基线从未调用过的工具名占比 | 0.066 | 0.071 | −0.005 | n.s. |
| StepsRatio | 下游 LLM 步数 / 基线剩余步数 | 2.64 | 2.57 | +0.064 | n.s. |

**解读**:干预使下游"又窄又漏"——漏覆盖基线分支 6pp(显著),探索熵更低,
且没有用新探索补偿。偏离方向是"没走完原来的路",不是"跑偏到新路"。

## L2 决策节奏(怎么决策)

| 指标 | 定义/公式 | SLM | identity | Δ [95%CI] | 判定 |
|---|---|---|---|---|---|
| DirectActionRate | P(A\|D):决策后直接动作 | 0.540 | 0.488 | +0.052 [+0.007,+0.104] | **显著** |
| BatchDecisionRate | 一次决策 ≥2 个工具调用的占比 | 0.409 | 0.425 | −0.016 | n.s. |
| ActionsPerDecision | (nA+nF)/nD | 2.48 | 2.55 | −0.074 | n.s. |
| ObservePerDecision | nO/nD | 0.399 | 0.431 | −0.032 | n.s. |
| ReasonOnlyRate | nR/n_llm(空转轮占比) | 0.085 | 0.063 | +0.023 [+0.001,+0.053] | **显著** |
| ThinkThenObserve | P(O\|D) | 0.181 | 0.214 | −0.033 | n.s. |
| ObserveChaining | P(O\|O) | 0.226 | 0.225 | +0.001 | n.s. |
| ObsBeforeAction | P(O\|A) | 0.023 | 0.024 | −0.001 | n.s. |
| **ToolEntropy** | 成功工具名分布的归一化香农熵(探索多样性) | 0.535 | 0.657 | −0.121 [−0.205,−0.046] | **显著** |

**解读**:SLM 组下游"更果断但更多空转"——决策后直接行动的比率更高、
探索熵更低(路径更窄);空转轮显著增多是下游陷入无效思考的节奏签名。

## L3 循环 / 回溯 / 重试(重复走)

| 指标 | 定义/公式 | SLM | identity | Δ [95%CI] | 判定 |
|---|---|---|---|---|---|
| RevisitRate | 调用"之前已调用过的工具名"占比 | 0.301 | 0.315 | −0.014 | n.s.(类别梯度 B 0.16→F 0.55) |
| SameToolPersistence | P(name_t+1 = name_t) | 0.247 | 0.246 | +0.001 | n.s.(F 类 0.52) |
| DupCallRate | 同(工具,参数)重复调用率 | 0.053 | 0.071 | −0.017 [−0.033,−0.002] | **显著** |
| LoopPatternCount | 重复子序列模式(X,Y,X,Y 与 X,Y,X)计数 | 18.5 | 16.9 | +1.58 | n.s. |
| FailRetryRate | P(D\|F) | 0.245 | 0.285 | −0.040 | n.s. |
| DeadEndRate | P(F\|D):决策踩进失败的概率 | 0.274 | 0.294 | −0.019 | n.s. |

**解读**:全局无显著差异,但 RevisitRate / SameToolPersistence 沿
B(0.16/0.14)→ C → E → D(0.53)→ F(0.55/0.52)单调上升——"反复回头
调同一批工具"是失败类的控制流签名(描述符,非全局响应)。

## L4 失败螺旋(失败动力学)

| 指标 | 定义/公式 | SLM | identity | Δ | 判定 |
|---|---|---|---|---|---|
| hall | 失败执行占比(nF/n_tools) | 0.317 | 0.350 | −0.033 | n.s. |
| FailChaining | P(F\|F) | 0.281 | 0.309 | −0.028 | n.s. |
| MaxFailRun | 最长连续失败段 | 3.58 | 4.07 | −0.49 | n.s. |
| RepairRate | 协议修复事件占比 | 0.005 | 0.005 | −0.001 | n.s.(C 类 span 上 4-6×) |

**解读**:失败螺旋是装置的背景属性,不是干预诱发的;repair 在个别
span 上被差异化触发(2 个 C 类)。

## L5 终止 / 出口 / 状态写入(怎么收尾)

| 指标 | 定义/公式 | SLM | identity | Δ [95%CI] | 判定 |
|---|---|---|---|---|---|
| ControlFlowDeviation | 终止性工具使用是否与基线分叉(0/1) | 0.094 | 0.000 | +0.094 [+0.019,+0.189] | **显著** |
| AbnormalExit | 末事件非 complete_task(1=非正常终止) | 0.023 | 0.008 | +0.015 | 边界 |
| FirstCompleteNormPos | 首次 complete_task 位置/总长 | 0.977 | 0.992 | −0.015 | 边界 |
| FirstWriteNormPos | 首个写操作位置/总长 | 0.977 | 0.992 | −0.015 | 边界 |
| WriteRatio | 写操作占(写+观察)比 | 0.658 | 0.624 | +0.034 [−0.001,+0.073] | 边界 |

**解读**:SLM 组更频繁地改写了终止路径(ControlFlowDeviation 显著),
异常出口略增——终止行为的扰动是干预响应的一部分。

## L6 动力学(时间解析)

| 指标 | 定义 | 当前值 |
|---|---|---|
| d_t 曲线 | 逐位置分歧度(工具名匹配减 identity 噪声地板) | B 类峰值 t=3(+0.40)后衰减;D 类平台 +0.86→+0.56;TAG 持久 |
| τ_on | 分歧出现位置 | 17/53 从未分歧;44% 第一步即分歧 |
| L_prop | 传播深度(d_t>θ 的最远位置) | 与 τ_on 同分布(集中前 3 步) |
| L_rec | 恢复距离(此后全部 ≤θ;∞=永不恢复) | B 类 45% 在 3-6 步内恢复;D 类 7/7 永不恢复 |
| persistence | d_t>θ 的位置占比 | D 类≈1.0 |
| amplification | 分歧后最大 d / 起始 d | 阈值 0.3/0.4/0.5 下结论稳健 |

## L7 轨迹距离(整条签名序列)

| 指标 | 定义 | 当前值 |
|---|---|---|
| d_cross | 归一化结构编辑距离(SLM vs identity,签名序列 Wagner-Fischer) | 0.573 |
| d_self | identity 组内编辑距离(重跑噪声地板) | 0.446 |
| **Δ = d_cross − d_self** | 干预净分歧 | **+0.128** |
| t★ | 首次分歧点(最长公共前缀 + 1) | 中位 2.0;44% ≤1;89% ≤3 |

## 功能定位汇总

**核心响应变量(全局显著,7+2 个)**:
ControlFlowDeviation、DirectActionRate、KeyChainCoverage、
ReasonOnlyRate、DupCallRate、ToolEntropy、MissingBranchRate(显著);
WriteRatio、AbnormalExit(边界)。

**机制描述符(全局不显著、但类别间梯度/个案信息强)**:
RevisitRate、SameToolPersistence(失败类签名)、RepairRate(C 类个案)、
FailChaining / MaxFailRun / hall(背景螺旋)、BranchWidth / NewBranchRate
(方向佐证)、d_t / τ_on / L_rec(动力学形态)、d_cross / t★(轨迹距离)。

**与成败分层的关系**(success_stratification):
成败最强判别器 = KeyChainCoverage(0.89 vs 0.59)+ ToolSeqJaccard
(0.51 vs 0.28)+ MaxFailRun(3.3 vs 6.0)+ AbnormalExit(0 vs 0.21);
失败 ≠ 探索太多(熵无差异),而是"没走对"。

---

## 冗余分析(2026-09-11,53 span 级 SLM 均值相关矩阵,|r|≥0.7)

**严格重复(删其一)**:
- FirstWriteNormPos ≡ FirstCompleteNormPos(同定义同值)→ 删前者
  (或将前者重定义为"第一个非只读工具执行位置");
- DeadEndRate ≈ 1 − DirectActionRate(r=−0.80,本数据中决策后非 A 即 F)
  → 保留 DirectActionRate,DeadEndRate 为推导量。

**高度冗余族(选代表)**:
- 规模族(tools/llm/BatchDecisionRate/ActionsPerDecision/StepsRatio/BranchWidth,
  两两 +0.70~+0.78):保留 tools/llm 为基线,其余降备用;
- 回溯族(LoopPatternCount×SameToolPersistence +0.76、×RevisitRate +0.73):
  删 LoopPatternCount;
- 观察族(ObserveChaining×ObservePerDecision +0.79、×WriteRatio −0.79):
  ObserveChaining 降备用。

**高相关但语义独立(保留)**:
- ControlFlowDeviation×KeyChainCoverage(−0.71)、DupCallRate×ToolEntropy(+0.72)、
  NovelInfoRate×hall(−0.71)、螺旋×规模交叉(MaxFailRun×llm +0.79 等——
  共变是"失败轮更长"的科学事实,不是定义重复)。

**精简结论**:核心表 40 → 31(删 3、降级 6);降级变量不进核心表,需用时
按引用查询。
