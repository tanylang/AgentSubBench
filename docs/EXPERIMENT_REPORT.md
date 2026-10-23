# Span 卸载可卸载性实验报告

> 实验期:2026-08-25 ~ 2026-09-03
> 数据集:AppWorld simplified traces(3 个任务、20 个 span)
> 采样:每 span × 5 轮,SLM 组 100 次重放 + identity 组 100 次重放

---

## 1. 研究目标与评估范式

**总目标**:证明 Agent 对"局部决策函数替换"(LLM→SLM)存在稳定且结构相关的
响应机制,建立 intervention dynamics / sensitivity 模型,并从中推导
每个 span 的 Offloadability。

**评估范式**(2026-09-03 确立):
- 废弃 "原始 LLM trace = 唯一 ground truth trajectory";
- 采用 "LLM execution distribution = baseline behavior distribution"
  (由 identity 注入组的重放采样估计);
- 核心问题:SLM 干预是否使 execution response **超出 baseline natural
  variation**;
- 判定量:标准化效应量 d = (μ_slm − μ_base) / σ_base,|d| > 1 视为超出。

---

## 2. 系统与实验装置

- **环境**:AppWorld 模拟器(in-process direct 模式);
- **决策模型**:baseline = DeepSeek V4 Flash;SLM = Qwen/Qwen3.5-9B;
- **干预装置**:单 span 反事实重放引擎 `SingleSpanReplayEngine`
  (restore → SLM → parse/verify → inject → 下游续跑 → 最终评估);
  - SLM 轮:注入 SLM 输出;
  - identity 轮:注入基线记录输出(不调用 SLM),即"零干预"对照;
- **续跑后端**:simplified(AppWorld 内置 agent 本体,任务级终止条件);
- **度量体系**(每轮记录):
  - 本地层:`local_outcome`(parse/schema/verifier 三关)+ 等价验证器
    `equivalence`(工具名多重集 / 参数 / 执行结果 / 文本语义 四维
    requirement);
  - 下游层:`downstream_trace`(逐轮 llm.call + tool.execution 结构化
    轨迹:步数、工具执行数、失败数、幻觉名、repair 数);
  - 最终层:`strict_success`(= final_task_success ∧ ¬collateral_damage)。

---

## 3. 数据收集

| 组 | 采样 | 记录数 | 状态 |
|---|---|---|---|
| SLM 组(干预) | 20 span × 5 轮 | 100 | ✅ |
| identity 组(对照) | 20 span × 5 轮 | 100 | ✅(经 2 次 bug 修复后重跑部分数据) |

采集过程中的关键修复(已全部落地):
1. 上下文重建对齐(角色/配对/截断/合成 id/重复指令);
2. DeepSeek reasoning_content 协议修复(所有 assistant tool_calls 消息
   携带非空 rc);
3. 未配对 tool_calls 400 的协议修复(被 agent skip 的非法名补齐中性
   失败回执);
4. freeform identity 双重包装修复。

---

## 4. 分析方法

1. **三层指标**:本地(等价验证器通过率)、下游(步数/工具/幻觉率)、
   最终(strict_success);
2. **配对对照**:同 span 的 SLM 组与 identity 组;
3. **增量效应**:d = (μ_slm − μ_base) / σ_base(按组内轮次计算);
4. **结构分层**:node_pattern / control_role / 轨迹位置 × 响应类别。

---

## 5. 结果

### 5.1 总体指标(每组 100 次重放)

| 指标 | SLM 组(干预) | identity 组(对照) | 原 trace 剩余参照 |
|---|---|---|---|
| strict_success | **99%** | 97% | — |
| 等价验证器通过率 | 45% | 71%¹ | — |
| 平均 LLM 步/轮 | 9.7 | 10.0 | **3.9** |
| 平均工具执行/轮 | 30.0 | 35.0 | **4.3** |
| 幻觉失败率 | 30% | 26% | 0% |
| repair 触发/轮 | 0.5 | 0.6 | — |

¹ identity 的 eq 不足 100% 是因为执行结果比较对"不可复现值"(登录
token、时间戳)天然敏感——它是该度量的校准基线:SLM 的 45% 应相对
71% 解读(可复现结果上的相对一致率 ≈ 63%)。

### 5.2 关键发现一:背景效应巨大

identity(零干预)下游平均 10 步 / 35 工具执行,是原 trace 剩余步数
(3.9 步 / 4.3 工具)的约 3×/8×——**续跑机制本身就会重做大量工作、
产生 26% 的幻觉失败执行**,即使注入的是完美输出。

### 5.3 关键发现二:卸载的增量效应小且双向

- strict:SLM 99% vs identity 97% → 卸载不降低任务成功率;
- 下游成本:Δ ≈ −0.3 步 / −5 工具(不显著);
- 幻觉率:Δ ≈ +4%(轻微);
- 结论:**在聚合层面,SLM 干预未使系统行为超出 baseline 自然波动**。

### 5.4 20 span 判定表(以 baseline 分布为参照)

| 类别 | 数量 | span |
|---|---|---|
| 范围内(可卸载) | 13/20 (65%) | 全部 tool_selection(3)、final_synthesis(3)、7 个 tool_argument_generation |
| 超出(更差) | 4/20 | 21abae1_1 inv=1(tools +5.2σ)、23cf851_1 inv=1、23cf851_2 inv=3(+44.5σ¹)、inv=4 |
| 超出(更好) | 3/20 | 23cf851_1 inv=2/3/4(SLM 偏离恰好短路了基线的幻觉螺旋) |
| C(成功率受损)/D | 0 | — |

¹ 基线零方差 + SLM 偏差叠加的脆弱点估计。

### 5.5 H2 结构分析:响应是结构相关的

- 所有"超出"都集中在 **tool_argument_generation(local_executor)**
  (14 个中 7 个超出);tool_selection(global_planner)与
  final_synthesis(terminal)零响应(各 3 个全在范围内);
- 轨迹位置不是主要区分维度(前段 2/6、后段 2/14 超出);
- 同结构内方向分化(更差 vs 更好)→ 结构决定"是否敏感",
  干预内容(偏离度 δ)决定"方向"——H3 的线索。

---

## 6. 当前研究假设状态

| 假设 | 状态 |
|---|---|
| H1 稳定性(同 span 干预响应收敛到稳定分布) | ✅ 初步支持(两组均可估计 σ,轮间有可测方差) |
| H2 结构性(响应可由结构特征预测) | ✅ 初步支持(响应集中在 local_executor 类) |
| H3 干预强度(响应与 δ 单调相关) | 线索明确,待偏离度梯度实验 |
| H4 可推导性(Ō(j)=f(xⱼ,δⱼ)) | 待 H3 后建模(需更大样本) |

---

## 7. 已知局限

1. n=5 的 σ 估计噪声大,效应量点估计脆弱(方向稳健,精确值需更多轮次);
2. |d|>1 的阈值为约定;
3. 等价验证器的执行结果分量对不可复现值(token/时间戳)敏感;
4. 样本仅 3 个任务、20 span,覆盖的 (node_pattern × state_effect)
   组合有限;
5. 下游幻觉背景(26%~30%)未消除(方法 A 工具清单锚定待实现)。

---

## 8. 下一步

1. **扩样本**:20 → 50 → 155 span(装置直接可跑);
2. **H3 实验**:偏离度梯度(多 SLM 候选 / 温度档位);
3. **方法 A**:工具清单锚定,降低背景幻觉,提升 Δ 信噪比;
4. **建模**:拟合 Ō(j) = f(xⱼ, δⱼ),留出 20% 验证(H4)。

---

*附:数据文件(results/)*
- `offload_profile.jsonl` / `_summary.json` — SLM 组
- `offload_profile (1).jsonl` — identity 组(部分,含早期污染数据)
- `offload_profile_id_21abae1_1.jsonl`、`offload_profile_id_23cf851_1b.jsonl`、
  `offload_profile_id_23cf851_2.jsonl` — identity 组(修复后)
