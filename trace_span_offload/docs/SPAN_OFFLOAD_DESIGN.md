# Span 卸载工作原理(Stage 10 概览,与当前实现一致)

> 目标问题:**哪些 span 可以安全地用 SLM(小模型)替换基线 LLM 来执行?**
>
> 思路一句话:先把"span 可卸载性"从信仰变成**可测量的量**,再用静态 allowlist 把测量结果固化为**逐 span 的卸载决策**。
>
> 详细运行方法见 README「当前实测流程」。

---

## 一、整体结构图

```
┌─────────────────────────────────────────────────────────────┐
│  输入:冻结 Agent 的成功 native traces(check_point 格式)      │
│  · 每个 llm.call span 带完整会话 + 环境增量 + 真实状态 hash    │
│  · 记录请求的原生 tools 参数(gen_ai.request.tools)           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ① extract  │  Span 提取
                               │  · 每个 LLM 调用 → 一个 span 实例
                               │  · 六条 eligibility 门控(状态可恢复、请求完整…)
                               │  · 三层条件化标签(span 特征 / 任务 / Agent)
                               ▼
                  ┌───────────────────────────┐
                  │        spans.jsonl        │
                  └─────────────┬─────────────┘
                               │
                    ② replay   │  单 span 反事实重放
                               │  · 每次只把 1 个 span 换成 SLM,其余不变
                               │  · restore:重建 span 前环境(工具重放)
                               │  · SLM 拿到同 prompt + 同原生 tools 协议
                               │  · 本地验证 → 注入 → 下游续跑 → 最终评估
                               ▼
                  ┌───────────────────────────┐
                  │       replays.jsonl       │
                  └─────────────┬─────────────┘
                               │
                    ③ analyze  │  条件化卸载性分析
                               │  · O_m(φ):每类 span 被 SLM 替换后的严格成功率
                               │  · 带 95% 置信区间 + 成本/延迟节省
                               ▼
                  ┌───────────────────────────┐
                  │       卸载性 profile       │
                  │  (哪些 span 类可安全卸载)   │
                  └─────────────┬─────────────┘
                               │
                    ④ validate │  静态 allowlist 多 span 验证
                               │  · 冻结规则:只读 + 结构化输出 + 强校验 + 置信下界≥阈值
                               │  · 逐 span 决策:命中 → SLM,否则 → baseline
                               ▼
                  ┌───────────────────────────┐
                  │  输出:逐 span 卸载决策规则   │
                  └───────────────────────────┘
```

---

## 二、核心思想:单 span 反事实重放

一次实验**只干预一个 span**:

```
span 前状态(restore 重建) ──► [ span j:换用 SLM ] ──► 下游:baseline 大模型继续 ──► 最终评估器
                              │
                              ├─ 同一个 prompt / 同一个原生 tools 协议
                              ├─ 本地验证(解析 + schema + 工具名校验)
                              └─ 失败按 fallback_mode 处理:
                                   none              → 注入失败信号,strict 如实记 0
                                   recorded_baseline → 兜底注入 baseline 记录输出(测 ceiling)
                                   live_baseline     → 现场调大模型重算
```

因为只有 span j 被改变,最终结果的变化**只能归因于这个 span**。每个 span 是**独立实验**("下一个 span 的卸载"由 batch 循环独立进行,互不污染)。

### 下游续跑 = 真实 AutoGen agent

下游续跑**不重写 agent**,而是实例化生成 trace 的**同一份 AutoGen agent 源码**
(`appworld_agents.code.autogen.agent`):恢复的会话以原生消息格式 seed 进
AssistantAgent 的上下文,工具调用经真实工具函数在 AppWorld 环境执行。
autogen 缺失时才回退 span_offload 的简化循环(记录中如实标注)。

### 卸载性度量

```
严格成功 S^strict = 最终任务成功 且 无附带损伤(如误发邮件、误删数据)

卸载性  O_m(φ) = P( 严格成功 | span 特征为 φ、基线 LLM 原本成功、替换模型为 SLM m )
```

- 只统计**基线原本就成功**的 run(失败是替换造成的,不是任务本来就做不成)
- **identity ceiling** = `fallback=recorded_baseline` 下的成功率,是桥 + baseline 的
  **噪声地板**(受模型采样非确定性、环境恢复近似影响,不是 100%);O_strict 相对 ceiling 解读
- 成本口径:`节省 = 基线 span 成本 − (SLM 成本 + 校验成本 + fallback 成本)`

---

## 三、span 分类:三层条件化标签

| 层 | 内容 | 例子 |
|---|---|---|
| **span 特征 φ** | 这个 span 本身是什么 | 做什么(tool_selection / tool_argument_generation / final_synthesis…)、角色、只读还是改状态、输出契约类型、校验手段 |
| **任务条件 z** | 任务本身的性质 | 跨 app 与否、任务长度、只读/改状态、附带风险等级、评估方式 |
| **Agent 条件 a** | 运行在哪个 agent 上 | 代表 agent 身份、架构类型 |

**约束**:重放结果(成功/失败/成本/模型身份)是"答案",**禁止**混进 span 特征标签。

---

## 四、当前关键结论(50e1ac9_1,13 span,M2.5 / DeepSeek V4 Flash)

| 测量 | 结果 | 含义 |
|---|---|---|
| identity ceiling(recorded_baseline) | 10/13(修复后预期 12/13) | 噪声地板 |
| O_strict(none,SLM 带原生 tools) | 2/13(本地层 5/13) | **M2.5 自身卸载成功率** |
| 协议的重要性 | 文本协议下 M2.5 本地层 ≈0;原生 tools 协议后 5/13 | **SLM 必须拿到与原 agent 相同的 function calling 协议** |
| 三层分解 | 本地层 5/13 格式可接受 → 下游零异常 → 最终 2/13 成功 | 语义正确性是 M2.5 更大的短板 |

## 五、诚实分层:什么能测

| 层 | 状态 | 说明 |
|---|---|---|
| **本地层**(SLM 输出可解析、过契约校验) | ✅ 全量可测 | 记录的消息 + 原生 tools 协议 |
| **live 下游/最终层** | ✅ 已解锁 | check_point 完整会话 + 真实 AppWorld 环境 + 真实 AutoGen 续跑 |
| **遗留近似** | 如实标注 | env_changes 为空时恢复走工具重放;占位 hash 记为 mismatch;span 1 缺指令由适配器补入并标记 |

## 六、整体逻辑一句话

```
成功运行记录 ──切出 span──► 逐 span 用 SLM 做受控替换实验(真实环境 + 真实 agent 续跑)
──统计──► 每类 span 的"替换后仍成功"概率(相对 identity ceiling 解读)
──冻结成白名单规则──► 运行时逐 span 决策:命中 → SLM(省成本),不命中 → baseline LLM(保质量)
```
