# Stage 10: Conditional Span Offloadability Pilot

单 span 反事实 SLM 替换测量流水线（spec 14.1–14.9）的参考实现。

```
Prefix → SLM_s → Baseline LLM for later spans → Final Evaluator
```

一次实验只干预一个 span，把结果变化归因于当前 span。**不实现**完整在线多 span routing runtime（14.9 边界，留给 TraceOffload）。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 离线 smoke（无需任何 API，脚本化确定性 LLM，全流程验证）
python -m span_offload.cli demo --offline --out data_demo

# 2) 接入真实 SLM 后
cp config.example.json config.json
#   编辑 config.json：slm.base_url / models / api key（或用环境变量）
python -m span_offload.cli demo --config config.json --out data
```

环境变量注入（推荐，避免密钥进文件）：

```bash
export STAGE10_SLM_BASE_URL=http://<host>:<port>/v1
export STAGE10_SLM_API_KEY=...
export STAGE10_SLM_MODEL=qwen3-4b-instruct
export STAGE10_BASELINE_BASE_URL=...   # 基线 LLM（可与 SLM 同源不同模型）
export STAGE10_BASELINE_MODEL=qwen3-32b-instruct
```

**SLM 协议**：`ChatModelClient` 走 OpenAI 兼容 `/chat/completions`（vLLM / SGLang / Ollama / TGI 等自部署 SLM 均支持）。如果你的 API 协议不同，只需改 `span_offload/llm_client.py` 一个文件。

## 流水线命令（真实 AppWorld 数据的用法）

```bash
# 提取：原生 traces -> spans.jsonl（14.1-14.3 六条 eligibility 条件）
python -m span_offload.cli extract --traces data/traces --manifest tasks.json --out data/spans.jsonl

# 重放：pilot spans × 候选 SLM 单 span 反事实重放（14.4-14.5）
python -m span_offload.cli replay --spans data/spans.jsonl --config config.json \
       --out data/replays.jsonl --models qwen3-4b-instruct --limit 100

# 分析：条件化卸载性 profile + task/agent/architecture 条件化 + 完整性检验（14.6-14.7）
python -m span_offload.cli analyze --replays data/replays.jsonl --out data/report --min-samples 10

# 效用验证：静态 allowlist 多 span 端到端 sanity check（14.8）
python -m span_offload.cli validate --config config.json --replays data/replays.jsonl \
       --spans data/spans.jsonl --out data/report
```

`demo` 命令 = 用 harness 的两个代表性 Agent（repA planner-executor / repB reactive）+ 8 个 mock AppWorld 任务走完五步（traces → extract → replay → analyze → validate），产物在 `data/report/report.md`（`data_demo/report/report.md` 是离线示例输出）。

## 代码结构（对应 spec 章节）

| 模块 | 内容 |
|---|---|
| `span_offload/schemas.py` | 三层条件化结构（14.2.1-14.2.6）：SpanDescriptor / TaskCondition / AgentCondition / ReplayInput；canonical 词汇表；label-leakage 守卫（14.2.5） |
| `span_offload/span_extraction.py` | 从 native trace 提取 s=(H,t,r,j)，六条 eligibility（14.1） |
| `span_offload/replay.py` | 单 span 反事实重放引擎（14.4），五层 outcome（14.5），strict success = I[final ∧ ¬collateral] |
| `span_offload/runtime.py` | AgentRuntime 抽象接口 = 真实 AppWorld 集成的**唯一对接点** |
| `span_offload/analysis.py` | O_m^strict(φ) profile + Wilson 95% CI（14.6）；task 条件化 + Δ_task（14.7.1）；within-agent / matched T^matched / architecture / joint（14.7.2-14.7.4）；descriptor completeness test（14.7.5） |
| `span_offload/allowlist.py` | 静态 allowlist（read_only + structured_output + strong_pre_commit_verifier + LCB≥θ）与 5 配置对比 + oracle 上界（14.8） |
| `span_offload/harness/` | mock AppWorld（日历/邮件/文档 + 评估器 + 冻结任务清单）、两个代表性 Agent、脚本化离线 LLM、MockAppWorldRuntime（作为 AgentRuntime 实现的参考） |
| `tests/run_offline_test.py` | 33 项端到端断言（无 pytest 依赖），`python tests/run_offline_test.py` |

## 接入真实 AppWorld（traces 是外部输入）

主分析路径的输入 = **successful native traces（Stage 9 冻结代表性 Agent 产出）** + 冻结任务清单 + SLM API。`demo` 内部生成 traces 只是冒烟便利。

### 已有适配器：AppWorld OTEL 格式（smolagents 版）

你的真实 trace 格式（如 `0d8a4ee_1.json`：SERVER/CLIENT/INTERNAL spans，`gen_ai.*` attributes，`MessageRole.X` 枚举角色）已由 `span_offload/appworld_adapter.py` 适配：

```bash
python -m span_offload.cli extract --traces data/appworld --manifest data/appworld/task_manifest.json \
       --out data/appworld/spans.jsonl --trace-format appworld
```

映射规则（source event_type + 消息内容，source 标签保留在 semantic_constraints 里）：
- 消息含 "List all APIs that may be needed" → `tool_selection`（API prediction 阶段）
- `reasoning` / "please provide an answer" → `final_synthesis`（freeform）
- 输出为 `{"answer":..., "status":...}` → `final_synthesis`（json）
- 输出为 tool-call list：末条消息以 "Error:" 开头 → `tool_argument_generation` + repair_controller + `after_failure=True`；否则 local_executor
- INTERNAL tool.execution 挂到前一个 LLM span；`agent.outcome=="success"` → baseline 成功；工具名查 `APWORLD_TOOL_EFFECTS` 决定 state_effect（未知工具保守标记 mutation）

对当前 `0d8a4ee_1.json` 的提取结果：28 个 LLM span 全部提取（tool_selection 1 / tool_argument_generation 25 / final_synthesis 2）。

### 两层可测量性（诚实分层）

**Local 层（今天就能测）**：`gen_ai.input.messages` 已记录真实请求 → `replay --mode trace-local` 用同一 prompt 调 SLM，走原解析器 + 本地契约检查（工具名合法、arguments 是 JSON 对象、answer/status 结构），产出每类 span 的 p_parse / p_schema / p_contract。记录打 `outcome_source="trace_local"`，**绝不进入 O_strict 分母**：

```bash
python -m span_offload.cli replay   --spans data/appworld/spans.jsonl --config config.json \
       --out data/appworld/replays.jsonl --trace-format appworld --mode trace-local
python -m span_offload.cli analyze  --replays data/appworld/replays.jsonl --out data/appworld/report \
       --source trace_local
```

**Downstream/Final 层（需 exporter 补齐后自动解锁）**：当前 trace 缺两个东西，census 以 ineligibility 原因诚实报告：
1. `environment_state_not_checkpointed` —— 没有每步的 AppWorld 环境快照；
2. `agent_state_not_checkpointed: conversation tail window only` —— 只记录了会话尾部窗口（~3 条），框架完整状态（完整消息列表/memory）无法恢复。

exporter 补上这两个字段后，同一批 span 自动变为 eligible，`replay`（live 模式）+ `analyze` 即产出完整 O_strict profiles。

### 其他 trace 格式

- 方案 A：写转换器直接产出 canonical JSON（字段清单见 `span_offload/trace_adapter.py` 顶部注释），用 `validate_canonical_trace()` 自检；
- 方案 B：保留原生格式，子类化 `TraceAdapter.load()` 在读取时映射（`appworld_adapter.py` 即示例）。

关键点：`artifacts` 存状态快照/消息/工具 schema 等大对象，span 只存 `*_ref` 引用（14.3）；每个 span 必须带 `application_state_hash`（重放时校验状态一致性）。

### AgentRuntime

真实重放还需实现 `AgentRuntime` 的 9 个方法（restore / resolve_request / parse / schema_check / verify_local / inject / run_to_node / run_to_completion / rollback）——`span_offload/harness/mock_runtime.py` 是完整参考实现。`replay` 阶段只需要 SLM（后续 span 一律用 trace 里记录的 baseline 继续）。

## 关键约定

- **descriptor 不泄漏 label**（14.2.5）：候选模型身份与 replay outcome 一律不进 φ_s；`assert_no_label_leakage` 硬守卫。
- **每个 cell 至少报告**：sample_count、point_estimate、95% CI、local / downstream / final-task success、collateral-damage rate；样本不足只标 `insufficient_support`，不给结论（14.6）。
- **所有 SLM candidate 从同一 span-before 快照启动**：每次 replay 重新 restore 并校验 `application_state_hash`，无跨 replay 状态污染（14.3）。
- **baseline-failed runs** 保留在 census 中做 failure analysis，不进卸载性分母（14.1）。
- **cost/latency 口径**：`cost_saving = baseline_span_cost − (slm + verifier + fallback cost)`；`verifier_cost=0`（全部本地确定性校验，无 verifier LLM 调用，已文档化）。
- offline mock 模式下 SLM 与 baseline 脚本相同，strict success 恒为 1 —— 它验证的是测量机制本身；真实 SLM 质量信号必须用真实 API 测量。

---

# 当前实测流程（真实 AppWorld + 真实 AutoGen 续跑）

以下为实际研究使用的运行方法（Windows / Python 3.12）。

## 1. 环境准备

```bash
# ① 虚拟环境
python -m venv .venv_appworld
.venv_appworld/Scripts/pip install -q -r requirements.txt

# ② AppWorld(第三方仓库;先获取 apps.bundle,仓库 git-lfs 额度不稳定时
#    可用 ghcr 镜像提取,见 fetch_appworld_bundles.py)
git clone --depth 1 https://github.com/stonybrooknlp/appworld.git benchmarks/appworld
cd benchmarks/appworld && git lfs pull   # 若 LFS 失败:python ../../scripts/utils/fetch_appworld_bundles.py
cd ..
.venv_appworld/Scripts/pip install -q -e benchmarks/appworld
cd benchmarks/appworld && ..\..\.venv_appworld\Scripts\appworld.exe install --repo
..\..\.venv_appworld\Scripts\appworld.exe download data   # 任务数据(~35MB)
cd ..

# ③ 真实 AutoGen agent 依赖(trace 的生成者,续跑后端)
.venv_appworld/Scripts/pip install -q "autogen-agentchat==0.7.5" "autogen-ext==0.7.5" \
    litellm joblib PyYAML
.venv_appworld/Scripts/pip install -q -e benchmarks/appworld/experiments --no-deps

# ④ 验证
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 .venv_appworld/Scripts/python.exe -c \
  "from appworld_agents.code.autogen.agent import get_autogen_tools; print('OK')"
```

> Windows 控制台为 GBK 编码,所有命令务必带 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`,
> 否则 rich/emoji 输出会崩溃。

## 2. 配置模型 API

```bash
cp config.example.json config.json   # 填入 slm / baseline_llm 的 base_url、api_key、model
```

- `slm` = 候选小模型(实测:MiniMax-M2.5,`https://api.minimaxi.com/v1`)
- `baseline_llm` = 大模型(实测:DeepSeek V4 Flash,`https://api.deepseek.com`,
  与 trace 生成模型一致;思考模式保持开启)
- **config.json 含密钥,勿上传仓库**(已在 .gitignore 排除)

## 3. 运行方法

### 3.1 提取(新格式 checkpoint trace)

trace 由 AutoGen agent 采集,每个 llm.call span 带 `check_point` 对象:

```json
"check_point": {
  "task_id": "...",
  "application_state_hash": "<sha256>",          // 真实值(占位空串 hash 会被如实标记)
  "env_changes": {"<app>": [[sql, params, committed], ...]},   // 增量 DB 变更
  "env_changes_baseline": {...},                  // span 0 的初始状态
  "conversation": {"has_full_conversation": true,
                   "messages": [...], "output_text": "..."}
}
```

`gen_ai.request.tools` 记录请求的原生工具 schema(续跑原样复现 function calling 协议)。

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m span_offload.cli extract \
    --traces check_point_traces/autogen_traces \
    --manifest data/appworld/task_manifest.json \
    --out data/appworld/spans.jsonl --trace-format appworld
```

### 3.2 单 span live 反事实重放(核心测量)

下游续跑默认用**真实 AutoGen agent 源码**(`appworld_agents.code.autogen.agent`,
即生成 trace 的同一份代码);autogen 缺失时回退 span_offload 简化循环。

```bash
# A. M2.5 真实卸载成功率(不兜底;本地层不过 → strict=0)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \
    scripts/runners/run_live_replay.py --trace 50e1ac9_1 --fallback none \
    --out data/appworld/replays_none.jsonl

# B. identity ceiling(兜底注入 baseline 记录输出 = 桥保真度上限/噪声地板)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \
    scripts/runners/run_live_replay.py --trace 50e1ac9_1 --fallback recorded_baseline \
    --out data/appworld/replays_ceiling.jsonl
```

每次重放:`restore`(重建 span 前环境:工具重放)→ SLM 同 prompt + 同原生 tools 协议
→ 本地验证 → 注入 → 真实 AutoGen agent + baseline 大模型续跑 → AppWorld 官方评估器。
每个 span 一条 ReplayRecord(五层 outcome)。

### 3.3 多 span 策略验证(14.8 附属,四配置对比)

```bash
# skip:本地层失败不兜底,直接进入下一个 span 的卸载
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \
    -m span_offload.cli validate-appworld --config config.json \
    --tasks 50e1ac9_1 --out data_appworld --on-local-failure skip

# fallback:本地层失败用大模型重算该 span 后继续下游
... --on-local-failure fallback
```

## 4. 实测结果速览(50e1ac9_1,13 span,M2.5 / DeepSeek V4 Flash)

| 测量 | 结果 | 含义 |
|---|---|---|
| identity ceiling(recorded_baseline) | 10/13(修复后预期 12/13) | 桥 + baseline 的噪声地板 |
| O_strict(none,SLM 带原生 tools) | 2/13(本地层 5/13) | M2.5 自身卸载成功率 |
| 关键结论 | 文本协议下 M2.5 本地层 ≈0;给原生 tools 协议后 5/13 本地层、2/13 严格成功 | SLM 必须拿到与原 agent 相同的 function calling 协议 |

## 5. 已知注意事项

- exporter 占位 hash(空串 sha256)会被如实记为 `snapshot hash mismatch`,待 exporter 算真值
- `env_changes` 为空时 restore 走工具重放近似(语义等价,细节可能偏差)
- span 1(首个工具回合)窗口缺任务指令时,适配器从 span 0 自动补入并标记 `instruction_reconstructed`
- 本机曾观测 `time.perf_counter()` 间歇跳变(epoch 级数值),新循环代码统一用墙钟
- 两个 API 密钥(MiniMax / DeepSeek)在 config.json,严禁入库

---

# 六 agent span 卸载(Stage 10 扩展)

`trace_record` 的 `OrderedTraceCollector` 为六个 agent 采集了统一格式的
checkpoint trace(`check_point_traces/` 下六个批次目录)。span 卸载流水线
对六个 agent 通用:extract → 单 span live 反事实重放 → 分析,`--traces`
换成对应批次目录即可:

```bash
for batch in appworld_simplified_batch1 appworld_autogen_batch1 \
             appworld_legacy_batch1 appworld_openai_agents_batch1 \
             appworld_smolagents_batch1 appworld_tau2_batch1; do
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \
    scripts/runners/run_live_replay.py --traces check_point_traces/$batch \
    --fallback none --out data/appworld/replays_$batch.jsonl
done
```

离线自检(不调 API):`python tests/run_agent_traces_test.py`(38 项)+
`python tests/run_offline_test.py`(33 项)。

任务清单:原 pilot manifest 只有 0d8a4ee_1 一个条目;六个批次共 65 个任务
的 TaskCondition 由 `scripts/manifests/build_task_manifest.py` 从 AppWorld
ground_truth(required_apps / required_apis / metadata)派生写回
`data/appworld/task_manifest.json`(pilot 手工条目原样保留,旧文件备份为
`data/appworld/task_manifest.pilot.backup.json`)。

远程服务器(singularity + sbatch):全部 shell 脚本位于 **`slurm/`** 目录
(`span_offload_server.sh` / `span_offload_slurm.sh` / `multi_replay_slurm.sh` /
`span_offload_env.sh` / 三个 `diag_*.sh` / `spotcheck_slurm.sh`)。提交命令
一律用绝对路径 `sbatch /share/home/tanylang/trace_span_offload/slurm/<脚本>`:

## 按 agent 的会话/协议适配(harness 通用设计)

所有 per-agent 差异收敛在两处,replay 引擎与 AppWorld 桥完全共享:

**1. 适配层(span_offload/appworld_adapter.py)** —— 会话拼接
`_splice_agent_conversation` 按 trace 自报的 agent 名选择拼接规则:

| agent | 会话记录形态 | 拼接结果 |
|---|---|---|
| Simplified / AutoGen | 完整会话(assistant 占位 / `[FunctionCall(...)]` 文本) | 既有 `_splice_tool_calls` |
| Legacy | 完整会话,assistant 回合无调用结构("None" 占位),含教程演示回合 | 最后一个任务指令之后的回合由 span 输出回填原生 tool_calls,tool 消息挂 tool_call_id |
| tau2 | 会话即任务回合全序 | 同上(无教程门控) |
| smolagents | `MessageRole.TOOL_CALL`/`TOOL_RESPONSE` 文本形态 | **保持文本工具协议**("Calling tools:"/"Observation:"),与 smolagents 1.19 的真实请求形态一致(tools + tool_choice="required") |
| OpenAI-SDK | **会话未记录**(流式 hook 拿不到请求消息) | adapter 用 span 输出 + 工具执行记录重建 transcript(`openai_prompt_reconstructed` 标记);live 重放时 runtime 用任务上下文重渲染真实 system prompt(与采集脚本同一模板) |

**2. 续跑层(span_offload/harness/appworld_runtime.py)** —— 后端按
trace agent 身份解析(`_backend_for`)。**硬性原则:续跑必须由真实 agent
源码驱动;框架依赖缺失 / 身份未知时该 span 如实报错
(backend_unavailable),绝不降级到近似循环。**

| 后端 | 机制 | 依赖 |
|---|---|---|
| simplified / autogen | 真实 agent 源码续跑(既有) | appworld-agents / autogen |
| legacy | 真实 legacy `OpenAILanguageModel` + `generate_second_step_text` 循环体逐行复刻(agent 的 solve 是单体不可断点续跑) | appworld-agents |
| smolagents | 真实 `ToolCallingAgent` 协议:OpenAIServerModel + 文本历史 + TaskCompleter 收尾 | `smolagents` PyPI |
| tau2 | 真实 `LLMAgent`(`get_init_state(message_history=…)` 原生支持中途恢复) | `tau2-bench` |
| openai | 真实 SDK Agent(Runner.run_streamed + function_tool 直连 in-process world;MCP 传输由直连工具等价替代并如实记录) | `openai-agents` SDK |

(generic 通用循环仅保留为历史代码,任何路径都不会自动使用它。)

协议旗标:`legacy`/`smolagents` 的原生协议为 `tool_choice="required"`
(采集脚本同款);`smolagents` 另设 `text_protocol`(历史保持文本形态)。
这些旗标同时作用于 SLM 请求(resolve_request)与续跑请求。

**已知限制(诚实记录)**:
- OpenAI-SDK trace 的 span 0(API 预测)无真实会话记录,保留占位请求
  并被提取器如实标记 ineligible(`lm_request_incomplete`);span ≥1 由
  runtime 重渲染完整 prompt
- tau2 的 span 0 同理(collector 只记录了 "API prediction" 占位消息)
- tau2 trace 无 `pass` 字段,基线成功判定回退到 `agent.outcome`(自我声称,
  比验证器 pass 更宽松)
- 本机 shell 的 `VIRTUAL_ENV` 指向无关 conda 环境(实测 AoI 项目),
  appworld 的 import 链会把它混入 sys.path;harness 在桥建立前后自动
  剔除(`_scrub_foreign_syspath`),后端探测也只信任本解释器环境内的包
- **世界冻结时钟 × litellm import 的交互**(服务器 OOM 根因,已修):
  AppWorld 用 freezegun 冻结 `time.monotonic/perf_counter/time`(且会
  叠加 2 个冻结工厂);tau2/smolagents 的 import 链在 import 期拉取
  litellm 价目表,服务器网络下该请求必然失败,冻结时钟使其重试永不
  超时 → 无限循环 + 每轮分配 → cgroup OOM(实测 ~2.4GB/s 增长)。
  runtime 在**任何世界创建之前**预加载框架 import 链(`_prewarm_*`,
  restore 开头,真实时钟下 litellm 一次失败即回退本地表);后续 import
  全部走模块缓存、零网络。世界内失败的 LLM 请求仍有同类冻结时钟
  隐患(成功请求不受影响,simplified 批次实测可正常跑通)——属既有
  潜在风险,非本次引入

---

# 当前研究主线(2026-09):53-span 干预响应实验

## 一、装置原理:单 span 反事实重放 + 分布参照评估

**核心问题**:当一个 Agent 中的 LLM span 被 SLM 替换,局部干预怎样改变下游执行?
影响什么时候被吸收 / 修正 / 持续 / 放大?

**实验范式**(不再以原始 trace 为唯一 ground truth):

```
span j ──► restore(工具重放重建 span 前环境)
        ├─ SLM 组:注入 SLM 输出      → 下游 baseline LLM 现场续跑 ×5 轮
        └─ identity 组:注入基线记录输出 → 下游 baseline LLM 现场续跑 ×5 轮
                    └──── 唯一差别 = 注入内容 ────┘
```

- **identity 组 = 基线行为分布的经验估计**("如果 span j 输出和原来一样,
  重跑一遍下游会怎样")——它把装置噪声 + LLM 重跑随机性全部量化;
- **干预效应 = SLM 组分布超出 identity 组自然波动的部分**:
  配对 Δ(span 级均值差 + bootstrap CI)、标准化效应量 d = (μ_slm−μ_id)/σ_id、
  A-F 响应分类、方差分解;
- 每个 span 是独立实验:一次只干预一个 span,结果变化只归因于它。

**五层测量**(每条重放记录):
本地层(parse/schema/verifier + 四维等价)→ 下游层(逐轮事件流:
D 决策/R 空转/O 观察/A 写/F 失败/Rp 修复)→ 最终层(strict =
成功 ∧ 无附带损伤)→ 成本层 → 统计推断层(跨 span 配对检验)。

**续跑后端**(按 trace agent 身份自动解析,**硬性原则:续跑必须由真实
agent 源码驱动;依赖缺失时该 span 如实报错,绝不降级**):
simplified / autogen / legacy / smolagents / tau2 / openai-SDK。
两条装置级防护:
- 防护 A:续跑完成判定统一读环境事实 `world.task_completed()`,不信任后端返回值;
- 防护 B:autogen 消息进模型上下文前经 `_autogen_model_inputs` 白名单闸口
  (类型对照模型客户端注册表),UI 摘要类消息永不进上下文。
原则:**不修改任何 agent 的控制逻辑**——harness 只做环境恢复、注入、
续跑驱动与测量;输入流必须忠实于原 trace,不新增任何内容。

## 二、代码结构

| 路径 | 内容 |
|---|---|
| `span_offload/` | 装置主包:schemas(三层条件化标签)、span_extraction(eligibility 六门控)、replay(单 span 反事实重放引擎)、harness/appworld_runtime.py(六后端续跑 + 两防护)、harness/appworld_env.py(环境桥)、appworld_adapter.py(trace 解析/会话拼接) |
| `slurm/` | 服务器运行脚本:multi_replay_slurm.sh(提取+重放主入口)、span_offload_server.sh/slurm.sh、span_offload_env.sh(框架自动安装)、diag_*/spotcheck |
| `scripts/` | 分析脚本(见下)+ run_offload_profile.py(重放驱动) |
| `results/` | 53-span 干净数据集 + 全部分析产物(见下) |
| `check_point_traces/` | 六个 agent 批次的原始 checkpoint trace |

## 三、运行命令

### 3.1 提取(spans 清单 = 实验对象清单)

```bash
# 本地(Windows,务必带 UTF8 环境变量)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \
  -m span_offload.cli extract \
  --traces check_point_traces/appworld_simplified_batch1 \
  --manifest data/appworld/task_manifest.json \
  --out out_span_offload/spans.jsonl --trace-format appworld

# 服务器:由 slurm/multi_replay_slurm.sh 自动完成(SPANS_FILE 不存在时提取),
# 批次与输出文件名用环境变量指定:
TRACE_DIR=check_point_traces/appworld_autogen_batch1 \
SPANS_FILE=out_span_offload/spans_autogen.jsonl \
  sbatch /share/home/tanylang/trace_span_offload/slurm/multi_replay_slurm.sh
```

⚠️ spans.jsonl 决定"在哪些 span 上做实验"——指向哪个批次,实验就是哪个批次的。

### 3.2 重放(核心测量,SLM×5 + identity×5)

```bash
# 全量批次(simplified,53 span 数据集即此命令的产物之一)
TRACE_IDS="21abae1_1 23cf851_1 23cf851_2 23cf851_3 383cbac_1 383cbac_3 4ec8de5_1" \
ROUNDS=5 IDENTITY_ROUNDS=5 MODE=both FALLBACK=recorded_baseline \
SPAN_OFFLOAD_CONT_BACKEND=simplified MAX_PARALLEL=2 \
  sbatch /share/home/tanylang/trace_span_offload/slurm/multi_replay_slurm.sh

# identity 单独收集(MODE=identity,不调 SLM,更快)
TRACE_IDS="21abae1_1 23cf851_1" MODE=identity ROUNDS=5 IDENTITY_ROUNDS=5 \
FALLBACK=recorded_baseline \
  sbatch /share/home/tanylang/trace_span_offload/slurm/multi_replay_slurm.sh

# 冒烟(新批次先跑 1-3 个 span)
TRACE_IDS="23cf851_2" N_SPANS=3 ROUNDS=5 IDENTITY_ROUNDS=5 MODE=both \
FALLBACK=recorded_baseline SPAN_OFFLOAD_CONT_BACKEND=autogen MAX_PARALLEL=1 \
  sbatch /share/home/tanylang/trace_span_offload/slurm/multi_replay_slurm.sh
```

### 3.3 分析(全部在本地跑,读 results/)

```bash
PY=PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe

# ① 主响应分析:A-F 分类、配对 Δ、d、方差分解、可卸载候选(S1-S10)
$PY scripts/analysis/response_analysis.py                 # → results/response_analysis_clean53.json

# ② 22 变量下游分析(轮级配对 bootstrap,显式传文件列表)
$PY scripts/analysis/analyze_offloadability.py \
  --inputs results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl \
            results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl \
            results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl \
            results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl \
            results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl \
            results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl \
            results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl \
            results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl \
  --traces-dir check_point_traces/appworld_simplified_batch1 \
  --out results/offloadability_variables_clean53.json --all

# ③ 动力学:d_t 曲线、分歧出现/传播/恢复距离(τ_on/L_prop/L_rec)
$PY scripts/analysis/dynamics_analysis.py                 # → results/dynamics_53.json (+png 用系统 python)

# ④ 控制流专项:分支宽度/探索熵/回溯率/异常出口等
$PY scripts/analysis/controlflow_analysis.py              # → results/controlflow_53.json

# ⑤ 归一化结构编辑距离 + 首次分歧点(参照 arXiv:2604.27586)
$PY scripts/analysis/trace_edit_distance.py               # → results/trace_edit_distance_53.json

# ⑥ 按最终任务成败分层的指标签名
$PY scripts/analysis/success_stratification.py            # → results/success_stratification.json
```

## 四、数据集与产物(results/)

命名规范:`offload_profile_<arm?>_<agent>_<baseline>_<slm>_<task>.jsonl`(与 slurm
输出标签一致;`id_` = 纯 identity 文件;模型名中的 `/ . 空格` 全部替换为 `_`)。

| 文件 | 内容 |
|---|---|
| `offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl` | batch-1 SLM 组(20 span × 5) |
| `offload_profile_id_simplified_..._{21abae1_1,23cf851_1}.jsonl` | 重跑 identity(各 6 span × 5) |
| `offload_profile_simplified_..._{23cf851_3,383cbac_1,383cbac_3,4ec8de5_1}.jsonl` | 新任务 SLM+identity |
| `offload_profile_id_simplified_..._23cf851_2.jsonl` | 23cf851_2 干净 identity(8 span × 5) |
| `offload_profile_id_contaminated_simplified_..._batch1.jsonl` | 修复前污染 identity(仅用于 S9 敏感性对照) |
| `offload_profile_tau2_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl` | tau2 后端批次 |
| `offload_profile_autogen_qwen27b_..._{23cf851_2,23cf851_3}.jsonl` | autogen 后端 + qwen27b baseline 批次 |
| `offload_profile_autogen_deepseek-v4-flash_..._{23cf851_2,50e1ac9_1}.jsonl` | autogen 后端 + deepseek baseline |
| `offload_profile_simplified_..._{6bdbc26_1,fac291d_1,fac291d_2}.jsonl` | 新任务批次 |
| `response_analysis_clean53.json` / `offloadability_variables_clean53.json` / `dynamics_53.json` / `controlflow_53.json` / `trace_edit_distance_53.json` / `success_stratification.json` / `span_id_mapping.json` | 各分析脚本产物 |
| `_archive/` | 被取代/废批次的历史文件 |

## 五、关键结论速览(53 span × 10 轮,全干净基线)

- **5/5 核心响应变量显著**(配对 Δ):ParamMatchRate −0.185、KeyChainCoverage
  −0.072、ControlFlowDeviation +0.094、DirectActionRate +0.052、WriteRatio +0.034;
  strict/tools/llm/hall 不显著——**干预改变执行的"质地",不改变"规模与结局"**;
- **H1 成立**:σ_slm/σ_id 中位数 0.52,干预不引入额外方差;干预效应与自然噪声
  同量级(σ 比 0.93);
- **A-F 分类**:B 良性分歧 29(55%)、F 放大失败 9、D 代价适应 7、E 基线不稳 6、
  C 纠正性 2、A 透明 0(验证器噪声地板);可卸载候选 27/53(51%);
- **动力学**:分歧高度集中在前 3 步(44% 第一步即分叉);B 类 45% 在 3-6 步内恢复,
  D 类 7/7 永不恢复;归一化编辑距离 Δ=+0.128(identity 地板 0.446);
- **成败签名**:失败轮 = 关键链覆盖低(0.59 vs 0.89)+ 幻觉螺旋深 + 异常出口;
  失败不是"探索太多"而是"没走对";
- **污染基线敏感性**:旧污染基线把 Δtools 夸大 5.4 倍且翻转 Δllm 符号——
  identity 对照装置必要性的最硬证据。
