# 6 个 Agent 的实现源码（参考副本）+ AutoGen 框架源码

本目录是研究中 6 个 agent 所用实现源码的**参考副本**（按 agent 组织）。
运行环境仍使用原位置的代码（editable 安装 / 子模块），移动会破坏导入。
另有 **AutoGen 框架源码**（`autogen/`）——与参考副本不同，它是运行时
真实生效的源码树（见下文）。

## 目录 ↔ 运行源码映射

| 本目录 | 运行源码位置 | 说明 |
|---|---|---|
| `agent_legacy/` | `benchmarks/appworld/experiments/code/legacy/` | AppWorld Legacy FunctionCallingAgent（plain 子目录为所用架构；recoma 为未使用的另一架构） |
| `agent_simplified/` | `benchmarks/appworld/experiments/code/simplified/` | AppWorld SimplifiedFunctionCallingAgent |
| `agent_smolagents/` | `benchmarks/appworld/experiments/code/smolagents/` | smolagents ToolCallingAgent 官方流程 |
| `agent_openai_agents/` | `benchmarks/appworld/experiments/code/openai_agents/` | OpenAI Agents SDK + MCP 流程 |
| `agent_autogen_appworld/` | `benchmarks/appworld/experiments/code/autogen/` | AutoGen 集成的 AppWorld 侧代码（工具包装/API 预测） |
| `agent_tau2/` | `benchmarks/tau2-bench/src/tau2/agent/` + `utils/llm_utils.py` | tau2-bench LLMAgent（原生，仅生成消息；驱动循环在 `benchmarks/tau2-bench/benchmarks/run_benchmark_tau2_llmagent.py`） |
| `appworld_common/` | `benchmarks/appworld/experiments/code/common/` | 共享组件（APIPredictor、usage tracker 等） |

## AutoGen 框架源码（运行时激活，非副本）

- **`autogen/`**：microsoft/autogen 源码检出（`autogen_agentchat` 0.7.5，
  AssistantAgent 本体），含本地补丁（如 `_assistant_agent.py` 的 run loop 行为）。
  由 `span_offload/vendor_autogen.py` 在启动时挂到 `sys.path` 前部，替代已安装
  的 wheel；仓库路径可用 `SPAN_OFFLOAD_AUTOGEN_REPO` 环境变量覆盖。

## 其他外部框架（未复制，PyPI 依赖）

- **smolagents**：PyPI 1.19.0（ToolCallingAgent 本体）
- **openai-agents SDK**：PyPI（`agents` 包，SDK Agent / Runner / MCP）

## 版本锚点

- appworld 仓库 commit（位于 `benchmarks/appworld`）：`a072b7a`
- tau2-bench 仓库 commit（运行时克隆到 `benchmarks/tau2-bench`）：`363133a`
- 本目录的副本未冻结版本；如需严格快照，请按上述 commit 校验原位置代码

## 采集与运行基础设施（不在本目录）

- 卸载流水线核心包：`span_offload/`（适配器、重放、分析）
- 实验脚本与调度：`scripts/`、`slurm/`
- 环境/任务：`benchmarks/appworld/src/`（世界模拟与 app API）
- 任务数据：`data/`（按 benchmark 分子目录）
