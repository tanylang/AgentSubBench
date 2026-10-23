"""
AppWorld adapter layer — unified agent integration and trace collection.

Benchmark scripts import the following modules directly as needed:
- adapter.hooks         Global llm_hook / tool_hook registry (instrumented at source, injected externally; no-op when no collector is registered)
- adapter.tools         Tool generation via generate_tools (oai/autogen/smolagents/tau2/openai_sdk)
                        and the unified execution entry point appworld_execute_tool (execute + canonical recording)
- adapter.builtin_hooks Sentinel injection for built-in agents (simplified/legacy), class-level
                        patching of AppWorld.execute, and AutoGen's _trace_tracer injection shim (AutoGenTracerShim)
- adapter.model_config  Model configuration (THINKING_MODE, etc.)

Single trace exit point: OrderedTraceCollector in benchmarks/trace_collector.py
(the llm.call / tool.execution span format matches all existing traces).
"""
