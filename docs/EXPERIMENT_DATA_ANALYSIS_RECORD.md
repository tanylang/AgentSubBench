# 实验数据结果分析记录(截至 2026-10-19)

> 本文件记录当前仓库内全部实验数据及其分析结果,重点是**论文(agentsubbench_arxiv.tex,ICLR 2027 投稿版)尚未收录**的部分。
> 生成方式:从 results/、out_span_offload/、check_point_traces/ 的实际产物中直接提取与复算。
> 模型约定:baseline = Qwen3.8-27B,candidate = Qwen/Qwen3.5-9B;主对比 T_span − B;strict success = final_success ∧ ¬collateral。

---

## 一、论文已收录的 6 个 cohort 与本地数据位置(已验证吻合)

| 论文 cohort | 点数(任务) | 重复/臂 | live 记录 | 本地数据位置 | 验证 |
|---|---|---|---|---|---|
| AppWorld Simplified screen 48 | 48 (17) | 5 | 720 | out_span_offload/simplified/screening/、legacy/(dedup 720 行) | mixed-B 1/48 ✓ |
| AppWorld selected follow-up 18 | 18 (12) | 15 | 810 | out_span_offload/simplified/confirm/ | mixed-B 6/18 ✓;Pearson −0.302、MAD 47.4pp、4 次符号反转 ✓ |
| AppWorld Tau2 44 | 44 (4) | 10 | 1,320 | out_span_offload/tau2/screening/ | mixed-B 11/44 ✓ |
| AppWorld Simplified expanded 84 | 84 (12) | 10 | 2,520 | out_span_offload/simplified/batch1/simplified/screening/ | mixed-B 23/84 ✓ |
| LoCoMo LangGraph conv-26 | 257 (95) | 10 | 7,710 | out_span_offload/langgraph/batch1/257span/、conv-26/offload/ | mixed-B 43/257 ✓ |
| SWE-bench mini-SWE-agent 134 | 134 (4 Django) | 10 | 4,020 | **本机不存在**(仅集群 /share/home/tanylang/) | — |

论文数字复验:conv-26 strict T 0.7949 vs B 0.9012 = **−10.62pp** CI[−14.8,−6.8]、net_saving **+2,185.5** token——与论文 −10.6pp/+2.19k 完全吻合(results/locomo/offloadability_langgraph_exante.json)。

---

## 二、论文未收录的实验与分析结果

### 2.1 LoCoMo 10 对话全量 cohort(论文只用了 1/10)

**数据**:out_span_offload/langgraph/conv-{26,30,41,42,43,44,47,48,49,50}/ + 合并文件
- out_span_offload/merged/_tmp_loco10_recs_merged.jsonl(78,690 行 live 记录;800MB)
- out_span_offload/merged/_tmp_loco10_spans_merged.jsonl(5,542 行 span 清单,1,986 个 task)
- 规模:**2,623 span × 3 臂 × 10 轮 = 78,690 条**(论文之外 70,980 条)
- 分析:results/main_comparisons/main_comparison_langgraph_10conv.json

**质量与 token(10conv 总体)**:
- strict:T 0.8326 vs B 0.9309,**Δ = −9.8pp** CI[−11.0,−8.7],显著
- NetSaving_tokens:**+2,859** CI[2,108,3,620];TotalTokens 27,311 vs 30,170
- tools/span −0.257、llm 步 −0.100

**时间响应(从原始记录 replay_meta.elapsed_ms 直接复算,目标调用开始→终止)**:

| 对话 | B 均值 | T_span 均值 | 时间节省 |
|---|---|---|---|
| conv-26 | 10.28s | 10.06s | +2.1% |
| conv-30 | 7.63s | 6.62s | +13.2% |
| conv-41 | 12.61s | 11.50s | +8.8% |
| conv-42 | 14.87s | 14.65s | +1.5% |
| conv-43 | 13.04s | 11.71s | +10.2% |
| conv-44 | 11.16s | 10.28s | +7.9% |
| conv-47 | 12.08s | 10.93s | +9.5% |
| conv-48 | 10.97s | 10.46s | +4.7% |
| conv-49 | 10.65s | 10.00s | +6.1% |
| conv-50 | 11.95s | 10.53s | +11.9% |
| **10conv 整体** | **11.79s** | **10.94s** | **+7.2%(Δ=0.84s)** |

注意:conv-26 旧批次(batch1/257span)为 B=11.67s / T=10.70s → **8.3%(0.96s,论文来源,精确吻合)**;10/17 重跑版为 +2.1%。墙钟时间跨批次漂移明显(B 臂均值差 1.4s),报告时间指标需注明测量批次。

**时间指标复验(2026-10-21)**:上表已由 scripts/analysis/recompute_langgraph_completion_time.py 从原始合并记录逐行复算,10 个对话全部与表中数字吻合(输出 results/locomo/completion_time_langgraph_10conv.json)。配对(span 级)Δ:10conv 均值 **−0.84s**,CI95 [−1.55, −0.14](显著);旧批次 257span 复算 B=11.67s / T=10.70s / Δ=−0.96s(8.3%),与论文 v7 表 LoCoMo 行完全一致。

**逐对话 strict 对比(stratified_offloadability_langgraph_10conv.json)**:
conv-26 −11.4pp、conv-30 −9.9、conv-41 −9.8、conv-42 −7.8、conv-43 −10.8、conv-44 −11.7、conv-47 −6.7、conv-48 −10.9、conv-49 −7.4、conv-50 −12.2。

### 2.2 LoCoMo conv-30/41/42/43 四对话 cohort(中间批次)

**数据**:out_span_offload/merged/_tmp_loco_recs_merged.jsonl(31,800 行)、_tmp_loco_spans_merged.jsonl(2,188 行)
- 规模:**1,060 span × 30 = 31,800 条**
- 分析:results/main_comparisons/main_comparison_langgraph_conv30_41_42_43.json

**关键数字**:strict T 0.8440 vs B 0.9392,**Δ = −9.5pp** CI[−11.5,−7.7];NetSaving **+3,522** CI[2,348,4,818];tools −0.269、llm −0.121。l67 轨迹距离:d_delta 0.148 CI[0.13,0.16]、identical_ratio 0.710。

**时间(2026-10-21 补算,此前缺失)**:整体 B 12.65s vs T_span 11.76s,**Δ = −0.90s(−7.1%)**;配对 span 级 CI95 [−2.59, +0.80]——**不显著**(10conv 的 −0.84s CI[−1.55,−0.14] 显著)。输出:results/locomo/completion_time_langgraph_4conv.json。

### 2.3 LoCoMo 卸载前(ex-ante)特征判别表

来源:results/locomo/locomo_exante_features_10conv.json(2,623 span)与 locomo_exante_features.json(1,060 span)。
结局:offload_ok = DiD≥0(T 不劣于 identity);d_success = strict_T − strict_B。
offload_ok 比例:10conv 79.1%、4conv 80.5%。

**10conv(2623 span)**:

| 特征(卸载前可算) | mean_ok | mean_fail | MW p | ρ(d_success) | 显著 |
|---|---|---|---|---|---|
| up_tool_calls | 0.91 | 1.75 | 1e-12 | −0.254 [−0.30,−0.20] | Y |
| up_llm_steps | 0.76 | 1.30 | 1e-12 | −0.228 [−0.27,−0.18] | Y |
| n_tool_calls_self | 0.64 | 1.05 | 1e-12 | −0.194 [−0.24,−0.15] | Y |
| inside_loop | 0.56 | 0.72 | 5.9e-09 | −0.096 [−0.13,−0.06] | Y |
| max_remaining_steps | 21.3 | 27.0 | 1e-12 | −0.090 [−0.13,−0.05] | Y |
| category(官方1-5) | 3.16 | 2.86 | 2.4e-07 | +0.057 [+0.02,+0.10] | Y |
| position(early→late) | 0.79 | 0.64 | 2e-05 | +0.060 [+0.02,+0.10] | Y |
| baseline_tokens | 14,138 | 14,582 | 0.33 | −0.034 | 不显著 |
| up_write_spans / after_failure | 0 | 0 | — | 0(零方差) | — |

**4conv(1060 span)**:方向一致且显著:up_tool_calls −0.232、n_tool_calls_self −0.241、up_llm_steps −0.209、inside_loop −0.104、max_remaining_steps −0.078;**category/position 不再显著**;baseline_tokens 仍不显著(p=0.43)。
注意:10conv 中 up_llm_steps/up_tool_calls/up_search_spans/last_write_gap 的 ρ 完全相同(共线,同属"上游活动量"信号)。

**关键外推性发现**:conv-26(论文 cohort)上 baseline_tokens 判别显著且方向为"token 越多越易卸载"(mean_ok 14,863 vs mean_fail 13,043,p=0.045);10conv 大样本中该信号消失(15,010 vs 15,635,p=0.43,方向相反)——论文单对话结论不可外推。

### 2.4 LoCoMo 卸载后(ex-post)显著响应指标表

来源:results/main_comparisons/main_comparison_langgraph_10conv.json 与 conv30_41_42_43.json(T_span:B,**17/32 指标显著**,两 cohort 显著集一致):

| 指标 | 10conv Δ | 4conv Δ |
|---|---|---|
| strict | −0.098 [−0.11,−0.09] | −0.095 [−0.11,−0.08] |
| NetSaving_tokens | +2,859 [2,108,3,620] | +3,522 [2,348,4,818] |
| TotalTokens | −2,859 | −3,522 |
| tools/span | −0.257 | −0.269 |
| llm 步/span | −0.100 | −0.121 |
| ActionsPerDecision | −0.217 | −0.221 |
| DirectActionRate | −0.112 | −0.120 |
| ToolSeqJaccard | +0.112 | +0.120 |
| BranchWidth | −0.113 | −0.120 |
| BatchDecisionRate | −0.105 | −0.101 |
| ReasonOnlyRate | +0.055 | +0.060 |
| RevisitRate | −0.039 | −0.042 |
| SameToolPersistence | −0.065 | −0.070 |
| WriteRatio | −0.112 | −0.120 |
| NovelInfoRate | −0.106 | −0.112 |

不显著:KeyChainCoverage、ObserveChaining、FailRetryRate、RepairRate、AbnormalExit、DeadEndRate 等(失败/修复类无变化)。

### 2.5 LoCoMo 条件分层可卸载性表(卸载前条件 × 卸载后结局)

来源:results/locomo/offloadability_langgraph_exante_conv30_41_42_43.json(1060 span;conv-26 版在 offloadability_langgraph_exante.json,与论文数字吻合)。
总体:offload_ok_rate **80.5%**,d_success −9.5pp,net_saving +3,522,zero_detour 0.676,collateral Wilson UCB 0.0003。

| 分层(卸载前条件) | n | offload_ok_rate | lift | d_success |
|---|---|---|---|---|
| **control_phase=execution** | 202 | **0.455** [0.39,0.52] | **−0.349** | **−0.338** |
| control_phase=initial | 429 | 0.907 | +0.102 | −0.019 |
| control_phase=terminal | 429 | 0.867 | +0.062 | −0.057 |
| **position=middle** | 137 | **0.526** | **−0.279** | **−0.310** |
| position=early | 471 | 0.860 | +0.055 | −0.043 |
| position=late | 452 | 0.832 | +0.027 | −0.085 |
| **category=3** | 34 | **0.471** | **−0.334** | **−0.359** |
| category=1 | 133 | 0.699 | −0.106 | −0.094 |
| category=2 | 216 | 0.866 | +0.061 | −0.085 |
| category=4 | 677 | 0.823 | +0.018 | −0.086 |
| cls=tool_argument_generation_read_only | 631 | 0.762 | −0.042 | −0.121 |
| cls=final_synthesis_read_only | 429 | 0.867 | +0.062 | −0.057 |
| **cls2=tool_argument_generation_late** | 23 | **0.174** | **−0.631** | **−0.600** |
| cls2=tool_argument_generation_middle | 137 | 0.526 | −0.279 | −0.310 |
| cls2=tool_argument_generation_early | 471 | 0.860 | +0.055 | −0.043 |

**连续特征 × 结局相关(同文件 corr 段)**:
- baseline_tokens×net_saving ρ=**0.340** [0.28,0.40] 显著;×direct_saving ρ=**0.786**;×downstream_comp 不显著;×d_success 不显著
- max_remaining_steps×net_saving ρ=**0.225** 显著;×d_success ρ=**0.146** 显著(靠后更易卸载成功);×downstream_comp ρ=−0.060 显著
- feature_disc:n_allowed_tools 无判别力(p=1.0)

10conv 版分层表在 results/locomo/stratified_offloadability_langgraph_10conv.json:middle −0.302、cat3 −0.258、cat5(n=4)−0.400、execution −0.338、final_synthesis −0.063、tool_arg −0.122、early −0.048、late −0.084。

### 2.6 LoCoMo 任务分类(999 QA)

results/locomo/locomo_task_classification.json:conv-26/30/41/42/43 共 **999 个 QA 任务**(conv-42 260、43 242、26 199、41 193、30 105);全部 memory_class=memory_read;baseline 成功率 **42.9%**(429/570);category 1-5 计数 142/156/46/418/237;position early/mid/late ≈335/333/331;offload_by_class.memory_read(8,970 记录/臂):B 0.9381 / I 0.9649 / T 0.8436(T vs B −9.5pp)。

### 2.7 AppWorld 未收录批次

| 批次 | 位置 | 规模 | 状态 |
|---|---|---|---|
| Tau2 后端 batch1 扩展筛查 | out_span_offload/tau2/batch1/ | 19 任务文件,**6,801 条**(0d8a4ee、530b157、6171bbc、68ee2c9、6bdbc26、b119b1f、d4e9306 等) | 有重放数据,无汇总分析 |
| Tau2 batch2 全新实例集 | out_span_offload/tau2/batch2/ + check_point_traces/appworld_tau2_qwen3_8-27b_batch2 | 33 文件,**9,720 条**;65 条新 trace(024c982、07bb666、0d01c76…f3f60f0) | C0 审计未跑(占位 0 字节) |
| autogen 后端筛查 | out_span_offload/autogen/screening/ | 4 任务(23cf851_1/3、4ec8de5_3、50e1ac9_1),278 条 | + spans_autogen_qwen27b.jsonl(699 span 清单) |
| no400 排除敏感性(s48) | out_span_offload/_no400_s48/ + results/main_comparisons/main_comparison_no400_s48.json | 25 spans,733 条 | NetSaving T:B −70,660(不显著) |
| no400 排除敏感性(s84) | out_span_offload/_no400_s84/ + results/main_comparisons/main_comparison_no400_s84.json | 48 spans(84 剔 36),1,879 条 | NetSaving T:B −1,443(不显著) |
| no400 排除敏感性(t44) | out_span_offload/_no400_t44/ + results/main_comparisons/main_comparison_no400_t44.json | 44 spans,1,320 条 | NetSaving T:B −475 |
| with400 对照 | results/main_comparisons/main_comparison_batch1_with400.json | 32 spans,934 条 | NetSaving T:B +7,470(不显著) |
| 53-span 早期 screen 全套 | results/appworld_53span/controlflow_53.json、controlflow_table.json、trace_edit_distance_53.json、dynamics_53.json、response_analysis_clean53.json、offloadability_variables_clean53.json、success_stratification.json、cost_decomposition_b27b.json | 53 spans × 5 轮 = 265/臂 | 48-screen 的前身(7 任务:21abae1_1、23cf851_1/2/3、383cbac_1/3、4ec8de5_1);响应类 B29/C2/D7/E6/F9;经济口径直接节省 $0.00387/次 |
| 其他后端 trace 采集批次 | check_point_traces/appworld_{autogen,legacy,openai_agents,smolagents,tau2}_batch1 等 | 各 65 条 trace(27 实例 × 后缀);oai_qwen3_8-27b 仅 5 文件 | 论文只用 simplified 与 tau2 |
| 早期 pilot | docs/EXPERIMENT_REPORT.md | 3 任务 20 span,100+100 重放 | baseline=DeepSeek V4 Flash,已被 Qwen 系列取代 |

### 2.8 中间快照(被最终 cohort 取代,留存记录)

- results/main_comparisons/main_comparison_27b.json、27b_allpairs.json(32 span,675 条)
- results/main_comparisons/main_comparison_batch1_intermediate.json(24)、batch1_latest.json(42)、batch1_with400.json(32)
- results/main_comparisons/main_comparison_confirm_preview8.json(8)、confirm_15span.json(15)
- results/tau2/tau2_analysis.json(22 span)、tau2_taxonomy_analysis.json(22)
- results/cross_agent/cross_agent_comparison.json(48/11/22)、fine_taxonomy_cross_agent.json
- results/misc/c0_gate_all_agents.json(autogen 75/75、tau2 52/52、simplified 53/53、langgraph 577/577,合计 757/757)
- results/tau2/tau2_final_manifest.json(never_run 7、removed_incomplete 1:50e1ac9_2@12 墙钟耗尽)

### 2.9 探索性分析文件(论文无对应章节)

- results/cross_agent/did_success_signature.json — 签名级 DiD(simplified_did_sig={ObserveChaining},tau2 无,交集 0)
- results/misc/latency_calibration.json — 六组 OLS 时延标定(R² 0.79–0.99)
- results/misc/minimal_metric_set.json — 18 指标最小核心集 + 3 条冗余规则(基于 48 screen + confirm20)
- docs/input_side_offloadability_metrics.md — **96 合并 span 的输入侧(上游)预决策指标分析**:进展度三分层(高进展 T=0.83);L4b 硬拒绝信号(**19/19 全败**,跨两个引擎版本稳定:零工具调用 + assistant 教程>6 条);L4a 预测位点 T=0.74 对照
- docs/preregistration_batch3.md — Batch 3 冻结预注册(2026-09-15:ε=0.05、δ=0.05、18-span 确认集、block_seed=0)
- results/tau2/tau2_interim_findings.md — 中期发现(含已撤回的"空输出轮优势"假象;三框架 I 臂 51%/77-92%/13%)

### 2.10 SWE-bench:本机数据与论文不一致(重要)

- **论文 134 点 cohort(4 个 Django 实例 10097/10880/10914/13933)的原始数据、重放产物、分析结果全部不在本机**——全仓 grep 只在 tex 文本中出现这 4 个编号;生成管线在 slurm/swebench_replay_slurm.sh,输出目录 out_span_offload/minisweagent/ 未同步。
- 本机存在的是**另一批论文未收录的 SWE 数据**:
  - check_point_traces/swebench_minisweagent/:27 条 native trace(356,407 行),**2,691 span,24 个任务**:astropy-13033/7166、django-10554/11138/11141/11299/11400/11490/11880/11885/12325/12406/12754/13344/13410/14007/14787/15022/15499/15561/15973、matplotlib-22719、flask-5014、requests-2317、xarray-4356、pytest-10081、scikit-learn-11578
  - data/swebench/task_manifest.json:**58 任务,带 pilot/heldout 划分**(heldout:7 个 psf__requests-*(1142/1724/1766/1921/2931/5414/6028)+ pytest-dev__pytest-10051)——论文称"task-disjoint 前瞻验证尚未测量",但 heldout 划分已定义
- 23cf851/21abae1/383cbac 等 hash 型 id 均为 **AppWorld 任务 id**(venmo 等),与 SWE-bench 无关。

---

## 三、发现的"数据 vs 论文"矛盾清单

1. **follow-up 轮数**:论文表报 18 点 × 15 轮 = 810 条,但 results/main_comparisons/main_comparison_confirm20_final.json 与 out_span_offload/simplified/confirm_merged20.jsonl 是 **20 轮 / 1,080 条**(含额外臂);论文稳定性数字(Pearson −0.302、MAD 47.4pp、4 次反转)正是从这批 20 轮数据复现的。
2. **LoCoMo 时间指标**:论文 conv-26 的 8.3%/0.96s 精确复现自旧批次 out_span_offload/langgraph/batch1/257span/;10/17 重跑版(conv-26/offload/)同 engine 下为 +2.1%。新 cohort(10conv/4conv)的分析 JSON 中**均无时间指标行**——时间维度已在本文 §2.1 直接从原始记录补算。
3. **conv-26 的 baseline_tokens 信号**:论文 cohort 上显著(p=0.045,"token 多→可卸载"),10conv 大样本中消失且反向(p=0.43)。
4. **C0 口径**:results/misc/c0_gate_all_agents.json 为 757/757(四 agent);论文 tex 引用六-cohort C0 表(v12_replay_audit),该表文件未检入本仓库,且本地无 SWE 的 C0 数据。
5. **Tau2 汇总缺口**:results/tau2/tau2_interim_findings.md 记录"tau2 summaries 仅覆盖 33 点而原始覆盖 44 点,summary 重生成必须先于进一步实验"(旧版问题,最终 44 点已由 tau2_final_manifest.json + tau2_analysis_v2.json 覆盖)。

---

## 四、文件索引(分析产物 → 数据 → 脚本)

- LoCoMo 分析脚本:scripts/analysis/{main_comparison,offloadability_langgraph_exante,locomo_exante_features,locomo_task_classification,stratified_offloadability}.py
- 复算时间脚本口径:replay_meta.elapsed_ms = 目标调用开始→完成/终止(论文附录 E 定义);原始记录该字段覆盖率 100%
- 全量原始数据 4.0GB 在 out_span_offload/(LoCoMo 合并文件 _tmp_loco10_recs_merged.jsonl 800MB);checkpoint trace 2.0GB 在 check_point_traces/
- 集群侧(未同步):/share/home/tanylang/trace_span_offload/ 下的 out_span_offload/minisweagent/(SWE 134 点重放产物)与对应分析结果

---

## 五、论文 LoCoMo cohort 迁移到 10conv(2026-10-21)

论文源码(paper/AgentSubBench.zip → paper/_overleaf_src/)已将 LoCoMo cohort 从 conv-26 单对话(257 span/95 题/7,710 条)原地更新为 10 对话全量(2,623 span/1,056 题/78,690 条),两个主文(arxiv 与 iclr2027,内容相同)与 updates/ 全部同步。关键新数字(全部由 scripts/analysis/recompute_langgraph_paper_numbers.py 从原始记录复算,并对旧 cohort 逐一验证复现论文数字后才采用):

| 指标 | 旧(论文) | 新(10conv) |
|---|---|---|
| strict T / B / I | 79.5 / 90.1 / 94.0 | **83.3 / 93.1 / 96.0**(21,838 / 24,417 / 25,172 per 26,230) |
| ΔQ(pp) | −10.6 [−14.9,−6.6] | **−9.8 [−11.2,−8.4]** |
| Net tokens(k) | +2.19 [−0.07,+4.53] | **+2.86 [+1.99,+3.76]**(direct +0.32 / comp −2.54) |
| ΔW(s) | −0.96 [−1.72,−0.19](8.3%) | **−0.84 [−1.51,−0.10](7.2%)** |
| T local eq. | 41.4 | **44.0** |
| mixed-B | 43/257 | **384/2,623**;baseline 分布向量 [32,20,18,36,37,31,48,44,66,84,2207] |
| B−I / T−B / T−I | −3.9 / −10.6 / −14.5 | **−2.9 / −9.8 / −12.7** |
| phase strata | initial 95 −1.8;execution 67 −24.5;terminal 95 −9.7 | **initial 1,056 −2.4;execution 511 −32.6;terminal 1,056 −6.3** |
| 题数口径 | 95 = conv-26 进入卸载池的题(总 199) | **1,056 = 10conv 进入卸载池的题;1,986 = 10 对话全部题(930 题 baseline 失败被剔除)**;final_synthesis=terminal=1,056,tool_arg=1,567 |

注意:
- **C0 口径缺口**:v12_replay_audit 中 LoCoMo 源审计仍为 577/577(旧 cohort 的审计数,本机无 10conv 的 C0 产物);表中 measured points 已更新为 2,623。如需审计覆盖全 cohort,需在集群重跑 run_langgraph_c0.py。
- **bootstrap CI**:v9_new_cohorts 的新 CI 按论文声明的任务簇 bootstrap(10k 次、seed 20260924)复算;旧 cohort 复算端点与论文印刷值差 ≤0.2pp(实现细节差异,点估计完全吻合)。
- 修改文件清单:agentsubbench_arxiv.tex / agentsubbench_iclr2027.tex(8 处×2)+ updates/{e1,e3,e4,v4_cross_agent,v7_baseline_distribution,v7_completion_time,v7_controls,v9_cross_workload,v9_detailed_tables,v9_main_summary,v9_new_cohorts,v10_response_figure,v11_local_final_figure,v11_overview_table,v12_replay_audit,v13_final_atlas_inventory,v13_local_final_detail,v13_main_atlas_table,v13_resampling_detail,v13_resource_detail}.tex。本机无 LaTeX,未编译验证。
