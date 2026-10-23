# -*- coding: utf-8 -*-
"""Three-group (simplified 48 / simplified 84 / tau2 44) offloadability metric summary table: A combined strata / B single axis / C input side."""
import json, sys, glob, statistics
sys.path.insert(0, '.')
from collections import defaultdict, Counter

GROUPS = {
    's48': (sorted(glob.glob(r'out_span_offload\_no400_s48\*.jsonl')), 5),
    's84': (sorted(glob.glob(r'out_span_offload\_no400_s84\*batch1_*.jsonl')), 5),
    't44': (sorted(glob.glob(r'out_span_offload\_no400_t44\*.jsonl')), 3),
}


def load(files, min_rounds):
    per = defaultdict(lambda: defaultdict(list))
    node_map = {}
    for fp in files:
        for line in open(fp, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            arm = (d.get('slm_output') or {}).get('target_arm')
            if arm not in ('T_span', 'B', 'I'):
                continue
            sid = d.get('span_id')
            node_map[(d.get('task_id'), d.get('invocation_index'))] = (
                d.get('span_descriptor') or {}).get('node_pattern')
            per[sid][arm].append({
                'strict': bool(d.get('strict_success')),
                'eq': bool((d.get('local_outcome') or {}).get('equivalence', {}).get('passed')),
            })
            if 'feats' not in per[sid]:
                sd = d.get('span_descriptor') or {}
                so = d.get('slm_output') or {}
                dt = d.get('downstream_trace') or {}
                prefix = (dt.get('prefix') or {}).get('messages') or []
                seq = []
                for m in prefix:
                    for c_ in (m.get('tool_calls') or []):
                        name = (c_.get('function') or {}).get('name') or c_.get('name') or ''
                        if name:
                            seq.append(name)
                last_form = '无assistant'
                for m in reversed(prefix):
                    if m.get('role') == 'assistant':
                        c = str(m.get('content') or '').strip()
                        tcs = m.get('tool_calls') or []
                        lines = [l for l in c.split('\n') if l.strip()]
                        if tcs:
                            last_form = '工具调用'
                        elif len(lines) >= 3 and any('.' in l or '__' in l for l in lines):
                            last_form = '结构化名单'
                        elif len(c) >= 40:
                            last_form = '长文本'
                        elif c:
                            last_form = '短应答'
                        else:
                            last_form = '空正文'
                        break
                last6 = seq[-6:]
                rep = 0.0
                if len(last6) >= 2:
                    cc = Counter(last6)
                    rep = sum(v - 1 for v in cc.values() if v > 1) / len(last6)
                base_calls = so.get('baseline_tool_calls') or []
                per[sid]['feats'] = {
                    'node': sd.get('node_pattern'),
                    'state': '|'.join((sd.get('mechanism_tags') or {}).get('state_effect') or ['?']),
                    'inv': d.get('invocation_index'),
                    'task': d.get('task_id'),
                    'prompt_msgs': len(prefix),
                    'rep': rep,
                    'last_form': last_form,
                    'n_base_calls': len(base_calls),
                }
    rows = []
    for sid, s in per.items():
        t, b, i = s['T_span'], s['B'], s['I']
        if len(t) < min_rounds or len(b) < min_rounds or len(i) < min_rounds:
            continue
        f = s['feats']
        f['prev_node'] = node_map.get((f['task'], f['inv'] - 1))
        f['pos'] = 'early' if f['inv'] <= 1 else ('middle' if f['inv'] <= 3 else 'late')
        f['batch'] = 'single' if f['n_base_calls'] <= 1 else 'multi'
        f['write'] = 'external_state_mutation' in (f['state'] or '')
        rows.append((f, sum(r['strict'] for r in t) / len(t),
                     sum(r['eq'] for r in t) / len(t),
                     sum(r['strict'] for r in b) / len(b),
                     sum(r['strict'] for r in i) / len(i)))
    return rows


results = {name: load(files, mr) for name, (files, mr) in GROUPS.items()}
for name, rows in results.items():
    print('%s: n=%d span' % (name, len(rows)))

GROUPS_DEF = [
    ('A1 late×single', lambda r: r[0]['pos'] == 'late' and r[0]['batch'] == 'single'),
    ('A2 late×multi', lambda r: r[0]['pos'] == 'late' and r[0]['batch'] == 'multi'),
    ('A3 middle×multi', lambda r: r[0]['pos'] == 'middle' and r[0]['batch'] == 'multi'),
    ('A4 early×single', lambda r: r[0]['pos'] == 'early' and r[0]['batch'] == 'single'),
    ('A5 early×multi', lambda r: r[0]['pos'] == 'early' and r[0]['batch'] == 'multi'),
    ('A6 write×late', lambda r: r[0]['write'] and r[0]['pos'] == 'late'),
    ('A7 read_only×early', lambda r: (not r[0]['write']) and r[0]['pos'] == 'early'),
    ('A8 convergence≥0.5×late', lambda r: r[0]['rep'] >= 0.5 and r[0]['pos'] == 'late'),
    ('B1 node=final_synthesis', lambda r: r[0]['node'] == 'final_synthesis'),
    ('B2 node=tool_selection', lambda r: r[0]['node'] == 'tool_selection'),
    ('B3 node=tool_argument_generation', lambda r: r[0]['node'] == 'tool_argument_generation'),
    ('B4 state=write', lambda r: r[0]['write']),
    ('B5 state=read_only', lambda r: not r[0]['write']),
    ('B6 position=early', lambda r: r[0]['pos'] == 'early'),
    ('B7 position=middle', lambda r: r[0]['pos'] == 'middle'),
    ('B8 position=late', lambda r: r[0]['pos'] == 'late'),
    ('B9 baseline=single', lambda r: r[0]['batch'] == 'single'),
    ('B10 baseline=multi', lambda r: r[0]['batch'] == 'multi'),
    ('C1 preceding=structured list', lambda r: r[0]['last_form'] == '结构化名单'),
    ('C2 preceding=tool call', lambda r: r[0]['last_form'] == '工具调用'),
    ('C3 preceding=short reply/empty body', lambda r: r[0]['last_form'] in ('短应答', '空正文')),
    ('C4 progress ≤33', lambda r: r[0]['prompt_msgs'] <= 33),
    ('C5 progress 34-42', lambda r: 33 < r[0]['prompt_msgs'] <= 42),
    ('C6 progress >42', lambda r: r[0]['prompt_msgs'] > 42),
    ('C7 convergence rep=0', lambda r: r[0]['rep'] == 0),
    ('C8 convergence rep∈(0,0.5)', lambda r: 0 < r[0]['rep'] < 0.5),
    ('C9 convergence rep≥0.5', lambda r: r[0]['rep'] >= 0.5),
    ('C10 preceding type=tool_selection', lambda r: r[0]['prev_node'] == 'tool_selection'),
    ('C11 preceding type=tool_argument_generation', lambda r: r[0]['prev_node'] == 'tool_argument_generation'),
]

print()
print('%-26s | %-22s | %-22s | %-22s' % ('group', 's48 (n, T, Δ_TB)', 's84 (n, T, Δ_TB)', 't44 (n, T, Δ_TB)'))
for label, cond in GROUPS_DEF:
    cells = []
    for name in ('s48', 's84', 't44'):
        xs = [r for r in results[name] if cond(r)]
        if not xs:
            cells.append('(n=0)')
            continue
        cells.append('n=%d T=%.2f Δ=%+.2f' % (
            len(xs), statistics.mean(r[1] for r in xs),
            statistics.mean(r[1] - r[3] for r in xs)))
    print('%-26s | %-22s | %-22s | %-22s' % (label, cells[0], cells[1], cells[2]))
