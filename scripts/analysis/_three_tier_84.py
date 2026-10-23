# -*- coding: utf-8 -*-
"""batch1 84-span three-tier offloadability analysis (A combined strata / B single axis / C input side)."""
import json, sys, glob, statistics
sys.path.insert(0, '.')
from collections import defaultdict, Counter

files = sorted(glob.glob(r'out_span_offload\simplified\batch1\simplified\screening\*batch1_*.jsonl'))
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
            gc = sd.get('graph_context') or {}
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
                'inside_loop': bool(gc.get('inside_loop')),
            }
rows = []
for sid, s in per.items():
    t, b, i = s['T_span'], s['B'], s['I']
    if len(t) < 5 or len(b) < 5 or len(i) < 5:
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
n = len(rows)
print('paired spans (each arm ≥5 rounds): %d\n' % n)


def agg(xs):
    if not xs:
        return '(n=0)'
    return ('n=%-3d T=%.2f eq=%.2f B=%.2f I=%.2f '
            'Δ_TB=%+.2f Δ_TI=%+.2f success rate (Δ≥0)=%.0f%%') % (
        len(xs), statistics.mean(r[1] for r in xs), statistics.mean(r[2] for r in xs),
        statistics.mean(r[3] for r in xs), statistics.mean(r[4] for r in xs),
        statistics.mean(r[1] - r[3] for r in xs), statistics.mean(r[1] - r[4] for r in xs),
        100 * sum(1 for r in xs if r[1] - r[3] >= 0) / len(xs))


def sec(title, groups):
    print('### %s ###' % title)
    for label, cond in groups:
        xs = [r for r in rows if cond(r)]
        print('  %-36s %s' % (label, agg(xs)))
    print()


sec('A. combined-strata metrics (cross)', [
    ('late × single (late position + baseline single-shot)', lambda r: r[0]['pos'] == 'late' and r[0]['batch'] == 'single'),
    ('late × multi', lambda r: r[0]['pos'] == 'late' and r[0]['batch'] == 'multi'),
    ('middle × multi (mid multi-batch)', lambda r: r[0]['pos'] == 'middle' and r[0]['batch'] == 'multi'),
    ('early × single (opening single-shot)', lambda r: r[0]['pos'] == 'early' and r[0]['batch'] == 'single'),
    ('early × multi (opening multi-batch)', lambda r: r[0]['pos'] == 'early' and r[0]['batch'] == 'multi'),
    ('write × late (write action + late)', lambda r: r[0]['write'] and r[0]['pos'] == 'late'),
    ('read_only × early (scouting + opening)', lambda r: (not r[0]['write']) and r[0]['pos'] == 'early'),
    ('convergence (rep≥0.5) × late', lambda r: r[0]['rep'] >= 0.5 and r[0]['pos'] == 'late'),
    ('final_synthesis (closing tier)', lambda r: r[0]['node'] == 'final_synthesis'),
])
sec('B. single-axis strata metrics', [
    ('node=final_synthesis', lambda r: r[0]['node'] == 'final_synthesis'),
    ('node=tool_selection', lambda r: r[0]['node'] == 'tool_selection'),
    ('node=tool_argument_generation', lambda r: r[0]['node'] == 'tool_argument_generation'),
    ('state=write (external state mutation)', lambda r: r[0]['write']),
    ('state=read_only', lambda r: not r[0]['write']),
    ('position=early (inv≤1)', lambda r: r[0]['pos'] == 'early'),
    ('position=middle (inv2-3)', lambda r: r[0]['pos'] == 'middle'),
    ('position=late (inv≥4)', lambda r: r[0]['pos'] == 'late'),
    ('baseline output=single', lambda r: r[0]['batch'] == 'single'),
    ('baseline output=multi', lambda r: r[0]['batch'] == 'multi'),
    ('inside loop (inside_loop)', lambda r: r[0]['inside_loop']),
])
sec('C. input-side metrics (computable before offload)', [
    ('preceding=structured list', lambda r: r[0]['last_form'] == '结构化名单'),
    ('preceding=tool call', lambda r: r[0]['last_form'] == '工具调用'),
    ('preceding=short reply/empty body', lambda r: r[0]['last_form'] in ('短应答', '空正文')),
    ('progress ≤33', lambda r: r[0]['prompt_msgs'] <= 33),
    ('progress 34-42', lambda r: 33 < r[0]['prompt_msgs'] <= 42),
    ('progress >42', lambda r: r[0]['prompt_msgs'] > 42),
    ('convergence rep=0 (exploration)', lambda r: r[0]['rep'] == 0),
    ('convergence rep∈(0,0.5)', lambda r: 0 < r[0]['rep'] < 0.5),
    ('convergence rep≥0.5', lambda r: r[0]['rep'] >= 0.5),
    ('preceding type=tool_selection (predicted list)', lambda r: r[0]['prev_node'] == 'tool_selection'),
    ('preceding type=tool_argument_generation (tool call)', lambda r: r[0]['prev_node'] == 'tool_argument_generation'),
])
