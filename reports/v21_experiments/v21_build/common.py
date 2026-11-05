"""Shared numbers and helpers for the v20 reports. Every number is read from n20.json (built by analyse.py from the drop)."""
import json, re, glob, sys
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart, line_chart

N = json.load(open('/home/claude/v21/n21.json'))
for _k, _v in N['held'].items():
    _v['both_half'], _v['bothn_half'] = _v['both'], _v['bothn']; _v['both'], _v['bothn'] = _v['both1'], _v['bothn1']
V19 = open('/mnt/user-data/outputs/v205/HC_MoE_Report_v20_5.html', encoding='utf-8').read()
HEAD = V19[:V19.index('<body>')]
BOOT = V19[V19.index('function hideUnderscore'):]

EXTRA_CSS = """
main ul:not(.so-far), main ol { padding-left: 1.3rem; margin-bottom: 1rem; }
main p, main li { overflow-wrap: anywhere; }
main ul:not(.so-far) li, main ol li { margin-bottom: 0.4rem; color: var(--body-text); }
.assess ul { padding-left: 1.1rem; }
.chart-wrap h3 { font-size: 1rem; }
@media (max-width: 640px) {
  main { padding: 0 1rem; margin: 1rem auto; }
  .report-header { padding: 1.4rem; }
  .report-header h1 { font-size: 1.4rem; }
  nav { padding: 0.6rem 1rem; }
  .chart-wrap, .card { padding: 1rem; }
}
"""
def head(title):
    h = re.sub(r'<title>[^<]*</title>', f'<title>{title}</title>', HEAD, count=1)
    i = h.rindex('</style>')
    return h[:i] + EXTRA_CSS + h[i:]

def f3(x, d=3):
    if x is None: return '&ndash;'
    return ('+' if x >= 0 else '&minus;') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'

FAMC = {'Qwen': '#f59e0b', 'gemma': '#f87171', 'ernie': '#60a5fa', 'liquid': '#2dd4bf'}
FAMN = {'Qwen': 'Qwen', 'gemma': 'Gemma', 'ernie': 'ERNIE', 'liquid': 'LFM2'}
MAIN = ['lambada', 'competition_math', 'fineweb-edu', 'race', 'structured-wikipedia', 'hellaswag', 'pg19', 'tinystories', 'mmlu']
UNSEEN = ['cnn_dailymail', 'scientific_papers', 'ultrachat', 'dialogsum', 'humaneval', 'mbpp']
SHORT = {'lambada': 'lambada', 'competition_math': 'comp_math', 'fineweb-edu': 'fineweb', 'race': 'race', 'structured-wikipedia': 'struct-wiki',
         'hellaswag': 'hellaswag', 'pg19': 'pg19', 'tinystories': 'tinystories', 'mmlu': 'mmlu', 'cnn_dailymail': 'cnn', 'scientific_papers': 'sci',
         'ultrachat': 'ultrachat', 'dialogsum': 'dialogsum', 'humaneval': 'humaneval', 'mbpp': 'mbpp'}
CELL = {(c['model'], c['bm']): c for c in N['cells']}
P = ['0.6B', '1.7B', '4B', '8B']; RAT = {'0.6B': '5.5', '1.7B': '1.94', '4B': '0.82', '8B': '0.41'}
D = N['depth']; B = N['base']; H = N['held']
ROB = {(r['model'], r['layer'], r['dataset']): r for r in N['rob']}
LPOS = {(r['model'], r['layer']): r for r in N['lpos']}
POOL = {(r['model'], r['block_name'], r['dataset']): r for r in N['pool']}
ESEL = {r['layer']: r for r in N['esel']}
FLOOR = 0.0128

def cnt(m, b, arm='c'): return CELL[(m, b)][arm + '_cnt']
def robust(m, arm='c', sign=1):
    return [b for (mm, b), c in CELL.items() if mm == m and (c[arm + '_cnt'] >= 9 if sign > 0 else c[arm + '_cnt'] <= -9)]

def cnt_html(c):
    if c >= 9: return f'<span class="badge badge-sig">+{c}/10</span>'
    if c <= -9: return f'<span class="badge" style="background:rgba(248,113,113,0.2);color:#fca5a5">&minus;{-c}/10</span>'
    return f'{"+" if c > 0 else ("&minus;" if c < 0 else "")}{abs(c)}/10'

def line_gaps(cid, labels, series, *, colors, x_title='', y_title='', null_band=None, height=340, dashed=()):
    """Line chart that tolerates missing points (None) and draws them as gaps bridged by spanGaps."""
    ds = []
    for i, (name, vals) in enumerate(series.items()):
        c = colors[i % len(colors)]
        ds.append(dict(type='line', label=name, data=[None if v is None else round(float(v), 5) for v in vals], borderColor=c, backgroundColor=c,
                       borderWidth=2.5, tension=0.2, pointRadius=3, fill=False, spanGaps=True, **({'borderDash': [6, 4]} if name in dashed else {})))
    if null_band:
        n = len(labels); com = dict(type='line', pointRadius=0, borderWidth=1.5, borderDash=[5, 4], borderColor='rgba(148,163,184,0.7)', fill=False, order=99)
        ds += [dict(data=[null_band] * n, label='typical floor (±%.4f)' % null_band, **com), dict(data=[-null_band] * n, label='_lo', **com)]
    return {'_canvas': cid, '_height': height, 'type': 'line', 'data': {'labels': list(labels), 'datasets': ds},
            'options': {'responsive': True, 'maintainAspectRatio': False,
                        'plugins': {'legend': {'labels': {'filter': 'HIDE_UNDERSCORE', 'boxWidth': 12, 'font': {'size': 11}}}},
                        'scales': {'x': {'title': {'display': bool(x_title), 'text': x_title}},
                                   'y': {'title': {'display': bool(y_title), 'text': y_title}, 'grid': {'color': 'var-chart-grid'}}}}}

def card(n, title, cid, cap, h=340):
    return (f'<div class="chart-wrap"><h3 style="margin-top:0">Figure {n} &mdash; {title}</h3><div style="height:{h}px;position:relative"><canvas id="{cid}"></canvas></div>'
            f'<div class="fig-caption">{cap}</div></div>\n')

def li(tier, claim, ev, since=''):
    s = f'<span class="since">{since}</span>' if since else ''
    return f'<li><span class="tier tier-{tier}">{tier}</span><div class="body"><div class="claim">{claim}{s}</div><div class="ev">{ev}</div></div></li>'

def tbl(head_cells, rows, num_from=None):
    th = ''.join(f'<th{" class=\"num\"" if num_from is not None and i >= num_from else ""}>{h}</th>' for i, h in enumerate(head_cells))
    tr = ''.join('<tr>' + ''.join(f'<td{" class=\"num\"" if num_from is not None and i >= num_from else ""}>{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'

def ts_series(fam, bms):
    L = D[fam]['layers']; cells = D[fam]['cells']
    return L, {SHORT[b]: [cells[b].get(str(l), {}).get('ts') if str(l) in cells.get(b, {}) else None for l in L] for b in bms}

def dcnt(fam, b, l):
    return D[fam]['cells'][b].get(str(l), {}).get('cnt')

def save(path, html):
    open(path, 'w', encoding='utf-8').write(html)
    print('wrote', path, len(html))

# ---------- shared figure specs ----------
def fig_cells(cid='f_cells'):
    labs = MAIN + UNSEEN
    ser = {FAMN[f]: [CELL[(f, b)]['c_ts'] if (f, b) in CELL else None for b in labs] for f in ['Qwen', 'ernie', 'liquid', 'gemma']}
    spec = bar_chart(cid, [SHORT[b] for b in labs], ser, null_band=FLOOR, colors=[FAMC['Qwen'], FAMC['ernie'], FAMC['liquid'], FAMC['gemma']],
                     y_title='two-sided strength (r)', height=360)
    return spec

def fig_dose(cid='f_dose'):
    F = N['dose_fam']
    return line_gaps(cid, [f'{p} ({RAT[p]}×)' for p in P], {'raw r(HC, IG)': F['r_raw'], 'r(HC, IG | CE_small)': F['part_s_c'], 'r(HC, IG | CE_big)': F['part_b_c']},
                     colors=['#94a3b8', '#f59e0b', '#60a5fa'], x_title='Qwen3 dense partner (active-parameter ratio, large / small)', y_title='family mean r, nine benchmarks',
                     null_band=FLOOR, dashed=('raw r(HC, IG)',))

def fig_qdepth(cid='f_qdepth'):
    L, s = ts_series('Qwen', ['race', 'fineweb-edu', 'lambada', 'competition_math'])
    return line_gaps(cid, [f'L{l}' for l in L], s, colors=['#f59e0b', '#60a5fa', '#a78bfa', '#94a3b8'], x_title='Qwen3-30B-A3B MoE layer',
                     y_title='two-sided strength (r)', null_band=FLOOR)
