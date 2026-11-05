"""HC-MoE v19: v18 + reports/v18_experiments (three drops). All numbers recomputed from parquet."""
import json, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart

UP = '/mnt/user-data/uploads/HC-MoE-Alon/reports/'
E = UP + 'v18_experiments/'
s = open(UP + 'HC_MoE_Report_v18.html', encoding='utf-8').read()
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def npos(x): return int((x > 0).sum())

# ================================================================ 1. hidden-state norm in the v16 pipeline
H = pd.read_parquet(E + 'experiment_last_layer_baselines_hnorm_v16/last_layer_baselines_hnorm_v16_records.parquet')
assert len(H) == 50 and not H[['r_ref', 'r_joint_3', 'r_joint_4']].isna().any().any()
BX = pd.read_parquet(UP + '26_10_15/experiment_last_layer_baselines_extended/last_layer_baselines_extended_all.parquet')
hm = H.merge(BX, on=['model', 'dataset', 'estimator', 'variant'], suffixes=('', '_x'))
DEV = hm.assign(d=np.maximum((hm.r_ref - hm.r_ref_x).abs(), (hm.r_joint_3 - hm.r_joint).abs())).groupby(['model', 'dataset']).d.max()
REPRO = {k: bool(v < 1e-6) for k, v in DEV.items()}
assert REPRO[('Qwen', 'competition_math')] and REPRO[('Qwen', 'lambada')] and REPRO[('Qwen', 'race')] and REPRO[('gemma', 'race')]
assert not REPRO[('Qwen', 'fineweb-edu')]
def hagg(g):
    r = g.r_ref.mean()
    return pd.Series(dict(n=int(g.n_tokens.iloc[0]), ref=r, j3=g.r_joint_3.mean(), j4=g.r_joint_4.mean(), ret3=g.r_joint_3.mean() / r, ret4=g.r_joint_4.mean() / r,
        cfg50=int((g.r_joint_4 / g.r_ref >= 0.5).sum()), keep=int((np.sign(g.r_joint_4) == np.sign(r)).sum()),
        ci_ex=int(((g.ci_r4_lower > 0) if r > 0 else (g.ci_r4_upper < 0)).sum()), lo=g.ci_r4_lower.mean(), hi=g.ci_r4_upper.mean(),
        rlo=g.ci_ret4_lower.median(), rhi=g.ci_ret4_upper.median(), dr2=g.delta_r2_4.mean(), dlo=g.ci_delta_r2_lower.mean(), dcfg=int((g.ci_delta_r2_lower > 0).sum())))
HN = H.groupby(['model', 'dataset']).apply(hagg, include_groups=False)
TA = ['competition_math', 'lambada', 'race']                       # fineweb-edu excluded: does not reproduce
GE50 = [d for d in TA if HN.loc[('Qwen', d), 'ret4'] >= 0.5]
LT25 = [d for d in TA if HN.loc[('Qwen', d), 'ret4'] < 0.25]
assert GE50 == ['competition_math'] and LT25 == ['race']
ret = lambda d, k='ret4': float(HN.loc[('Qwen', d), k])

# ================================================================ 2. code vs prose at matched loss
D = pd.read_parquet(E + 'experiment_code_against_prose_matched_loss/code_against_prose_decile_records.parquet')
assert len(D) == 800 and not D[['r_partial', 'mean_ce_small']].isna().any().any()
DA = D.groupby(['model', 'dataset', 'decile']).agg(ce=('mean_ce_small', 'first'), n=('n_tokens', 'first'), r=('r_partial', 'mean'), pos=('r_partial', npos), floor=('floor', 'first')).reset_index()
PROSE, CODE = ['cnn_dailymail', 'scientific_papers'], ['mbpp', 'humaneval']
MATCH = {}
for m in ['Qwen', 'gemma']:
    A = DA[DA.model == m]; rows = []
    for c in CODE:
        for _, r in A[A.dataset == c].iterrows():
            if r.ce < 0.3: continue
            pv = np.mean([np.interp(r.ce, A[A.dataset == p].sort_values('ce').ce, A[A.dataset == p].sort_values('ce').r) for p in PROSE])
            rows.append(dict(code=c, decile=int(r.decile), ce=r.ce, rc=r.r, rp=pv, pos=int(r.pos)))
    MATCH[m] = pd.DataFrame(rows)
QM, GM = MATCH['Qwen'], MATCH['gemma']
Q_BELOW = int((QM.rc < QM.rp).sum()); G_BELOW = int((GM.rc < GM.rp).sum())
assert Q_BELOW == len(QM) == 10 and G_BELOW >= 9
PROSE_POS = int(((DA.model == 'Qwen') & DA.dataset.isin(PROSE) & (DA.decile >= 2) & (DA.pos == 10)).sum())   # of 18
def dv(m, d, k, f='r'): return float(DA[(DA.model == m) & (DA.dataset == d) & (DA.decile == k)][f].iloc[0])

# ================================================================ 3. Gemma finer scaling (losses only)
GS = pd.DataFrame(json.load(open(E + 'gemma_shared_expert_ablation/gemma_shared_expert_ablation_summary.json')))
assert set(GS.target_layer) == {29} and GS.routing_file.isna().all()
import glob as _g
HAS_CORR = len(_g.glob(E + 'gemma_shared_expert_ablation/*records*')) > 0
assert not HAS_CORR
BASE = {'race': 2.1523418394734875, 'competition_math': 0.8167406628552398}       # v17 last-layer alpha=1 avg_loss
GS['inc'] = GS.apply(lambda r: 100 * (r.avg_loss / BASE[r.dataset] - 1), axis=1)

# ================================================================ helpers
def card(n, title, cid, cap, h=340):
    return (f'<div class="card"><h3 style="margin-top:0">Figure {n} &mdash; {title}</h3><div style="height:{h}px;position:relative">'
            f'<canvas id="{cid}"></canvas></div><p style="font-size:0.85rem;color:#64748b;margin-top:0.75rem">{cap}</p></div>\n')
def table(head, rows, num_from=2):
    th = ''.join(f'<th class="num">{h}</th>' if i >= num_from else f'<th>{h}</th>' for i, h in enumerate(head))
    tr = ''.join('<tr>' + ''.join(f'<td class="num">{c}</td>' if i >= num_from else f'<td>{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'
def note(t): return f'<p style="font-size:0.85rem;color:#64748b;margin:0.5rem 0 0">{t}</p>\n'
def scatter(cid, series, colors, ytitle='within-string r(HC, IG | CE_small)'):
    ds = []
    for (name, pts, dash), c in zip(series, colors):
        ds.append(dict(label=name, data=[dict(x=round(float(x), 4), y=round(float(y), 5)) for x, y in pts], showLine=True, borderColor=c, backgroundColor=c,
                       borderWidth=2.5, pointRadius=3, tension=0.2, fill=False, **({'borderDash': [6, 4]} if dash else {})))
    return {'_canvas': cid, 'type': 'scatter', 'data': {'datasets': ds}, 'options': {'responsive': True, 'maintainAspectRatio': False,
            'plugins': {'legend': {'labels': {'boxWidth': 12, 'font': {'size': 11}}}},
            'scales': {'x': {'type': 'linear', 'title': {'display': True, 'text': 'mean small-model loss in the decile (nats)'}},
                       'y': {'title': {'display': True, 'text': ytitle}, 'grid': {'color': 'var-chart-grid'}}}}}

# ================================================================ figures 47-49
CH = [('Qwen', 'competition_math'), ('Qwen', 'lambada'), ('Qwen', 'race'), ('gemma', 'race')]
specs = [bar_chart('fig_hn16', [f'{"Gemma" if m == "gemma" else m} / {d}' for m, d in CH],
                   {'three router baselines': [HN.loc[k, 'ret3'] for k in CH], 'plus the hidden-state norm': [HN.loc[k, 'ret4'] for k in CH]},
                   y_title='share of within-string r(HC, IG | CE_small) retained', colors=['#94a3b8', '#f87171'])]
specs[0]['data']['datasets'].append(dict(data=[0.5] * 4, label='rule: half', type='line', pointRadius=0, borderWidth=1.5, borderDash=[5, 4], borderColor='rgba(248,113,113,0.8)', fill=False, order=-1))
specs[0]['data']['datasets'].append(dict(data=[0.25] * 4, label='rule: a quarter', type='line', pointRadius=0, borderWidth=1.5, borderDash=[2, 3], borderColor='rgba(148,163,184,0.8)', fill=False, order=-1))
COL = {'cnn_dailymail': '#60a5fa', 'scientific_papers': '#2dd4bf', 'mbpp': '#f87171', 'humaneval': '#f59e0b'}
for m, cid in (('Qwen', 'fig_cp_q'), ('gemma', 'fig_cp_g')):
    ser = []
    for dset in PROSE + CODE:
        A = DA[(DA.model == m) & (DA.dataset == dset)].sort_values('ce')
        ser.append((f'{dset} ({"prose" if dset in PROSE else "code"})', list(zip(A.ce, A.r)), dset in PROSE))
    specs.append(scatter(cid, ser, [COL[d] for d in PROSE + CODE]))

# ================================================================ sections 29-31
hrow = [[("Gemma" if m == "gemma" else m), d, 'yes' if REPRO[(m, d)] else '<b>no</b>', f3(HN.loc[(m, d), 'ref']), f3(HN.loc[(m, d), 'j3']), f3(HN.loc[(m, d), 'j4']),
         f'{f3(HN.loc[(m,d),"lo"])} to {f3(HN.loc[(m,d),"hi"])}', pct(HN.loc[(m, d), 'ret3']), pct(HN.loc[(m, d), 'ret4']),
         f'{int(HN.loc[(m,d),"cfg50"])}/10', f'{int(HN.loc[(m,d),"keep"])}/10', f'{100*HN.loc[(m,d),"dr2"]:.3f}%'] for m, d in [('Qwen', 'competition_math'), ('Qwen', 'lambada'), ('Qwen', 'race'), ('Qwen', 'fineweb-edu'), ('gemma', 'race')]]
sec29 = f'''<section id="hn16"><h2>29. The hidden-state norm, inside the pipeline that reproduces</h2><div class="grid-4"><div class="stat-box"><div class="val">4 / 5</div><div class="lbl">cells whose three-baseline arm reproduces v16 exactly</div></div><div class="stat-box"><div class="val">{pct(ret("competition_math"))} / {pct(ret("lambada"))} / {pct(ret("race"))}</div><div class="lbl">retained after four baselines: competition_math, lambada, race</div></div><div class="stat-box"><div class="val">{pct(ret("race","ret3"))} &rarr; {pct(ret("race"))}</div><div class="lbl">race, three &rarr; four baselines</div></div><div class="stat-box"><div class="val">{pct(HN.loc[("gemma","race"),"ret4"])}</div><div class="lbl">Gemma race retained (inversion not absorbed)</div></div></div>
<p>v18&rsquo;s hidden-state-norm drop did not reproduce v16 and was not graded. This rerun adds the norm of the hidden state entering the last MoE layer as a fourth column to the v16 baseline code, and checks its three-baseline arm against v16 before reading the fourth. 50 rows.</p>
<p><b>The rule, fixed in v18:</b> only if the three-baseline arm reproduces v16 to 1e-6 &mdash; at least half surviving four baselines in three of four tier-A cells &rarr; the baseline finding stands with four; below a quarter in two or more &rarr; restate in terms of the norm.</p>
<div class="finding null"><h4>Reproduces in four of five cells; with the norm added, neither branch of the rule is met</h4><p><b>Reproduction.</b> competition_math, lambada, race and Gemma/race reproduce v16 exactly in all ten configurations (largest difference 0). fineweb-edu does not (difference up to {float(DEV[("Qwen","fineweb-edu")]):.2f}): the drop again uses the 207,061-token subset rather than v16&rsquo;s 477,782 tokens. It is excluded, leaving three tier-A cells.</p><p><b>Four baselines.</b> The share of Qwen&rsquo;s scale-specific association that survives router entropy, maximum probability, logit gap <i>and</i> the hidden-state norm: competition_math {pct(ret("competition_math"))} (nine of ten configurations at or above half), lambada {pct(ret("lambada"))} (four of ten), race {pct(ret("race"))} (none). One cell clears half, so the first branch fails; one is below a quarter, so the second branch fails too. The result sits between the branches, and it is not uniform: the norm takes little from competition_math and lambada ({pct(ret("competition_math","ret3"))} &rarr; {pct(ret("competition_math"))}, {pct(ret("lambada","ret3"))} &rarr; {pct(ret("lambada"))}) and most of what was left in race ({pct(ret("race","ret3"))} &rarr; {pct(ret("race"))}).</p><p><b>What remains.</b> The four-baseline partial is positive in every configuration in competition_math and lambada, with intervals excluding zero in {int(HN.loc[("Qwen","competition_math"),"ci_ex"])} and {int(HN.loc[("Qwen","lambada"),"ci_ex"])} of ten; in race it is {f3(ret("race","j4"))}, positive in {int(HN.loc[("Qwen","race"),"keep"])} of ten and excluding zero in {int(HN.loc[("Qwen","race"),"ci_ex"])}. Gemma&rsquo;s race inversion is untouched ({pct(HN.loc[("gemma","race"),"ret4"])}).</p></div>
<div class="finding scale"><h4>Reading</h4><p>HC carries information that none of the four simpler last-layer statistics carry, clearly in two of three tier-A cells. In race most of what survived the router statistics is shared with the size of the hidden state the router reads, so race&rsquo;s association is largely about how large that state is, not how anomalous the routing is. The defensible claim narrows to: beyond router confidence and hidden-state size, HC keeps about three quarters of its scale-specific association in competition_math and half in lambada.</p></div>
{card(47, 'Retention with three and with four baselines', 'fig_hn16', 'Share of within-string r(HC, IG | CE<sub>small</sub>) retained, ratio of ten-configuration means; cells whose three-baseline arm reproduces v16 exactly. Dashed lines: the rule&rsquo;s half and quarter.', h=300)}
{table(['family', 'benchmark', 'reproduces v16', 'reference', 'after 3', 'after 4', 'CI (after 4)', 'retained (3)', 'retained (4)', 'configs &ge; 50%', 'sign kept', 'R&sup2; increment'], hrow)}
</section>
'''

def mrows(M): return [[r.code, str(r.decile), f'{r.ce:.2f}', f'{f3(r.rc)} ({r.pos}/10)', f3(r.rp), f3(r.rc - r.rp)] for r in M.itertuples()]
sec30 = f'''<section id="codeprose"><h2>30. Code against prose at matched small-model loss</h2><div class="grid-4"><div class="stat-box"><div class="val">{Q_BELOW} / {len(QM)}</div><div class="lbl">Qwen code deciles below prose at the same loss</div></div><div class="stat-box"><div class="val">{f3(QM.rc.sub(QM.rp).mean())}</div><div class="lbl">Qwen mean gap, code &minus; prose</div></div><div class="stat-box"><div class="val">{PROSE_POS} / 18</div><div class="lbl">Qwen prose deciles 2&ndash;10 positive in every configuration</div></div><div class="stat-box"><div class="val">{G_BELOW} / {len(GM)}</div><div class="lbl">Gemma code deciles below prose at the same loss</div></div></div>
<p>Section 28 found the code effect grows with the small model&rsquo;s loss on a name. v18 asked whether that is what any hard token looks like, or specific to code. Tokens of two code benchmarks (mbpp, humaneval) and two prose benchmarks (cnn_dailymail, scientific_papers) were split into deciles of CE<sub>small</sub>, and the within-string r(HC, IG | CE<sub>small</sub>) computed in each, both families, ten configurations: 800 rows. Because the deciles are per benchmark, code and prose deciles cover different losses; the comparison below interpolates the prose curve to each code decile&rsquo;s mean loss (code deciles below 0.3 nats, where prose has no comparable tokens, are left out).</p>
<p><b>The rule, fixed in v18:</b> if prose shows the same fall into negative values at high loss, the code exception is a hard-token effect that code has more of; if prose stays positive at matched loss, it is specific to code.</p>
<div class="finding scale"><h4>Specific to code: at matched loss, prose rises and code falls</h4><p><b>Qwen.</b> In prose the scale-specific association <i>grows</i> with the small model&rsquo;s loss: cnn_dailymail goes from {f3(dv("Qwen","cnn_dailymail",1))} in the easiest decile to {f3(dv("Qwen","cnn_dailymail",9))} in the ninth, scientific_papers from {f3(dv("Qwen","scientific_papers",1))} to {f3(dv("Qwen","scientific_papers",9))}, positive in every configuration in deciles 2&ndash;10 of both. In code it falls: mbpp reaches {f3(dv("Qwen","mbpp",10))} in its hardest decile. At the same mean loss, every one of the {len(QM)} code deciles sits below the prose curve, by {f3(QM.rc.sub(QM.rp).mean())} on average. By the rule&rsquo;s second branch, the code exception is specific to code: hard prose tokens are where HC marks extra gain, hard code tokens are where it marks less.</p><p><b>Gemma</b>, inverted overall, is near zero or mildly negative in prose at every loss (cnn_dailymail {f3(dv("gemma","cnn_dailymail",10))}, scientific_papers {f3(dv("gemma","scientific_papers",10))} in the hardest decile) and falls steeply in code (humaneval {f3(dv("gemma","humaneval",10))}). {G_BELOW} of {len(GM)} code deciles sit below prose at matched loss. The same code-specific fall appears in both families, on top of each family&rsquo;s own sign.</p></div>
{card(48, 'Qwen: scale-specific association against small-model loss, code and prose', 'fig_cp_q', 'Each point is a decile of CE<sub>small</sub> within one benchmark; ten-configuration means. Dashed: prose. Solid: code.')}
{card(49, 'Gemma: the same', 'fig_cp_g', 'Gemma&rsquo;s prose is near zero or mildly negative at every loss; its code falls far below it.')}
{table(['code benchmark', 'decile', 'mean CE_small', 'code r', 'prose r at the same loss', 'gap'], mrows(QM), num_from=1)}{note('Qwen. Prose value = mean of cnn_dailymail and scientific_papers, each linearly interpolated to the code decile&rsquo;s mean loss.')}
<div class="caveat"><b>What this does not show.</b> Two code and two prose benchmarks. Deciles 1&ndash;5 of code have mean loss below 0.3 nats &mdash; half of code tokens are nearly free for the small model &mdash; so the matched comparison rests on the upper half of code. The interpolated prose value is a curve read-off, not a separate measurement.</div></section>
'''

grow = [[r.dataset, f'{r.alpha:g}', f'{r.avg_loss:.4f}', f'{r.perplexity:.3f}', f'+{r.inc:.2f}%'] for r in GS.itertuples()]
sec31 = f'''<section id="gfine"><h2>31. Gemma&rsquo;s shared expert, finer scaling: losses only</h2>
<p>v18 proposed scaling the shared experts of <i>all</i> layers more finely (&alpha; = 0.875, 0.75, 0.70, 0.65) inside the 10% loss budget, recomputing HC, to see whether the race inversion ever halves. The drop contains losses for those &alpha; values with only the <b>last layer&rsquo;s</b> shared expert scaled (layer 29), no recomputed routing, and no correlation records.</p>
<div class="finding null"><h4>Not the proposed experiment, and not analysable as delivered</h4><p>The last-layer arm cannot change HC, and the drop has no HC or CE<sub>small</sub> series to correlate with the new losses, so r(HC, IG) cannot be computed here. What the losses do show: scaling the last layer&rsquo;s shared expert down to 0.65 raises mean CE<sub>big</sub> by at most {GS.inc.max():.2f}%, in line with v17&rsquo;s last-layer arm (+1.1% at 0.5). The all-layer finer scan remains open.</p></div>
{table(['benchmark', '&alpha; (last layer only)', 'mean CE_big', 'perplexity', 'change vs &alpha; = 1'], grow, num_from=1)}
</section>
'''

# ================================================================ insert & renumber
i = s.index('<section id="shared"')
s = s[:i] + sec29 + sec30 + sec31 + s[i:]
NEW = ('hn16', 'codeprose', 'gfine')
def shift_h2(m):
    sid, num, rest = m.group(1), int(m.group(2)), m.group(3)
    return m.group(0) if sid in NEW or num < 29 else f'<section id="{sid}"><h2>{num + 3}.{rest}'
s = re.sub(r'<section id="([^"]+)"><h2>(\d+)\.([^<]*)', shift_h2, s)
be = s.index('<script>\nconst CHART_SPECS'); body, tail = s[:be], s[be:]
a, b = body.index('<section id="hn16"'), body.index('<section id="shared"')
def fs(m): n = int(m.group(1)); return m.group(0) if n < 47 else f'Figure {n + 3}'
def ss(m): n = int(m.group(2)); return m.group(0) if n < 29 else f'{m.group(1)}{n + 3}'
pre = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[:a]))
post = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[b:]))
s = pre + body[a:b] + post + tail
nav = re.search(r'<a href="#strata">[^<]*</a>', s).group(0)
s = s.replace(nav, nav + '<a href="#hn16">Hidden-state norm (rerun)</a><a href="#codeprose">Code vs prose</a><a href="#gfine">Gemma, finer scaling</a>', 1)
m = re.search(r'const CHART_SPECS = (\[.*?\]);\nfunction hideUnderscore', s, flags=re.S)
s = s[:m.start(1)] + json.dumps(json.loads(m.group(1)) + specs) + s[m.end(1):]
open('/home/claude/v19/stage1.html', 'w', encoding='utf-8').write(s)
json.dump(dict(HN={f'{k[0]}|{k[1]}': v for k, v in HN.round(5).to_dict('index').items()}, REPRO={f'{k[0]}|{k[1]}': v for k, v in REPRO.items()},
               Q_BELOW=Q_BELOW, G_BELOW=G_BELOW, NQ=len(QM), NG=len(GM), QGAP=float(QM.rc.sub(QM.rp).mean()), GGAP=float(GM.rc.sub(GM.rp).mean()),
               PROSE_POS=PROSE_POS, GSMAX=float(GS.inc.max()),
               DV={f'{m}|{d}|{k}': dv(m, d, k) for m in ['Qwen', 'gemma'] for d in PROSE + CODE for k in range(1, 11)}),
          open('/home/claude/v19/n19.json', 'w'), indent=1, default=float)
print('stage1 ok', len(s))
