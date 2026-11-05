"""HC-MoE v18: v17 + reports/v17_experiments (four experiments). All numbers recomputed from parquet; every drop checked against earlier drops."""
import json, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart, line_chart

UP = '/mnt/user-data/uploads/HC-MoE-Alon/reports/'
E = UP + 'v17_experiments/'
s = open(UP + 'HC_MoE_Report_v17.html', encoding='utf-8').read()
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def npos(x): return int((x > 0).sum())

# ================================================================ 1. partner sweep with fineweb regenerated
Q = pd.read_parquet(E + 'qwen_8b_partner_sweep/qwen_8b_sweep_records.parquet')
QO = pd.read_parquet(UP + '26_10_15/v16_experiments/qwen_8b_partner_sweep/qwen_8b_sweep_records.parquet')
assert len(Q) == 360 and not Q.isna().any().any()
mm = Q.merge(QO, on=['dataset', 'partner', 'estimator', 'variant'], suffixes=('', '_o'))
dd = (mm.r_partial_ce_small - mm.r_partial_ce_small_o).abs().groupby(mm.dataset).max()
assert (dd.drop('fineweb-edu') == 0).all() and dd['fineweb-edu'] > 0.05     # only fineweb changed
P = ['0.6B', '1.7B', '4B', '8B']; RATIO = {'0.6B': 5.5, '1.7B': 1.94, '4B': 0.82, '8B': 0.41}
FW = Q[Q.dataset == 'fineweb-edu']; FWN = int(FW.n_tokens.iloc[0])
FWG = FW[(FW.estimator == 'GMM') & (FW.variant == 'standard')].set_index('partner')
assert abs(FWG.loc['0.6B', 'r_partial_ce_small'] - 0.076) < 0.001 and abs(FWG.loc['4B', 'r_partial_ce_small'] - 0.064) < 0.001   # = experiment 3
FWA = FW.groupby('partner').agg(raw=('r_raw', 'mean'), s=('r_partial_ce_small', 'mean'), spos=('r_partial_ce_small', npos)).loc[P]
QC = Q.groupby(['dataset', 'partner']).agg(raw=('r_raw', 'mean'), s=('r_partial_ce_small', 'mean'), b=('r_partial_ce_big', 'mean'),
                                          spos=('r_partial_ce_small', npos), floor=('floor', 'first'), ig=('mean_ig', 'first')).reset_index()
FAM9 = QC.groupby('partner')[['raw', 's', 'b', 'ig']].mean().loc[P]
FAM8 = QC[QC.dataset != 'fineweb-edu'].groupby('partner')[['raw', 's', 'b', 'ig']].mean().loc[P]
PS = QC.pivot(index='dataset', columns='partner', values='s')[P]
MONO9 = int((PS.diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum())
POS8 = QC[(QC.partner == '8B') & (QC.s > QC.floor) & (QC.spos >= 9)].dataset.tolist()
NEG8 = QC[(QC.partner == '8B') & (QC.s < -QC.floor) & (QC.spos <= 1)].dataset.tolist()

# ================================================================ 2. hidden-state norm
H = pd.read_parquet(E + 'experiment_last_layer_baselines_hnorm/last_layer_baselines_hnorm_records.parquet')
assert len(H) == 50 and not H.isna().any().any()
BX = pd.read_parquet(UP + '26_10_15/experiment_last_layer_baselines_extended/last_layer_baselines_extended_all.parquet')
hm = H.merge(BX, on=['model', 'dataset', 'estimator', 'variant'], suffixes=('', '_x'))
REF_DEV = float((hm.r_ref - hm.r_ref_x).abs().max())
qs = Q[Q.partner == '0.6B']; hq = H[H.model == 'Qwen'].merge(qs, on=['dataset', 'estimator', 'variant'])
RAW_DEV = float((hq.r_ref - hq.r_partial_ce_small).abs().max()); RAW_CORR = float(np.corrcoef(hq.r_ref, hq.r_partial_ce_small)[0, 1])
assert REF_DEV > 0.02 and RAW_DEV > 0.02                                      # reproduces neither earlier statistic
def hagg(g):
    r = g.r_ref.mean(); return pd.Series(dict(n=int(g.n_tokens.iloc[0]), ref=r, j3=g.r_joint_3.mean(), j4=g.r_joint_4.mean(),
        ret3=g.r_joint_3.mean() / r, ret4=g.r_joint_4.mean() / r, j43=g.r_joint_4.mean() / g.r_joint_3.mean(),
        j4keep=int((np.sign(g.r_joint_4) == np.sign(r)).sum()), dr2=g.delta_r2.mean()))
HC_ = H.groupby(['model', 'dataset']).apply(hagg, include_groups=False)
def bagg(g): return g.r_joint.mean() / g.r_ref.mean()
OLD3 = BX.groupby(['model', 'dataset']).apply(bagg, include_groups=False)
CELLS_H = [('Qwen', 'competition_math'), ('Qwen', 'lambada'), ('Qwen', 'race'), ('Qwen', 'fineweb-edu'), ('gemma', 'race')]

# ================================================================ 3. Gemma all layers
GA = pd.read_parquet(E + 'gemma_shared_expert_ablation/gemma_all_layers_ablation_records.parquet')
assert len(GA) == 60 and not GA.isna().any().any()
GL = pd.read_parquet(UP + '26_10_15/v16_experiments/gemma_shared_expert_ablation/gemma_scaling_records.parquet')
G = GA.groupby(['dataset', 'alpha']).agg(ce=('mean_ce_big', 'first'), cep=('ce_increase_pct', 'first'), raw=('r_raw', 'mean'),
                                         part=('r_partial', 'mean'), pos=('r_partial', npos), floor=('floor', 'first'))
def g_(d, a, k): return float(G.loc[(d, a), k])
CE_SAME = abs(g_('race', 1.0, 'ce') - GL[(GL.dataset == 'race') & (GL.alpha == 1.0)].mean_ce_big.iloc[0]) < 1e-9
assert CE_SAME
A1 = {d: (g_(d, 1.0, 'raw'), float(GL[(GL.dataset == d) & (GL.alpha == 1.0)].r_raw.mean())) for d in ['race', 'competition_math']}
SHR = {d: 1 - g_(d, 0.75, 'part') / g_(d, 1.0, 'part') for d in ['race', 'competition_math']}
assert all(v < 0.5 for v in SHR.values()) and g_('race', 0.75, 'cep') < 10 and g_('competition_math', 0.75, 'cep') < 10

# ================================================================ 4. first-occurrence strata
SJ = pd.DataFrame(json.load(open(E + 'experiment_code_first_occurrence_stratification/code_first_occurrence_stratification_summary.json')))
SR = pd.read_parquet(E + 'experiment_code_first_occurrence_stratification/code_first_occurrence_stratification_records.parquet')
assert len(SR) == 680 and not SR[['r_within']].isna().any().any()
SR['cell'] = SR.model + '_' + SR.dataset
ST = SR.groupby(['cell', 'stratum']).agg(n=('n_tokens', 'first'), r=('r_within', 'mean'), pos=('r_within', npos), lo=('ci_low', 'mean'), hi=('ci_high', 'mean'))
CELLS = ['Qwen_mbpp', 'Qwen_humaneval', 'gemma_mbpp', 'gemma_humaneval']
def st(c, k, f='r'): return float(ST.loc[(c, k), f])
PREV = {'Qwen_humaneval': (-0.0169, -0.0308), 'Qwen_mbpp': (-0.0281, -0.1061), 'gemma_humaneval': (-0.1606, -0.2935), 'gemma_mbpp': (-0.0287, -0.1152)}
REP = {c: (abs(st(c, 'ALL_TOKENS') - PREV[c][0]) < 6e-4, abs(st(c, 'ALL_FIRST_ID') - PREV[c][1]) < 6e-4) for c in CELLS}
assert REP['Qwen_humaneval'] == (True, True) and REP['Qwen_mbpp'] == (True, True)
TIERS = ['FIRST_ID__T1_LowLoss', 'FIRST_ID__T2_MedLoss', 'FIRST_ID__T3_HighLoss']
TYPES = ['FIRST_ID__Convention', 'FIRST_ID__EnglishWord', 'FIRST_ID__Unconventional']
# rule: negative confined to high-loss, unconventional names?
CONFINED = [c for c in CELLS if st(c, 'FIRST_ID__T3_High__Unconventional') < 0 and st(c, 'FIRST_ID__T1_LowLoss', 'pos') >= 5 and st(c, 'FIRST_ID__EnglishWord', 'pos') >= 5]
assert CONFINED == []
GRAD = [c for c in CELLS if abs(st(c, TIERS[0])) < min(abs(st(c, TIERS[1])), abs(st(c, TIERS[2])))]
ENGNEG = all(st(c, 'FIRST_ID__EnglishWord', 'pos') == 0 for c in CELLS)

# ================================================================ helpers
def card(n, title, cid, cap, h=340):
    return (f'<div class="card"><h3 style="margin-top:0">Figure {n} &mdash; {title}</h3><div style="height:{h}px;position:relative">'
            f'<canvas id="{cid}"></canvas></div><p style="font-size:0.85rem;color:#64748b;margin-top:0.75rem">{cap}</p></div>\n')
def table(head, rows, num_from=2):
    th = ''.join(f'<th class="num">{h}</th>' if i >= num_from else f'<th>{h}</th>' for i, h in enumerate(head))
    tr = ''.join('<tr>' + ''.join(f'<td class="num">{c}</td>' if i >= num_from else f'<td>{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'
def note(t): return f'<p style="font-size:0.85rem;color:#64748b;margin:0.5rem 0 0">{t}</p>\n'
def floor_band(spec, floors):
    for lab, sg in (('measured floor', 1), ('_floor_lo', -1)):
        spec['data']['datasets'].append(dict(data=[round(sg * f, 5) for f in floors], label=lab, type='line', pointRadius=9, pointHoverRadius=9,
            pointStyle='line', showLine=False, borderWidth=2.5, borderColor='rgba(226,232,240,0.95)', fill=False, order=-1))
    return spec

# ================================================================ figures 43-46
xl = [f'{p} ({RATIO[p]}×)' for p in P]
specs = [line_chart('fig_sweep9', xl, {'r(HC, IG | CE_small), nine benchmarks': list(FAM9.s), 'same, eight benchmarks (v17)': list(FAM8.s),
            'raw r(HC, IG), nine benchmarks': list(FAM9.raw), 'fineweb-edu alone, | CE_small': list(FWA.s)},
            x_title='small partner (active-parameter ratio, large / small)', colors=['#f59e0b', '#fcd34d', '#94a3b8', '#60a5fa'], null_band=0.0128)]
hl = [f'{"Gemma" if m == "gemma" else m} / {d}' for m, d in CELLS_H]
specs.append(bar_chart('fig_hnorm', hl, {'3 baselines, v16 drop': [OLD3.loc[k] for k in CELLS_H], '3 baselines, this drop': [HC_.loc[k, 'ret3'] for k in CELLS_H],
            '4 baselines (with norm), this drop': [HC_.loc[k, 'ret4'] for k in CELLS_H]}, y_title='share of reference retained', colors=['#94a3b8', '#f87171', '#fca5a5']))
gl = [f'{d} α={a:g}' for d in ['race', 'competition_math'] for a in (1.0, 0.75, 0.5)]
specs.append(floor_band(bar_chart('fig_gall', gl, {'raw r(HC, IG)': [g_(d, a, 'raw') for d in ['race', 'competition_math'] for a in (1.0, 0.75, 0.5)],
            'r(HC, IG | CE_small)': [g_(d, a, 'part') for d in ['race', 'competition_math'] for a in (1.0, 0.75, 0.5)]}, colors=['#64748b', '#f87171']),
            [g_(d, a, 'floor') for d in ['race', 'competition_math'] for a in (1.0, 0.75, 0.5)]))
SL = ['low loss', 'medium loss', 'high loss', 'convention (i, n, res…)', 'English word', 'unconventional']
specs.append(bar_chart('fig_strata', SL, {c.replace('_', ' / ').replace('gemma', 'Gemma'): [st(c, k) for k in TIERS + TYPES] for c in CELLS},
            y_title='within-string r(HC, IG | CE_small)', colors=['#f59e0b', '#fbbf24', '#f87171', '#fca5a5']))

# ================================================================ sections 25-28
srow = [[d] + [f'{f3(PS.loc[d,p])}' for p in P] for d in PS.index]
sec25 = f'''<section id="sweep9"><h2>25. fineweb-edu regenerated: the dose-response on nine benchmarks</h2><div class="grid-4"><div class="stat-box"><div class="val">{f3(FAM9.s.iloc[0])} &rarr; {f3(FAM9.s.iloc[-1])}</div><div class="lbl">family-mean scale-specific association, nine benchmarks, 0.6B &rarr; 8B</div></div><div class="stat-box"><div class="val">{f3(FWG.loc["0.6B","r_partial_ce_small"])}</div><div class="lbl">fineweb-edu, GMM/standard, 0.6B (experiment 3: +0.076)</div></div><div class="stat-box"><div class="val">{FWN:,}</div><div class="lbl">fineweb-edu tokens in the sweep (main grid 477,782)</div></div><div class="stat-box"><div class="val">{f3(FWA.s.iloc[-1])}</div><div class="lbl">fineweb-edu with the 8B partner</div></div></div>
<p>v17 excluded the sweep&rsquo;s fineweb-edu cell: its HC and loss series were misaligned (r &asymp; 0 under every partner). The drop regenerates it. The other 320 rows are byte-for-byte the v17 values, so nothing else moves.</p>
<p><b>The rule, fixed in v17:</b> if the 0.6B value reproduces +0.0707 and the family mean still falls monotonically to within the floor at 8B, the dose-response stands on all nine benchmarks.</p>
<div class="finding scale"><h4>Aligned now, and it reproduces experiment 3 &mdash; but on a 207,061-token subset, not the main grid&rsquo;s tokens</h4><p>Under GMM/standard the regenerated cell gives {f3(FWG.loc["0.6B","r_partial_ce_small"])} with the 0.6B partner and {f3(FWG.loc["4B","r_partial_ce_small"])} with 4B, exactly experiment 3&rsquo;s +0.076 and +0.064. Its ten-configuration raw r with 0.6B is {f3(FWA.raw.iloc[0])}, not +0.0707, because the sweep uses a {FWN:,}-token subset of fineweb-edu (43% of the main grid&rsquo;s 477,782), the same subset experiment 3 used. The rule&rsquo;s reproduction check is therefore not met as written; the cell is consistent with its own lineage, and its alignment is restored.</p><p><b>The dose-response, nine benchmarks.</b> Family mean of r(HC, IG | CE<sub>small</sub>): {" &rarr; ".join(f3(v) for v in FAM9.s)} (eight benchmarks in v17: {" &rarr; ".join(f3(v) for v in FAM8.s)}). Monotone at the family level and inside the floor at 8B. fineweb-edu itself falls by about half rather than to zero ({" &rarr; ".join(f3(v) for v in FWA.s)}) and stays positive in every configuration with the 8B partner, joining {" and ".join([d for d in POS8 if d != "fineweb-edu"])} as the benchmarks whose scale-specific association survives a partner with more active parameters. Strictly decreasing in {MONO9} of nine benchmarks.</p></div>
{card(43, 'The dose-response with fineweb-edu restored', 'fig_sweep9', 'Ten configurations per point. fineweb-edu (blue) falls by about half rather than to zero; the family mean reaches the floor at 8B either way.')}
{table(['benchmark'] + [f'{p} ({RATIO[p]}&times;)' for p in P], srow, num_from=1)}
{note('r(HC, IG | CE<sub>small</sub>), raw level, ten-configuration means.')}
</section>
'''

hrow = [[("Gemma" if m == "gemma" else m), d, f'{int(HC_.loc[(m,d),"n"]):,}', f3(HC_.loc[(m, d), 'ref']), f3(float(BX[(BX.model==m)&(BX.dataset==d)].r_ref.mean())),
         pct(OLD3.loc[(m, d)]), pct(HC_.loc[(m, d), 'ret3']), pct(HC_.loc[(m, d), 'ret4']), f'{HC_.loc[(m,d),"j43"]:.2f}', f'{int(HC_.loc[(m,d),"j4keep"])}/10'] for m, d in CELLS_H]
sec26 = f'''<section id="hnorm"><h2>26. The hidden-state norm: a drop that does not reproduce</h2><div class="grid-4"><div class="stat-box"><div class="val">not graded</div><div class="lbl">v17 rule on the hidden-state norm</div></div><div class="stat-box"><div class="val">{REF_DEV:.3f}</div><div class="lbl">largest gap between this drop&rsquo;s reference and v16&rsquo;s</div></div><div class="stat-box"><div class="val">{HC_.j43.min():.2f}&ndash;{HC_.j43.max():.2f}</div><div class="lbl">adding the norm to three baselines (ratio, within this drop)</div></div><div class="stat-box"><div class="val">5 / 5</div><div class="lbl">cells keeping sign after all four</div></div></div>
<p>The norm of the hidden state entering the last MoE layer was the fourth planned baseline, missing three times. This drop adds it and reruns the joint partial with three and with four baselines on Qwen&rsquo;s four tier-A cells and Gemma/race. 50 rows.</p>
<p><b>The rule, fixed in v17:</b> at least half surviving all four baselines in three of four tier-A cells, with the R&sup2; increment excluding zero in all four &rarr; the baseline finding stands with four baselines.</p>
<div class="finding null"><h4>Not graded: the drop&rsquo;s reference statistic matches neither earlier statistic, and its three-baseline arm does not reproduce v15&ndash;v16</h4><p>The reference partial should be the within-string r(HC, IG | CE<sub>small</sub>) that the v15 and v16 drops computed, which reproduced v13 to three decimals. Here it differs from those values by up to {REF_DEV:.3f} in the same configurations; it also differs from the raw-level partial in the partner sweep by up to {RAW_DEV:.3f} (correlation {RAW_CORR:.2f}). The three-baseline arm, a direct replication, gives retention of {", ".join(pct(HC_.loc[k,"ret3"]) for k in CELLS_H)} for the five cells, where v16 measured {", ".join(pct(OLD3.loc[k]) for k in CELLS_H)}. The fineweb-edu cell also uses the 207,061-token subset rather than the main grid. Until the pipeline is reconciled, the rule cannot be applied to these numbers.</p><p><b>What can be read.</b> Within the drop, adding the norm to the three baselines changes the remaining association by a factor of {HC_.j43.min():.2f}&ndash;{HC_.j43.max():.2f}, and every cell keeps its sign in all ten configurations. That is a relative comparison inside one pipeline, so it is suggestive: the norm does not appear to absorb what the three router statistics leave. It is not graded.</p></div>
{card(44, 'Retention after the baselines: this drop against v16', 'fig_hnorm', 'Grey: v16&rsquo;s three-baseline retention (reproduced v15 exactly). Red: the same three baselines in this drop. Light red: four baselines. The red bars should match the grey ones and do not.', h=300)}
{table(['family', 'benchmark', 'tokens', 'reference, this drop', 'reference, v16', 'retained (3), v16', 'retained (3), this drop', 'retained (4), this drop', '4 vs 3 baselines', 'sign kept (4)'], hrow)}
<div class="caveat"><b>What to do.</b> Rerun the hidden-state norm inside the v16 baseline pipeline, which reproduces the series&rsquo; numbers, and check that its three-baseline arm returns v16&rsquo;s values before reading the fourth.</div></section>
'''

grow = [[d, f'{a:g}', f'{g_(d,a,"ce"):.3f}', f'+{g_(d,a,"cep"):.1f}%' if a < 1 else '&mdash;', f3(g_(d, a, 'raw')), f'{f3(g_(d,a,"part"))} ({int(g_(d,a,"pos"))}/10)',
         'inside' if g_(d, a, 'cep') < 10 else '<b>outside</b>'] for d in ['race', 'competition_math'] for a in (1.0, 0.75, 0.5)]
sec27 = f'''<section id="gall"><h2>27. Gemma&rsquo;s shared experts in every layer</h2><div class="grid-4"><div class="stat-box"><div class="val">{SHR["race"]:.0%}</div><div class="lbl">race inversion smaller at &alpha; = 0.75 (rule needs 50%)</div></div><div class="stat-box"><div class="val">+{g_("race",0.75,"cep"):.1f}%</div><div class="lbl">race CE<sub>big</sub> at &alpha; = 0.75 (budget 10%)</div></div><div class="stat-box"><div class="val">{SHR["competition_math"]:.0%}</div><div class="lbl">competition_math smaller at &alpha; = 0.75 (+{g_("competition_math",0.75,"cep"):.1f}% CE)</div></div><div class="stat-box"><div class="val">flips</div><div class="lbl">both, at &alpha; = 0.5 (outside budget)</div></div></div>
<p>v17 found that halving the last layer&rsquo;s shared expert left Gemma&rsquo;s race inversion intact, but that arm could not change HC. This one scales the shared expert of <i>every</i> MoE layer by &alpha; = 1, 0.75 and 0.5, which changes the last layer&rsquo;s routing, and recomputes both HC and CE<sub>big</sub>. 60 rows.</p>
<p><b>The rule, fixed in v17:</b> if the race inversion shrinks by half at a setting where mean CE<sub>big</sub> rises less than 10%, the shared experts are implicated through routing; if not, the shared-expert explanation is dropped.</p>
<div class="finding gemma"><h4>Not met: a 41% shrink inside the budget, short of half &mdash; and the shrink grows with the dose</h4><p>At &alpha; = 0.75 mean CE<sub>big</sub> rises {g_("race",0.75,"cep"):.1f}% on race and the inversion goes {f3(g_("race",1.0,"part"))} &rarr; {f3(g_("race",0.75,"part"))}, {SHR["race"]:.0%} smaller; on competition_math, {f3(g_("competition_math",1.0,"part"))} &rarr; {f3(g_("competition_math",0.75,"part"))} ({SHR["competition_math"]:.0%}) at +{g_("competition_math",0.75,"cep"):.1f}%. At &alpha; = 0.5 both flip sign ({f3(g_("race",0.5,"part"))}, {f3(g_("competition_math",0.5,"part"))}), but CE rises {g_("race",0.5,"cep"):.0f}% and {g_("competition_math",0.5,"cep"):.0f}%, far outside the budget. The rule&rsquo;s second branch therefore applies by its letter: the shared-expert explanation is dropped as <i>the</i> explanation.</p><p>That is too strong a reading of what the data show. Weakening every shared expert by a quarter removes two fifths of the race inversion at a 3% loss cost, which last-layer ablation (v17, 6% at half strength) never did. The shared experts plausibly contribute to Gemma&rsquo;s inversion through routing; they do not account for it within a working model. Recorded as a partial contribution, not an explanation.</p></div>
<div class="finding null"><h4>The unablated baseline does not reproduce the main study on competition_math</h4><p>At &alpha; = 1 nothing is ablated and CE<sub>big</sub> is identical to v17&rsquo;s, yet the recomputed HC gives a raw r of {f3(A1["competition_math"][0])} on competition_math against {f3(A1["competition_math"][1])} in v17 and the main study; race agrees ({f3(A1["race"][0])} against {f3(A1["race"][1])}). The HC recomputation differs from the original on competition_math. The shrinkage above is measured against this arm&rsquo;s own &alpha; = 1, so it is internally consistent, but the competition_math numbers should not be compared with the rest of the series.</p></div>
{card(45, 'Gemma with every layer&rsquo;s shared expert scaled by &alpha;', 'fig_gall', 'Ten-configuration means; white ticks are floors. &alpha; = 0.5 is outside the 10% CE budget for both benchmarks.')}
{table(['benchmark', '&alpha;', 'mean CE_big', 'CE change', 'raw r', 'r | CE_small', 'CE budget'], grow, num_from=1)}
</section>
'''

trow = [[c.replace('_', ' / ').replace('gemma', 'Gemma')] + [f'{f3(st(c,k))} ({int(st(c,k,"pos"))}/10)' for k in TIERS + TYPES] for c in CELLS]
sec28 = f'''<section id="strata"><h2>28. First-occurrence identifiers by loss and name type</h2><div class="grid-4"><div class="stat-box"><div class="val">0 / 4</div><div class="lbl">cells where the effect is confined to high-loss, unconventional names</div></div><div class="stat-box"><div class="val">{len(GRAD)} / 4</div><div class="lbl">cells where low-loss names are the weakest tier</div></div><div class="stat-box"><div class="val">4 / 4</div><div class="lbl">cells negative in every configuration for English-word names</div></div><div class="stat-box"><div class="val">{f3(st("Qwen_humaneval","FIRST_ID__Convention"))}</div><div class="lbl">Qwen humaneval, conventional names (10/10 positive)</div></div></div>
<p>The code effect sits on the first appearance of a name (section 22). v17 proposed that it is about unpredictable naming. First-occurrence identifiers were split three ways by the small model&rsquo;s loss (terciles) and three ways by name type &mdash; a convention (<code>i</code>, <code>n</code>, <code>res</code>), an English word, or neither &mdash; and the within-string r(HC, IG | CE<sub>small</sub>) computed in each of the nine cells of the cross. 680 rows. The two Qwen cells reproduce section 22 exactly; the two Gemma cells differ slightly (humaneval {f3(st("gemma_humaneval","ALL_FIRST_ID"))} against {f3(PREV["gemma_humaneval"][1])}).</p>
<p><b>The rule, fixed in v17:</b> if the negative association is confined to high-loss, unconventional names, code&rsquo;s exception is stated as unpredictable naming.</p>
<div class="finding null"><h4>Not confined: it spans name types, and English words carry it everywhere</h4><p>English-word names are negative in every cell and every configuration (Qwen mbpp {f3(st("Qwen_mbpp","FIRST_ID__EnglishWord"))}, humaneval {f3(st("Qwen_humaneval","FIRST_ID__EnglishWord"))}; Gemma mbpp {f3(st("gemma_mbpp","FIRST_ID__EnglishWord"))}, humaneval {f3(st("gemma_humaneval","FIRST_ID__EnglishWord"))}), as are unconventional names. Low-loss names are negative too in three of four cells. The unpredictable-naming account is not supported by the rule.</p><p><b>What the strata do show.</b> In {len(GRAD)} of four cells the association is weakest among the names the small model finds easiest and stronger at medium and high loss &mdash; Qwen mbpp {f3(st("Qwen_mbpp",TIERS[0]))}, {f3(st("Qwen_mbpp",TIERS[1]))}, {f3(st("Qwen_mbpp",TIERS[2]))}; Gemma humaneval {f3(st("gemma_humaneval",TIERS[0]))}, {f3(st("gemma_humaneval",TIERS[1]))}, {f3(st("gemma_humaneval",TIERS[2]))}. Conventional names are the weakest type in three of four cells and positive in Qwen humaneval. So the code effect grows with how hard a new name is for the small model, whatever kind of name it is: high-HC occurrences of a hard new name are ones the large model also fails to gain on.</p></div>
{card(46, 'First-occurrence identifiers: by small-model loss and by name type', 'fig_strata', 'Within-string r(HC, IG | CE<sub>small</sub>), ten-configuration means. Left three: loss terciles. Right three: name types.')}
{table(['cell', 'low loss', 'medium loss', 'high loss', 'convention', 'English word', 'unconventional'], trow, num_from=1)}
<div class="caveat"><b>What this does not show.</b> The strata are small (240&ndash;6,019 tokens), and the floor shown in the drop is the full benchmark&rsquo;s, so it understates each stratum&rsquo;s noise. Loss terciles are defined on the small model&rsquo;s loss, which the partial also controls for; the tier comparison is between groups, the partial within them.</div></section>
'''

# ================================================================ insert & renumber
i = s.index('<section id="shared"')
s = s[:i] + sec25 + sec26 + sec27 + sec28 + s[i:]
NEW = ('sweep9', 'hnorm', 'gall', 'strata')
def shift_h2(m):
    sid, num, rest = m.group(1), int(m.group(2)), m.group(3)
    return m.group(0) if sid in NEW or num < 25 else f'<section id="{sid}"><h2>{num + 4}.{rest}'
s = re.sub(r'<section id="([^"]+)"><h2>(\d+)\.([^<]*)', shift_h2, s)
be = s.index('<script>\nconst CHART_SPECS'); body, tail = s[:be], s[be:]
a, b = body.index('<section id="sweep9"'), body.index('<section id="shared"')
def fs(m): n = int(m.group(1)); return m.group(0) if n < 43 else f'Figure {n + 4}'
def ss(m): n = int(m.group(2)); return m.group(0) if n < 25 else f'{m.group(1)}{n + 4}'
pre = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[:a]))
post = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[b:]))
s = pre + body[a:b] + post + tail
nav = re.search(r'<a href="#sweep">[^<]*</a>', s).group(0)
s = s.replace(nav, nav + '<a href="#sweep9">Nine-benchmark sweep</a><a href="#hnorm">Hidden-state norm</a><a href="#gall">Gemma, all layers</a><a href="#strata">Identifier strata</a>', 1)
m = re.search(r'const CHART_SPECS = (\[.*?\]);\nfunction hideUnderscore', s, flags=re.S)
s = s[:m.start(1)] + json.dumps(json.loads(m.group(1)) + specs) + s[m.end(1):]
open('/home/claude/v18/stage1.html', 'w', encoding='utf-8').write(s)
json.dump(dict(FAM9=FAM9.round(5).to_dict(), FAM8=FAM8.round(5).to_dict(), FWA=FWA.round(5).to_dict(), FWN=FWN, POS8=POS8, NEG8=NEG8, MONO9=MONO9,
               REF_DEV=REF_DEV, RAW_DEV=RAW_DEV, J43=[float(HC_.j43.min()), float(HC_.j43.max())],
               OLD3={f'{k[0]}|{k[1]}': float(OLD3.loc[k]) for k in CELLS_H}, RET3={f'{k[0]}|{k[1]}': float(HC_.loc[k, 'ret3']) for k in CELLS_H},
               SHR=SHR, G={f'{k[0]}|{k[1]}': v for k, v in G.round(5).to_dict('index').items()}, A1=A1,
               ST={f'{k[0]}|{k[1]}': v for k, v in ST.round(5).to_dict('index').items()}, GRAD=GRAD),
          open('/home/claude/v18/n18.json', 'w'), indent=1, default=float)
print('stage1 ok', len(s), 'POS8', POS8, 'NEG8', NEG8, 'GRAD', GRAD, 'j43', HC_.j43.round(2).tolist())
