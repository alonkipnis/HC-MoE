"""Build HC-MoE v16 from v15 plus the second 26_10_15 drop (four experiments).
Every number is recomputed here from the drop's parquet files."""
import json, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart

UP = '/mnt/user-data/uploads/HC-MoE-Alon/reports/'
D = UP + '26_10_15/'
s = open(UP + 'HC_MoE_Report_v15.html', encoding='utf-8').read()

def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100 * x:.0f}%'
def pos(x): return int((x > 0).sum())

# ================================================================ data
# --- extended baselines
B = pd.read_parquet(D + 'experiment_last_layer_baselines_extended/last_layer_baselines_extended_all.parquet')
assert B.groupby(['model', 'dataset']).size().eq(10).all() and len(B) == 170 and not B.isna().any().any()
OLD = pd.read_csv(D + 'experiment_last_layer_baselines/last_layer_baselines_all_configurations.csv')
mm = OLD.merge(B, on=['model', 'dataset', 'estimator', 'variant'])
B_REPRO = (len(mm), max((mm.r_ref_s - mm.r_ref).abs().max(), (mm.r_joint_s - mm.r_joint).abs().max()))
assert B_REPRO[0] == 80 and B_REPRO[1] < 1e-12
assert not any('norm' in c.lower() for c in B.columns)          # hidden-state norm absent
def bagg(g):
    rs, rj = g.r_ref.mean(), g.r_joint.mean()
    return pd.Series(dict(tier=g.tier.iloc[0], floor=g.floor.iloc[0], ref=rs, ref_pos=pos(g.r_ref), joint=rj,
        jlo=g.ci_joint_lower.mean(), jhi=g.ci_joint_upper.mean(), ret=rj / rs, cfg50=int((g.retention >= 0.5).sum()),
        ret_b=g.r_joint_b.mean() / g.r_ref_b.mean(), r2b=g.r2_base.mean(), dr2=g.delta_r2.mean(),
        dlo=g.ci_delta_r2_lower.mean(), dhi=g.ci_delta_r2_upper.mean(), dr2_cfg=int((g.ci_delta_r2_lower > 0).sum()),
        p_ent=g.pred_ent.mean(), p_mp=g.pred_mp.mean(), p_lg=g.pred_lg.mean()))
BX = B.groupby(['model', 'dataset']).apply(bagg, include_groups=False)
# Delta R^2 equals joint partial^2 * (1 - R^2_base) to rounding -> same test as the joint partial
DR2_IDENT = float((B.r_joint ** 2 * (1 - B.r2_base) - B.delta_r2).abs().max())   # per configuration
assert DR2_IDENT < 1e-9
QBX = BX.loc['Qwen']; GBX = BX.loc['gemma']
TA = ['competition_math', 'lambada', 'race', 'fineweb-edu']
assert (QBX.loc[TA, 'dlo'] > 0).all()
OLDP = OLD.groupby('dataset')[['r_ent_ig_given_s', 'r_mp_ig_given_s', 'r_gap_ig_given_s']].first()

# --- code dissection
CF = pd.read_parquet(D + 'experiment_code_token_dissection/code_full_within_partials.parquet')
CP = pd.read_parquet(D + 'experiment_code_token_dissection/code_per_class_partials.parquet')
CL = pd.read_parquet(D + 'experiment_code_token_dissection/code_loco_ablation.parquet')
assert len(CF) == 40 and len(CP) == 280 and len(CL) == 280
CFa = CF.groupby('cell').agg(r=('r_within', 'mean'), npos=('r_within', pos), floor=('floor', 'first'))
CPa = CP.groupby(['cell', 'class']).agg(n=('n_tokens', 'first'), share=('share', 'first'), r=('r_within', 'mean'), npos=('r_within', pos))
CLa = CL.groupby(['cell', 'omitted_class']).agg(r=('r_within_loco', 'mean'), npos=('r_within_loco', pos))
CELLS = ['Qwen_mbpp', 'Qwen_humaneval', 'gemma_mbpp', 'gemma_humaneval']
CLASSES = ['IDENTIFIER', 'KEYWORD', 'OPERATOR_PUNCT', 'WHITESPACE_INDENT', 'NEWLINE', 'NUMBER', 'STRING_LITERAL']
# rule: does removing a single class turn Qwen positive (>=9/10 and > floor) on BOTH code sets?
def turns_pos(cell, cls):
    r = CLa.loc[(cell, cls)]; return r.npos >= 9 and r.r > CFa.loc[cell, 'floor']
ONE_CLASS = [c for c in CLASSES if turns_pos('Qwen_mbpp', c) and turns_pos('Qwen_humaneval', c)]
assert ONE_CLASS == []
ID_NEG = all(CPa.loc[(c, 'IDENTIFIER'), 'npos'] == 0 for c in CELLS)
assert ID_NEG

# --- dialogue / format
DL = pd.read_parquet(D + 'experiment_dialogue_format_decomposition/dialogue_ablation_records.parquet')
assert len(DL) == 60
DLa = DL.groupby(['model', 'dataset']).agg(floor=('floor', 'first'), raw=('r_raw', 'mean'), raw_pos=('r_raw', pos),
        nofmt=('r_no_format', 'mean'), nofmt_pos=('r_no_format', pos), top10=('r_no_top10', 'mean'), top10_pos=('r_no_top10', pos))
DJ = json.load(open(D + 'experiment_dialogue_format_decomposition/dialogue_format_decomposition_summary.json'))['top_negative_between_contributors']
TOPDS = pd.DataFrame(DJ['Qwen_dialogsum']).head(4); TOPUC = pd.DataFrame(DJ['Qwen_ultrachat']).head(4)
def det_pos(r): return r.nofmt > r.floor and r.nofmt_pos >= 9
FMT_DS = det_pos(DLa.loc[('Qwen', 'dialogsum')]); FMT_UC = det_pos(DLa.loc[('Qwen', 'ultrachat')])
assert (not FMT_DS) and FMT_UC

# --- lambada nulls
LJ = json.load(open(D + 'experiment_null_top_tokens_lambada/lambada_null_invariance_summary.json'))
LC = pd.DataFrame(LJ['configurations'])
GMM_SHARE = LC.loc[LC.config == 'GMM_standard', 'share_contractions'].iloc[0]
LAMB_MET = bool((LC.share_contractions >= 0.5 * GMM_SHARE).all())
assert LAMB_MET
LM = pd.read_parquet(D + 'experiment_null_top_tokens_lambada/lambada_jaccard_matrix.parquet')
others = [c for c in LM.columns if c != 'GMM_standard']
J_OTHERS = LM.loc[others, others].values[np.triu_indices(len(others), 1)]
J_GMM = LM.loc['GMM_standard', others].values

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
        spec['data']['datasets'].append(dict(data=[round(sg * f, 5) for f in floors], label=lab, type='line', pointRadius=9,
            pointHoverRadius=9, pointStyle='line', showLine=False, borderWidth=2.5, borderColor='rgba(226,232,240,0.95)', fill=False, order=-1))
    return spec

# ================================================================ figures 32-37
specs = []
QORD = TA + ['cnn_dailymail', 'scientific_papers', 'ultrachat', 'dialogsum', 'humaneval', 'mbpp']
specs.append(floor_band(bar_chart('fig_bx_all', QORD, {
    'HC, within | CE_small': [QBX.loc[d, 'ref'] for d in QORD], 'after the three baselines': [QBX.loc[d, 'joint'] for d in QORD]},
    colors=['#f59e0b', '#f87171']), [QBX.loc[d, 'floor'] for d in QORD]))
specs.append(bar_chart('fig_bx_r2', QORD, {'R&sup2; increment from adding HC (%)': [100 * QBX.loc[d, 'dr2'] for d in QORD]},
    y_title='% of within-string IG variance', colors=['#60a5fa']))
specs[-1]['data']['datasets'][0]['label'] = 'R² increment from adding HC (%)'
specs.append(bar_chart('fig_code_class', CLASSES, {c.replace('_', ' / ').replace('gemma', 'Gemma'): [CPa.loc[(c, k), 'r'] for k in CLASSES] for c in CELLS},
    colors=['#f59e0b', '#fbbf24', '#f87171', '#fca5a5']))
specs.append(bar_chart('fig_code_loco', ['none (full benchmark)'] + CLASSES,
    {c.replace('_', ' / ').replace('gemma', 'Gemma'): [CFa.loc[c, 'r']] + [CLa.loc[(c, k), 'r'] for k in CLASSES] for c in CELLS},
    colors=['#f59e0b', '#fbbf24', '#f87171', '#fca5a5']))
DORD = [('Qwen', 'dialogsum'), ('Qwen', 'ultrachat'), ('Qwen', 'mbpp'), ('gemma', 'dialogsum'), ('gemma', 'ultrachat'), ('gemma', 'mbpp')]
specs.append(floor_band(bar_chart('fig_fmt', [f'{"Gemma" if m == "gemma" else m} / {d}' for m, d in DORD], {
    'raw r(HC, IG)': [DLa.loc[k, 'raw'] for k in DORD], 'format tokens removed (pre-registered)': [DLa.loc[k, 'nofmt'] for k in DORD],
    'ten largest negative contributors removed': [DLa.loc[k, 'top10'] for k in DORD]}, colors=['#64748b', '#60a5fa', '#475569']),
    [DLa.loc[k, 'floor'] for k in DORD]))
sp = bar_chart('fig_lamb', [c.replace('_', ' ') for c in LC.config], {
    "share of ' and ' n' in the top ventile": list(LC.share_contractions), 'Jaccard overlap with GMM/standard': list(LC.jaccard_vs_gmm_std)},
    y_title='share / overlap', colors=['#f59e0b', '#94a3b8'])
sp['data']['datasets'].append(dict(data=[round(0.5 * GMM_SHARE, 4)] * len(LC), label='rule: half of GMM/standard share', type='line', pointRadius=0,
                                   borderWidth=1.5, borderDash=[5, 4], borderColor='rgba(248,113,113,0.8)', fill=False, order=-1))
specs.append(sp)

# ================================================================ sections 17-20
q = QBX; g = GBX
rows = []
for m, df in (('Qwen', QBX), ('Gemma', GBX)):
    for d in (QORD if m == 'Qwen' else ['race', 'cnn_dailymail', 'scientific_papers', 'ultrachat', 'dialogsum', 'humaneval', 'mbpp']):
        r = df.loc[d]
        rows.append([m, d, 'tier A' if r.tier == 'Tier-A' else 'unseen', f'&plusmn;{r.floor:.4f}', f3(r.ref), f'{r.ref_pos}/10', f3(r.joint),
                     f'{f3(r.jlo)} to {f3(r.jhi)}', pct(r.ret) if abs(r.ref) > r.floor else '&mdash;', f'{100*r.dr2:.3f}%',
                     f'{100*r.dlo:.3f} to {100*r.dhi:.3f}', f'{r.dr2_cfg}/10'])
bx_table = table(['family', 'benchmark', 'cell', 'floor', 'HC within | CE_small', 'positive', 'after 3 baselines', 'CI (after)', 'retained',
                  'R&sup2; increment', 'CI (%)', 'configs CI &gt; 0'], rows)
sec17 = f'''<section id="bx"><h2>17. Baselines, extended: every unseen cell and the R&sup2; increment</h2><div class="grid-4"><div class="stat-box"><div class="val">4 / 4</div><div class="lbl">Qwen tier-A cells: R&sup2; increment from HC excludes zero</div></div><div class="stat-box"><div class="val">{100*q.loc[TA,"dr2"].min():.2f}&ndash;{100*q.loc[TA,"dr2"].max():.2f}%</div><div class="lbl">of within-string IG variance that HC adds</div></div><div class="stat-box"><div class="val">{pct(q.loc["ultrachat","ret"])}</div><div class="lbl">ultrachat retained after baselines</div></div><div class="stat-box"><div class="val">not run</div><div class="lbl">hidden-state norm, again</div></div></div>
<p>v15 proposed completing the baseline experiment: add the hidden-state norm, run every unseen cell, and measure the within-string R&sup2; increment from adding HC to the baselines. The drop has the unseen cells (Qwen&rsquo;s six and all seven Gemma cells) and the R&sup2; increment. <b>It does not have the hidden-state norm</b>: the joint partial is the same three-baseline partial as v15, identical in the {B_REPRO[0]} shared rows. 170 of 170 rows, no missing values.</p>
<p><b>The rule, fixed in v15:</b> (1) at least half surviving all four baselines in three of four tier-A cells &rarr; the v15 finding stands; below a quarter in two or more &rarr; restate. (2) If HC&rsquo;s R&sup2; increment excludes zero in every tier-A cell, HC adds to the best simple router statistics, not only to each one. Part (1) cannot be evaluated without the norm; on three baselines it is unchanged from v15 ({pct(q.loc["competition_math","ret"])}, {pct(q.loc["lambada","ret"])}, {pct(q.loc["race","ret"])}, {pct(q.loc["fineweb-edu","ret"])}).</p>
<div class="finding scale"><h4>Part (2) is met: HC adds to the three baselines together in every tier-A cell &mdash; by about a tenth of a percent of the variance</h4><p>The increment&rsquo;s interval excludes zero in all four tier-A cells: competition_math {100*q.loc["competition_math","dr2"]:.3f}%, lambada {100*q.loc["lambada","dr2"]:.3f}%, race {100*q.loc["race","dr2"]:.3f}%, fineweb-edu {100*q.loc["fineweb-edu","dr2"]:.3f}% of within-string IG variance (means of ten configurations). Per configuration it is less uniform: the interval excludes zero in {q.loc["race","dr2_cfg"]}, {q.loc["fineweb-edu","dr2_cfg"]}, {q.loc["competition_math","dr2_cfg"]} and {q.loc["lambada","dr2_cfg"]} of ten configurations for race, fineweb-edu, competition_math and lambada.</p><p><b>What it adds.</b> The increment is the joint partial in other units: it equals r<sub>joint</sub>&sup2; &times; (1 &minus; R&sup2;<sub>baselines</sub>) exactly, in all 170 configurations. So it confirms the v15 test rather than adding an independent one. What it does add is scale: the baselines and the small model&rsquo;s loss together explain {100*q.loc[TA,"r2b"].min():.0f}&ndash;{100*q.loc[TA,"r2b"].max():.0f}% of within-string IG variance in the tier-A cells, and HC adds about 0.1% to that. That is the honest size of the claim.</p></div>
<div class="finding scale"><h4>Out of domain: three of four passes survive the baselines; the code reversals do too</h4><p>ultrachat keeps all of its association ({f3(q.loc["ultrachat","ref"])} &rarr; {f3(q.loc["ultrachat","joint"])}, interval {f3(q.loc["ultrachat","jlo"])} to {f3(q.loc["ultrachat","jhi"])}), alongside cnn_dailymail ({pct(q.loc["cnn_dailymail","ret"])}) and scientific_papers ({pct(q.loc["scientific_papers","ret"])}); dialogsum does not ({pct(q.loc["dialogsum","ret"])}, as in v15). The code reversals are not router-confidence effects: humaneval goes {f3(q.loc["humaneval","ref"])} &rarr; {f3(q.loc["humaneval","joint"])} and mbpp {f3(q.loc["mbpp","ref"])} &rarr; {f3(q.loc["mbpp","joint"])}, both intervals below zero.</p></div>
<div class="finding gemma"><h4>Gemma: every unseen cell keeps its sign after the baselines</h4><p>The inversions survive essentially intact (race {pct(g.loc["race","ret"])}, cnn_dailymail {pct(g.loc["cnn_dailymail","ret"])}, humaneval {pct(g.loc["humaneval","ret"])}, mbpp {pct(g.loc["mbpp","ret"])}, ultrachat {pct(g.loc["ultrachat","ret"])}), and so does dialogsum&rsquo;s positive part ({pct(g.loc["dialogsum","ret"])}). Whatever drives Gemma&rsquo;s scale-specific associations, it is not what entropy, maximum probability or the logit gap measure.</p></div>
<div class="finding null"><h4>The two drops disagree on how well the baselines predict IG on their own</h4><p>Each drop reports every baseline&rsquo;s own association with IG. For HC the two agree exactly; for the baselines they do not in every cell. In competition_math, maximum probability is {f3(q.loc["competition_math","p_mp"])} here and {f3(OLDP.loc["competition_math","r_mp_ig_given_s"])} in the first drop, and entropy {f3(q.loc["competition_math","p_ent"])} against {f3(OLDP.loc["competition_math","r_ent_ig_given_s"])}; in race and the unseen prose cells the two agree to within 0.01. The drops do not say how these were computed, so figure 31 in section 16 (first-drop values) and the table below can differ. The v15 reading is unaffected under either set: the simpler statistics predict IG about as well as HC or better &mdash; in competition_math, maximum probability is now {q.loc["competition_math","p_mp"]/q.loc["competition_math","ref"]:.1f}&times; HC.</p></div>
{card(32, 'HC&rsquo;s scale-specific association before and after the three baselines, every Qwen cell', 'fig_bx_all', 'Within strings, | CE<sub>small</sub>; means of ten configurations; white ticks are each cell&rsquo;s floor. The first four are the studied tier-A cells.')}
{card(33, 'What HC adds to the baselines, in variance explained', 'fig_bx_r2', 'R&sup2; increment from adding HC to a within-string model of IG on CE<sub>small</sub> and the three baselines, as a percentage of IG variance. Positive by construction; what matters is the interval in the table.', h=280)}
{bx_table}{note('&lsquo;retained&rsquo; is a ratio of means and is shown only where the reference is outside its floor. Configs CI &gt; 0 counts configurations whose R&sup2;-increment interval excludes zero.')}
<div class="caveat"><b>What this does not show.</b> The hidden-state norm, planned in v14.1 and again in v15, is still missing, so part (1) of the rule is open. And an R&sup2; increment of 0.1% is significant here because the samples are large; it is not large.</div></section>
'''

# code section
crow = []
for k in CLASSES:
    crow.append([k.lower().replace('_', ' / ')] + [f'{f3(CPa.loc[(c, k), "r"])} ({CPa.loc[(c, k), "npos"]}/10)' for c in CELLS] +
                [f'{pct(CPa.loc[(c, k), "share"])}' for c in CELLS[:2]])
code_table = table(['token class', 'Qwen / mbpp', 'Qwen / humaneval', 'Gemma / mbpp', 'Gemma / humaneval', 'share, Qwen mbpp', 'share, Qwen humaneval'], crow, num_from=1)
lrow = [[k.lower().replace('_', ' / ')] + [f'{f3(CLa.loc[(c, k), "r"])} ({CLa.loc[(c, k), "npos"]}/10)' for c in CELLS] for k in CLASSES]
loco_table = table(['class removed', 'Qwen / mbpp', 'Qwen / humaneval', 'Gemma / mbpp', 'Gemma / humaneval'], lrow, num_from=1)
sec18 = f'''<section id="code"><h2>18. Why code reverses</h2><div class="grid-4"><div class="stat-box"><div class="val">0</div><div class="lbl">single token classes whose removal turns Qwen positive on both code sets</div></div><div class="stat-box"><div class="val">4 / 4</div><div class="lbl">code cells where identifiers are negative in every configuration</div></div><div class="stat-box"><div class="val">{f3(CLa.loc[("Qwen_mbpp","IDENTIFIER"),"r"])}</div><div class="lbl">Qwen mbpp without identifiers ({CLa.loc[("Qwen_mbpp","IDENTIFIER"),"npos"]}/10 positive)</div></div><div class="stat-box"><div class="val">{f3(CLa.loc[("Qwen_humaneval","IDENTIFIER"),"r"])}</div><div class="lbl">Qwen humaneval without identifiers (0/10)</div></div></div>
<p>Both code benchmarks fail for Qwen, and Gemma&rsquo;s scale-specific part is negative on both too. v15 asked whether one token class carries it. Tokens of mbpp and humaneval were classed as identifiers, keywords, operators and punctuation, whitespace and indentation, newlines, numbers and string literals. For each class, the within-string r(HC, IG | CE<sub>small</sub>) was computed on that class alone, and on the benchmark with that class removed. Both families, ten configurations; 40, 280 and 280 rows, none missing. The full-benchmark values reproduce section 15 exactly.</p>
<p><b>The rule, fixed in v15:</b> if removing one class turns Qwen positive on both code sets, code is a token-class exception; if the negative part is spread across classes, code is a domain where last-layer routing anomalies mark tokens the large model handles worse.</p>
<div class="finding null"><h4>No single class: code is a domain effect, by the rule&rsquo;s second branch</h4><p>Removing identifiers turns Qwen&rsquo;s mbpp positive ({f3(CLa.loc[("Qwen_mbpp","IDENTIFIER"),"r"])}, {CLa.loc[("Qwen_mbpp","IDENTIFIER"),"npos"]} of ten configurations). It does not do the same for humaneval ({f3(CLa.loc[("Qwen_humaneval","IDENTIFIER"),"r"])}, none of ten), and no other removal turns humaneval positive: every one of the seven leaves it negative in every configuration. In humaneval the negative part is spread over identifiers ({f3(CPa.loc[("Qwen_humaneval","IDENTIFIER"),"r"])}), operators and punctuation ({f3(CPa.loc[("Qwen_humaneval","OPERATOR_PUNCT"),"r"])}) and whitespace ({f3(CPa.loc[("Qwen_humaneval","WHITESPACE_INDENT"),"r"])}). So by the rule, code is reported as a domain where high-HC occurrences of a token are ones the large model handles worse, holding the small model&rsquo;s loss fixed.</p></div>
<div class="finding scale"><h4>Identifiers are negative in every code cell, in both families</h4><p>Within identifiers alone the association is negative in all four cells and all forty configurations: Qwen mbpp {f3(CPa.loc[("Qwen_mbpp","IDENTIFIER"),"r"])}, Qwen humaneval {f3(CPa.loc[("Qwen_humaneval","IDENTIFIER"),"r"])}, Gemma mbpp {f3(CPa.loc[("gemma_mbpp","IDENTIFIER"),"r"])}, Gemma humaneval {f3(CPa.loc[("gemma_humaneval","IDENTIFIER"),"r"])}. Identifiers are 36&ndash;41% of code tokens. In a benchmark of short functions, most identifiers repeat names already written in the prompt or the signature, which both models can copy; that is one candidate for why routing anomalies on them do not mark extra gain for the large model. It is a hypothesis, and section 31 proposes the test. Keywords are negative in three of four cells.</p></div>
{card(34, 'The scale-specific association within each code token class', 'fig_code_class', 'Within-string r(HC, IG | CE<sub>small</sub>) computed on each class alone; means of ten configurations. Identifiers (first group) are negative in all four cells.')}
{card(35, 'The full benchmark with one class removed', 'fig_code_loco', 'First group: the full benchmark. Only one removal changes a sign: identifiers in Qwen/mbpp.')}
{code_table}{note('Means of ten configurations; in brackets, the number of configurations above zero.')}
{loco_table}
<div class="caveat"><b>What this does not show.</b> The per-class and leave-one-out values are not additive: the small-model control is re-estimated on each subset, so the full-benchmark value is not a weighted average of the class values. That is why removing identifiers can turn mbpp positive although the other classes are mostly negative on their own. Two code benchmarks, both Python.</div></section>
'''

# format section
dsr, ucr = DLa.loc[('Qwen', 'dialogsum')], DLa.loc[('Qwen', 'ultrachat')]
frow = [[('Gemma' if m == 'gemma' else m), d, f'&plusmn;{DLa.loc[(m,d),"floor"]:.4f}', f'{f3(DLa.loc[(m,d),"raw"])} ({DLa.loc[(m,d),"raw_pos"]}/10)',
         f'{f3(DLa.loc[(m,d),"nofmt"])} ({DLa.loc[(m,d),"nofmt_pos"]}/10)', f'{f3(DLa.loc[(m,d),"top10"])} ({DLa.loc[(m,d),"top10_pos"]}/10)'] for m, d in DORD]
fmt_table = table(['family', 'benchmark', 'floor', 'raw r', 'format tokens removed', 'ten largest negative contributors removed'], frow)
def toks(df):
    return ', '.join(f'&lsquo;{t}&rsquo; ({"flagged" if f else "not flagged"}; HC {h:.1f}, IG {i:+.2f})' for t, f, h, i in zip(df.token_str, df.is_format, df.hc_mean, df.ig_mean))
sec19 = f'''<section id="format"><h2>19. The raw sign on dialogue and chat: format tokens</h2><div class="grid-4"><div class="stat-box"><div class="val">{f3(ucr.nofmt)}</div><div class="lbl">Qwen ultrachat raw r, format tokens removed ({ucr.nofmt_pos}/10 positive)</div></div><div class="stat-box"><div class="val">{f3(dsr.nofmt)}</div><div class="lbl">Qwen dialogsum, format tokens removed ({dsr.nofmt_pos}/10)</div></div><div class="stat-box"><div class="val">1 of 4</div><div class="lbl">pieces of dialogsum&rsquo;s speaker tag flagged as format</div></div><div class="stat-box"><div class="val">3 / 3</div><div class="lbl">Gemma cells still negative without format tokens</div></div></div>
<p>On dialogsum and ultrachat, Qwen&rsquo;s within-string association is positive but its raw correlation is negative or null, so the part between token types is negative. v15 asked which token strings carry it. The drop decomposes r(HC, IG) into between- and within-string contributions per string for dialogsum, ultrachat and mbpp, both families, and recomputes the raw correlation three ways: without tokens flagged as formatting, without the ten largest negative contributors, and without the twenty largest. Sixty rows, none missing.</p>
<p><b>The rule, fixed in v15:</b> if removing format tokens turns Qwen&rsquo;s raw r positive on dialogsum <i>and</i> ultrachat, the reversal is a formatting effect; if it stays negative, it is about content.</p>
<div class="finding scale"><h4>ultrachat: a formatting effect</h4><p>Without format tokens Qwen&rsquo;s raw r on ultrachat is {f3(ucr.nofmt)}, positive in {ucr.nofmt_pos} of ten configurations and outside the floor (&plusmn;{ucr.floor:.4f}); with them it was {f3(ucr.raw)}. The largest negative contributors are the chat-role tokens and line-ending punctuation: {toks(TOPUC)}. Role tokens get some of the highest HC values in the benchmark, and the large model is <i>worse</i> on them than the small one.</p></div>
<div class="finding null"><h4>dialogsum: the test as run is inconclusive, because the format list was incomplete</h4><p>Without flagged format tokens Qwen&rsquo;s dialogsum raw r is {f3(dsr.nofmt)}: positive on average, just outside the floor, but only {dsr.nofmt_pos} of ten configurations agree, so the sign is undetermined by the series&rsquo; criterion. Strictly, the rule is not met. The reason is visible in the contributors. dialogsum marks speakers as <code>#Person1#:</code>, which the tokenizer splits into four pieces, and the four largest negative contributors are exactly those pieces: {toks(TOPDS)}. Only &lsquo;Person&rsquo; was flagged. The pre-registered test therefore removed one of the four pieces of the tag.</p><p>Removing the ten largest negative contributors &mdash; which include all four pieces &mdash; gives {f3(dsr.top10)} ({dsr.top10_pos} of ten). That number is not evidence: those ten strings were chosen for pushing the correlation down, so removing them raises it by construction, and six of them are ordinary words. Removing the complete tag and nothing else is the test that settles it, and the drop has what it needs (section 31).</p></div>
<div class="finding gemma"><h4>Gemma&rsquo;s raw inversion is not a formatting effect</h4><p>Without format tokens Gemma stays negative on dialogsum ({f3(DLa.loc[("gemma","dialogsum"),"nofmt"])}), ultrachat ({f3(DLa.loc[("gemma","ultrachat"),"nofmt"])}) and mbpp ({f3(DLa.loc[("gemma","mbpp"),"nofmt"])}), in every configuration. Even without its ten largest negative contributors it stays negative on all three.</p></div>
{card(36, 'Raw r(HC, IG) with format tokens removed', 'fig_fmt', 'Means of ten configurations; white ticks are floors. Dark bars remove strings selected for their negative contribution, and are shown only for scale.')}
{fmt_table}{note('In brackets: configurations above zero. mbpp has no strings flagged as format among its largest contributors; its largest is &lsquo;function&rsquo;, from the problem statement.')}
</section>
'''

# lambada section
lrows = [[r.config.replace('_', ' / '), f'{r.surge:+.2f}', f'{pct(r.share_apostrophe)}', f'{pct(r.share_n)}', f'{pct(r.share_contractions)}',
          f'{pct(r.top10_share)}', f'{r.jaccard_vs_gmm_std:.2f}'] for r in LC.itertuples()]
lamb_table = table(['configuration', 'surge (nats)', "share of &lsquo; '&rsquo;", "share of &lsquo; n&rsquo;", 'both', 'ten most frequent strings', 'Jaccard vs GMM/standard'], lrows, num_from=1)
sec20 = f'''<section id="lambnull"><h2>20. Lambada&rsquo;s top tokens under every null</h2><div class="grid-4"><div class="stat-box"><div class="val">10 / 10</div><div class="lbl">configurations where the two contraction strings keep at least half their GMM share</div></div><div class="stat-box"><div class="val">{pct(LC.share_contractions.min())}&ndash;{pct(LC.share_contractions.max())}</div><div class="lbl">their share of the top ventile</div></div><div class="stat-box"><div class="val">{J_GMM.min():.2f}&ndash;{J_GMM.max():.2f}</div><div class="lbl">overlap of GMM/standard with the other nine</div></div><div class="stat-box"><div class="val">{J_OTHERS.min():.2f}&ndash;{J_OTHERS.max():.2f}</div><div class="lbl">overlap among the other nine</div></div></div>
<p>v13 described lambada&rsquo;s top HC ventile as dominated by two near-exclusive strings, the apostrophe and &lsquo; n&rsquo; of split contractions, from GMM/standard alone. v15 (carried from v13) asked whether that description holds under every null. The top ventile was listed for all ten configurations.</p>
<p><b>The rule:</b> if those two strings keep at least half their GMM share under every null, the description holds across nulls.</p>
<div class="finding scale"><h4>Met, and GMM/standard is the configuration that understates it</h4><p>The two strings make up {pct(GMM_SHARE)} of the top ventile under GMM/standard and {pct(LC.share_contractions[LC.config != "GMM_standard"].min())}&ndash;{pct(LC.share_contractions.max())} under the other nine. The surge in IG at the top is positive in all ten ({LC.surge.min():+.2f} to {LC.surge.max():+.2f} nats). The other nine agree with one another more than any of them agrees with GMM/standard (Jaccard {J_OTHERS.min():.2f}&ndash;{J_OTHERS.max():.2f} among themselves, {J_GMM.min():.2f}&ndash;{J_GMM.max():.2f} with it). So lambada&rsquo;s token-type description is not an artefact of the null; if anything, the series&rsquo; reference configuration is the outlier. That contrasts with competition_math (v13), where the top-ventile tokens depended strongly on the null.</p></div>
{card(37, 'Lambada&rsquo;s top ventile under each configuration', 'fig_lamb', 'Amber: share of the two contraction strings. Grey: overlap of the top-ventile token set with GMM/standard&rsquo;s. Dashed red: the rule&rsquo;s threshold.', h=320)}
{lamb_table}
<div class="caveat"><b>What this does not show.</b> It is about which tokens reach the top, not about the correlation; the Qwen3-8B partner, the other half of this P2 item, was not run.</div></section>
'''

# ================================================================ insert and renumber
i = s.index('<section id="shared"')
NEW = sec17 + sec18 + sec19 + sec20
s = s[:i] + NEW + s[i:]
NEW_IDS = ('bx', 'code', 'format', 'lambnull')
def shift_h2(m):
    sid, num, rest = m.group(1), int(m.group(2)), m.group(3)
    if sid in NEW_IDS or num < 17: return m.group(0)
    return f'<section id="{sid}"><h2>{num + 4}.{rest}'
s = re.sub(r'<section id="([^"]+)"><h2>(\d+)\.([^<]*)', shift_h2, s)
be = s.index('<script>\nconst CHART_SPECS'); body, tail = s[:be], s[be:]
a, b = body.index('<section id="bx"'), body.index('<section id="shared"')
def fs(m): n = int(m.group(1)); return m.group(0) if n < 32 else f'Figure {n + 6}'
def ss(m): n = int(m.group(2)); return m.group(0) if n < 17 else f'{m.group(1)}{n + 4}'
pre = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[:a]))
post = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[b:]))
s = pre + body[a:b] + post + tail
nav = re.search(r'<a href="#baselines">[^<]*</a>', s).group(0)
s = s.replace(nav, nav + '<a href="#bx">Baselines, extended</a><a href="#code">Why code reverses</a><a href="#format">Format tokens</a><a href="#lambnull">Lambada under every null</a>', 1)
m = re.search(r'const CHART_SPECS = (\[.*?\]);\nfunction hideUnderscore', s, flags=re.S)
s = s[:m.start(1)] + json.dumps(json.loads(m.group(1)) + specs) + s[m.end(1):]

json.dump(dict(B_REPRO=B_REPRO, DR2_IDENT=DR2_IDENT, FMT_DS=FMT_DS, FMT_UC=FMT_UC, GMM_SHARE=GMM_SHARE,
               QBX=QBX.drop(columns='tier').round(6).to_dict(), GBX=GBX.drop(columns='tier').round(6).to_dict(),
               DLa={f'{k[0]}|{k[1]}': v for k, v in DLa.round(5).to_dict('index').items()},
               CLa={f'{k[0]}|{k[1]}': v for k, v in CLa.round(5).to_dict('index').items()},
               CPa={f'{k[0]}|{k[1]}': v for k, v in CPa.round(5).to_dict('index').items()},
               CFa=CFa.round(5).to_dict('index'), J=[float(J_OTHERS.min()), float(J_OTHERS.max()), float(J_GMM.min()), float(J_GMM.max())],
               LCmin=float(LC.share_contractions.min()), LCmax=float(LC.share_contractions.max())),
          open('/home/claude/v16/n16.json', 'w'), indent=1, default=float)
open('/home/claude/v16/stage1.html', 'w', encoding='utf-8').write(s)
print('stage1 ok', len(s), 'DR2 identity max dev', DR2_IDENT)
