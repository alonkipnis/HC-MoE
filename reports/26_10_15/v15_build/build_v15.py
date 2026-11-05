"""Build HC-MoE v15 from v14.1 plus the 26_10_15 drop (three P0 experiments).

Every new number is recomputed here from the drop's CSVs; the drop's decision
summaries are not used (they average per-configuration ratios).
"""
import json, re, sys
import numpy as np, pandas as pd

sys.path.insert(0, [p for p in __import__('glob').glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')][0])
from report_builder import bar_chart

UP = '/mnt/user-data/uploads/HC-MoE-Alon/reports/'
s = open('/home/claude/v15/v141.html', encoding='utf-8').read()

# ---------------------------------------------------------------- data
W = pd.read_csv(UP + '26_10_15/experiment_widen_unseen_sample/widen_unseen_all_configurations.csv')
B = pd.read_csv(UP + '26_10_15/experiment_last_layer_baselines/last_layer_baselines_all_configurations.csv')
O = pd.read_csv(UP + '26_10_11/experiment_unseen_benchmarks/unseen_benchmarks_all_configurations.csv')
assert len(W) == 120 and W.groupby(['model', 'dataset']).size().eq(10).all() and not W.isna().any().any()
assert len(B) == 80 and B.groupby(['model', 'dataset']).size().eq(10).all() and not B.isna().any().any()

key = ['model', 'dataset', 'estimator', 'variant']
M = O.merge(W, on=key, suffixes=('_o', '_w'))
REPRO_ROWS = len(M)
REPRO_MAX = max((M[c + '_o'] - M[c + '_w']).abs().max() for c in
                ['r_raw', 'r_spearman', 'raw_part_b', 'raw_part_s', 'r_within', 'within_part_b', 'within_part_s'])

def wcell(g):
    f = g.measured_floor.iloc[0]
    return pd.Series(dict(
        n=int(g.n_tokens.iloc[0]), ess=float(g.ess.iloc[0]), defl=g.n_tokens.iloc[0] / g.ess.iloc[0], floor=f,
        raw=g.r_raw.mean(), raw_lo=g.r_raw_ci_lo.mean(), raw_hi=g.r_raw_ci_hi.mean(), raw_pos=int((g.r_raw > 0).sum()),
        sp=g.r_spearman.mean(), within=g.r_within.mean(),
        wB=g.within_part_b.mean(), wB_pos=int((g.within_part_b > 0).sum()),
        wS=g.within_part_s.mean(), wS_pos=int((g.within_part_s > 0).sum()),
        wS_lo=g.within_part_s_ci_lo.mean(), wS_hi=g.within_part_s_ci_hi.mean()))
WC = W.groupby(['model', 'dataset']).apply(wcell, include_groups=False)

def passes(r):  # v14.1 rule: positive & outside floor under both controls within strings, >= 9/10 agreeing
    return (r.wS > r.floor) and (r.wB > r.floor) and r.wS_pos >= 9 and r.wB_pos >= 9
QW = WC.loc['Qwen'].copy(); QW['pass'] = QW.apply(passes, axis=1)
GW = WC.loc['gemma'].copy()
N_PASS = int(QW['pass'].sum())
assert N_PASS == 4 and set(QW.index[QW['pass']]) == {'cnn_dailymail', 'dialogsum', 'scientific_papers', 'ultrachat'}
assert (GW.raw_pos == 0).all() and (GW.raw < -GW.floor).all()

def bcell(g):
    rs, rj = g.r_ref_s.mean(), g.r_joint_s.mean()
    return pd.Series(dict(
        tier=g.tier.iloc[0], n=int(g.n_tokens.iloc[0]), layer=int(g.layer.iloc[0]),
        ref=rs, joint=rj, ret=rj / rs, jlo=g.r_joint_s_ci_lo.mean(), jhi=g.r_joint_s_ci_hi.mean(),
        ret_lo=g.ret_joint_ci_lo.median(), ret_hi=g.ret_joint_ci_hi.median(),
        ret_ent=g.r_part_ent.mean() / rs, ret_mp=g.r_part_mp.mean() / rs, ret_gap=g.r_part_gap.mean() / rs,
        sign_kept=int((np.sign(g.r_joint_s) == np.sign(rs)).sum()), cfg50=int((g.ret_joint >= 0.5).sum()),
        ret_b=g.r_joint_b.mean() / g.r_ref_b.mean(),
        hc_ent=g.r_hc_ent.mean(), hc_mp=g.r_hc_mp.mean(), hc_gap=g.r_hc_gap.mean(),
        ent_ig=g.r_ent_ig_given_s.mean(), mp_ig=g.r_mp_ig_given_s.mean(), gap_ig=g.r_gap_ig_given_s.mean(),
        two_sided=g.r_ref_2s.abs().max()))
BC = B.groupby(['model', 'dataset']).apply(bcell, include_groups=False)
QA = BC.loc['Qwen'][BC.loc['Qwen'].tier == 'Tier-A']
assert len(QA) == 4 and int((QA.ret >= 0.5).sum()) == 3 and (QA.jlo > 0).all()
DS_RET = BC.loc[('Qwen', 'dialogsum'), 'ret']
# within-string values reproduce v13's (0.027, 0.035, 0.073, 0.069; Gemma race -0.031)
for d, v in dict(competition_math=0.027, lambada=0.035, **{'fineweb-edu': 0.073}, race=0.069).items():
    assert abs(QA.loc[d, 'ref'] - v) < 0.0006, d
assert abs(BC.loc[('gemma', 'race'), 'ref'] + 0.031) < 0.0006

DEFL_MED = float(np.median(WC.defl))

# ---------------------------------------------------------------- helpers
def f3(x, d=3):
    return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100 * x:.0f}%'
def esc_name(d): return d.replace('_', '_')
NICE = dict(cnn_dailymail='cnn_dailymail (news)', scientific_papers='scientific_papers', ultrachat='ultrachat (chat)',
            dialogsum='dialogsum (dialogue)', humaneval='humaneval (code)', mbpp='mbpp (code)')
ORDER = ['cnn_dailymail', 'scientific_papers', 'ultrachat', 'dialogsum', 'humaneval', 'mbpp']

def card(fig_no, title, canvas, caption, h=340):
    return (f'<div class="card"><h3 style="margin-top:0">Figure {fig_no} &mdash; {title}</h3>'
            f'<div style="height:{h}px;position:relative"><canvas id="{canvas}"></canvas></div>'
            f'<p style="font-size:0.85rem;color:#64748b;margin-top:0.75rem">{caption}</p></div>\n')

def floor_lines(spec, floors):
    for lab, sign in (('measured floor (per benchmark)', 1), ('_floor_lo', -1)):
        spec['data']['datasets'].append(dict(
            data=[round(sign * f, 5) for f in floors], label=lab, type='line', pointRadius=9, pointHoverRadius=9, pointStyle='line',
            showLine=False, borderWidth=2.5, borderColor='rgba(226,232,240,0.95)', fill=False, order=-1))
    return spec

def table(head, rows, num_from=2):
    th = ''.join(f'<th class="num">{h}</th>' if i >= num_from else f'<th>{h}</th>' for i, h in enumerate(head))
    trs = ''.join('<tr>' + ''.join(f'<td class="num">{c}</td>' if i >= num_from else f'<td>{c}</td>'
                                   for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'

def note(t): return f'<p style="font-size:0.85rem;color:#64748b;margin:0.5rem 0 0">{t}</p>\n'

# ---------------------------------------------------------------- figures (28-31 new)
new_specs = []
for fam, df, cid in (('Qwen', QW, 'fig_oodw_qwen'), ('Gemma', GW, 'fig_oodw_gemma')):
    labs = [NICE[d] for d in ORDER]
    sp = bar_chart(cid, labs, {'raw r(HC, IG)': [df.loc[d, 'raw'] for d in ORDER],
                               'within strings | CE_big': [df.loc[d, 'wB'] for d in ORDER],
                               'within strings | CE_small': [df.loc[d, 'wS'] for d in ORDER]},
                   colors=['#64748b', '#60a5fa', '#f59e0b'])
    new_specs.append(floor_lines(sp, [df.loc[d, 'floor'] for d in ORDER]))

bl = ['competition_math', 'lambada', 'race', 'fineweb-edu', 'cnn_dailymail', 'scientific_papers', 'dialogsum']
bl_lab = [d + (' *' if d in ('cnn_dailymail', 'scientific_papers', 'dialogsum') else '') for d in bl]
QB = BC.loc['Qwen']
sp = bar_chart('fig_base_ret', bl_lab, {
    'entropy only': [QB.loc[d, 'ret_ent'] for d in bl], 'max probability only': [QB.loc[d, 'ret_mp'] for d in bl],
    'logit gap only': [QB.loc[d, 'ret_gap'] for d in bl], 'all three': [QB.loc[d, 'ret'] for d in bl]},
    y_title='share of within-string r(HC, IG | CE_small) retained', colors=['#94a3b8', '#a78bfa', '#2dd4bf', '#f59e0b'])
sp['data']['datasets'].append(dict(data=[0.5] * len(bl), label='pre-registered line (50%)', type='line', pointRadius=0,
                                   borderWidth=1.5, borderDash=[5, 4], borderColor='rgba(248,113,113,0.8)', fill=False, order=99))
new_specs.append(sp)
pr = ['competition_math', 'lambada', 'race', 'fineweb-edu']
new_specs.append(bar_chart('fig_base_pred', pr, {
    'HC': [QB.loc[d, 'ref'] for d in pr], 'router entropy': [QB.loc[d, 'ent_ig'] for d in pr],
    'max router probability': [QB.loc[d, 'mp_ig'] for d in pr], 'top-1 minus top-2 logit gap': [QB.loc[d, 'gap_ig'] for d in pr],
    'HC after all three': [QB.loc[d, 'joint'] for d in pr]},
    y_title='within-string r(·, IG | CE_small)', colors=['#f59e0b', '#94a3b8', '#a78bfa', '#2dd4bf', '#f87171']))

# ---------------------------------------------------------------- section: widened out-of-domain
q, g = QW, GW
rows = []
for fam, df in (('Qwen', q), ('Gemma', g)):
    for d in ORDER:
        r = df.loc[d]
        verdict = ('' if fam == 'Gemma' else ('<b>pass</b>' if r['pass'] else 'fail'))
        rows.append([fam, NICE[d], f'{int(r.n):,}', f'{r.ess:,.0f}', f'{r.defl:.2f}&times;', f'&plusmn;{r.floor:.4f}',
                     f3(r.raw), f'{int(r.raw_pos)}/10', f3(r.sp), f3(r.wB), f'{int(r.wB_pos)}/10', f3(r.wS), f'{int(r.wS_pos)}/10',
                     f'{f3(r.wS_lo)} to {f3(r.wS_hi)}', verdict])
widen_table = table(['family', 'benchmark', 'tokens', 'measured ESS', 'deflation', 'floor', 'raw r', 'raw positive',
                     'Spearman', 'within | CE_big', 'positive', 'within | CE_small', 'positive', 'CI (| CE_small)', 'v14.1 rule'],
                    rows, num_from=2)

prev = {('Qwen', 'dialogsum'): 85010, ('Qwen', 'mbpp'): 21643, ('gemma', 'dialogsum'): 85927, ('gemma', 'mbpp'): 18254}
prev_floor = {('Qwen', 'dialogsum'): 0.0067, ('Qwen', 'mbpp'): 0.0133, ('gemma', 'dialogsum'): 0.0067, ('gemma', 'mbpp'): 0.0145}
ess_rows = [[('Qwen' if m == 'Qwen' else 'Gemma'), d, f'{prev[(m, d)]:,}', f'{WC.loc[(m, d), "ess"]:,.0f}',
             f'&plusmn;{prev_floor[(m, d)]:.4f}', f'&plusmn;{WC.loc[(m, d), "floor"]:.4f}',
             f'{int(WC.loc[(m, d), "n"]):,}'] for (m, d) in prev]
ess_table = table(['family', 'benchmark', 'ESS inferred (v14)', 'ESS measured (v15)', 'floor inferred', 'floor measured', 'tokens'], ess_rows)

sec_oodw = f'''<section id="oodwide"><h2>15. Out of domain, widened: six unseen benchmarks</h2><div class="grid-4"><div class="stat-box"><div class="val">{N_PASS} / 6</div><div class="lbl">Qwen benchmarks meeting the prediction (both controls, within strings)</div></div><div class="stat-box"><div class="val">0 / 2</div><div class="lbl">Qwen passes on code (mbpp, humaneval)</div></div><div class="stat-box"><div class="val">6 / 6</div><div class="lbl">Gemma raw r(HC, IG) negative, every configuration</div></div><div class="stat-box"><div class="val">&plusmn;{q.loc["dialogsum","floor"]:.4f}</div><div class="lbl">dialogsum floor, measured (v14 inferred &plusmn;0.0067)</div></div></div>
<p>v14.1 set two P0 experiments on unseen text. The first completed the v14 drop: measured effective sample sizes (the 26_09_03 autocorrelation method), token counts and raw-level bootstrap intervals for mbpp and dialogsum. The second widened the sample with four benchmarks fixed before looking &mdash; <b>cnn_dailymail</b> (news), <b>scientific_papers</b>, <b>humaneval</b> (a second code set) and <b>ultrachat</b> (open-ended chat). Same pipeline as v14: both families, five estimators, two variants, raw and within-token-string correlations under both difficulty controls. 120 of 120 rows present, no missing values. Every cell is graded against its own measured floor.</p>
<p><b>The prediction, fixed in v14.1:</b> Qwen positive and outside the floor under both controls within strings, read with the series' consistency criterion (at least 9 of 10 configurations agreeing). <b>The rule:</b> at least four of six &rarr; domain-general with named exceptions; two or fewer &rarr; specific to the studied benchmark types.</p>
<div class="finding scale"><h4>Qwen meets the prediction on four of six &mdash; every benchmark except the two code sets</h4><p><b>Passes.</b> cnn_dailymail ({f3(q.loc["cnn_dailymail","wS"])} under | CE<sub>small</sub>, {f3(q.loc["cnn_dailymail","wB"])} under | CE<sub>big</sub>, all ten configurations), scientific_papers ({f3(q.loc["scientific_papers","wS"])} and {f3(q.loc["scientific_papers","wB"])}, all ten), ultrachat ({f3(q.loc["ultrachat","wS"])} and {f3(q.loc["ultrachat","wB"])}, all ten) and dialogsum ({f3(q.loc["dialogsum","wS"])} in nine of ten and {f3(q.loc["dialogsum","wB"])}). The first two are comparable to Qwen's studied tier-A cells within strings (+0.027 to +0.073). ultrachat and dialogsum pass by less than one floor width.</p><p><b>Failures.</b> Both code benchmarks. mbpp reverses ({f3(q.loc["mbpp","wS"])} under | CE<sub>small</sub>, eight of ten negative), as in v14. humaneval is negative in all ten configurations ({f3(q.loc["humaneval","wS"])}) and sits on its floor (&plusmn;{q.loc["humaneval","floor"]:.4f}) &mdash; its interval, {f3(q.loc["humaneval","wS_lo"])} to {f3(q.loc["humaneval","wS_hi"])}, excludes zero. On code, holding the small model's loss fixed, high-HC occurrences of a token are ones the large model handles <i>worse</i>.</p><p><b>By the rule</b> the claim is restated as domain-general with one named exception, code. That is the first branch, met with no slack: one more failure would have left the result between the two branches.</p></div>
<div class="finding null"><h4>What carries over is the within-string association, not the raw sign</h4><p>Qwen's raw r(HC, IG) is positive on cnn_dailymail ({f3(q.loc["cnn_dailymail","raw"])}), scientific_papers ({f3(q.loc["scientific_papers","raw"])}) and humaneval ({f3(q.loc["humaneval","raw"])}), negative on dialogsum ({f3(q.loc["dialogsum","raw"])}) and mbpp ({f3(q.loc["mbpp","raw"])}), and undetermined on ultrachat ({f3(q.loc["ultrachat","raw"])}, two of ten positive). On the nine studied benchmarks it was positive everywhere. On dialogsum and ultrachat the within-string association is positive while the raw one is negative, so the between-token-type component is negative there; on humaneval the raw correlation is positive while the scale-specific part is negative. The part of the association that generalises is the one the series has treated as the scale-specific claim since v11 &mdash; same token string, either model's loss held fixed &mdash; and the raw correlation is the part that depends on which token types the text contains.</p></div>
<div class="finding gemma"><h4>Gemma: the raw inversion holds everywhere; the scale-specific part does not</h4><p>Raw r(HC, IG) is negative on all six benchmarks in all ten configurations: cnn_dailymail {f3(g.loc["cnn_dailymail","raw"])}, scientific_papers {f3(g.loc["scientific_papers","raw"])}, ultrachat {f3(g.loc["ultrachat","raw"])}, dialogsum {f3(g.loc["dialogsum","raw"])}, mbpp {f3(g.loc["mbpp","raw"])} and humaneval {f3(g.loc["humaneval","raw"])}, the strongest Gemma cell the series has measured (ESS {g.loc["humaneval","ess"]:,.0f}, floor &plusmn;{g.loc["humaneval","floor"]:.4f}). Within strings the large-model control is negative on all six as well. The small-model control is not: strictly negative on cnn_dailymail ({f3(g.loc["cnn_dailymail","wS"])}, nine of ten) and humaneval ({f3(g.loc["humaneval","wS"])}, ten of ten); positive on dialogsum ({f3(g.loc["dialogsum","wS"])}, ten of ten); null on scientific_papers ({f3(g.loc["scientific_papers","wS"])}); negative but split on mbpp and ultrachat (eight of ten each). Gemma's raw inversion is the most general result in the series. Its scale-specific part is still the least settled.</p></div>
{card(28, 'Qwen on six unseen benchmarks: raw and within-string correlations', 'fig_oodw_qwen', 'Means of ten configurations. The prediction concerns the blue and amber bars; grey ticks are each benchmark&rsquo;s measured floor. The four left-hand benchmarks pass; the two code sets fail on the amber bar.')}
{card(29, 'Gemma on the same six benchmarks', 'fig_oodw_gemma', 'Raw and | CE<sub>big</sub> are negative everywhere; the | CE<sub>small</sub> part is positive on dialogsum and null on scientific_papers.')}
{widen_table}{note('Means of ten configurations; &lsquo;positive&rsquo; counts configurations above zero. CI is the passage-bootstrap interval averaged over configurations. Floors and ESS are measured per cell. The rule column applies only to Qwen, for which the prediction was written.')}
<h3>Completing the v14 drop</h3>
<p>The v14 results rested on floors inferred from bootstrap interval widths. Measured directly, the inference was close: within 8% for Qwen and 5&ndash;36% for Gemma, with mbpp's Gemma floor somewhat smaller than inferred. <b>dialogsum's measured floor is &plusmn;{q.loc["dialogsum","floor"]:.4f}</b>, and Qwen's within-string | CE<sub>small</sub> value of {f3(q.loc["dialogsum","wS"], 4)} stays outside it with nine of ten configurations positive. By the v14.1 rule, the letter of the v13 pass stands. The {REPRO_ROWS} rows this drop shares with the 26_10_11 drop are identical (largest difference {'zero' if REPRO_MAX == 0 else f'{REPRO_MAX:.0e}'}), so every v14 number stands as published. Identical output shows the two runs used the same inputs; the drop does not say whether the Gemma loss files were regenerated, so it confirms consistency rather than independently confirming that those inputs were complete.</p>
{ess_table}
<div class="caveat"><b>What this does not show.</b> Two code benchmarks, both failing, are enough to name code as an exception and not enough to say why. humaneval is the smallest cell (ESS {q.loc["humaneval","ess"]:,.0f}) and its failure sits on the floor. Two of the four passes (ultrachat, dialogsum) are about a quarter the size of Qwen's studied cells within strings, and section 16 shows that dialogsum's does not survive simpler router statistics. These cells are graded by the pre-registered rule only; ARF and tiers were not computed for them.</div></section>
'''

# ---------------------------------------------------------------- section: baselines
QAo = ['competition_math', 'lambada', 'race', 'fineweb-edu']
brow = []
for (m, d) in [('Qwen', x) for x in QAo] + [('Qwen', 'cnn_dailymail'), ('Qwen', 'scientific_papers'), ('Qwen', 'dialogsum'), ('gemma', 'race')]:
    r = BC.loc[(m, d)]
    brow.append(['Qwen' if m == 'Qwen' else 'Gemma', d, 'tier A' if r.tier == 'Tier-A' else 'unseen', f'{int(r.n):,}',
                 f3(r.ref), f3(r.joint), f'{f3(r.jlo)} to {f3(r.jhi)}', pct(r.ret), f'{pct(r.ret_lo)}&ndash;{pct(r.ret_hi)}',
                 pct(r.ret_ent), pct(r.ret_mp), pct(r.ret_gap), f'{int(r.cfg50)}/10', pct(r.ret_b), f'{r.hc_ent:+.2f}'])
base_table = table(['family', 'benchmark', 'cell', 'tokens', 'HC, within | CE_small', 'after all three', 'CI (after)',
                    'retained', 'retained, CI', 'entropy only', 'max prob only', 'logit gap only', 'configs &ge; 50%',
                    'retained | CE_big', 'corr(HC, entropy)'], brow, num_from=3)

qa = QA
sec_base = f'''<section id="baselines"><h2>16. Beyond simpler last-layer router signals</h2><div class="grid-4"><div class="stat-box"><div class="val">3 / 4</div><div class="lbl">Qwen tier-A cells keeping &ge; 50% after all baselines</div></div><div class="stat-box"><div class="val">4 / 4</div><div class="lbl">remainder&rsquo;s CI excludes zero</div></div><div class="stat-box"><div class="val">{pct(qa.ret.min())}&ndash;{pct(qa.ret.max())}</div><div class="lbl">retention range, tier-A cells</div></div><div class="stat-box"><div class="val">{pct(DS_RET)}</div><div class="lbl">dialogsum retained (its pass is absorbed)</div></div></div>
<p>HC is computed from the router of the last MoE layer, the one feeding the output head (layer {int(BC.loc[("Qwen","race"),"layer"])} of Qwen3-30B-A3B, {int(BC.loc[("gemma","race"),"layer"])} of gemma-4-26B-A4B). The question set in v14.1: does it carry anything that simpler statistics of the same router do not? Three baselines were computed per token: <b>router entropy</b>, the <b>maximum router probability</b>, and the <b>top-1 minus top-2 logit gap</b>. The statistic is the series&rsquo; scale-specific one, r(HC, IG | CE<sub>small</sub>) within token strings, partialled on each baseline and on all three together, for Qwen&rsquo;s four tier-A cells, Gemma&rsquo;s race and three of Qwen&rsquo;s unseen cells. Ten configurations, passage bootstrap. The reference values reproduce v13&rsquo;s within-string numbers to three decimals in all five studied cells.</p>
<p><b>The rule, fixed in v14.1:</b> if at least half survives all baselines in three or more of Qwen&rsquo;s four tier-A cells, HC carries information the simpler signals do not; if less than a quarter survives, the result is restated in terms of the baseline that absorbs it.</p>
<div class="finding scale"><h4>Met in three of four cells; the remainder is outside zero in all four</h4><p>Retained after all three baselines: competition_math {pct(qa.loc["competition_math","ret"])} ({f3(qa.loc["competition_math","ref"])} &rarr; {f3(qa.loc["competition_math","joint"])}), lambada {pct(qa.loc["lambada","ret"])} ({f3(qa.loc["lambada","ref"])} &rarr; {f3(qa.loc["lambada","joint"])}), race {pct(qa.loc["race","ret"])} ({f3(qa.loc["race","ref"])} &rarr; {f3(qa.loc["race","joint"])}), fineweb-edu {pct(qa.loc["fineweb-edu","ret"])} ({f3(qa.loc["fineweb-edu","ref"])} &rarr; {f3(qa.loc["fineweb-edu","joint"])}). The rule&rsquo;s first branch holds. No cell falls toward the quarter line, the sign is kept in every configuration of every cell, and the remaining partial&rsquo;s interval excludes zero in all four.</p><p><b>How firmly.</b> Only competition_math clears 50% with room: every configuration does, and its interval is {pct(qa.loc["competition_math","ret_lo"])}&ndash;{pct(qa.loc["competition_math","ret_hi"])}. lambada and race clear it by 6 and 4 points, with intervals that cross the line ({pct(qa.loc["lambada","ret_lo"])}&ndash;{pct(qa.loc["lambada","ret_hi"])}, {pct(qa.loc["race","ret_lo"])}&ndash;{pct(qa.loc["race","ret_hi"])}) and six of ten configurations above it. Under the large-model control instead, retention is {pct(qa.ret_b.min())}&ndash;{pct(qa.ret_b.max())}. The defensible statement is that HC carries information these statistics do not, in every tier-A cell; that half or more of the scale-specific association is that information holds at the point estimate in three cells.</p></div>
<div class="finding null"><h4>The simpler signals are not weaker predictors of Intelligence Gain</h4><p>The drop also reports each baseline&rsquo;s own within-string association with IG under the small-model control. They are of the same size as HC&rsquo;s. In race the maximum router probability shows {f3(QB.loc["race","mp_ig"])} against HC&rsquo;s {f3(QB.loc["race","ref"])}; in competition_math {f3(QB.loc["competition_math","mp_ig"])} against {f3(QB.loc["competition_math","ref"])}; in fineweb-edu the logit gap shows {f3(QB.loc["fineweb-edu","gap_ig"])} against {f3(QB.loc["fineweb-edu","ref"])}; only in lambada is HC the largest. So the last MoE layer&rsquo;s routing carries information about where scale helps in several forms, and HC is one of them. What this experiment establishes is that HC&rsquo;s share is <i>distinct</i> &mdash; it survives the others &mdash; not that HC is the best single summary of the router. That is the claim a paper can make.</p></div>
<div class="finding null"><h4>No single baseline absorbs much; the three together do</h4><p>Router entropy alone keeps {pct(qa.ret_ent.min())}&ndash;{pct(qa.ret_ent.max())} of the tier-A associations; maximum probability absorbs little (it keeps {pct(qa.ret_mp.min())}&ndash;{pct(qa.ret_mp.max())}); the logit gap keeps {pct(qa.ret_gap.min())}&ndash;{pct(qa.ret_gap.max())}. Each alone takes at most about a quarter, and none takes much in every cell; together they take more than any one. The cell that loses most, fineweb-edu, is the one where HC and entropy are most correlated ({qa.loc["fineweb-edu","hc_ent"]:+.2f}); in competition_math, where they are nearly unrelated ({qa.loc["competition_math","hc_ent"]:+.2f}), nothing is absorbed.</p></div>
<div class="finding gemma"><h4>Out of domain, and Gemma</h4><p>On the unseen benchmarks, cnn_dailymail keeps {pct(QB.loc["cnn_dailymail","ret"])} and scientific_papers {pct(QB.loc["scientific_papers","ret"])}, both with intervals excluding zero. <b>dialogsum keeps {pct(DS_RET)}</b> ({f3(QB.loc["dialogsum","ref"])} &rarr; {f3(QB.loc["dialogsum","joint"])}, interval {f3(QB.loc["dialogsum","jlo"])} to {f3(QB.loc["dialogsum","jhi"])}): its narrow pass in section 15 is carried by what the simpler statistics already see. Gemma&rsquo;s race inversion is not absorbed at all ({f3(BC.loc[("gemma","race"),"ref"])} &rarr; {f3(BC.loc[("gemma","race"),"joint"])}, {pct(BC.loc[("gemma","race"),"ret"])}).</p></div>
{card(30, 'What each baseline leaves of Qwen&rsquo;s scale-specific association', 'fig_base_ret', 'Share of within-string r(HC, IG | CE<sub>small</sub>) retained, ratio of ten-configuration means. Dashed red: the pre-registered 50% line. * unseen benchmarks.')}
{card(31, 'HC and the simpler signals as predictors, Qwen tier-A cells', 'fig_base_pred', 'Each statistic&rsquo;s within-string association with IG under the small-model control. The last bar is HC after all three baselines are partialled out.', h=320)}
{base_table}{note('Retained = mean partial after baselines / mean reference partial, over ten configurations. The drop&rsquo;s own summary averages per-configuration ratios instead, which is unstable when a reference is near zero (dialogsum: 44% that way, 15% as a ratio of means). &lsquo;retained, CI&rsquo; is the median over configurations of the passage-bootstrap interval for the ratio.')}
<div class="caveat"><b>What this does not show.</b> (1) The fourth planned baseline, the norm of the hidden state entering the layer, is not in the drop. (2) ultrachat, mbpp and humaneval were not run. (3) The drop also reports a partial on both difficulty controls at once; IG is an exact linear function of the two losses, so that partial is zero by construction (largest |value| {BC.two_sided.max():.3f}) and is not used. (4) Retention is a share of an association that is itself small: what survives in fineweb-edu is {f3(qa.loc["fineweb-edu","joint"])}.</div></section>
'''

# ---------------------------------------------------------------- insert new sections, renumber
i = s.find('<section id="shared"')
s = s[:i] + sec_oodw + sec_base + s[i:]

def renum_h2(m):
    n = int(m.group(2))
    return f'{m.group(1)}{n + 2 if n >= 15 and m.group(3) not in ("oodwide", "baselines") else n}.'
# h2 numbers: sections after the inserted ones shift by 2
def shift_h2(match):
    sid, num, rest = match.group(1), int(match.group(2)), match.group(3)
    if sid in ('oodwide', 'baselines') or num < 15:
        return match.group(0)
    return f'<section id="{sid}"><h2>{num + 2}.{rest}'
s = re.sub(r'<section id="([^"]+)"><h2>(\d+)\.([^<]*)', shift_h2, s)

body_end = s.find('<script>\nconst CHART_SPECS')
body, tail = s[:body_end], s[body_end:]
# figure numbers >= 28 in existing content shift by 4 (new are 28-31, written after shift)
def shift_fig(m):
    n = int(m.group(1))
    return m.group(0) if n < 28 else f'Figure {n + 4}'
a, b = body.find('<section id="oodwide"'), body.find('<section id="shared"')
body = re.sub(r'Figure (\d+)', shift_fig, body[:a]) + body[a:b] + re.sub(r'Figure (\d+)', shift_fig, body[b:])
# text references to sections >= 15 shift by 2 (only outside new sections)
def shift_sec(m):
    n = int(m.group(2))
    return m.group(0) if n < 15 else f'{m.group(1)}{n + 2}'
body = re.sub(r'([Ss]ection )(\d+)', shift_sec, body[:a]) + body[a:b] + re.sub(r'([Ss]ection )(\d+)', shift_sec, body[b:])
s = body + tail

# nav links: add two entries after the ood link
s = s.replace('<a href="#ood">', '<a href="#ood">', 1)
nav_ood = re.search(r'<a href="#ood">[^<]*</a>', s).group(0)
s = s.replace(nav_ood, nav_ood + '<a href="#oodwide">Out of domain, widened</a><a href="#baselines">Beyond simpler router signals</a>', 1)

# chart specs
m = re.search(r'const CHART_SPECS = (\[.*?\]);\nfunction hideUnderscore', s, flags=re.S)
specs = json.loads(m.group(1)) + new_specs
s = s[:m.start(1)] + json.dumps(specs) + s[m.end(1):]

open('/home/claude/v15/stage1.html', 'w', encoding='utf-8').write(s)
json.dump(dict(QW=QW.round(5).reset_index().to_dict('records'), GW=GW.round(5).reset_index().to_dict('records'),
               BC=BC.drop(columns='tier').round(5).reset_index().to_dict('records'), N_PASS=N_PASS, DS_RET=DS_RET,
               DEFL_MED=DEFL_MED, REPRO_ROWS=REPRO_ROWS, REPRO_MAX=float(REPRO_MAX)),
          open('/home/claude/v15/numbers.json', 'w'), indent=1, default=str)
print('stage1 ok', len(s), 'specs', len(specs), 'defl med', DEFL_MED)
