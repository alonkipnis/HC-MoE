"""HC-MoE v17: v16 + reports/26_10_15/v16_experiments (four experiments). All numbers recomputed from parquet."""
import json, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart, line_chart

UP = '/mnt/user-data/uploads/HC-MoE-Alon/reports/'
E = UP + '26_10_15/v16_experiments/'
s = open(UP + 'HC_MoE_Report_v16.html', encoding='utf-8').read()
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def npos(x): return int((x > 0).sum())

# ================================================================ data
# 1. speaker tag
T = pd.read_parquet(E + 'experiment_dialogsum_speaker_ablation/dialogsum_tag_ablation_records.parquet')
assert len(T) == 20 and not T.isna().any().any()
TQ, TG = T[T.model == 'Qwen'], T[T.model == 'gemma']
tq = dict(floor=TQ.floor.iloc[0], n=int(TQ.n_total.iloc[0]), nc=int(TQ.n_clean.iloc[0]), full=TQ.r_full.mean(), full_pos=npos(TQ.r_full),
          person=TQ.r_no_person_only.mean(), person_pos=npos(TQ.r_no_person_only), notag=TQ.r_no_speaker_tags.mean(), notag_pos=npos(TQ.r_no_speaker_tags),
          lo=TQ.ci_clean_lower.mean(), hi=TQ.ci_clean_upper.mean(), sp=TQ.spearman_no_speaker_tags.mean())
tg = dict(floor=TG.floor.iloc[0], full=TG.r_full.mean(), notag=TG.r_no_speaker_tags.mean(), notag_pos=npos(TG.r_no_speaker_tags), nc=int(TG.n_clean.iloc[0]), n=int(TG.n_total.iloc[0]))
TAG_MISS = TQ[TQ.r_no_speaker_tags <= 0][['estimator', 'variant', 'r_no_speaker_tags']].values.tolist()
TAG_RULE = tq['notag'] > tq['floor'] and tq['notag_pos'] >= 9
assert not TAG_RULE and tq['notag_pos'] == 8
assert abs(tq['full'] + 0.0349) < 1e-3 and abs(tg['full'] + 0.0856) < 1e-3          # reproduces v15/v16
NONT = TQ[~TQ.estimator.str.startswith('StudentT')].r_no_speaker_tags
# 2. identifiers
C = pd.read_parquet(E + 'experiment_code_identifier_copying/code_copying_partials.parquet')
assert len(C) == 40 and not C.isna().any().any()
CC = C.groupby('cell').agg(n_first=('n_first', 'first'), n_rep=('n_rep', 'first'), floor=('floor', 'first'), full=('r_full', 'mean'),
                           allid=('r_all_identifiers', 'mean'), rep=('r_repeated_identifiers', 'mean'), rep_pos=('r_repeated_identifiers', npos),
                           first=('r_first_identifiers', 'mean'), first_pos=('r_first_identifiers', npos),
                           wo_rep=('r_without_repeated_id', 'mean'), wo_all=('r_without_all_id', 'mean'))
CELLS = ['Qwen_mbpp', 'Qwen_humaneval', 'gemma_mbpp', 'gemma_humaneval']
COPY = [c for c in CELLS if abs(CC.loc[c, 'rep']) >= 2 * abs(CC.loc[c, 'first']) and CC.loc[c, 'rep'] < 0 and CC.loc[c, 'first'] >= -CC.loc[c, 'floor']]
assert COPY == []
FIRST_STRONGER = [c for c in CELLS if CC.loc[c, 'first'] < CC.loc[c, 'rep']]
# 3. Gemma shared expert
GS = pd.read_parquet(E + 'gemma_shared_expert_ablation/gemma_scaling_records.parquet')
assert len(GS) == 60 and not GS.isna().any().any()
GA = GS.groupby(['dataset', 'alpha']).agg(ce_big=('mean_ce_big', 'first'), ce_small=('mean_ce_small', 'first'), ig=('mean_ig', 'first'),
                                          raw=('r_raw', 'mean'), part=('r_partial_ce_small', 'mean'), pos=('r_partial_ce_small', npos), floor=('floor', 'first'))
def ga(d, a, k): return float(GA.loc[(d, a), k])
SHRINK_R = 1 - ga('race', 0.5, 'part') / ga('race', 1.0, 'part')
CE_UP_R = ga('race', 0.5, 'ce_big') / ga('race', 1.0, 'ce_big') - 1
SHRINK_M = 1 - ga('competition_math', 0.5, 'part') / ga('competition_math', 1.0, 'part')
CE_UP_M = ga('competition_math', 0.5, 'ce_big') / ga('competition_math', 1.0, 'ce_big') - 1
assert SHRINK_R < 0.5 and CE_UP_R < 0.10
assert abs(ga('race', 1.0, 'raw') + 0.0575) < 1e-3 and abs(ga('competition_math', 1.0, 'raw') + 0.0783) < 1e-3
SUMM = pd.DataFrame(json.load(open(E + 'gemma_shared_expert_ablation/gemma_shared_expert_ablation_summary.json')))
PPL = {(r.dataset, r.alpha): r.perplexity for r in SUMM.itertuples()}
# 4. Qwen partner sweep
Q = pd.read_parquet(E + 'qwen_8b_partner_sweep/qwen_8b_sweep_records.parquet')
assert len(Q) == 360 and not Q.isna().any().any()
P = ['0.6B', '1.7B', '4B', '8B']; RATIO = {'0.6B': 5.5, '1.7B': 1.94, '4B': 0.82, '8B': 0.41}
FW = Q[(Q.dataset == 'fineweb-edu')]
FW_N = int(FW.n_tokens.iloc[0]); FW_R = float(FW[FW.partner == '0.6B'].r_raw.mean())
assert FW_N != 477782 and abs(FW_R) < 0.005           # broken cell: main study n=477,782, r=+0.0707
QK = Q[Q.dataset != 'fineweb-edu']
QC = QK.groupby(['dataset', 'partner']).agg(raw=('r_raw', 'mean'), s=('r_partial_ce_small', 'mean'), b=('r_partial_ce_big', 'mean'),
                                           spos=('r_partial_ce_small', npos), floor=('floor', 'first'), ig=('mean_ig', 'first')).reset_index()
FAM = QC.groupby('partner')[['raw', 's', 'b', 'ig']].mean().loc[P]
PS = QC.pivot(index='dataset', columns='partner', values='s')[P]; PR = QC.pivot(index='dataset', columns='partner', values='raw')[P]
PSP = QC.pivot(index='dataset', columns='partner', values='spos')[P]
NEG8 = QC[(QC.partner == '8B') & (QC.s < -QC.floor) & (QC.spos <= 1)].dataset.tolist()
POS8 = QC[(QC.partner == '8B') & (QC.s > QC.floor) & (QC.spos >= 9)].dataset.tolist()
assert len(NEG8) < 3
MONO = int((PS.diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum())
# 0.6B column reproduces the main study's raw cells (10-config means)
MAIN = {'lambada': .1307, 'competition_math': .1000, 'race': .0618, 'structured-wikipedia': .0571, 'hellaswag': .0321, 'pg19': .0289, 'tinystories': .0228, 'mmlu': .0172}
REPRO_Q = max(abs(PR.loc[d, '0.6B'] - v) for d, v in MAIN.items())
assert REPRO_Q < 6e-4

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

# ================================================================ figures 38-42
specs = []
labs = [f'{e} / {v}' for e, v in zip(TQ.estimator, TQ.variant)]
specs.append(floor_band(bar_chart('fig_tag', labs, {'raw r, all tokens': list(TQ.r_full), "&lsquo;Person&rsquo; removed": list(TQ.r_no_person_only),
    'whole tag removed': list(TQ.r_no_speaker_tags)}, colors=['#64748b', '#93c5fd', '#60a5fa']), [tq['floor']] * 10))
specs[-1]['data']['datasets'][1]['label'] = "'Person' removed"
CL = [c.replace('_', ' / ').replace('gemma', 'Gemma') for c in CELLS]
specs.append(bar_chart('fig_copy', CL, {'all identifiers': [CC.loc[c, 'allid'] for c in CELLS], 'first occurrences': [CC.loc[c, 'first'] for c in CELLS],
    'repeats of a name already in context': [CC.loc[c, 'rep'] for c in CELLS]}, colors=['#94a3b8', '#f87171', '#60a5fa']))
gl = [f'{d} &alpha;={a:g}'.replace('&alpha;', 'α') for d in ['race', 'competition_math'] for a in (1.0, 0.5, 0.0)]
specs.append(floor_band(bar_chart('fig_shared', gl, {'raw r(HC, IG)': [ga(d, a, 'raw') for d in ['race', 'competition_math'] for a in (1.0, 0.5, 0.0)],
    'r(HC, IG | CE_small)': [ga(d, a, 'part') for d in ['race', 'competition_math'] for a in (1.0, 0.5, 0.0)]}, colors=['#64748b', '#f87171']),
    [ga(d, a, 'floor') for d in ['race', 'competition_math'] for a in (1.0, 0.5, 0.0)]))
xl = [f'{p} ({RATIO[p]}&times;)'.replace('&times;', '×') for p in P]
specs.append(line_chart('fig_sweep_fam', xl, {'raw r(HC, IG)': list(FAM.raw), 'r(HC, IG | CE_small)': list(FAM.s), 'r(HC, IG | CE_big)': list(FAM.b)},
    x_title='small partner (active-parameter ratio, large / small)', colors=['#94a3b8', '#f59e0b', '#60a5fa'], null_band=0.0128))
specs.append(line_chart('fig_sweep_bm', xl, {d: list(PS.loc[d]) for d in PS.index}, x_title='small partner',
    y_title='r(HC, IG | CE_small)', colors=['#f59e0b', '#60a5fa', '#f87171', '#2dd4bf', '#a78bfa', '#84cc16', '#94a3b8', '#e879f9']))

# ================================================================ sections 21-24
sec21 = f'''<section id="tag"><h2>21. The whole speaker tag on dialogsum</h2><div class="grid-4"><div class="stat-box"><div class="val">{f3(tq["notag"])}</div><div class="lbl">Qwen raw r without the tag (was {f3(tq["full"])})</div></div><div class="stat-box"><div class="val">{tq["notag_pos"]} / 10</div><div class="lbl">configurations positive (rule needs 9)</div></div><div class="stat-box"><div class="val">{1 - tq["nc"]/tq["n"]:.0%}</div><div class="lbl">of tokens are speaker-tag pieces</div></div><div class="stat-box"><div class="val">{f3(tg["notag"])}</div><div class="lbl">Gemma without the tag (0/10 positive)</div></div></div>
<p>v16 found that the format test on dialogsum had removed one of the four pieces of the speaker tag <code>#Person1#:</code>. This experiment removes every piece, identified by position in the tag so that digits and &lsquo;#&rsquo; elsewhere stay in, and nothing else: {tq["n"]-tq["nc"]:,} of {tq["n"]:,} tokens for Qwen. Ten configurations, both families. The full-text values reproduce sections 15 and 19.</p>
<p><b>The rule, fixed in v16:</b> if Qwen turns positive with at least 9 of 10 configurations, the out-of-domain raw-sign reversal is restated as a formatting effect on both dialogue and chat; if it stays undetermined or negative, dialogsum&rsquo;s reversal is about content.</p>
<div class="finding null"><h4>Positive in eight of ten: by the letter, the second branch; in size, mostly the tag</h4><p>Without the tag Qwen&rsquo;s raw r is {f3(tq["notag"])} (mean interval {f3(tq["lo"])} to {f3(tq["hi"])}; Spearman {f3(tq["sp"])}), against {f3(tq["full"])} with it: a change of {f3(tq["notag"]-tq["full"])}, larger than the original effect. Eight configurations are positive. The two that are not are StudentT/standard ({f3(TAG_MISS[0][2])}) and StudentTSkewed/standard ({f3(TAG_MISS[1][2])}); the six non-StudentT configurations range {f3(NONT.min())} to {f3(NONT.max())}. The rule needs nine, so by the letter dialogsum&rsquo;s reversal stays &lsquo;about content&rsquo;, and the claim that it is formatting is not established.</p><p>Read beside section 19, the evidence points one way: the tag carries most of the reversal, and whether removing it leaves a clearly positive correlation depends on the null estimator &mdash; the StudentT nulls disagree. That makes it an estimator-sensitive result, which is what the rule&rsquo;s 9-of-10 criterion exists to catch.</p></div>
<div class="finding gemma"><h4>Gemma: the tag accounts for about half of the inversion, and the rest is not formatting</h4><p>Gemma goes from {f3(tg["full"])} to {f3(tg["notag"])} without the tag, negative in all ten configurations. Its raw inversion on dialogue survives the removal of formatting, as on chat and code in section 19.</p></div>
{card(38, 'Qwen on dialogsum: raw r with the speaker tag removed, by configuration', 'fig_tag', 'Grey: all tokens. Light blue: only &lsquo;Person&rsquo; removed (the v16 test). Blue: the whole tag removed. White ticks: the measured floor. The two StudentT standard configurations are the exceptions.')}
<div class="caveat"><b>What this does not show.</b> Removing 19% of tokens changes the sample as well as the formatting. The rule is applied as written; a looser 8-of-10 criterion would pass, and the series has declined to loosen criteria after seeing results.</div></section>
'''

crow = [[c.replace('_', ' / ').replace('gemma', 'Gemma'), f'{int(CC.loc[c,"n_first"]):,}', f'{int(CC.loc[c,"n_rep"]):,}', f3(CC.loc[c, 'allid']),
         f'{f3(CC.loc[c,"first"])} ({int(CC.loc[c,"first_pos"])}/10)', f'{f3(CC.loc[c,"rep"])} ({int(CC.loc[c,"rep_pos"])}/10)', f3(CC.loc[c, 'full']), f3(CC.loc[c, 'wo_rep']), f3(CC.loc[c, 'wo_all'])] for c in CELLS]
sec22 = f'''<section id="copy"><h2>22. Is the identifier effect copying?</h2><div class="grid-4"><div class="stat-box"><div class="val">0 / 4</div><div class="lbl">cells where repeats carry the negative association</div></div><div class="stat-box"><div class="val">{len(FIRST_STRONGER)} / 4</div><div class="lbl">cells where first occurrences are more negative than repeats</div></div><div class="stat-box"><div class="val">{f3(CC.loc["gemma_humaneval","first"])}</div><div class="lbl">Gemma humaneval, first occurrences</div></div><div class="stat-box"><div class="val">{f3(CC.loc["gemma_mbpp","rep"])}</div><div class="lbl">Gemma mbpp, repeats (positive)</div></div></div>
<p>v16 found identifiers negative in every code cell of both families, and proposed copying as a cause: in short functions most names repeat ones already written in the prompt or signature, which both models can copy. Identifiers were split into first occurrences and repeats of a name already in context, and the within-string r(HC, IG | CE<sub>small</sub>) computed for each. Four cells, ten configurations, 40 rows.</p>
<p><b>The rule, fixed in v16:</b> if repeats carry the negative association (at least twice as strong, interval excluding zero) and first occurrences are null or positive, code&rsquo;s exception is a copying effect; if both are negative, it stays a domain effect.</p>
<div class="finding null"><h4>Not copying: first occurrences carry the effect at least as strongly as repeats</h4><p>First occurrences are negative in every cell and every configuration: Qwen mbpp {f3(CC.loc["Qwen_mbpp","first"])}, Qwen humaneval {f3(CC.loc["Qwen_humaneval","first"])}, Gemma mbpp {f3(CC.loc["gemma_mbpp","first"])}, Gemma humaneval {f3(CC.loc["gemma_humaneval","first"])}. Repeats are weaker in three cells and positive in one (Gemma mbpp {f3(CC.loc["gemma_mbpp","rep"])}, nine of ten). The copying prediction is the reverse of what the data show; by the rule, code stays a domain effect.</p><p>What the split does show is where the effect sits: on the first appearance of a name. A new identifier is a naming choice that neither model can predict from context, so the large model&rsquo;s extra capacity has little to work with there, while the router still treats those tokens as anomalous. That is a hypothesis about why code differs, not a tested one.</p></div>
{card(39, 'Code identifiers: first occurrences against repeats', 'fig_copy', 'Within-string r(HC, IG | CE<sub>small</sub>), means of ten configurations. Copying predicted the blue bars to carry the effect; the red ones do.', h=300)}
{table(['cell', 'first occurrences', 'repeats', 'all identifiers', 'first only', 'repeats only', 'full benchmark', 'without repeats', 'without identifiers'], crow, num_from=1)}
{note('Means of ten configurations; in brackets, configurations above zero. As in section 18, subset values are not additive.')}
</section>
'''

grow = [[d, f'{a:g}', f'{ga(d,a,"ce_big"):.3f}', f'{PPL[(d,a)]:.1f}', f'{ga(d,a,"ig"):+.3f}', f3(ga(d, a, 'raw')), f'{f3(ga(d,a,"part"))} ({int(ga(d,a,"pos"))}/10)']
        for d in ['race', 'competition_math'] for a in (1.0, 0.5, 0.0)]
sec23 = f'''<section id="sharedx"><h2>23. Gemma&rsquo;s shared expert</h2><div class="grid-4"><div class="stat-box"><div class="val">{SHRINK_R:.0%}</div><div class="lbl">race inversion shrinks at &alpha; = 0.5 (rule needs 50%)</div></div><div class="stat-box"><div class="val">+{CE_UP_R:.1%}</div><div class="lbl">mean CE<sub>big</sub> at &alpha; = 0.5</div></div><div class="stat-box"><div class="val">{PPL[("race",1.0)]:.1f} &rarr; {PPL[("race",0.0)]:.0f}</div><div class="lbl">race perplexity, &alpha; = 1 &rarr; 0</div></div><div class="stat-box"><div class="val">not run</div><div class="lbl">all-layer arm</div></div></div>
<p>gemma-4-26B-A4B has one always-active shared expert per MoE layer. v14.1 asked whether Gemma&rsquo;s scale-specific inversion is carried by it. The output of the last MoE layer&rsquo;s shared expert (layer 29) was scaled by &alpha; = 1, 0.5 and 0 and the large model&rsquo;s loss recomputed on race and competition_math. HC comes from that layer&rsquo;s router, which reads the hidden state before the expert outputs are added, so HC is unchanged by construction and only CE<sub>big</sub>, and with it IG, moves. Ten configurations; 60 rows. At &alpha; = 1 the correlations reproduce the main study (race {f3(ga("race",1.0,"raw"),4)}, competition_math {f3(ga("competition_math",1.0,"raw"),4)}).</p>
<p><b>The rule:</b> if the race inversion shrinks by half or more at &alpha; = 0.5 while mean CE<sub>big</sub> rises by less than 10%, the shared expert is implicated; if unchanged at &alpha; = 0, it is ruled out.</p>
<div class="finding gemma"><h4>Not implicated at &alpha; = 0.5; &alpha; = 0 breaks the model, so the ruling-out branch cannot be read</h4><p><b>&alpha; = 0.5.</b> Mean CE<sub>big</sub> rises {CE_UP_R:.1%} on race and {CE_UP_M:.1%} on competition_math, inside the rule&rsquo;s budget. The inversion barely moves: r(HC, IG | CE<sub>small</sub>) goes {f3(ga("race",1.0,"part"))} &rarr; {f3(ga("race",0.5,"part"))} on race ({SHRINK_R:.0%} smaller) and {f3(ga("competition_math",1.0,"part"))} &rarr; {f3(ga("competition_math",0.5,"part"))} on competition_math ({SHRINK_M:.0%}). The first branch is not met.</p><p><b>&alpha; = 0.</b> Removing the last layer&rsquo;s shared expert entirely collapses the model: race perplexity {PPL[("race",1.0)]:.1f} &rarr; {PPL[("race",0.0)]:.0f}, competition_math {PPL[("competition_math",1.0)]:.1f} &rarr; {PPL[("competition_math",0.0)]:.1f}, and mean IG turns strongly negative ({ga("race",0.0,"ig"):+.2f} nats on race): the &lsquo;large&rsquo; model is now far worse than the small one. The race correlation flips to {f3(ga("race",0.0,"part"))}, but with the model broken, IG no longer measures what scale buys, so the flip carries no information about the inversion. The second branch is neither met nor refuted.</p><p><b>Reading.</b> Halving the shared expert&rsquo;s contribution leaves Gemma&rsquo;s inversion essentially intact. Within the range where the model still works, there is no sign that the last layer&rsquo;s shared expert carries it. The all-layer arm, which would also change HC, was not run.</p></div>
{card(40, 'Gemma, last-layer shared expert scaled by &alpha;', 'fig_shared', 'Ten-configuration means; white ticks are floors. At &alpha; = 0 the model collapses (perplexity &times;40 on race), so those bars are not interpretable.')}
{table(['benchmark', '&alpha;', 'mean CE_big', 'perplexity', 'mean IG', 'raw r', 'r | CE_small'], grow, num_from=1)}
</section>
'''

srow = [[d] + [f'{f3(PS.loc[d,p])} ({int(PSP.loc[d,p])}/10)' for p in P] + [' &rarr; '.join(f3(PR.loc[d, p], 2) for p in P)] for d in PS.index]
sec24 = f'''<section id="sweep"><h2>24. An 8B partner: the compute gap taken to zero</h2><div class="grid-4"><div class="stat-box"><div class="val">{f3(FAM.s.iloc[0])} &rarr; {f3(FAM.s.iloc[-1])}</div><div class="lbl">family-mean scale-specific association, 0.6B &rarr; 8B partner</div></div><div class="stat-box"><div class="val">{len(NEG8)} / 8</div><div class="lbl">benchmarks strictly negative with 8B (rule needs 3)</div></div><div class="stat-box"><div class="val">{FAM.ig.iloc[0]:.2f} &rarr; {FAM.ig.iloc[-1]:.2f}</div><div class="lbl">mean IG (nats), 0.6B &rarr; 8B</div></div><div class="stat-box"><div class="val">excluded</div><div class="lbl">fineweb-edu: pipeline error</div></div></div>
<p>Qwen3-30B-A3B was held fixed and paired with four dense partners &mdash; Qwen3-0.6B, 1.7B, 4B and, new here, 8B &mdash; active-parameter ratios {RATIO["0.6B"]}&times;, {RATIO["1.7B"]}&times;, {RATIO["4B"]}&times; and {RATIO["8B"]}&times;. Unlike experiment 3 (v8&ndash;v12, GMM/standard only), every partner runs on all ten configurations and all nine studied benchmarks: 360 rows. The 0.6B column reproduces the main study&rsquo;s raw cells to within {REPRO_Q:.4f}, and the GMM/standard values for 0.6B&ndash;4B reproduce experiment 3.</p>
<div class="caveat"><b>fineweb-edu is excluded.</b> In this drop it has {FW_N:,} tokens against the main study&rsquo;s 477,782, and r(HC, IG) is {FW_R:+.4f} with the 0.6B partner, where the main study measured +0.0707 on the same pair &mdash; and essentially zero under every partner and both controls. That pattern is what misaligned token series produce. Experiment 3 measured fineweb-edu correctly (+0.076 &rarr; +0.064 under the small-model control). The family means below use the other eight benchmarks.</div>
<p><b>The rule, fixed in v13:</b> if three or more benchmarks turn negative with the 8B partner, the active-compute account of lambada&rsquo;s 4B reversal is supported.</p>
<div class="finding null"><h4>The active-compute account is not supported</h4><p>With the 8B partner two benchmarks are strictly negative under the small-model control: {", ".join(NEG8)}. The rule needs three. Lambada, whose 4B reversal the account was built to explain, is <i>not</i> more negative with a partner that has more active parameters: {f3(PS.loc["lambada","4B"])} with 4B, {f3(PS.loc["lambada","8B"])} with 8B. The 4B reversal stays unexplained, and no longer has a candidate.</p></div>
<div class="finding scale"><h4>The scale-specific association goes to zero as the gap closes</h4><p>Across the eight benchmarks, the family mean of r(HC, IG | CE<sub>small</sub>) is {", ".join(f3(v) for v in FAM.s)} for the four partners; the raw correlation {", ".join(f3(v) for v in FAM.raw)}; mean IG {", ".join(f"{v:.2f}" for v in FAM.ig)} nats. With an 8B partner the small model is within 0.15 nats of the large one on average, and the association that marks where scale helps is gone at the family level. Per benchmark the path is noisier: strictly decreasing in {MONO} of eight, and two benchmarks, {" and ".join(POS8)}, stay positive with every partner. As a dose-response this extends experiment 3 to a fourth point, on the full configuration grid.</p></div>
{card(41, 'Qwen&rsquo;s association as the small partner grows (family means, eight benchmarks)', 'fig_sweep_fam', 'Ten configurations per point. Dashed: the main study&rsquo;s typical floor (&plusmn;0.0128). The large model is held fixed throughout.')}
{card(42, 'The scale-specific association per benchmark', 'fig_sweep_bm', 'r(HC, IG | CE<sub>small</sub>), ten-configuration means. pg19 and race stay positive throughout; competition_math falls furthest.')}
{table(['benchmark', '0.6B', '1.7B', '4B', '8B', 'raw r, 0.6B &rarr; 8B'], srow, num_from=1)}
{note('Scale-specific association r(HC, IG | CE<sub>small</sub>); in brackets, configurations above zero. Raw, not within token strings, as in experiment 3.')}
</section>
'''

# ================================================================ insert & renumber
i = s.index('<section id="shared"')
s = s[:i] + sec21 + sec22 + sec23 + sec24 + s[i:]
NEW = ('tag', 'copy', 'sharedx', 'sweep')
def shift_h2(m):
    sid, num, rest = m.group(1), int(m.group(2)), m.group(3)
    return m.group(0) if sid in NEW or num < 21 else f'<section id="{sid}"><h2>{num + 4}.{rest}'
s = re.sub(r'<section id="([^"]+)"><h2>(\d+)\.([^<]*)', shift_h2, s)
be = s.index('<script>\nconst CHART_SPECS'); body, tail = s[:be], s[be:]
a, b = body.index('<section id="tag"'), body.index('<section id="shared"')
def fs(m): n = int(m.group(1)); return m.group(0) if n < 38 else f'Figure {n + 5}'
def ss(m): n = int(m.group(2)); return m.group(0) if n < 21 else f'{m.group(1)}{n + 4}'
pre = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[:a]))
post = re.sub(r'([Ss]ection )(\d+)', ss, re.sub(r'Figure (\d+)', fs, body[b:]))
s = pre + body[a:b] + post + tail
nav = re.search(r'<a href="#lambnull">[^<]*</a>', s).group(0)
s = s.replace(nav, nav + '<a href="#tag">Speaker tag</a><a href="#copy">Identifier copying</a><a href="#sharedx">Gemma shared expert</a><a href="#sweep">8B partner</a>', 1)
m = re.search(r'const CHART_SPECS = (\[.*?\]);\nfunction hideUnderscore', s, flags=re.S)
s = s[:m.start(1)] + json.dumps(json.loads(m.group(1)) + specs) + s[m.end(1):]
open('/home/claude/v17/stage1.html', 'w', encoding='utf-8').write(s)

json.dump(dict(tq=tq, tg=tg, TAG_MISS=TAG_MISS, CC=CC.round(5).to_dict('index'), FIRST_STRONGER=FIRST_STRONGER,
               SHRINK_R=SHRINK_R, CE_UP_R=CE_UP_R, SHRINK_M=SHRINK_M, GA={f'{k[0]}|{k[1]}': v for k, v in GA.round(5).to_dict('index').items()},
               PPL={f'{k[0]}|{k[1]}': v for k, v in PPL.items()}, FAM=FAM.round(5).to_dict(), PS=PS.round(5).to_dict(), NEG8=NEG8, POS8=POS8, MONO=MONO,
               FW_N=FW_N, FW_R=FW_R, REPRO_Q=REPRO_Q), open('/home/claude/v17/n17.json', 'w'), indent=1, default=float)
print('stage1 ok', len(s))
