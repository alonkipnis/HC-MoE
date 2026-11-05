from common import *

c = CELL; F = N['dose_fam']; DB = N['dose_bm']
q5 = ['fineweb-edu', 'lambada', 'race', 'cnn_dailymail', 'scientific_papers']
cpq_s = N['cp']['Qwen|ce_mean_decile|mean_part_s_c']; cpq_b = N['cp']['Qwen|ce_mean_decile|mean_part_b_c']
ed = D['ernie']['cells']; E18 = ESEL[18]; E27 = ESEL[27]
GAB = {(r['dataset'], r['stats_mode'], r['alpha']): r for r in N['gab']}
cpe_s = N['cp']['ernie|ce_mean_decile|mean_part_s_c']

specs = [fig_dose('w_dose'), fig_cells('w_cells')]
for d_ in specs[1]['data']['datasets']:
    if d_.get('label', '').startswith('null envelope'): d_['label'] = 'typical floor (±0.0128)'

findings = [
 ('A', 'In Qwen3, HC marks the tokens where scale helps, beyond either model&rsquo;s difficulty.',
  'Both partials positive in 10/10 configurations on fineweb, lambada, race, cnn and sci (' + '; '.join(f'{SHORT[b]} {f3(c[("Qwen", b)]["c_s"])} / {f3(c[("Qwen", b)]["c_b"])}' for b in q5) + '). Replicates in both held-out halves; survives winsorising, ranks, removing the strongest expert, both floor methods and the old scale.'),
 ('B', 'The same association appears in ERNIE and LFM2; race is positive in three of four families.',
  'ERNIE race, fineweb, tinystories, and out of domain cnn and mbpp (10/10); LFM2 race, pg19 (10/10), lambada (9/10), none of its unseen benchmarks; all replicate in held-out halves. Other benchmarks in both families go the other way.'),
 ('B', 'The association shrinks as the dense partner closes the compute gap, but race and fineweb persist.',
  f'Family-mean r | CE<sub>small</sub> {" &rarr; ".join(f3(v) for v in F["part_s_c"])} and r | CE<sub>big</sub> {" &rarr; ".join(f3(v) for v in F["part_b_c"])} from the 0.6B to the 8B partner; race and fineweb pass 10/10 even with the 8B partner.'),
 ('B', 'HC is distinct from router confidence, but shares much with the hidden-state norm.',
  f'Three router statistics leave &ge;50% of both partials in 10 of 13 positive cells tested. With the norm added (Qwen), lambada keeps {pct(B["Qwen_L47_lambada"]["R4s"])}, race {pct(B["Qwen_L47_race"]["R4s"])}, fineweb {pct(B["Qwen_L47_fineweb-edu_max512_hnorm"]["R4s"])}. HC adds 0.01&ndash;0.4% of within-string IG variance.'),
 ('B', 'In Qwen the effect is a property of late routing; in ERNIE it peaks at mid depth.',
  f'Qwen race and fineweb pass only in layers 41&ndash;47; earlier layers track difficulty (r | CE<sub>small</sub> &gt; 0, r | CE<sub>big</sub> &lt; 0); lambada passes from layer 5. ERNIE passes at layers 14&ndash;22, and a layer chosen on half the passages beats the last layer on the other half ({E18["half_b_rob_pos_cells"]} vs {E27["half_b_rob_pos_cells"]} robust cells), but there the router statistics explain r | CE<sub>big</sub>.'),
 ('A', 'Gemma 4 is the inverted family.',
  'Never positive at the last layer; both partials negative in 9&ndash;10/10 on humaneval, race and news, with 94% kept after all router statistics. Positive only at layer 7; inverted from mid depth. Not explained by the shared experts: at a 0.75 scale (+3.4% loss) the race inversion shrinks ' + f'{GAB[("race","alpha_self_stats",0.75)]["inversion_shrink_pct_ces"]:.0f}% / {GAB[("race","alpha_self_stats",0.75)]["inversion_shrink_pct_ceb"]:.0f}%' + ' (rule: &ge;50%).'),
 ('B', 'In Qwen and Gemma, code is where the signal breaks down; ERNIE is the exception.',
  f'At matched loss, Qwen&rsquo;s code sits below prose in {cpq_s["below"]}/10 deciles on r | CE<sub>small</sub> and {cpq_b["below"]}/10 on r | CE<sub>big</sub>; Gemma 10/10 on both. ERNIE shows no gap ({cpe_s["below"]}/10), and mbpp is its strongest positive cell.'),
]
fl = ''.join(f'<li><span class="tier tier-{t}">{t}</span><div class="body"><div class="claim"><b>{i+1}. {cl}</b></div><div class="ev">{e}</div></div></li>' for i, (t, cl, e) in enumerate(findings))

exps = [
 ('1', 'Last-layer grid under Convention C', '60 cells &times; 10 configurations; both partials within token strings', '5 Qwen cells 10/10', 'v20.5 &sect;4'),
 ('1', 'Held-out passage halves', 'Both partials in each half', 'All five Qwen cells 10/10 in both halves', 'v20.5 &sect;10'),
 ('1', 'Robustness battery', 'Winsorised, ranks, p<sub>min</sub> control, drop strongest expert', '9&ndash;10/10 in every robust cell', 'v20.5 &sect;10'),
 ('1', 'Two floor methods; old scale as sensitivity arm', 'ESS from CE<sub>small</sub> vs IG; Convention A', 'Verdict changes in 1 of 48 cells; Qwen cells 10/10 under A', 'v20.5 &sect;2, &sect;4'),
 ('2', 'Same grid, ERNIE and LFM2, incl. six unseen benchmarks', 'Two-sided test at the last layer; pre-registered out-of-domain rule', 'race 10/10 in both; ERNIE also cnn, mbpp; LFM2 none out of domain', 'v20.5 &sect;4, &sect;8'),
 ('3', 'Four-partner compute sweep under C', 'Qwen3-30B-A3B with 0.6B, 1.7B, 4B, 8B partners; 9 benchmarks', f'Family r | CE<sub>small</sub> {f3(F["part_s_c"][0])} &rarr; {f3(F["part_s_c"][3])}; race, fineweb persist', 'v20.5 &sect;5'),
 ('4', 'Router baselines under C', 'Entropy, top probability, logit gap; + hidden-state norm', '10 of 13 cells keep &ge;50%; norm absorbs race, fineweb', 'v20.5 &sect;6'),
 ('5', 'Depth profiles, 10 configurations', 'Qwen 19, Gemma 18, ERNIE 15, LFM2 8 layers', 'Qwen late (41&ndash;47); ERNIE mid; Gemma inverts from mid depth', 'v20.5 &sect;7'),
 ('5', 'Out-of-sample layer selection; pooled HC; lambada position', 'Choose on half A, test on half B', 'ERNIE L18 replicates; pooling adds nothing in Qwen; lambada not the final word', 'v20.5 &sect;7, &sect;10'),
 ('6', 'Gemma on 15 benchmarks under C', 'Two-sided test, held-out halves, baselines', 'Inverted in 3 cells 9&ndash;10/10; never positive', 'v20.5 &sect;4, &sect;8'),
 ('6', 'All-layer shared-expert ablation, HC recomputed', 'Scale 1, 0.75, 0.5; race, fineweb, comp_math', 'Inversion shrinks &le;27% within budget; rule not met', 'v20.5 &sect;9'),
 ('7', 'Code vs prose at matched loss', 'Deciles of mean loss, both partials, four families', 'Qwen 10/10 and 8/10; Gemma 10/10; ERNIE, LFM2 no gap', 'v20.5 &sect;10'),
]
et = ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{cc}</td><td>{d}</td><td style="white-space:nowrap">{e}</td></tr>' for a, b, cc, d, e in exps)

strengths = [
 ('A pipeline audited end to end.', 'One function computes every number; with a single flag it reproduces all 72 original configurations exactly, and every table is regenerated from the records.'),
 ('A strict, pre-registered test.', 'Both partials beyond autocorrelation-corrected floors, same sign, in 9 of 10 null configurations, within the same token string, on held-out halves.'),
 ('A dose-response.', 'With the large model fixed, the association shrinks as the dense partner grows &mdash; the closest the study comes to causal evidence.'),
 ('More than one family.', 'race passes in Qwen, ERNIE and LFM2, ERNIE generalises to unseen news and code, and Gemma provides a robust contrast.'),
 ('Localised in depth.', 'The Qwen effect lives in the last seven MoE layers, and the depth profile shows why earlier layers fail (they track difficulty, not the gap).'),
]
weak = [
 ('Small effects.', 'r 0.02&ndash;0.14; HC adds at most 0.4% of within-string IG variance beyond router statistics.'),
 ('Shared with the hidden-state norm.', 'The norm removes most of race and fineweb in Qwen; it was measured only there.'),
 ('Heterogeneous families.', 'ERNIE and LFM2 are negative on some benchmarks; LFM2 passes nothing out of domain.'),
 ('Several earlier claims did not survive the fix.', 'competition_math, dialogue and chat.'),
 ('Mostly observational.', 'The one intervention (Gemma shared experts) rejects an explanation but does not supply one.'),
]
li2 = lambda xs: ''.join(f'<li><b>{a}</b> {b}</li>' for a, b in xs)

abstract = ("Mixture-of-Experts (MoE) language models route each token to a few experts, but whether that routing reveals where model scale pays off is unknown. "
 "For each token, we treat a large MoE model&rsquo;s per-expert routing scores as p-values against an estimated null and summarize them with Higher Criticism (HC), a statistic designed to detect a few strong departures among many tests. "
 "We relate HC to Intelligence Gain (IG), the per-token loss reduction over a dense small model from the same family, and require the association to hold with either model&rsquo;s loss controlled, within occurrences of the same token string, in at least nine of ten null-estimation configurations and above autocorrelation-corrected noise floors. "
 "Across four model pairs (Qwen3, ERNIE-4.5, Gemma 4, LFM2), routing anomalies in the final MoE layer mark where scale helps in three families: in Qwen3-30B-A3B on web, cloze and reading-comprehension text and on unseen news and scientific papers, replicating on held-out passages and surviving outlier, rank and single-expert controls, and on reading comprehension in ERNIE-4.5 and LFM2 as well, with ERNIE-4.5 generalising to unseen news and code. "
 "Holding the large model fixed, the association weakens as the dense partner grows from 0.6B to 8B parameters, yet persists on two benchmarks even when the dense model has 2.4&times; the MoE model&rsquo;s active parameters. "
 "A depth sweep localises Qwen3&rsquo;s effect to the last seven MoE layers, where earlier layers track only token difficulty. "
 "Gemma 4 shows a robust reversed association that weakening its shared experts does not remove, and in Qwen3 and Gemma 4 code is where the signal breaks down. "
 "The association is largely distinct from router entropy, top-expert probability and logit gap but shared in part with the hidden-state norm, and it is small (r &le; 0.14), positioning HC as a diagnostic of scaling benefit rather than a routing signal.")
leave = ['The history of the standardization fix: the paper describes the corrected method; the old scale can appear only as a one-line sensitivity check.',
 'Everything computed with wikitext standardization (the first layer sweep, the v18 shared-expert result) and the v20 ablation rerun that reused unablated routing.',
 'Pooled multi-layer HC (no gain), the lambada final-word analysis, and the detailed Gemma layer-27 anomaly.',
 'Exploration from earlier cycles: retired IG-percentile slices, top-ventile surges, rejected code hypotheses (copying, naming), the dialogue tag test.']
stay = [('ERNIE&rsquo;s and LFM2&rsquo;s negative cells, and Gemma&rsquo;s inversion,', 'in the main table. The cross-family claim is about specific benchmarks, and dropping the cells that go the other way would overstate it.'),
 ('All six unseen benchmarks for every family,', 'not only those that pass: the out-of-domain test was pre-registered on six; for Qwen dialogue, chat and both code sets fail, and LFM2 passes none.'),
 ('ERNIE&rsquo;s positive code result,', 'beside Qwen&rsquo;s and Gemma&rsquo;s code break: dropping it would turn a family-specific pattern into a false general claim about code.'),
 ('The hidden-state norm,', 'named as the statistic that shares most of the signal; claiming HC is distinct from &ldquo;router statistics&rdquo; without it would be contradicted by our own numbers.'),
 ('ERNIE&rsquo;s mid-depth result with its qualification,', 'if it is reported at all: the router statistics explain the CE<sub>big</sub> side there.'),
 ('Effect sizes and the variance increment,', 'stated once in the abstract and once in the limitations.')]

body = f'''<body>
<nav><span class="brand">HC-MoE</span><a href="#setup">Setup</a><a href="#findings">Findings</a><a href="#experiments">Experiments</a><a href="#dose">Dose-response</a><a href="#assessment">Strengths &amp; weaknesses</a><a href="#abstract">Paper abstract</a></nav>
<main>
<header class="report-header"><h1>HC-MoE: Summary of Work and Paper Draft</h1>
<div class="subtitle">What the study has established through v20.5 under the corrected standardization, the experiments behind each claim, an honest assessment, and a draft abstract</div>
<div class="date">October 29, 2026 &middot; through report v20.5</div>
<div class="abstract"><b>In one sentence.</b> Anomalous routing in the last MoE layers marks the tokens where a larger model helps most &mdash; robustly in Qwen3, on specific benchmarks (and in ERNIE out of domain) in ERNIE and LFM2, in reverse in Gemma &mdash; and the effect shrinks, without fully vanishing, as the compute gap between the models closes.</div></header>

<section id="setup"><h2>1. Setup</h2>
<p>The question is whether a large MoE model&rsquo;s routing tells us, token by token, where it beats a smaller model of the same family. For each token we treat the per-expert routing scores of the large model as p-values against an estimated null, after standardizing each benchmark against its own per-expert median and MAD, and summarize them with Higher Criticism (HC). We relate HC to <b>Intelligence Gain</b>, IG = CE<sub>small</sub> &minus; CE<sub>big</sub>.</p>
{tbl(['Family', 'Large (MoE)', 'Small (dense)', 'Routing in the last MoE layer', 'Active ratio'], [['Qwen', 'Qwen3-30B-A3B', 'Qwen3-0.6B', '128 experts, top-8, no shared', '5.5&times;'], ['ERNIE', 'ERNIE-4.5-21B-A3B', 'ERNIE-4.5-0.3B', '64 experts, top-6, 2 shared', '10&times;'], ['Gemma', 'gemma-4-26B-A4B', 'gemma-4-E2B', '128 experts, top-8, 1 shared', '1.7&times;'], ['Liquid', 'LFM2-8B-A1B', 'LFM2-700M', '32 experts, top-4', '2.0&times;']], num_from=4)}
<ul class="plain">
<li><b>Test:</b> within the same token string, r(HC, IG | CE<sub>small</sub>) and r(HC, IG | CE<sub>big</sub>); a cell passes if both are beyond its floor with the same sign in &ge;9 of 10 configurations (5 null estimators &times; 2 HC variants).</li>
<li><b>Grid:</b> 9 studied and 6 unseen benchmarks for every family (60 cells); the last MoE layer, plus depth profiles in all four families.</li>
<li><b>Inference:</b> floors at each cell&rsquo;s effective sample size (the larger of two estimates), passage-level bootstrap, held-out halves, decision rules written before each experiment.</li></ul></section>

<section id="findings"><h2>2. Main findings</h2>
<p>Grade A = holds under every check run; B = holds, with a named sensitivity or a limited sample.</p>
<div class="tier-legend"><span><span class="tier tier-A">A</span> robust under every check</span><span><span class="tier tier-B">B</span> holds, with a named sensitivity</span></div>
<ul class="so-far">{fl}</ul>
<div class="caveat"><strong>Changed by the standardization fix.</strong> competition_math (previously the strongest Qwen cell) needs the old scale; dialogue and chat no longer pass out of domain; ERNIE and LFM2 move from &ldquo;null&rdquo; to positive on specific benchmarks.</div>
</section>

<section id="experiments"><h2>3. Experiments behind each finding</h2>
<p>Each claim rests on experiments that were run, checked against earlier drops, and graded against rules fixed beforehand.</p>
<div class="table-wrap"><table><thead><tr><th>Finding</th><th>Experiment</th><th>What it tested</th><th>Result</th><th>Report</th></tr></thead><tbody>{et}</tbody></table></div>
{card(1, 'Every last-layer cell', 'w_cells', 'Two-sided strength (the weaker partial when both agree in sign, zero otherwise), mean of ten configurations. ERNIE and LFM2 were not run on the six unseen benchmarks.', h=340)}
</section>

<section id="dose"><h2>4. Dose-response</h2>
{card(2, 'The association falls as the dense partner grows', 'w_dose', 'Qwen3-30B-A3B held fixed; family means over nine benchmarks and ten configurations, under the corrected standardization. Source: HC-MoE v20.5 &sect;5.')}
<p><b>Raw</b> is the plain correlation r(HC, IG). The two partials hold one model&rsquo;s loss fixed: with CE<sub>small</sub> fixed, IG varies only through how well the big model does; with CE<sub>big</sub> fixed, only through how badly the small model does. Both fall by about four fifths between the 0.6B and 8B partners, and the family mean is inside the floor at 8B &mdash; but race and fineweb still pass with the 8B partner. This is the figure the paper should lead with.</p>
</section>

<section id="assessment"><h2>5. Strengths and weaknesses</h2>
<div class="assess">
<div class="col strong"><h4>Strengths</h4><ul>{li2(strengths)}</ul></div>
<div class="col weak"><h4>Weaknesses</h4><ul>{li2(weak)}</ul></div>
</div></section>

<section id="abstract"><h2>6. Draft paper abstract</h2>
<div class="abstract-card" style="background:var(--surface);border:1px solid var(--border);border-left:5px solid var(--c-big);border-radius:8px;padding:1.5rem 1.75rem">
<div style="font-size:1.15rem;font-weight:700;color:var(--heading);margin-bottom:0.9rem">Routing Anomalies Mark Where Scale Helps: Higher Criticism on Mixture-of-Experts Routing</div>
<p style="line-height:1.75;margin:0">{abstract}</p></div>
<div class="assess" style="margin-top:1.5rem">
<div class="col open"><h4>What the paper can leave out</h4><p style="font-size:0.88rem">Exploration, dead ends and superseded computations; omitting them does not change what the claims mean.</p><ul>{''.join(f'<li>{x}</li>' for x in leave)}</ul></div>
<div class="col strong"><h4>What should stay</h4><p style="font-size:0.88rem">Leaving these out would change what the remaining claims mean, and a reviewer can find them.</p><ul>{li2(stay)}</ul></div>
</div></section>
</main>
<script>
const CHART_SPECS = {json.dumps(specs)};
{BOOT}'''
save('/home/claude/v205/HC_MoE_Work_Summary_v20_5.html', head('HC-MoE Work Summary v20.5') + body)
