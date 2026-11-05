from common import *

c = CELL; F = N['dose_fam']; DB = N['dose_bm']
q5 = ['fineweb-edu', 'lambada', 'race', 'cnn_dailymail', 'scientific_papers']
cpq_s = N['cp']['Qwen|ce_mean_decile|mean_part_s_c']; cpq_b = N['cp']['Qwen|ce_mean_decile|mean_part_b_c']
ed = D['ernie']['cells']
GAB = {(r['dataset'], r['stats_mode'], r['alpha']): r for r in N['gab']}
cpe_s = N['cp']['ernie|ce_mean_decile|mean_part_s_c']

specs = [fig_cells('s_cells'), fig_dose('s_dose'), fig_qdepth('s_qdepth')]
for d_ in specs[0]['data']['datasets']:
    if d_.get('label', '').startswith('null envelope'): d_['label'] = 'typical floor (±0.0128)'

body = f'''<body>
<nav><span class="brand">HC-MoE</span><a href="#setup">Setup</a><a href="#findings">Findings</a><a href="#figs">Evidence</a><a href="#limits">Limits</a><a href="#next">Next</a></nav>
<main>
<header class="report-header"><h1>HC-MoE: Summary of Findings</h1>
<div class="subtitle">A short companion to the full v20.5 report &mdash; what the study has established under the corrected standardization, how firmly, and what remains open</div>
<div class="date">October 29, 2026 &middot; summary of v20.5</div>
<div class="abstract"><b>In one paragraph.</b> For each token we compute Higher Criticism (HC) over the routing p-values of a large MoE model and ask whether it tracks Intelligence Gain (IG = CE<sub>small</sub> &minus; CE<sub>big</sub>) beyond the difficulty of either model. v20 fixes a scaling bug in the standardization (the MAD was applied twice) and regrades everything with a two-sided rule: both partial correlations, holding each model&rsquo;s loss fixed in turn, beyond the floor with the same sign in at least 9 of 10 configurations. <b>In Qwen3, HC marks the tokens where scale helps</b> on fineweb, lambada, race, news and scientific text, in every configuration and both held-out halves. <b>The same holds in ERNIE and LFM2 on race</b> and a few other benchmarks, and in ERNIE also on unseen news and code; <b>Gemma 4 is inverted</b>, and weakening its shared experts does not undo the inversion. The association shrinks as the dense partner closes the compute gap, but race and fineweb persist with an 8B partner. In Qwen the effect lives in the last seven MoE layers; ERNIE&rsquo;s is strongest at mid depth, where simple router statistics explain it. Effects are small (r 0.02&ndash;0.14).</div></header>

<section id="setup"><h2>1. Setup</h2>
{tbl(['Family', 'Large (MoE)', 'Small (dense)', 'Routing in the last MoE layer', 'Active ratio'], [['Qwen', 'Qwen3-30B-A3B', 'Qwen3-0.6B', '128 experts, top-8, no shared', '5.5&times;'], ['ERNIE', 'ERNIE-4.5-21B-A3B', 'ERNIE-4.5-0.3B', '64 experts, top-6, 2 shared', '10&times;'], ['Gemma', 'gemma-4-26B-A4B', 'gemma-4-E2B', '128 experts, top-8, 1 shared', '1.7&times;'], ['Liquid', 'LFM2-8B-A1B', 'LFM2-700M', '32 experts, top-4', '2.0&times;']], num_from=4)}
<ul>
<li><b>Statistic.</b> Per token, HC over the per-expert routing p-values; each benchmark&rsquo;s routing scores are standardized against its own per-expert median and MAD (corrected scale, &ldquo;Convention C&rdquo;), then compared with nulls fitted on wikitext.</li>
<li><b>Test.</b> Within the same token string, r(HC, IG | CE<sub>small</sub>) and r(HC, IG | CE<sub>big</sub>). A cell passes if both are beyond its floor with the same sign in &ge;9 of 10 configurations (5 null estimators &times; 2 HC variants).</li>
<li><b>Grid.</b> 9 studied and 6 unseen benchmarks for all four families (60 cells); last MoE layer, plus depth profiles (Qwen 19 layers, Gemma 18, ERNIE 15, LFM2 8).</li>
<li><b>Inference.</b> Floors at each cell&rsquo;s effective sample size (the larger of two estimates), passage-level bootstrap, held-out halves, decision rules fixed before each run. The old scale is kept as a sensitivity arm; it reproduces every earlier number.</li></ul></section>

<section id="findings"><h2>2. Findings</h2>
<p>Tier A: holds under every check run. Tier B: holds, with a named sensitivity or a limited sample.</p>
<ul class="so-far">
{li('A', '<b>In Qwen3, HC marks the tokens where scale helps</b> &mdash; beyond either model&rsquo;s difficulty, token by token.', 'Both partials positive in 10/10 configurations on fineweb, lambada and race, and on unseen news (cnn) and scientific papers: ' + '; '.join(f'{SHORT[b]} {f3(c[("Qwen", b)]["c_s"])} / {f3(c[("Qwen", b)]["c_b"])}' for b in q5) + '. Each replicates in both held-out halves, survives winsorising, ranks and removing the strongest expert, and holds under the old scale.')}
{li('B', '<b>The same association appears in ERNIE and LFM2 on specific benchmarks; race is positive in three of four families.</b>', f'ERNIE: race, fineweb, tinystories, and out of domain cnn and mbpp (10/10). LFM2: race, pg19 (10/10), lambada (9/10), but none of its six unseen benchmarks. Other benchmarks in both families go the other way.')}
{li('B', '<b>The association shrinks as the dense partner closes the compute gap.</b>', f'Large model fixed, partner 0.6B &rarr; 8B: family-mean r | CE<sub>small</sub> {" &rarr; ".join(f3(v) for v in F["part_s_c"])}, r | CE<sub>big</sub> {" &rarr; ".join(f3(v) for v in F["part_b_c"])}. race and fineweb still pass with the 8B partner, which has 2.4&times; the MoE model&rsquo;s active parameters.')}
{li('B', '<b>HC is distinct from router confidence, but shares much with the hidden-state norm.</b>', f'After entropy, top probability and logit gap, at least half of both partials remains in 10 of 13 positive cells tested. Adding the hidden-state norm (Qwen): lambada keeps {pct(B["Qwen_L47_lambada"]["R4s"])}, race {pct(B["Qwen_L47_race"]["R4s"])}, fineweb {pct(B["Qwen_L47_fineweb-edu_max512_hnorm"]["R4s"])} of r | CE<sub>small</sub>. HC adds 0.01&ndash;0.4% of within-string IG variance.')}
{li('B', '<b>In Qwen, the effect is a property of late routing.</b>', 'race and fineweb pass only in layers 41&ndash;47. Earlier layers track the small model&rsquo;s difficulty instead (r | CE<sub>small</sub> positive, r | CE<sub>big</sub> negative). lambada is the exception and passes from layer 5, through context tokens rather than the final word. Pooling the last layers does not beat layer 47.')}
{li('B', '<b>In ERNIE, the effect is stronger at mid depth, but there it is not distinct from router statistics.</b>', f'Layers 14&ndash;22 pass on race, fineweb, pg19 and tinystories; a layer chosen on half the passages (L18) beats the last layer on the other half. The gain is in r | CE<sub>small</sub> (race {f3(ed["race"]["18"]["s"])} vs {f3(ed["race"]["27"]["s"])}), and the router statistics remove r | CE<sub>big</sub> at L18&ndash;20.')}
{li('A', '<b>Gemma 4 is never positive at its last layer, and robustly inverted on humaneval, race and news.</b>', 'Both partials negative in 9&ndash;10 of 10 configurations in those cells, both held-out halves, 94% kept after all router statistics. With depth: positive only at layer 7, inverted from mid depth, most strongly at layer 27.')}
{li('B', '<b>Gemma&rsquo;s shared experts do not explain its inversion.</b>', f'Scaling every layer&rsquo;s shared expert to 0.75, with HC recomputed from the ablated model, shrinks the race inversion by {GAB[("race","alpha_self_stats",0.75)]["inversion_shrink_pct_ces"]:.0f}% / {GAB[("race","alpha_self_stats",0.75)]["inversion_shrink_pct_ceb"]:.0f}% at a +3.4% loss cost (rule: &ge;50%); it disappears only when loss rises 30&ndash;82%.')}
{li('B', '<b>In Qwen and Gemma, code is where the signal breaks down &mdash; not in ERNIE.</b>', f'At matched loss, Qwen&rsquo;s code sits below prose in {cpq_s["below"]}/10 deciles on r | CE<sub>small</sub> and {cpq_b["below"]}/10 on r | CE<sub>big</sub>; Gemma 10/10 on both. ERNIE shows no such gap ({cpe_s["below"]}/10), and mbpp is its strongest positive cell.')}
</ul>
<div class="caveat"><strong>What the fix removed.</strong> competition_math (previously tier A) needs the old scale to pass; dialogue and chat, which passed out of domain in v15, fail under the corrected scale. Gemma&rsquo;s earlier &ldquo;41%&rdquo; shared-expert result is retired; the redone ablation rejects the explanation.</div></section>

<section id="figs"><h2>3. Key evidence</h2>
{card(1, 'Every last-layer cell', 's_cells', 'Two-sided strength: the weaker of the two partials when both agree in sign, zero otherwise; mean of ten configurations. Dashed: typical floor. Unseen benchmarks on the right.', h=360)}
{card(2, 'Dose-response: the association as the dense partner grows', 's_dose', 'Qwen3-30B-A3B held fixed; family means over nine benchmarks and ten configurations.')}
{card(3, 'Depth: Qwen&rsquo;s effect arrives in the last layers', 's_qdepth', 'Two-sided strength by layer. race and fineweb are zero until layer 41; lambada is positive throughout.')}
</section>

<section id="limits"><h2>4. What is not established</h2><ul>
<li><b>Effect size.</b> Correlations are 0.02&ndash;0.14 and HC adds at most 0.4% of within-string IG variance beyond router statistics: a diagnostic of scaling benefit, not a routing signal.</li>
<li><b>Generality.</b> ERNIE and LFM2 are positive on some benchmarks and negative on others; LFM2 passes nothing out of domain.</li>
<li><b>The hidden-state norm</b> absorbs much of race and fineweb; it was measured only for Qwen.</li>
<li><b>Why Gemma inverts</b> is open: not the shared experts.</li>
<li><b>Why code</b> breaks the signal in Qwen and Gemma but carries it in ERNIE.</li>
<li><b>The difficulty correlations</b> (r(HC, CE)) have not yet been recomputed under the corrected scale.</li></ul></section>

<section id="next"><h2>5. Next steps</h2><ol>
<li>Recompute r(HC, CE<sub>big</sub>) and r(HC, CE<sub>small</sub>) under the corrected scale (existing data).</li>
<li>Hidden-state norm for every headline cell, and every Qwen layer between 36 and 41.</li>
<li>Dissect ERNIE&rsquo;s mbpp signal by token class.</li></ol>
<p style="font-size:0.85rem;color:#64748b">Full evidence, every check and every pre-registered rule: <code>HC_MoE_Report_v20_5.html</code>.</p></section>
</main>
<script>
const CHART_SPECS = {json.dumps(specs)};
{BOOT}'''
save('/home/claude/v205/HC_MoE_Summary_v20_5.html', head('HC-MoE Summary v20.5') + body)
