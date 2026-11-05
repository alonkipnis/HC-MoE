"""Short companion report for HC-MoE v17. Numbers come from the same recomputed sources as the full report."""
import json, re, sys, glob
sys.path.insert(0, glob.glob('/root/.claude/skills/synced/*/hc-stats-report/scripts')[0])
from report_builder import bar_chart, line_chart

full = open('/home/claude/v17/HC_MoE_Report_v17.html', encoding='utf-8').read()
head = full[:full.index('<body>')]
head = re.sub(r'<title>[^<]*</title>', '<title>HC-MoE Summary v17</title>', head, count=1)
boot = full[full.index('function hideUnderscore'):]          # theme + boot script, through </html>

cells = json.load(open('/home/claude/v17/cells36.json'))[:36]
R = {(c[0], c[1]): float(c[2].split()[0]) for c in cells}; TIER = {(c[0], c[1]): c[-1] for c in cells}
N15 = json.load(open('/home/claude/v15/numbers.json')); N16 = json.load(open('/home/claude/v16/n16.json')); N17 = json.load(open('/home/claude/v17/n17.json'))
QW = {r['dataset']: r for r in N15['QW']}; GW = {r['dataset']: r for r in N15['GW']}
QB = N16['QBX']; FAM = N17['FAM']; tq, tg, CC = N17['tq'], N17['tg'], N17['CC']
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
FLOOR = 0.0128
BM = ['lambada', 'competition_math', 'fineweb-edu', 'race', 'structured-wikipedia', 'hellaswag', 'pg19', 'tinystories', 'mmlu']
FAMS = ['Qwen', 'Gemma', 'ERNIE', 'Liquid']
famean = {f: sum(R[(f, b)] for b in BM) / 9 for f in FAMS}
qa = {f: sum(1 for b in BM if TIER[(f, b)] == 'A') for f in FAMS}
TA = ['competition_math', 'lambada', 'race', 'fineweb-edu']
P = ['0.6B', '1.7B', '4B', '8B']; RAT = {'0.6B': '5.5', '1.7B': '1.94', '4B': '0.82', '8B': '0.41'}
UN = ['cnn_dailymail', 'scientific_papers', 'ultrachat', 'dialogsum', 'humaneval', 'mbpp']
dr = [QB['dr2'][d] for d in TA]

specs = [
 bar_chart('s_fam', BM, {f: [R[(f, b)] for b in BM] for f in FAMS}, null_band=FLOOR, colors=['#f59e0b', '#f87171', '#60a5fa', '#2dd4bf']),
 line_chart('s_dose', [f'{p} ({RAT[p]}×)' for p in P], {'raw r(HC, IG)': [FAM['raw'][p] for p in P], 'r(HC, IG | CE_small)': [FAM['s'][p] for p in P]},
            x_title='Qwen3 dense partner (active-parameter ratio, large / small)', colors=['#94a3b8', '#f59e0b'], null_band=FLOOR),
 bar_chart('s_base', TA, {'HC, within strings | CE_small': [QB['ref'][d] for d in TA], 'after entropy, max probability, logit gap': [QB['joint'][d] for d in TA]},
           colors=['#f59e0b', '#f87171']),
 bar_chart('s_ood', UN, {'Qwen, within strings | CE_small': [QW[d]['wS'] for d in UN], 'Gemma, raw r': [GW[d]['raw'] for d in UN]}, colors=['#f59e0b', '#f87171']),
]
for lab, sg in (('measured floor', 1), ('_lo', -1)):
    specs[3]['data']['datasets'].append(dict(data=[round(sg * QW[d]['floor'], 5) for d in UN], label=lab, type='line', pointRadius=9, pointStyle='line',
                                             showLine=False, borderWidth=2.5, borderColor='rgba(226,232,240,0.95)', fill=False, order=-1))
def card(n, title, cid, cap, h=320):
    return (f'<div class="card"><h3 style="margin-top:0">Figure {n} &mdash; {title}</h3><div style="height:{h}px;position:relative"><canvas id="{cid}"></canvas></div>'
            f'<p style="font-size:0.85rem;color:#64748b;margin-top:0.75rem">{cap}</p></div>\n')
def li(tier, claim, ev):
    return f'<li><span class="tier tier-{tier}">{tier}</span><div class="body"><div class="claim">{claim}</div><div class="ev">{ev}</div></div></li>'

body = f'''<body>
<nav><span class="brand">HC-MoE</span><a href="#setup">Setup</a><a href="#findings">Findings</a><a href="#figs">Evidence</a><a href="#limits">Limits</a><a href="#next">Next</a></nav>
<main>
<header class="report-header"><h1>HC-MoE: Summary of Findings</h1>
<div class="subtitle">A short companion to the full v17 report &mdash; what the study has established, how firmly, and what remains open</div>
<div class="date">October 15, 2026 &middot; summary of v17</div>
<div class="abstract"><b>In one paragraph.</b> For each token we compute Higher Criticism (HC) over the routing p-values of a large MoE model&rsquo;s last MoE layer and ask whether it tracks token difficulty and Intelligence Gain (IG = CE<sub>small</sub> &minus; CE<sub>big</sub>), the loss reduction from a dense small model of the same family. Across four model pairs, nine studied benchmarks and six unseen ones, HC tracks difficulty in every family. In <b>Qwen3</b> it marks the tokens where scale helps: the association survives control for either model&rsquo;s loss within the same token string, replicates on held-out text, generalises to unseen prose and chat, and shrinks to zero as the dense partner grows from 0.6B to 8B. It is distinct from simpler router statistics, but small beyond them. <b>Gemma 4</b> shows a robust <i>reversed</i> association whose cause is still unknown. ERNIE and LFM2 show no consistent family-level effect. Code is the one domain where Qwen&rsquo;s association reverses.</div></header>

<section id="setup"><h2>1. Setup</h2>
<div class="table-wrap"><table><thead><tr><th>Family</th><th>Large (MoE)</th><th>Small (dense)</th><th>Routing in the measured layer</th><th class="num">Active ratio</th></tr></thead><tbody>
<tr><td>Qwen</td><td>Qwen3-30B-A3B</td><td>Qwen3-0.6B</td><td>128 experts, top-8, no shared</td><td class="num">5.5&times;</td></tr>
<tr><td>ERNIE</td><td>ERNIE-4.5-21B-A3B</td><td>ERNIE-4.5-0.3B</td><td>64 experts, top-6, 2 shared</td><td class="num">10&times;</td></tr>
<tr><td>Gemma</td><td>gemma-4-26B-A4B</td><td>gemma-4-E2B</td><td>128 experts, top-8, 1 shared</td><td class="num">1.7&times;</td></tr>
<tr><td>Liquid</td><td>LFM2-8B-A1B</td><td>LFM2-700M</td><td>32 experts, top-4</td><td class="num">2.0&times;</td></tr></tbody></table></div>
<ul>
<li><b>Statistic.</b> Per token, HC over the per-expert routing p-values of the <b>last MoE layer</b> of the large model (no other layer measured).</li>
<li><b>Targets.</b> CE<sub>big</sub>, CE<sub>small</sub> and IG; the scale-specific test is r(HC, IG) with either loss held fixed, within the same token string.</li>
<li><b>Grid.</b> Nine studied benchmarks; five null estimators &times; two HC variants = ten configurations per cell; a claim needs at least 9 of 10 configurations to agree.</li>
<li><b>Inference.</b> Floors at each cell&rsquo;s measured effective sample size (token autocorrelation costs about 2.6&times;), passage-level bootstrap, FDR control, held-out replication. Every follow-up experiment had its decision rule written before the data.</li></ul></section>

<section id="findings"><h2>2. Findings</h2>
<p>Tier A: holds under every check run. Tier B: holds, with a named sensitivity or a limited sample.</p>
<ul class="so-far">
{li('A', '<b>HC tracks token difficulty in every family</b>, with a family-specific sign.', '31 and 28 of 36 cells outside the floor against CE<sub>big</sub> and CE<sub>small</sub>. Gemma is negative on both; the other three positive.')}
{li('A', '<b>In Qwen3, HC marks the tokens where scale helps</b> &mdash; beyond either model&rsquo;s difficulty, token by token.', f'Mean r(HC, IG) {f3(famean["Qwen"])}, positive on all nine benchmarks, {qa["Qwen"]} tier A; strongest lambada +0.131 (95% CI +0.120 to +0.141). With either loss held fixed within the same token string, the four tier-A cells stay positive in every configuration (+0.027 to +0.073). Replicates on held-out halves (cell correlation 0.95); not driven by outlier tokens (winsorised 0.91&ndash;1.05 of r); carried by several experts (82% survives removing the strongest).')}
{li('B', '<b>The association scales with the compute gap and goes to zero when it closes.</b>', f'Large model fixed, dense partner 0.6B &rarr; 1.7B &rarr; 4B &rarr; 8B: family-mean scale-specific r {" &rarr; ".join(f3(FAM["s"][p]) for p in P)}; mean IG {" &rarr; ".join(f"{FAM["ig"][p]:.2f}" for p in P)} nats. Eight benchmarks (one cell of the sweep is a pipeline error and excluded); per benchmark the path is noisier, and pg19 and race stay positive throughout.')}
{li('B', '<b>HC carries information that simpler statistics of the same router do not &mdash; but not much more.</b>', f'After router entropy, maximum probability and logit gap are partialled out, {", ".join(pct(QB["ret"][d]) for d in TA)} of the scale-specific association remains in the four tier-A cells, every remainder excluding zero. HC adds {100*min(dr):.2f}&ndash;{100*max(dr):.2f}% of within-string IG variance beyond them. On their own, the simpler statistics predict IG about as well as HC.')}
{li('B', '<b>Out of domain, Qwen&rsquo;s association generalises &mdash; except on code.</b>', f'Pre-registered on six unseen benchmarks: positive under both controls on news ({f3(QW["cnn_dailymail"]["wS"])}), scientific papers ({f3(QW["scientific_papers"]["wS"])}), chat ({f3(QW["ultrachat"]["wS"])}) and dialogue ({f3(QW["dialogsum"]["wS"])}); negative on both code sets. The code reversal is a domain effect carried by first occurrences of identifiers, not one token class and not copying. The negative <i>raw</i> sign on chat comes from role tokens; on dialogue the speaker tag carries most of it (8 of 10 configurations).')}
{li('A', '<b>Gemma 4 shows a robust reversed association.</b>', f'Mean r(HC, IG) {f3(famean["Gemma"])} on the studied grid and negative on all six unseen benchmarks in every configuration (to {f3(GW["humaneval"]["raw"])} on humaneval). Ruled out as causes: the compute gap, format tokens, router confidence, and the last layer&rsquo;s shared expert at half strength. Its scale-specific part changes sign across benchmarks.')}
{li('B', '<b>ERNIE and LFM2 show no consistent family-level association with IG.</b>', f'Family means {f3(famean["ERNIE"])} and {f3(famean["Liquid"])}: strong cells of both signs cancel across benchmarks.')}
</ul></section>

<section id="figs"><h2>3. Key evidence</h2>
{card(1, 'r(HC, IG) by benchmark and family', 's_fam', 'Means of ten configurations on the studied grid. Dashed: the typical floor (&plusmn;0.0128). Qwen is positive everywhere, Gemma negative in eight of nine.', h=340)}
{card(2, 'Dose-response: the association as the dense partner grows', 's_dose', 'Qwen3-30B-A3B held fixed; family means over eight benchmarks and ten configurations.')}
{card(3, 'Distinct from simpler router statistics', 's_base', 'Qwen&rsquo;s four tier-A cells, within token strings with CE<sub>small</sub> held fixed; red after partialling the three router baselines.', h=280)}
{card(4, 'Six unseen benchmarks', 's_ood', 'Amber: Qwen&rsquo;s scale-specific association (the pre-registered test); white ticks are its measured floors. Red: Gemma&rsquo;s raw correlation. The two code sets are on the right.')}
</section>

<section id="limits"><h2>4. What is not established</h2><ul>
<li><b>One layer.</b> Every result is about the last MoE layer&rsquo;s router. Whether earlier layers carry the same information is untested (a layer sweep is deferred as costly).</li>
<li><b>Effect size.</b> Correlations are 0.02&ndash;0.13 and HC&rsquo;s increment over simple router statistics is about 0.1% of within-string IG variance. This is a diagnostic and interpretability result, not a usable routing signal yet.</li>
<li><b>Why Gemma inverts</b>, and why the effect reverses on code, are open.</li>
<li><b>The hidden-state norm</b>, the fourth planned baseline, has not been tested.</li>
<li><b>The dialogue formatting claim</b> misses its rule by one configuration; the StudentT nulls disagree.</li></ul></section>

<section id="next"><h2>5. Next steps</h2><ol>
<li>Regenerate the misaligned fineweb-edu cell of the partner sweep (an hour) so the dose-response covers all nine benchmarks.</li>
<li>Add the hidden-state-norm baseline (a few hours).</li>
<li>Ablate the shared experts of all of Gemma&rsquo;s layers, within a 10% loss budget, to test whether they shape its routing.</li>
<li>Stratify first-occurrence identifiers to test the unpredictable-naming account of code.</li></ol>
<p style="font-size:0.85rem;color:#64748b">Full evidence, every check and every pre-registered rule: <code>HC_MoE_Report_v17.html</code>.</p></section>
</main>
<script>
const CHART_SPECS = {json.dumps(specs)};
{boot}'''
html_out = head + body
open('/home/claude/v17/HC_MoE_Summary_v17.html', 'w', encoding='utf-8').write(html_out)
print('ok', len(html_out), famean, qa)
