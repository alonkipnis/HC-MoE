import json, re
s = open('/home/claude/v19/stage1.html', encoding='utf-8').read()
N = json.load(open('/home/claude/v19/n19.json'))
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def rep(a, b, cnt=1):
    global s
    n = s.count(a); assert n == cnt, (n, a[:100]); s = s.replace(a, b)
HN = lambda d, k='ret4', m='Qwen': N['HN'][f'{m}|{d}'][k]
r4 = f'{pct(HN("competition_math"))}, {pct(HN("lambada"))} and {pct(HN("race"))}'

rep('<h1>HC-MoE v18: Nine Benchmarks, Shared Experts, and a Drop That Does Not Reproduce</h1>', '<h1>HC-MoE v19: The Hidden-State Norm, and Code Against Prose</h1>')
i = s.index('<div class="subtitle">'); j = s.index('</div>', i)
s = s[:i] + '<div class="subtitle">The v18 proposals: the hidden-state norm rerun inside the pipeline that reproduces, code against prose at matched small-model loss, and a finer shared-expert scan in Gemma</div>' + s[j + 6:]
rep('October 18, 2026  &middot;  v18', 'October 18, 2026  &middot;  v19')
s = re.sub(r'<title>[^<]*</title>', '<title>HC-MoE v19</title>', s, count=1)
i = s.index('<div class="abstract">'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="abstract"><b>Abstract.</b> Three drops, each checked against what it should reproduce and graded against the rule written for it in v18. '
 f'<b>Hidden-state norm.</b> Rerun inside the v16 baseline code, the three-baseline arm reproduces v16 exactly in four of five cells; fineweb-edu, again on its 207,061-token subset, is excluded. With the norm added as a fourth baseline, Qwen keeps {r4} of its within-string scale-specific association in competition_math, lambada and race. One cell clears half and one is below a quarter, so neither branch of the rule is met. The norm takes little from competition_math and lambada and most of what remained in race; Gemma&rsquo;s inversion is untouched. '
 f'<b>Code against prose.</b> In prose, Qwen&rsquo;s scale-specific association <i>grows</i> with the small model&rsquo;s loss; in code it falls negative. At matched loss all {N["NQ"]} code deciles sit below prose (mean gap {f3(N["QGAP"])}), and {N["G_BELOW"]} of {N["NG"]} for Gemma. By the rule, the code exception is specific to code, not a hard-token effect. '
 '<b>Gemma, finer scaling.</b> The drop scaled only the last layer&rsquo;s shared expert and contains losses but no correlations, so it cannot be analysed; the all-layer scan remains open.</div>') + s[j:]

i = s.index('<h3>What this drop changes</h3>'); j = s.index('<div class="finding null"><h4>Revision v14.1')
s = s[:i] + ('<h3>What this drop changes</h3>\n'
 f'<div class="finding null"><h4>The hidden-state norm: between the rule&rsquo;s branches</h4><p><b>Norm in the v16 pipeline (P0).</b> Reproduces v16 exactly in four of five cells (fineweb-edu excluded). After four baselines: {r4}. Neither &ldquo;half in three cells&rdquo; nor &ldquo;below a quarter in two&rdquo;. The norm absorbs most of race&rsquo;s remainder. Section 29.</p></div>\n'
 f'<div class="finding scale"><h4>The code exception is specific to code</h4><p><b>Code vs prose at matched loss (P2).</b> Prose rises with loss, code falls; {N["Q_BELOW"]}/{N["NQ"]} Qwen and {N["G_BELOW"]}/{N["NG"]} Gemma code deciles below prose at the same loss. Rule&rsquo;s second branch. Section 30.</p></div>\n'
 '<div class="finding null"><h4>Gemma finer scaling: not analysable</h4><p><b>Finer shared-expert scan (P1).</b> Last layer only, losses only. Section 31.</p></div>\n') + s[j:]

a = s.index('<ul class="so-far">'); b = s.index('</ul>', a)
items = re.findall(r'<li>.*?</li>(?=<li>|$)', s[a + len('<ul class="so-far">'):b], flags=re.S)
assert len(items) == 18
def add_ev(k, text):
    it = items[k]; j = it.rindex('</div></div></li>'); items[k] = it[:j] + ' <b>v19:</b> ' + text + it[j:]
assert 'simpler router statistics' in items[3]
add_ev(3, f'with the hidden-state norm as a fourth baseline (pipeline reproducing v16 exactly), {r4} remain in competition_math, lambada and race; the norm absorbs most of race&rsquo;s remainder. Neither branch of the rule is met.')
assert 'code' in items[4]
add_ev(4, f'specific to code: at matched small-model loss, all {N["NQ"]} Qwen code deciles sit below prose (mean gap {f3(N["QGAP"])}); prose rises with loss, code falls. Same in Gemma ({N["G_BELOW"]}/{N["NG"]}).')
s = s[:a + len('<ul class="so-far">')] + ''.join(items) + s[b:]
i = s.index('<h3>Main findings so far</h3>'); j = s.index('<ul class="so-far">', i)
s = s[:i] + ('<h3>Main findings so far</h3>\n<p>What the project holds after this report. Every result is about each large model&rsquo;s <b>last MoE layer</b>. v19 completes the baseline set &mdash; HC survives all four in two of three reproducible tier-A cells &mdash; and shows the code exception is specific to code. Nothing is withdrawn.</p>') + s[j:]

i = s.index('<h3>Changed this cycle</h3>'); j = s.index('<h3>Checks done</h3>', i)
s = s[:i] + ('<h3>Changed this cycle</h3>\n<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Was</th><th>From</th><th>Now</th></tr></thead><tbody>'
 f'<tr><td>HC adds to simpler last-layer statistics</td><td>B, three router baselines</td><td>v15</td><td>Stays B, now four baselines: {r4} remain; clear in two of three cells.</td></tr>'
 '<tr><td>Code exception</td><td>domain effect, unexplained</td><td>v16</td><td>Specific to code: at matched loss prose rises, code falls.</td></tr>'
 '<tr><td>Gemma finer shared-expert scan</td><td>proposed</td><td>v18</td><td>Not analysable (last layer only, no correlations).</td></tr></tbody></table></div>\n') + s[j:]

rep('<p>Recorded as each check ran. Four new checks this cycle, one of them failing its reproduction test.</p>',
    '<p>Recorded as each check ran. Three new checks this cycle; one could not be analysed.</p>')
old = re.search(r'<tr><td><strong>First-occurrence strata</strong>.*?</tr>', s, flags=re.S).group(0)
s = s.replace(old, old +
 '<tr><td><strong>Hidden-state norm, v16 pipeline</strong><span class="since">v19</span></td><td>Whether HC adds to the norm of the hidden state entering the layer</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v19).</b> Three-baseline arm reproduces v16 exactly in 4/5 cells (fineweb-edu subset excluded). Four-baseline retention {r4}; neither rule branch met.</td></tr>'
 '<tr><td><strong>Code against prose at matched loss</strong><span class="since">v19</span></td><td>Whether the code exception is a hard-token effect</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v19).</b> CE<sub>small</sub> deciles, two code and two prose benchmarks, both families, 800 rows. Code below prose at matched loss: Qwen {N["Q_BELOW"]}/{N["NQ"]}, Gemma {N["G_BELOW"]}/{N["NG"]}. Specific to code.</td></tr>'
 '<tr><td><strong>Finer shared-expert scan</strong><span class="since">v19</span></td><td>Whether all-layer scaling halves Gemma&rsquo;s race inversion inside the budget</td><td><span class="status status-notrun">not analysable</span></td>'
 f'<td><b>New (v19).</b> Drop scaled the last layer only and contains losses but no correlations (max CE change +{N["GSMAX"]:.2f}%).</td></tr>', 1)
i = s.index('<div class="caveat"><b>Data gaps in this drop.</b>'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="caveat"><b>Data gaps in this drop.</b> fineweb-edu in the baseline rerun uses the 207,061-token subset and does not reproduce v16. The Gemma finer scan scaled only the last layer and has no correlation records.</div>') + s[j:]

qn = (f'<div class="finding scale"><h4>v19: four baselines, and code against prose</h4><p>Beyond router entropy, maximum probability, logit gap and the hidden-state norm, {pct(HN("competition_math"))} of the scale-specific association remains in competition_math and {pct(HN("lambada"))} in lambada; race keeps {pct(HN("race"))}. In prose the association grows with the small model&rsquo;s loss; in code it falls. Sections 29&ndash;30.</p></div>\n')
for sid, txt in (('qwen', qn),):
    i = s.index(f'<section id="{sid}">'); j = s.index('<div class="card">', i); s = s[:j] + txt + s[j:]

i = s.index('<div class="assess">'); j = s.index('</section>', i)
s = s[:i] + ('<div class="assess"><div class="col strong"><h4>Holds up</h4><ul>'
 '<li><b>On the nine studied benchmarks</b>, Qwen&rsquo;s &lsquo;where scale helps&rsquo; holds at every level tested.</li>'
 '<li><b>Dose-response on nine benchmarks</b>, to the floor at an 8B partner.</li>'
 f'<li><b>Distinct from four simpler last-layer statistics</b> in competition_math ({pct(HN("competition_math"))}) and lambada ({pct(HN("lambada"))}).</li>'
 '<li><b>Out of domain</b> on news, scientific text and chat; <b>code</b> is an exception specific to code, not to hard tokens.</li>'
 '<li><b>Gemma&rsquo;s raw inversion</b> on fifteen benchmarks; the shared experts contribute part of it.</li></ul></div>'
 '<div class="col weak"><h4>Not yet established</h4><ul>'
 f'<li><b>race</b> keeps only {pct(HN("race"))} after the hidden-state norm.</li>'
 '<li><b>How much HC adds</b>: small, about 0.1% of within-string IG variance.</li>'
 '<li><b>Why Gemma inverts</b> beyond the shared experts&rsquo; partial contribution.</li>'
 '<li><b>Why code reverses</b>: specific to code, mechanism unknown.</li>'
 '<li><b>One layer.</b></li></ul></div>'
 '<div class="col open"><h4>Open</h4><ul>'
 '<li><b>What in code makes hard tokens ones the large model gains less on</b>, while in prose they are where it gains most?</li>'
 '<li><b>Does an all-layer finer scan ever halve Gemma&rsquo;s race inversion</b> inside the loss budget?</li></ul></div></div>') + s[j:]

i = s.index('<section id="next">'); j = s.index('<section id="tables">')
h2 = re.match(r'<section id="next"><h2>[^<]*</h2>', s[i:]).group(0)
def exp(t, pr, cost, q, d, st, rule):
    return (f'<div class="exp"><h4>{t}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd><dt>Design</dt><dd>{d}</dd>'
            f'<dt>Statistic</dt><dd>{st}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>\n')
s = s[:i] + (h2 + '<h3>What happened to v18&rsquo;s list</h3>\n<div class="table-wrap"><table><thead><tr><th>v18 experiment</th><th>Status</th><th>Outcome</th></tr></thead><tbody>'
 '<tr><td>Hidden-state norm in the v16 pipeline (P0)</td><td><b>Done</b></td><td>Reproduces 4/5; between the rule&rsquo;s branches.</td></tr>'
 '<tr><td>Finer shared-expert scaling in Gemma (P1)</td><td>Delivered, <b>not analysable</b></td><td>Last layer only; no correlations.</td></tr>'
 '<tr><td>Code against prose at matched loss (P2)</td><td><b>Done</b></td><td>Specific to code.</td></tr>'
 '<tr><td>Layer sweep</td><td>Deferred</td><td>Last resort.</td></tr></tbody></table></div>\n<h3>Proposed</h3>\n'
 + exp('Finer all-layer shared-expert scan in Gemma', 'P1', '~half a day',
   'Does scaling every layer&rsquo;s shared expert halve the race inversion anywhere inside the 10% loss budget?',
   'All layers (not only the last) at &alpha; = 0.875, 0.75, 0.70, 0.65 on race and competition_math; recompute routing, HC and CE<sub>big</sub>; deliver correlation records, not only losses. Reconcile the &alpha; = 1 HC on competition_math with the main study first.',
   'r(HC, IG | CE<sub>small</sub>) against CE increase.',
   'If the race inversion halves at a setting inside the budget, the shared experts are implicated through routing; if it flattens before half, they remain a partial contributor.')
 + exp('fineweb-edu on the main grid&rsquo;s tokens', 'P2', 'an hour',
   'Two drops now use a 207,061-token subset of fineweb-edu. Do the partner sweep and the baselines hold on the full 477,782 tokens?',
   'Rerun fineweb-edu in the sweep and the four-baseline code on the main grid&rsquo;s token set.',
   'The 0.6B raw r (should be +0.0707) and four-baseline retention.',
   'If the 0.6B value reproduces +0.0707, fineweb-edu joins the four-baseline grading as a fourth tier-A cell.')
 + exp('What in code: a descriptive dissection of hard code tokens', 'P2', 'a few hours',
   'Hard prose tokens are where HC marks extra gain; hard code tokens are where it marks less. What are the hard code tokens?',
   'List the top-decile CE<sub>small</sub> code tokens with high HC and negative within-string IG residual; classify by syntactic role and position in the function.',
   'Composition of that set against the same set in prose.',
   'Descriptive; no claim changes unless one role accounts for most of the negative covariance.')
 + exp('Layer sweep (last resort)', 'Deferred', '~1 day per family if router logits were cached',
   'Does the relationship live only in the last MoE layer?', 'HC from every MoE layer&rsquo;s router.', 'The scale-specific partial by depth.',
   'Run only if the last-layer scope becomes decisive for a reviewer.')
 + '</section>\n') + s[j:]
open('/home/claude/v19/HC_MoE_Report_v19.html', 'w', encoding='utf-8').write(s)
print('ok', len(s))
