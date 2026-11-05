import json, re
s = open('/home/claude/v18/stage1.html', encoding='utf-8').read()
N = json.load(open('/home/claude/v18/n18.json'))
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def rep(a, b, cnt=1):
    global s
    n = s.count(a); assert n == cnt, (n, a[:100]); s = s.replace(a, b)
P = ['0.6B', '1.7B', '4B', '8B']
f9 = ' &rarr; '.join(f3(N['FAM9']['s'][p]) for p in P)
G = lambda d, a, k: N['G'][f'{d}|{a}'][k]
ST = lambda c, k: N['ST'][f'{c}|{k}']['r']

rep('<h1>HC-MoE v17: The Exceptions Tested, and the Compute Gap Taken to Zero</h1>', '<h1>HC-MoE v18: Nine Benchmarks, Shared Experts, and a Drop That Does Not Reproduce</h1>')
i = s.index('<div class="subtitle">'); j = s.index('</div>', i)
s = s[:i] + '<div class="subtitle">The four experiments proposed in v17: fineweb-edu regenerated, the hidden-state norm, Gemma&rsquo;s shared experts in every layer, and first-occurrence identifiers by loss and name type</div>' + s[j + 6:]
rep('October 15, 2026  &middot;  v17', 'October 18, 2026  &middot;  v18')
s = re.sub(r'<title>[^<]*</title>', '<title>HC-MoE v18</title>', s, count=1)
i = s.index('<div class="abstract">'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="abstract"><b>Abstract.</b> All four experiments proposed in v17 were run; each is graded against its pre-registered rule, and every drop is checked against the earlier ones it should reproduce. '
 f'<b>Dose-response, nine benchmarks.</b> fineweb-edu, misaligned in v17, is regenerated and now reproduces experiment 3 exactly. It does so on a 207,061-token subset, so the rule&rsquo;s check against the main grid&rsquo;s +0.0707 is not met as written. With it restored, the family-mean scale-specific association falls {f9} as Qwen3-30B-A3B&rsquo;s dense partner grows from 0.6B to 8B. fineweb-edu itself falls only by about half ({f3(N["FWA"]["s"]["0.6B"])} &rarr; {f3(N["FWA"]["s"]["8B"])}). '
 f'<b>Hidden-state norm: not graded.</b> The drop&rsquo;s reference statistic matches neither the within-string partial of v15&ndash;v16 nor the raw-level partial, and its three-baseline arm, a direct replication, gives {min(N['RET3'].values()):.0%}&ndash;{max(N['RET3'].values()):.0%} retention across the five cells where v16 measured {min(N['OLD3'].values()):.0%}&ndash;{max(N['OLD3'].values()):.0%}, cell by cell different. Within the drop, adding the norm to the three router baselines changes what remains by a factor of {N["J43"][0]:.2f}&ndash;{N["J43"][1]:.2f}, which is suggestive only. '
 f'<b>Gemma&rsquo;s shared experts, all layers.</b> Scaling every layer&rsquo;s shared expert to 0.75 shrinks the race inversion by {N["SHR"]["race"]:.0%} at +{G("race",0.75,"cep"):.1f}% CE, short of the rule&rsquo;s 50%. Beyond the budget both benchmarks flip. The shared experts contribute to the inversion; they do not explain it. '
 f'<b>Identifiers.</b> The code effect is not confined to hard, unconventional names: English-word names carry it in every cell. In three of four cells it grows with the small model&rsquo;s loss on the name.</div>') + s[j:]

i = s.index('<h3>What this drop changes</h3>'); j = s.index('<div class="finding null"><h4>Revision v14.1')
s = s[:i] + ('<h3>What this drop changes</h3>\n'
 f'<div class="finding scale"><h4>The dose-response holds on nine benchmarks</h4><p><b>Regenerate fineweb-edu (P0).</b> Aligned; reproduces experiment 3 (+0.076 &rarr; +0.064) on a 207,061-token subset, so the check against the main grid&rsquo;s +0.0707 is not met as written. Nine-benchmark family mean {f9}, inside the floor at 8B. Section 25.</p></div>\n'
 '<div class="finding null"><h4>The hidden-state-norm drop does not reproduce, and is not graded</h4><p><b>Hidden-state norm (P0).</b> Reference statistic and three-baseline arm both disagree with v16. Within the drop the norm absorbs little. Section 26.</p></div>\n'
 f'<div class="finding gemma"><h4>Gemma&rsquo;s shared experts: a partial contribution</h4><p><b>All-layer ablation (P1).</b> {N["SHR"]["race"]:.0%} shrink inside the budget; rule (50%) not met; both flip outside the budget. The unablated recomputed HC does not reproduce the main study on competition_math. Section 27.</p></div>\n'
 '<div class="finding null"><h4>Identifiers: not unpredictable naming</h4><p><b>First-occurrence strata (P2).</b> Rule not met; the effect spans name types and grows with the small model&rsquo;s loss in three of four cells. Section 28.</p></div>\n'
 '<div class="finding null"><h4>Not run</h4><p>The layer sweep, deferred as a last resort.</p></div>\n') + s[j:]

a = s.index('<ul class="so-far">'); b = s.index('</ul>', a)
items = re.findall(r'<li>.*?</li>(?=<li>|$)', s[a + len('<ul class="so-far">'):b], flags=re.S)
assert len(items) == 18
def add_ev(k, text):
    it = items[k]; j = it.rindex('</div></div></li>'); items[k] = it[:j] + ' <b>v18:</b> ' + text + it[j:]
assert 'simpler router statistics' in items[3]
add_ev(3, f'the hidden-state-norm drop does not reproduce v16 and is not graded; within it, the norm changes the remainder by {N["J43"][0]:.2f}&ndash;{N["J43"][1]:.2f}&times;.')
assert 'code' in items[4]
add_ev(4, f'not unpredictable naming either: English-word names carry it in every cell; in three of four cells it grows with the small model&rsquo;s loss on the name.')
assert 'Gemma' in items[8]
add_ev(8, f'weakening every layer&rsquo;s shared expert to 0.75 removes {N["SHR"]["race"]:.0%} of the race inversion at +{G("race",0.75,"cep"):.1f}% CE (rule 50%): a partial contribution through routing, not an explanation.')
assert 'compute gap' in items[9]
add_ev(9, f'nine benchmarks with fineweb-edu regenerated: {f9}; fineweb-edu, pg19 and race stay positive with the 8B partner.')
s = s[:a + len('<ul class="so-far">')] + ''.join(items) + s[b:]
i = s.index('<h3>Main findings so far</h3>'); j = s.index('<ul class="so-far">', i)
s = s[:i] + ('<h3>Main findings so far</h3>\n<p>What the project holds after this report. Every result is about each large model&rsquo;s <b>last MoE layer</b>. v18 completes the dose-response on nine benchmarks, adds a partial mechanism for Gemma (its shared experts), and narrows the code exception. Nothing is withdrawn; the hidden-state-norm baseline is still open because its drop does not reproduce.</p>') + s[j:]

i = s.index('<h3>Changed this cycle</h3>'); j = s.index('<h3>Checks done</h3>', i)
s = s[:i] + ('<h3>Changed this cycle</h3>\n<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Was</th><th>From</th><th>Now</th></tr></thead><tbody>'
 f'<tr><td>The compute gap scales the scale-specific signal</td><td>B, eight benchmarks</td><td>v17</td><td>Stays B, nine benchmarks: {f9}.</td></tr>'
 '<tr><td>Gemma&rsquo;s inversion and the shared experts</td><td>not implicated (last layer)</td><td>v17</td><td>Partial contribution through routing (41% at &alpha; = 0.75, all layers); not an explanation.</td></tr>'
 '<tr><td>Code exception is unpredictable naming</td><td>hypothesis</td><td>v17</td><td>Not supported; grows with small-model loss, spans name types.</td></tr>'
 '<tr><td>HC adds to the hidden-state norm</td><td>untested</td><td>v14.1</td><td>Still untested: the drop does not reproduce.</td></tr></tbody></table></div>\n') + s[j:]

rep('<p>Recorded as each check ran. Four new checks this cycle. The hidden-state-norm baseline is still missing, so the baseline check stays partial.</p>',
    '<p>Recorded as each check ran. Four new checks this cycle, one of them failing its reproduction test.</p>')
old = re.search(r'<tr><td><strong>Four-partner compute sweep</strong>.*?</tr>', s, flags=re.S).group(0)
s = s.replace(old, old +
 '<tr><td><strong>fineweb-edu regenerated in the sweep</strong><span class="since">v18</span></td><td>Whether the sweep&rsquo;s misaligned cell is fixed</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v18).</b> Aligned; reproduces experiment 3 exactly (GMM/standard +0.076, +0.064). Uses a {N["FWN"]:,}-token subset, so does not reproduce the main grid&rsquo;s +0.0707. The other 320 rows are identical to v17.</td></tr>'
 '<tr><td><strong>Hidden-state-norm baseline</strong><span class="since">v18</span></td><td>Whether HC adds to the norm of the hidden state entering the layer</td><td><span class="status status-notrun">failed reproduction</span></td>'
 f'<td><b>New (v18).</b> Reference differs from v16 by up to {N["REF_DEV"]:.3f} and from the raw-level partial by up to {N["RAW_DEV"]:.3f}; the three-baseline arm does not reproduce v16. Not graded.</td></tr>'
 '<tr><td><strong>Shared-expert ablation, all layers</strong><span class="since">v18</span></td><td>Whether Gemma&rsquo;s shared experts carry its inversion through routing</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v18).</b> &alpha; = 0.75: race {N["SHR"]["race"]:.0%} smaller at +{G("race",0.75,"cep"):.1f}% CE; competition_math {N["SHR"]["competition_math"]:.0%}. Rule not met. Unablated HC does not reproduce the main study on competition_math.</td></tr>'
 '<tr><td><strong>First-occurrence strata</strong><span class="since">v18</span></td><td>Whether the code effect is unpredictable naming</td><td><span class="status status-done">done</span></td>'
 '<td><b>New (v18).</b> 3 loss terciles &times; 3 name types, four cells, 680 rows. Rule not met. Qwen cells reproduce v17; Gemma cells differ slightly.</td></tr>', 1)
i = s.index('<div class="caveat"><b>Data gaps in this drop.</b>'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="caveat"><b>Data gaps in this drop.</b> The hidden-state-norm drop does not reproduce the v16 baseline pipeline. The all-layer ablation&rsquo;s recomputed HC does not reproduce the main study on competition_math. The partner sweep&rsquo;s fineweb-edu uses a 207,061-token subset of the main grid&rsquo;s 477,782.</div>') + s[j:]

qn = (f'<div class="finding scale"><h4>v18: the dose-response on nine benchmarks</h4><p>With fineweb-edu regenerated, the family-mean scale-specific association falls {f9} from the 0.6B to the 8B partner. fineweb-edu, pg19 and race stay positive throughout. Section 25.</p></div>\n')
gn = (f'<div class="finding gemma"><h4>v18: the shared experts contribute</h4><p>Weakening every layer&rsquo;s shared expert to 0.75 removes {N["SHR"]["race"]:.0%} of the race inversion at +{G("race",0.75,"cep"):.1f}% CE; at 0.5 the inversion flips, but the model is far outside the loss budget. Section 27.</p></div>\n')
for sid, txt in (('qwen', qn), ('gemma', gn)):
    i = s.index(f'<section id="{sid}">'); j = s.index('<div class="card">', i); s = s[:j] + txt + s[j:]

i = s.index('<div class="assess">'); j = s.index('</section>', i)
s = s[:i] + ('<div class="assess"><div class="col strong"><h4>Holds up</h4><ul>'
 '<li><b>On the nine studied benchmarks</b>, Qwen&rsquo;s &lsquo;where scale helps&rsquo; holds at every level tested, including against the router baselines together.</li>'
 f'<li><b>Dose-response on nine benchmarks</b>: {f9} as the dense partner grows from 0.6B to 8B.</li>'
 '<li><b>Out of domain</b> it generalises to news, scientific text and chat.</li>'
 '<li><b>Gemma&rsquo;s raw inversion</b> holds on fifteen benchmarks, and part of it runs through the shared experts.</li>'
 '<li><b>HC tracks token difficulty in all four families.</b></li></ul></div>'
 '<div class="col weak"><h4>Not yet established</h4><ul>'
 '<li><b>The hidden-state norm</b>: still untested, because its drop does not reproduce.</li>'
 '<li><b>How much HC adds</b>: 0.04&ndash;0.13% of within-string IG variance over the router baselines.</li>'
 '<li><b>Why Gemma inverts</b>: the shared experts account for about two fifths at most within a working model.</li>'
 '<li><b>Code</b>: not one token class, not copying, not unpredictable naming.</li>'
 '<li><b>One layer.</b></li></ul></div>'
 '<div class="col open"><h4>Open</h4><ul>'
 '<li><b>Why the code effect grows with the small model&rsquo;s loss on a name</b>, in both families.</li>'
 '<li><b>Why fineweb-edu, pg19 and race survive an 8B partner</b> when the others do not.</li>'
 '<li><b>What carries the rest of Gemma&rsquo;s inversion.</b></li></ul></div></div>') + s[j:]

i = s.index('<section id="next">'); j = s.index('<section id="tables">')
h2 = re.match(r'<section id="next"><h2>[^<]*</h2>', s[i:]).group(0)
def exp(t, pr, cost, q, d, st, rule):
    return (f'<div class="exp"><h4>{t}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd><dt>Design</dt><dd>{d}</dd>'
            f'<dt>Statistic</dt><dd>{st}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>\n')
s = s[:i] + (h2 + '<h3>What happened to v17&rsquo;s list</h3>\n<div class="table-wrap"><table><thead><tr><th>v17 experiment</th><th>Status</th><th>Outcome</th></tr></thead><tbody>'
 '<tr><td>Regenerate fineweb-edu (P0)</td><td><b>Done</b></td><td>Aligned; reproduces experiment 3, not the main grid (subset). Dose-response on nine.</td></tr>'
 '<tr><td>The hidden-state norm (P0)</td><td>Done, <b>failed reproduction</b></td><td>Not graded.</td></tr>'
 '<tr><td>Shared-expert ablation, all layers (P1)</td><td><b>Done</b></td><td>41% inside budget; rule not met; partial contribution.</td></tr>'
 '<tr><td>First-occurrence identifiers (P2)</td><td><b>Done</b></td><td>Not unpredictable naming; grows with loss.</td></tr>'
 '<tr><td>Layer sweep</td><td>Deferred</td><td>Last resort.</td></tr></tbody></table></div>\n<h3>Proposed</h3>\n'
 + exp('The hidden-state norm, inside the v16 pipeline', 'P0', 'a few hours, same forward passes',
   'The v18 drop does not reproduce v16. What does the norm do inside the pipeline that does?',
   'Add the norm as a fourth column to the v16 extended-baseline code, unchanged otherwise. First check that the three-baseline arm returns v16&rsquo;s values exactly; then add the fourth.',
   'Retention after four baselines; R&sup2; increment with interval.',
   'Only if the three-baseline arm reproduces v16 to 1e-6: at least half surviving four baselines in three of four tier-A cells &rarr; the baseline finding stands with four. Below a quarter in two or more &rarr; restate in terms of the norm.')
 + exp('Finer shared-expert scaling in Gemma', 'P1', '~half a day',
   'Inside the loss budget, all-layer scaling to 0.75 removes 41% of the race inversion. Does a setting inside the budget reach half?',
   'All layers at &alpha; = 0.875, 0.75, 0.70, 0.65 on race and competition_math, stopping where CE rises 10%. Recompute HC; first reconcile the &alpha; = 1 HC with the main study on competition_math.',
   'r(HC, IG | CE<sub>small</sub>) against CE increase, per benchmark.',
   'If the inversion halves at any setting inside the budget on race, the shared experts are implicated through routing. If the curve flattens before half, they are a partial contributor, as v18 reads it.')
 + exp('Code against prose at matched small-model loss', 'P2', 'a few hours, existing token series',
   'The code effect grows with the small model&rsquo;s loss on a name. Is that specific to code, or what any hard token looks like?',
   'Bin tokens by CE<sub>small</sub> deciles in mbpp, humaneval and two prose benchmarks (cnn_dailymail, scientific_papers); within-string r(HC, IG | CE<sub>small</sub>) per decile per benchmark.',
   'The decile profile for code against prose.',
   'If prose shows the same fall into negative values at high loss, the code exception is a hard-token effect that code has more of. If prose stays positive at matched loss, it is specific to code.')
 + exp('Layer sweep (last resort)', 'Deferred', '~1 day per family if router logits were cached',
   'Does the relationship live only in the last MoE layer?', 'HC from every MoE layer&rsquo;s router.', 'The scale-specific partial by depth.',
   'Run only if the last-layer scope becomes decisive for a reviewer.')
 + '</section>\n') + s[j:]
open('/home/claude/v18/HC_MoE_Report_v18.html', 'w', encoding='utf-8').write(s)
print('ok', len(s))
