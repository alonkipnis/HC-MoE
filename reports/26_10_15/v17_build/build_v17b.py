import json, re
s = open('/home/claude/v17/stage1.html', encoding='utf-8').read()
N = json.load(open('/home/claude/v17/n17.json'))
tq, tg, CC, GA, FAM, PS = N['tq'], N['tg'], N['CC'], N['GA'], N['FAM'], N['PS']
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def rep(a, b, cnt=1):
    global s
    n = s.count(a); assert n == cnt, (n, a[:100]); s = s.replace(a, b)
P = ['0.6B', '1.7B', '4B', '8B']
fs = ' &rarr; '.join(f3(FAM['s'][p]) for p in P)
ga = lambda d, a, k: GA[f'{d}|{a}'][k]

# header / abstract
rep('<h1>HC-MoE v16: What HC Adds, Why Code Reverses, and Format Tokens</h1>', '<h1>HC-MoE v17: The Exceptions Tested, and the Compute Gap Taken to Zero</h1>')
i = s.index('<div class="subtitle">'); j = s.index('</div>', i)
s = s[:i] + '<div class="subtitle">The four experiments proposed in v16 that were run: the whole speaker tag, identifier copying in code, Gemma&rsquo;s shared expert, and a Qwen3-8B partner</div>' + s[j + 6:]
rep('October 15, 2026  &middot;  v16', 'October 15, 2026  &middot;  v17')
s = re.sub(r'<title>[^<]*</title>', '<title>HC-MoE v17: Exceptions Tested</title>', s, count=1)
i = s.index('<div class="abstract">'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="abstract"><b>Abstract.</b> v16 proposed five experiments; four were run, each graded against the rule written for it beforehand. The hidden-state-norm baseline is missing for the third time. '
 f'<b>An 8B partner.</b> With Qwen3-30B-A3B held fixed and paired with dense partners of 0.6B, 1.7B, 4B and 8B, the family-mean scale-specific association r(HC, IG | CE<sub>small</sub>) falls {fs} on eight benchmarks and ten configurations. As the compute gap closes, the signal that marks where scale helps goes to zero. Only two benchmarks turn strictly negative with the 8B partner, so the active-compute account of lambada&rsquo;s 4B reversal is not supported (rule: three). The drop&rsquo;s fineweb-edu cell is a pipeline error and is excluded. '
 f'<b>Speaker tag.</b> Removing every piece of dialogsum&rsquo;s <code>#Person1#:</code> turns Qwen&rsquo;s raw correlation from {f3(tq["full"])} to {f3(tq["notag"])}, positive in eight of ten configurations. The rule needs nine, so by its letter the dialogue reversal is not established as formatting; the two StudentT standard nulls disagree. '
 f'<b>Identifiers.</b> Copying is ruled out: first occurrences of a name carry the negative association in code at least as strongly as repeats, in all four cells. Code stays a domain effect. '
 f'<b>Gemma&rsquo;s shared expert.</b> Halving the last layer&rsquo;s shared expert raises CE by {N["CE_UP_R"]:.1%} and shrinks the race inversion by {N["SHRINK_R"]:.0%}, far short of the rule&rsquo;s 50%. Removing it collapses the model (perplexity {N["PPL"]["race|1.0"]:.1f} &rarr; {N["PPL"]["race|0.0"]:.0f}), so that arm cannot be read. Within the range where the model works, the shared expert does not carry the inversion.</div>') + s[j:]

# what this drop changes
i = s.index('<h3>What this drop changes</h3>'); j = s.index('<div class="finding null"><h4>Revision v14.1')
s = s[:i] + ('<h3>What this drop changes</h3>\n'
 f'<div class="finding scale"><h4>The compute gap taken to zero: the scale-specific association follows it down</h4><p><b>Qwen3-8B partner (P2).</b> Family mean {fs} across 0.6B, 1.7B, 4B and 8B partners (eight benchmarks, ten configurations). Rule (three or more benchmarks negative with 8B &rarr; active-compute account supported): <b>not met</b>, two of eight. Lambada&rsquo;s 4B reversal does not deepen with 8B. fineweb-edu excluded as a pipeline error. Section 24.</p></div>\n'
 f'<div class="finding null"><h4>Speaker tag: eight of ten, one short of the rule</h4><p><b>Whole-tag removal (P0).</b> Qwen dialogsum {f3(tq["full"])} &rarr; {f3(tq["notag"])}, 8/10 positive; rule needs 9. By the letter, dialogsum&rsquo;s reversal is not established as formatting. Gemma stays negative ({f3(tg["notag"])}). Section 21.</p></div>\n'
 '<div class="finding null"><h4>Identifiers: not copying</h4><p><b>Copying test (P1).</b> Rule (repeats carry it, first occurrences null) <b>not met</b> in any cell; first occurrences are as negative or more. Code stays a domain effect. Section 22.</p></div>\n'
 f'<div class="finding gemma"><h4>Gemma&rsquo;s shared expert: not implicated</h4><p><b>Shared-expert ablation (P1).</b> At &alpha; = 0.5, CE +{N["CE_UP_R"]:.1%}, race inversion {N["SHRINK_R"]:.0%} smaller (rule: 50%). At &alpha; = 0 the model collapses, so the ruling-out branch is unreadable. All-layer arm not run. Section 23.</p></div>\n'
 '<div class="finding null"><h4>Not run</h4><p>The hidden-state-norm baseline (third time) and the layer sweep (deferred as a last resort).</p></div>\n') + s[j:]

# findings register
a = s.index('<ul class="so-far">'); b = s.index('</ul>', a)
items = re.findall(r'<li>.*?</li>(?=<li>|$)', s[a + len('<ul class="so-far">'):b], flags=re.S)
assert len(items) == 18
def add_ev(k, text):
    it = items[k]; j = it.rindex('</div></div></li>'); items[k] = it[:j] + ' <b>v17:</b> ' + text + it[j:]
def li(tier, claim, since, ev):
    return (f'<li><span class="tier tier-{tier}">{tier}</span><div class="body"><div class="claim">{claim}<span class="since">since {since}</span></div><div class="ev">{ev}</div></div></li>')
assert 'code' in items[4]
add_ev(4, f'not copying: first occurrences of a name are as negative or more than repeats in all four cells (Qwen mbpp {f3(CC["Qwen_mbpp"]["first"])} vs {f3(CC["Qwen_mbpp"]["rep"])}; Gemma humaneval {f3(CC["gemma_humaneval"]["first"])} vs {f3(CC["gemma_humaneval"]["rep"])}).')
assert 'formatting' in items[5]
items[5] = li('B', '<b>Qwen&rsquo;s negative raw sign on chat is a formatting effect</b>; on dialogue the speaker tag carries most of it, but the rule is missed by one configuration.', 'v16',
    f'ultrachat without format tokens +0.011 (9/10). v17: dialogsum without the whole tag {f3(tq["notag"])} (was {f3(tq["full"])}), 8/10 positive &mdash; StudentT and StudentTSkewed standard disagree; rule needs 9/10, so dialogue formatting is not established.')
assert 'Gemma' in items[8]
add_ev(8, f'not the last layer&rsquo;s shared expert: halving it shrinks the race inversion {N["SHRINK_R"]:.0%} (rule: 50%) at +{N["CE_UP_R"]:.1%} CE; removing it collapses the model.')
assert 'compute gap' in items[9]
add_ev(9, f'extended to an 8B partner on all ten configurations: family mean {fs} (eight benchmarks; fineweb-edu excluded as a pipeline error). The active-compute account of lambada&rsquo;s 4B reversal is not supported (two of eight negative with 8B; rule three).')
s = s[:a + len('<ul class="so-far">')] + ''.join(items) + s[b:]
i = s.index('<h3>Main findings so far</h3>'); j = s.index('<ul class="so-far">', i)
s = s[:i] + ('<h3>Main findings so far</h3>\n<p>What the project holds after this report. Every result is about the routing of each large model&rsquo;s <b>last MoE layer</b>. v17 strengthens the dose-response &mdash; the scale-specific association goes to zero as the compute gap closes &mdash; and closes three explanations: copying for code, the shared expert for Gemma, active compute for lambada. Nothing is withdrawn.</p>') + s[j:]

# changed this cycle
i = s.index('<h3>Changed this cycle</h3>'); j = s.index('<h3>Checks done</h3>', i)
s = s[:i] + ('<h3>Changed this cycle</h3>\n<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Was</th><th>From</th><th>Now</th></tr></thead><tbody>'
 f'<tr><td>The compute gap scales the scale-specific signal</td><td>B, three partners, GMM only</td><td>v8</td><td>Stays B, now four partners on all ten configurations: {fs}.</td></tr>'
 '<tr><td>Active-compute account of lambada&rsquo;s 4B reversal</td><td>hypothesis</td><td>v13</td><td>Not supported (rule not met); lambada not more negative with 8B.</td></tr>'
 '<tr><td>Dialogue raw-sign reversal is formatting</td><td>inconclusive</td><td>v16</td><td>Not established: 8 of 10 with the whole tag removed; rule needs 9.</td></tr>'
 '<tr><td>Code exception is copying</td><td>hypothesis</td><td>v16</td><td>Ruled out; first occurrences carry it.</td></tr>'
 '<tr><td>Gemma&rsquo;s inversion is carried by the shared expert</td><td>hypothesis</td><td>v11</td><td>Not implicated at &alpha; = 0.5; &alpha; = 0 unreadable.</td></tr></tbody></table></div>\n') + s[j:]

# checks
rep('<p>Recorded as each check ran. Three new checks this cycle and one extended. The hidden-state-norm baseline is still missing, so the baseline check stays partial.</p>',
    '<p>Recorded as each check ran. Four new checks this cycle. The hidden-state-norm baseline is still missing, so the baseline check stays partial.</p>')
old = re.search(r'<tr><td><strong>Lambada top tokens under every null</strong>.*?</tr>', s, flags=re.S).group(0)
s = s.replace(old, old +
 '<tr><td><strong>Whole speaker-tag removal</strong><span class="since">v17</span></td><td>Whether dialogsum&rsquo;s raw-sign reversal is the speaker tag</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v17).</b> All five tag pieces removed by position ({1-tq["nc"]/tq["n"]:.0%} of tokens). Qwen {f3(tq["full"])} &rarr; {f3(tq["notag"])}, 8/10; rule needs 9. Gemma {f3(tg["full"])} &rarr; {f3(tg["notag"])}, 0/10. Full-text values reproduce v15/v16.</td></tr>'
 '<tr><td><strong>Identifier copying</strong><span class="since">v17</span></td><td>Whether repeated names carry the code effect</td><td><span class="status status-done">done</span></td>'
 '<td><b>New (v17).</b> First occurrences vs repeats, four cells, 40 rows. Rule not met anywhere; first occurrences carry it.</td></tr>'
 '<tr><td><strong>Shared-expert ablation</strong><span class="since">v17</span></td><td>Whether Gemma&rsquo;s last-layer shared expert carries its inversion</td><td><span class="status status-partial">partial</span></td>'
 f'<td><b>New (v17).</b> &alpha; = 1, 0.5, 0 on race and competition_math; &alpha; = 1 reproduces the main study. &alpha; = 0.5: inversion {N["SHRINK_R"]:.0%} smaller. &alpha; = 0: model collapses. <b>Partial:</b> the all-layer arm was not run.</td></tr>'
 '<tr><td><strong>Four-partner compute sweep</strong><span class="since">v17</span></td><td>The dose-response, extended to an 8B partner on the full grid</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v17).</b> 360 rows; 0.6B column reproduces the main study to {N["REPRO_Q"]:.4f}. <b>fineweb-edu excluded:</b> {N["FW_N"]:,} tokens (main 477,782) and r = {N["FW_R"]:+.4f} (main +0.0707), the signature of misaligned series.</td></tr>', 1)
i = s.index('<div class="caveat"><b>Data gaps in this drop.</b>'); j = s.index('</div>', i) + 6
s = s[:i] + (f'<div class="caveat"><b>Data gaps in this drop.</b> The hidden-state norm is missing for the third time. The partner sweep&rsquo;s fineweb-edu cell is misaligned ({N["FW_N"]:,} tokens, r &asymp; 0 under every partner) and should be regenerated. The shared-expert ablation&rsquo;s all-layer arm was not run.</div>') + s[j:]

# family notes
qn = (f'<div class="finding scale"><h4>v17: the 8B partner</h4><p>Holding Qwen3-30B-A3B fixed, the family-mean scale-specific association falls {fs} as the dense partner grows from 0.6B to 8B; with the 8B partner the small model is within 0.15 nats of the large one and the association is gone at the family level. pg19 and race stay positive throughout. Section 24.</p></div>\n')
gn = (f'<div class="finding gemma"><h4>v17: not the shared expert</h4><p>Halving the last layer&rsquo;s shared expert leaves the race inversion {N["SHRINK_R"]:.0%} smaller; removing the speaker tag on dialogsum halves the raw inversion but leaves it negative in every configuration. Sections 21 and 23.</p></div>\n')
for sid, txt in (('qwen', qn), ('gemma', gn)):
    i = s.index(f'<section id="{sid}">'); j = s.index('<div class="card">', i); s = s[:j] + txt + s[j:]

# assessment
i = s.index('<div class="assess">'); j = s.index('</section>', i)
s = s[:i] + ('<div class="assess"><div class="col strong"><h4>Holds up</h4><ul>'
 '<li><b>On the nine studied benchmarks</b>, Qwen&rsquo;s &lsquo;where scale helps&rsquo; holds at every level tested, including against the router baselines together.</li>'
 f'<li><b>Dose-response.</b> With the large model fixed, the scale-specific association falls to zero as the partner grows from 0.6B to 8B ({fs}).</li>'
 '<li><b>Out of domain</b> it generalises to news, scientific text and chat.</li>'
 '<li><b>Gemma&rsquo;s raw inversion</b> holds on fifteen benchmarks and survives formatting, router baselines and a halved shared expert.</li>'
 '<li><b>HC tracks token difficulty in all four families.</b></li></ul></div>'
 '<div class="col weak"><h4>Not yet established</h4><ul>'
 '<li><b>How much HC adds.</b> 0.04&ndash;0.13% of within-string IG variance over the router baselines; the simpler statistics predict as well or better.</li>'
 '<li><b>The hidden-state norm</b> is still untested.</li>'
 '<li><b>Dialogue formatting.</b> 8 of 10 configurations; the StudentT nulls disagree.</li>'
 '<li><b>Why Gemma inverts.</b> Not the compute gap, not formatting, not router confidence, not the last layer&rsquo;s shared expert at &alpha; = 0.5.</li>'
 '<li><b>One layer.</b> Every result is last-MoE-layer routing.</li></ul></div>'
 '<div class="col open"><h4>Open</h4><ul>'
 '<li><b>Why first occurrences of identifiers</b>: unpredictable naming choices, or something about code?</li>'
 '<li><b>What lambada&rsquo;s 4B reversal is</b>, now that active compute is ruled out.</li>'
 '<li><b>Does the all-layer shared-expert arm move Gemma&rsquo;s HC, and with it the inversion?</b></li></ul></div></div>') + s[j:]

# next experiments
i = s.index('<section id="next">'); j = s.index('<section id="tables">')
h2 = re.match(r'<section id="next"><h2>[^<]*</h2>', s[i:]).group(0)
def exp(t, pr, cost, q, d, st, rule):
    return (f'<div class="exp"><h4>{t}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd><dt>Design</dt><dd>{d}</dd>'
            f'<dt>Statistic</dt><dd>{st}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>\n')
s = s[:i] + (h2 + '<h3>What happened to v16&rsquo;s list</h3>\n<div class="table-wrap"><table><thead><tr><th>v16 experiment</th><th>Status</th><th>Outcome</th></tr></thead><tbody>'
 '<tr><td>Remove the whole speaker tag (P0)</td><td><b>Done</b></td><td>8/10; rule missed by one configuration.</td></tr>'
 '<tr><td>The hidden-state norm (P0)</td><td>Not in the drop</td><td>Carried, third time.</td></tr>'
 '<tr><td>Why identifiers are negative (P1)</td><td><b>Done</b></td><td>Not copying; first occurrences carry it.</td></tr>'
 '<tr><td>Shared-expert ablation (P1)</td><td><b>Done, partial</b></td><td>Not implicated at &alpha; = 0.5; all-layer arm not run.</td></tr>'
 '<tr><td>A larger partner (P2)</td><td><b>Done</b></td><td>Dose-response to zero; active-compute account not supported; fineweb-edu cell broken.</td></tr>'
 '<tr><td>Layer sweep</td><td>Deferred</td><td>Last resort.</td></tr></tbody></table></div>\n<h3>Proposed</h3>\n'
 + exp('Regenerate fineweb-edu in the partner sweep', 'P0', 'an hour, existing losses',
   'The sweep&rsquo;s fineweb-edu cell has the wrong token count and r &asymp; 0 everywhere. Is the dose-response the same with it included?',
   'Rebuild the cell from the same 477,782-token series the main study uses; check that the 0.6B column reproduces +0.0707.',
   'Family-mean r(HC, IG | CE<sub>small</sub>) across the four partners, nine benchmarks.',
   'If the 0.6B value reproduces and the family mean still falls monotonically to within the floor at 8B, the v17 dose-response stands on all nine benchmarks.')
 + exp('The hidden-state norm', 'P0', 'a few hours, same forward passes',
   'The fourth planned baseline, missing three times.',
   'Norm of the hidden state entering the last MoE layer; joint partial and R&sup2; increment with all four baselines.',
   'Retention after four baselines; R&sup2; increment with interval.',
   'At least half surviving in three of four tier-A cells, increment excluding zero in all four &rarr; the baseline finding stands with four. Below a quarter in two or more &rarr; restate in terms of the norm.')
 + exp('Shared-expert ablation, all layers', 'P1', '~half a day',
   'Halving the last layer&rsquo;s shared expert leaves Gemma&rsquo;s inversion intact, but that arm cannot change HC. Does weakening every layer&rsquo;s shared expert?',
   'Scale all layers&rsquo; shared-expert outputs by 0.75 and 0.5; recompute HC and CE<sub>big</sub> on race and competition_math; stop if mean CE rises more than 10%.',
   'r(HC, IG | CE<sub>small</sub>) at each setting.',
   'If the race inversion shrinks by half at a setting within the CE budget, the shared experts are implicated through routing. If not, the shared-expert explanation is dropped.')
 + exp('First-occurrence identifiers', 'P2', 'a few hours, existing token series',
   'The code effect sits on first occurrences of names. Are they simply the least predictable tokens for both models?',
   'Within first-occurrence identifiers, stratify by the small model&rsquo;s loss and by whether the name is a common English word, a convention (i, n, res) or neither.',
   'Within-string partial per stratum.',
   'If the negative association is confined to high-loss, unconventional names, code&rsquo;s exception is stated as unpredictable naming.')
 + exp('Layer sweep (last resort)', 'Deferred', '~1 day per family if router logits were cached',
   'Does the relationship live only in the last MoE layer?',
   'HC from every MoE layer&rsquo;s router for Qwen&rsquo;s tier-A cells and Gemma/race.',
   'The scale-specific partial by depth.',
   'Run only if the last-layer scope becomes decisive for a reviewer.')
 + '</section>\n') + s[j:]
open('/home/claude/v17/HC_MoE_Report_v17.html', 'w', encoding='utf-8').write(s)
print('ok', len(s))
