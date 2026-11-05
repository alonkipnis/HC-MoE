import json, re
s = open('/home/claude/v16/stage1.html', encoding='utf-8').read()
N = json.load(open('/home/claude/v16/n16.json'))
Q = N['QBX']; G = N['GBX']; DL = N['DLa']; CL = N['CLa']; CP = N['CPa']
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def rep(a, b, cnt=1):
    global s
    n = s.count(a); assert n == cnt, (n, a[:100]); s = s.replace(a, b)
TA = ['competition_math', 'lambada', 'race', 'fineweb-edu']
dr = [Q['dr2'][d] for d in TA]
DRR = f'{100*min(dr):.2f}&ndash;{100*max(dr):.2f}%'
ucr, dsr = DL['Qwen|ultrachat'], DL['Qwen|dialogsum']

# ---------------------------------------------------------------- header
rep('<h1>HC-MoE v15: Beyond the Studied Benchmarks, Beyond Simpler Router Signals</h1>',
    '<h1>HC-MoE v16: What HC Adds, Why Code Reverses, and Format Tokens</h1>')
i = s.index('<div class="subtitle">'); j = s.index('</div>', i)
s = s[:i] + '<div class="subtitle">Four of v15&rsquo;s five proposed experiments: the extended baselines, a token-class dissection of code, a format-token decomposition of dialogue and chat, and lambada&rsquo;s top tokens under every null</div>' + s[j + 6:]
rep('October 15, 2026  &middot;  v15', 'October 15, 2026  &middot;  v16')
s = re.sub(r'<title>[^<]*</title>', '<title>HC-MoE v16: What HC Adds</title>', s, count=1)
i = s.index('<div class="abstract">'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="abstract"><b>Abstract.</b> v15 proposed six experiments. Four were run. The layer sweep is deferred as a last resort, and the shared-expert ablation and the Qwen3-8B partner are not in the drop. Each result is graded against the rule written for it in v15. '
 f'<b>Baselines, extended.</b> Adding HC to a within-string model of Intelligence Gain on the small model&rsquo;s loss and three router baselines (entropy, maximum probability, logit gap) raises R&sup2; by an amount whose interval excludes zero in all four of Qwen&rsquo;s tier-A cells. That meets the rule&rsquo;s second part: HC adds to the simple router statistics taken together, not only to each one. The amount is {DRR} of within-string IG variance. '
 f'Out of domain, three of Qwen&rsquo;s four passes survive the baselines (ultrachat {pct(Q["ret"]["ultrachat"])}, scientific_papers {pct(Q["ret"]["scientific_papers"])}, cnn_dailymail {pct(Q["ret"]["cnn_dailymail"])}; dialogsum {pct(Q["ret"]["dialogsum"])}), and so do both code reversals and every Gemma cell. The hidden-state norm is still missing, so the rule&rsquo;s first part is open. '
 f'<b>Code.</b> No single token class explains the reversal. Removing identifiers turns Qwen&rsquo;s mbpp positive ({f3(CL["Qwen_mbpp|IDENTIFIER"]["r"])}), but humaneval stays negative whichever class is removed. By the rule, code is a domain effect. Within identifiers alone the association is negative in all four code cells of both families, in every configuration. '
 f'<b>Format tokens.</b> Without its chat-role tokens, Qwen&rsquo;s raw correlation on ultrachat turns positive ({f3(ucr["nofmt"])}, {ucr["nofmt_pos"]} of 10 configurations): a formatting effect. On dialogsum the test is inconclusive ({f3(dsr["nofmt"])}, {dsr["nofmt_pos"]} of 10), because only one of the four tokens of the speaker tag <code>#Person1#:</code> was flagged as format. Gemma&rsquo;s raw inversion survives every removal. '
 f'<b>Lambada.</b> The two contraction strings dominate the top ventile under all ten configurations ({pct(N["LCmin"])}&ndash;{pct(N["LCmax"])}); GMM/standard, the series&rsquo; reference, is the configuration that understates them.</div>') + s[j:]

# ---------------------------------------------------------------- what this drop changes
i = s.index('<h3>What this drop changes</h3>'); j = s.index('<div class="finding null"><h4>Revision v14.1')
s = s[:i] + ('<h3>What this drop changes</h3>\n'
 f'<div class="finding scale"><h4>HC adds to the router baselines together &mdash; by about a tenth of a percent</h4><p><b>Complete the baselines (P0).</b> Rule part (2): if HC&rsquo;s R&sup2; increment excludes zero in every tier-A cell, HC adds to the best simple router statistics. <b>Met</b> in all four ({DRR} of within-string IG variance). Part (1), with the hidden-state norm, cannot be evaluated: the norm is not in the drop. Out of domain, ultrachat, scientific_papers and cnn_dailymail survive the baselines; dialogsum does not; the code reversals and every Gemma cell do. Section 17.</p></div>\n'
 f'<div class="finding null"><h4>Code is a domain effect, not one token class</h4><p><b>Why code reverses (P1).</b> Rule: if removing one class turns Qwen positive on both code sets, token-class exception; if spread, domain. <b>Second branch.</b> Identifiers carry mbpp, but humaneval stays negative whichever class is removed. Identifiers are negative in all four code cells, both families, every configuration. Section 18.</p></div>\n'
 f'<div class="finding null"><h4>Format tokens explain ultrachat&rsquo;s raw sign; dialogsum&rsquo;s test was under-specified</h4><p><b>What drives the negative raw sign (P1).</b> Rule: if removing format tokens turns Qwen positive on dialogsum and ultrachat, formatting. <b>Not met as written:</b> ultrachat turns positive ({f3(ucr["nofmt"])}, {ucr["nofmt_pos"]}/10); dialogsum is undetermined ({f3(dsr["nofmt"])}, {dsr["nofmt_pos"]}/10) because the format list flagged only one of the speaker tag&rsquo;s four tokens. Gemma is not a formatting effect. Section 19.</p></div>\n'
 '<div class="finding scale"><h4>Lambada&rsquo;s token description holds under every null</h4><p><b>Lambada&rsquo;s tokens under every null (P2, half).</b> Rule met: the two contraction strings keep at least half their GMM share in all ten configurations. Section 20.</p></div>\n'
 '<div class="finding null"><h4>Not run</h4><p>The layer sweep is deferred: running more layers is costly in this setup, so it is kept as a last resort rather than the next step. The shared-expert ablation in Gemma and the Qwen3-8B partner are not in the drop and are carried forward (section 31).</p></div>\n') + s[j:]

# ---------------------------------------------------------------- findings register
a = s.index('<ul class="so-far">'); b = s.index('</ul>', a)
items = re.findall(r'<li>.*?</li>(?=<li>|$)', s[a + len('<ul class="so-far">'):b], flags=re.S)
assert len(items) == 16
def add_ev(k, text):
    global items
    it = items[k]; j = it.rindex('</div></div></li>'); items[k] = it[:j] + ' <b>v16:</b> ' + text + it[j:]
def li(tier, claim, since, ev):
    return (f'<li><span class="tier tier-{tier}">{tier}</span><div class="body"><div class="claim">{claim}<span class="since">since {since}</span></div><div class="ev">{ev}</div></div></li>')
assert 'Qwen&rsquo;s HC marks' in items[1] or "Qwen's HC marks" in items[1]
add_ev(1, f'adding HC to the small model&rsquo;s loss and the three baselines raises within-string R&sup2; in all four tier-A cells, by {DRR} of IG variance.')
assert 'domain-general' in items[2]
add_ev(2, f'three of four passes survive the router baselines (ultrachat {pct(Q["ret"]["ultrachat"])}, scientific_papers {pct(Q["ret"]["scientific_papers"])}, cnn_dailymail {pct(Q["ret"]["cnn_dailymail"])}); dialogsum does not ({pct(Q["ret"]["dialogsum"])}).')
assert 'simpler router statistics' in items[3]
add_ev(3, f'R&sup2; increment from HC over the three baselines together excludes zero in all four tier-A cells ({DRR}); hidden-state norm still untested; the two drops disagree on the baselines&rsquo; own predictive values in competition_math, not on the conclusion.')
assert 'top-ventile surges' in items[4]
add_ev(4, f'not in lambada: its two contraction strings make up {pct(N["LCmin"])}&ndash;{pct(N["LCmax"])} of the top ventile under every null, and GMM/standard is the outlier (overlap {N["J"][2]:.2f}&ndash;{N["J"][3]:.2f} with the other nine, which overlap {N["J"][0]:.2f}&ndash;{N["J"][1]:.2f} among themselves).')
assert 'Gemma' in items[5]
add_ev(5, f'not a formatting effect (dialogsum {f3(DL["gemma|dialogsum"]["nofmt"])} and ultrachat {f3(DL["gemma|ultrachat"]["nofmt"])} without format tokens) and not absorbed by router baselines in any of seven cells.')
code_item = li('B', '<b>On code, the scale-specific association is negative in both families, and it is a property of the domain rather than of one token class.</b>', 'v16',
    f'No single class removal turns Qwen positive on both code sets (rule&rsquo;s second branch); humaneval stays negative under every removal. Within identifiers alone: Qwen mbpp {f3(CP["Qwen_mbpp|IDENTIFIER"]["r"])}, humaneval {f3(CP["Qwen_humaneval|IDENTIFIER"]["r"])}, Gemma mbpp {f3(CP["gemma_mbpp|IDENTIFIER"]["r"])}, humaneval {f3(CP["gemma_humaneval|IDENTIFIER"]["r"])}, every configuration negative. Survives the router baselines. Two Python benchmarks.')
fmt_item = li('B', '<b>Qwen&rsquo;s negative raw sign on chat is a formatting effect</b>; on dialogue the test was under-specified.', 'v16',
    f'ultrachat without format tokens {f3(ucr["nofmt"])} ({ucr["nofmt_pos"]}/10), with them {f3(ucr["raw"])}; the role tokens &lsquo;assistant&rsquo; and &lsquo;user&rsquo; carry the largest negative contributions. dialogsum {f3(dsr["nofmt"])} ({dsr["nofmt_pos"]}/10) with only one of the four speaker-tag tokens removed.')
items = items[:4] + [code_item, fmt_item] + items[4:]
s = s[:a + len('<ul class="so-far">')] + ''.join(items) + s[b:]
i = s.index('<h3>Main findings so far</h3>'); j = s.index('<ul class="so-far">', i)
s = s[:i] + ('<h3>Main findings so far</h3>\n<p>What the project holds after this report. Every result is a statement about the routing of each large model&rsquo;s <b>last MoE layer</b>. v16 sharpens three findings rather than adding a headline: HC adds to the router baselines taken together, by a small amount; code&rsquo;s exception is a domain effect; and the raw-sign reversal on chat is about format tokens. Nothing is withdrawn.</p>') + s[j:]

# ---------------------------------------------------------------- changed this cycle
i = s.index('<h3>Changed this cycle</h3>'); j = s.index('<h3>Checks done</h3>', i)
s = s[:i] + ('<h3>Changed this cycle</h3>\n<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Was</th><th>From</th><th>Now</th></tr></thead><tbody>'
 f'<tr><td>HC adds to simpler router statistics</td><td>B, to each one</td><td>v15</td><td>Stays B, now to all three together: R&sup2; increment excludes zero in every tier-A cell ({DRR}). Hidden-state norm still untested.</td></tr>'
 '<tr><td>Qwen out of domain, code excepted</td><td>B</td><td>v15</td><td>Stays B; three of four passes survive the router baselines.</td></tr>'
 '<tr><td>Code as the exception</td><td>named, unexplained</td><td>v15</td><td>New B: a domain effect, not one token class; identifiers negative in all four code cells.</td></tr>'
 '<tr><td>Negative raw sign on dialogue and chat</td><td>open</td><td>v14</td><td>New B: formatting on ultrachat; dialogsum&rsquo;s test under-specified.</td></tr>'
 '<tr><td>Lambada&rsquo;s top-ventile tokens</td><td>described under GMM/standard</td><td>v13</td><td>Hold under all ten configurations.</td></tr>'
 '<tr><td>Gemma&rsquo;s raw inversion</td><td>A</td><td>v5</td><td>Stays A; not a formatting effect, not absorbed by router baselines.</td></tr>'
 '<tr><td>Layer sweep</td><td>P0</td><td>v15</td><td>Deferred: a last resort, given the cost of running more layers.</td></tr></tbody></table></div>\n') + s[j:]

# ---------------------------------------------------------------- checks register
rep('<p>Recorded as each check ran. Three new checks this cycle. One planned component, the hidden-state-norm baseline, is missing from the drop and is marked partial.</p>',
    '<p>Recorded as each check ran. Three new checks this cycle and one extended. The hidden-state-norm baseline is still missing, so the baseline check stays partial.</p>')
old = re.search(r'<tr><td><strong>Last-layer router baselines</strong>.*?</tr>', s, flags=re.S).group(0)
new = ('<tr><td><strong>Last-layer router baselines</strong><span class="since">v15</span></td><td>Whether HC adds anything beyond simpler statistics of the same router</td><td><span class="status status-partial">partial</span></td>'
 f'<td>Entropy, maximum probability, logit gap. Qwen tier-A retention {pct(Q["ret"]["competition_math"])}, {pct(Q["ret"]["lambada"])}, {pct(Q["ret"]["race"])}, {pct(Q["ret"]["fineweb-edu"])}. <b>Extended in v16</b> to every unseen cell and the within-string R&sup2; increment: excludes zero in all four tier-A cells ({DRR}); the increment equals r<sub>joint</sub>&sup2;(1 &minus; R&sup2;) exactly. 170/170 rows; the 80 shared with v15 identical. <b>Partial:</b> the hidden-state norm has still not been computed.</td></tr>'
 '<tr><td><strong>Code token dissection</strong><span class="since">v16</span></td><td>Whether one token class carries the code reversal</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v16).</b> Seven classes, per class and leave-one-out, both families. Removing identifiers turns Qwen mbpp positive ({f3(CL["Qwen_mbpp|IDENTIFIER"]["r"])}); humaneval stays negative under every removal: domain effect. Identifiers negative in all four cells, 40/40 configurations. Values are not additive across classes.</td></tr>'
 '<tr><td><strong>Format-token decomposition</strong><span class="since">v16</span></td><td>Which token strings carry the negative between-string part out of domain</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v16).</b> ultrachat turns positive without format tokens ({f3(ucr["nofmt"])}, {ucr["nofmt_pos"]}/10); dialogsum undetermined ({f3(dsr["nofmt"])}, {dsr["nofmt_pos"]}/10) because three of the four speaker-tag tokens were not flagged. The top-10-contributor removal selects on the outcome and is not used as evidence.</td></tr>'
 '<tr><td><strong>Lambada top tokens under every null</strong><span class="since">v16</span></td><td>Whether lambada&rsquo;s token-type description depends on the null</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v16).</b> Contraction strings {pct(N["LCmin"])}&ndash;{pct(N["LCmax"])} of the top ventile in all ten configurations, against a rule threshold of {pct(0.5*N["GMM_SHARE"])}. GMM/standard is the outlier.</td></tr>')
s = s.replace(old, new, 1)
i = s.index('<div class="caveat"><b>Data gaps in this drop.</b>'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="caveat"><b>Data gaps in this drop.</b> The hidden-state norm is missing for the second time. The dialogsum format list flagged only &lsquo;Person&rsquo; of the speaker tag&rsquo;s four tokens. The shared-expert ablation and the Qwen3-8B partner were not run. The drop&rsquo;s baseline-only predictive values differ from the first drop&rsquo;s in competition_math; their definition is not documented.</div>') + s[j:]

# ---------------------------------------------------------------- family notes
qn = (f'<div class="finding scale"><h4>v16: what HC adds, and code</h4><p>Over the small model&rsquo;s loss and the three router baselines together, HC raises within-string R&sup2; in all four tier-A cells, by {DRR} of IG variance. On code the negative scale-specific association is spread across token classes in humaneval and carried by identifiers in mbpp; ultrachat&rsquo;s negative raw sign comes from the chat-role tokens (sections 17&ndash;19).</p></div>\n')
gn = (f'<div class="finding gemma"><h4>v16: not formatting, not router confidence</h4><p>Gemma&rsquo;s raw inversion survives the removal of format tokens on dialogsum, ultrachat and mbpp, and its scale-specific associations survive the router baselines in all seven cells. On code its identifiers carry the strongest negative association in the study (humaneval {f3(CP["gemma_humaneval|IDENTIFIER"]["r"])}). Sections 17&ndash;19.</p></div>\n')
for sid, txt in (('qwen', qn), ('gemma', gn)):
    i = s.index(f'<section id="{sid}">'); j = s.index('<div class="card">', i); s = s[:j] + txt + s[j:]

# ---------------------------------------------------------------- assessment
i = s.index('<div class="assess">'); j = s.index('</section>', i)
s = s[:i] + ('<div class="assess"><div class="col strong"><h4>Holds up</h4><ul>'
 '<li><b>On the nine studied benchmarks</b>, Qwen&rsquo;s &lsquo;where scale helps&rsquo; holds at every level tested, including against the router baselines taken together.</li>'
 '<li><b>Out of domain</b> it generalises to news, scientific text and chat, and those three survive the baselines.</li>'
 '<li><b>Gemma&rsquo;s raw inversion</b> holds on fifteen benchmarks, every configuration, and is neither a formatting nor a router-confidence effect.</li>'
 '<li><b>The exceptions are now characterised</b>: code is a domain effect; chat&rsquo;s raw sign is format tokens.</li>'
 '<li><b>HC tracks token difficulty in all four families.</b></li></ul></div>'
 '<div class="col weak"><h4>Not yet established</h4><ul>'
 f'<li><b>How much HC adds.</b> Significant but small: {DRR} of within-string IG variance over the baselines. The simpler router statistics predict IG as well or better.</li>'
 '<li><b>The hidden-state norm</b>, the fourth baseline, is still missing.</li>'
 '<li><b>dialogsum</b>: its pass does not survive the baselines, and its format test removed one of four tag tokens.</li>'
 '<li><b>One layer.</b> Every result is last-MoE-layer routing; the sweep is deferred.</li>'
 '<li><b>Gemma&rsquo;s scale-specific part</b> changes sign across benchmarks; the shared-expert ablation has not run.</li></ul></div>'
 '<div class="col open"><h4>Open</h4><ul>'
 '<li><b>Why are identifiers negative in code, in both families?</b> Copying from the prompt is one candidate.</li>'
 '<li><b>Does removing the whole speaker tag turn dialogsum positive?</b></li>'
 '<li><b>Does HC add to the hidden-state norm?</b></li>'
 '<li><b>Is Gemma&rsquo;s inversion carried by the shared expert?</b></li></ul></div></div>') + s[j:]

# ---------------------------------------------------------------- next experiments
i = s.index('<section id="next">'); j = s.index('<section id="tables">')
h2 = re.match(r'<section id="next"><h2>[^<]*</h2>', s[i:]).group(0)
def exp(t, pr, cost, q, d, st, rule):
    return (f'<div class="exp"><h4>{t}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd><dt>Design</dt><dd>{d}</dd>'
            f'<dt>Statistic</dt><dd>{st}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>\n')
s = s[:i] + (h2 + '<h3>What happened to v15&rsquo;s list</h3>\n<div class="table-wrap"><table><thead><tr><th>v15 experiment</th><th>Status</th><th>Outcome</th></tr></thead><tbody>'
 '<tr><td>Layer sweep (P0)</td><td>Deferred</td><td>Kept as a last resort: running more layers is costly in this setup.</td></tr>'
 f'<tr><td>Complete the baselines (P0)</td><td><b>Done, partial</b></td><td>R&sup2; increment excludes zero in all four tier-A cells ({DRR}); unseen cells added. Hidden-state norm missing again.</td></tr>'
 '<tr><td>Why code reverses (P1)</td><td><b>Done</b></td><td>Domain effect; identifiers negative in every code cell.</td></tr>'
 '<tr><td>What drives the negative raw sign out of domain (P1)</td><td><b>Done</b></td><td>ultrachat: formatting. dialogsum: inconclusive, incomplete format list.</td></tr>'
 '<tr><td>Shared-expert ablation in Gemma (P1)</td><td>Not in the drop</td><td>Carried.</td></tr>'
 '<tr><td>Qwen3-8B partner and lambada&rsquo;s tokens under every null (P2)</td><td><b>Half done</b></td><td>Lambada&rsquo;s description holds under every null; the 8B partner was not run.</td></tr></tbody></table></div>\n<h3>Proposed</h3>\n'
 + exp('Remove the whole speaker tag on dialogsum', 'P0', 'about an hour, existing records',
   'The pre-registered format test removed only &lsquo;Person&rsquo; from dialogsum&rsquo;s <code>#Person1#:</code>. Is the raw-sign reversal there the speaker tag?',
   'Recompute Qwen&rsquo;s and Gemma&rsquo;s raw r(HC, IG) on dialogsum with every token of the tag removed &mdash; &lsquo;#&rsquo;, &lsquo;Person&rsquo;, &lsquo;1&rsquo;, &lsquo;2&rsquo;, &lsquo;#:&rsquo; &mdash; identified by position in the tag, not by string, so that digits and &lsquo;#&rsquo; elsewhere stay in. Nothing else removed.',
   'Raw r without the tag, ten configurations, against the measured floor.',
   'If Qwen turns positive with at least 9 of 10 configurations, the out-of-domain raw-sign reversal is restated as a formatting effect on both dialogue and chat. If it stays undetermined or negative, dialogsum&rsquo;s reversal is about content.')
 + exp('The hidden-state norm', 'P0', 'a few hours, same forward passes',
   'The fourth planned baseline has been missing twice. Does HC add to it?',
   'Norm of the hidden state entering the last MoE layer, per token. Rerun the joint partial and the R&sup2; increment with all four baselines on Qwen&rsquo;s tier-A cells, the unseen cells and Gemma.',
   'Retention after all four baselines; R&sup2; increment with its interval.',
   'If at least half survives in three of four tier-A cells and the increment still excludes zero in all four, the v15&ndash;v16 finding stands with four baselines. If below a quarter in two or more, it is restated in terms of the norm.')
 + exp('Why identifiers are negative in code', 'P1', 'a few hours, existing token series',
   'Within identifiers, high HC marks occurrences the large model gains less on, in both families. Is that copying?',
   'Split code identifiers into first occurrences and repeats of a name already in the context (prompt, signature or earlier code). Within-string r(HC, IG | CE<sub>small</sub>) for each, both families, both code sets.',
   'The partial on repeated versus first-occurrence identifiers.',
   'If the negative association is carried by repeated identifiers (at least twice as strong, interval excluding zero) and first occurrences are null or positive, code&rsquo;s exception is stated as a copying effect. If both are negative, it stays a domain effect.')
 + exp('Shared-expert ablation in Gemma&rsquo;s large model', 'P1', '~half a day, existing models',
   'Is Gemma/race&rsquo;s scale-specific inversion carried by the always-active shared expert?',
   'Scale the last MoE layer&rsquo;s shared-expert output by 1, 0.5 and 0 and recompute CE<sub>big</sub> on race and competition_math (HC is unchanged by construction). Second arm: ablate all layers&rsquo; shared experts and recompute HC.',
   'r(HC, IG | CE<sub>small</sub>) at each setting; mean CE<sub>big</sub> at each setting.',
   'If the race inversion shrinks by half or more at 0.5 while mean CE rises by less than 10%, the shared expert is implicated. If unchanged at 0, it is ruled out.')
 + exp('A larger partner', 'P2', '~half a day',
   'Does a Qwen3-8B partner reverse other benchmarks, as the active-compute account of lambada&rsquo;s 4B reversal predicts?',
   'Add Qwen3-8B as a fourth small partner; two-sided control on the nine studied benchmarks.',
   'r(HC, IG | CE<sub>small</sub>) per benchmark with the 8B partner.',
   'If three or more benchmarks turn negative, the active-compute account is supported.')
 + exp('Layer sweep (last resort)', 'Deferred', '~1 day per family if router logits were cached; more otherwise',
   'Every result is about the last MoE layer. Does the relationship live only there?',
   'HC from every MoE layer&rsquo;s router for Qwen&rsquo;s tier-A cells and Gemma/race; within-string partials by depth.',
   'The scale-specific partial as a function of depth.',
   'Run only if a reviewer or a later result makes the last-layer scope decisive. Late-only &rarr; a property of late routing; from the middle layers &rarr; routing in general.')
 + '</section>\n') + s[j:]
open('/home/claude/v16/HC_MoE_Report_v16.html', 'w', encoding='utf-8').write(s)
print('ok', len(s))
