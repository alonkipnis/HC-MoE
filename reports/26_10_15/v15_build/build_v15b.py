import json, re
s = open('/home/claude/v15/stage1.html', encoding='utf-8').read()
N = json.load(open('/home/claude/v15/numbers.json'))
QW = {r['dataset']: r for r in N['QW']}; GW = {r['dataset']: r for r in N['GW']}
BC = {(r['model'], r['dataset']): r for r in N['BC']}
def f3(x, d=3): return ('+' if x >= 0 else '-') + f'{abs(x):.{d}f}'
def pct(x): return f'{100*x:.0f}%'
def rep(a, b, cnt=1):
    global s
    n = s.count(a); assert n == cnt, (n, a[:90]); s = s.replace(a, b)
def cut(start, end):
    """return (i, j) of the span from start marker up to (not incl.) end marker"""
    i = s.index(start); j = s.index(end, i); return i, j

qa = {d: BC[('Qwen', d)] for d in ['competition_math', 'lambada', 'race', 'fineweb-edu']}

# ------------------------------------------------------------ header + abstract
rep('<h1>HC-MoE v14.1: Out of Domain, at the Last MoE Layer</h1>',
    '<h1>HC-MoE v15: Beyond the Studied Benchmarks, Beyond Simpler Router Signals</h1>')
i, j = cut('<div class="subtitle">', '</div>')
s = s[:i] + '<div class="subtitle">The three P0 experiments from v14.1 &mdash; measured floors out of domain, four more unseen benchmarks, and last-layer router baselines</div>' + s[j + 6:]
rep('October 11, 2026  &middot;  v14.1', 'October 15, 2026  &middot;  v15')
i = s.index('<div class="abstract">'); j = s.index('</div>', i) + 6
abstract = (
 '<div class="abstract"><b>Abstract.</b> v14.1 set three P0 experiments, each with its decision rule written down first. All three ran, and all three meet their rules &mdash; two of them with little to spare. '
 f'<b>Unseen benchmarks.</b> Graded against measured floors, Qwen&rsquo;s scale-specific association &mdash; within token strings, either model&rsquo;s loss held fixed &mdash; is positive and outside the floor on four of six benchmarks the series had never used: news (cnn_dailymail, {f3(QW["cnn_dailymail"]["wS"])}), scientific papers ({f3(QW["scientific_papers"]["wS"])}), open-ended chat (ultrachat, {f3(QW["ultrachat"]["wS"])}) and dialogue (dialogsum, {f3(QW["dialogsum"]["wS"])}). '
 f'It fails on both code sets: mbpp reverses ({f3(QW["mbpp"]["wS"])}) and humaneval is negative in every configuration, at the edge of its floor ({f3(QW["humaneval"]["wS"])}). By the rule, the Qwen claim is restated as <b>domain-general with one named exception, code</b>. '
 'The raw correlation does not generalise the same way &mdash; positive on three, negative on two, undetermined on one &mdash; so what carries over is the within-string association, not the overall sign. '
 f'Gemma&rsquo;s raw inversion holds on all six in every configuration, strongest on humaneval ({f3(GW["humaneval"]["raw"])}). '
 f'<b>Measured floors.</b> dialogsum&rsquo;s floor measured directly is &plusmn;{QW["dialogsum"]["floor"]:.4f} (v14 inferred &plusmn;0.0067) and nine of ten configurations stay positive, so the letter of the v13 pass stands; the rows shared with v14 reproduce exactly. '
 f'<b>Last-layer baselines.</b> After partialling router entropy, the maximum router probability and the top-1/top-2 logit gap together, at least half of Qwen&rsquo;s scale-specific association survives in three of four tier-A cells (competition_math {pct(qa["competition_math"]["ret"])}, lambada {pct(qa["lambada"]["ret"])}, race {pct(qa["race"]["ret"])}; fineweb-edu {pct(qa["fineweb-edu"]["ret"])}), and what survives excludes zero in all four. '
 f'Two qualifications travel with this. The simpler signals predict Intelligence Gain about as well as HC does, so HC carries <i>distinct</i> information rather than more of it; and dialogsum&rsquo;s narrow out-of-domain pass does not survive them ({pct(N["DS_RET"])} retained). The hidden-state norm, the fourth planned baseline, is missing from the drop.</div>')
s = s[:i] + abstract + s[j:]

# ------------------------------------------------------------ section 1: what this drop changes
i, j = cut('<h3>What this drop changes</h3>', '<div class="finding null"><h4>Revision v14.1')
changes = ('<h3>What this drop changes</h3>\n'
 f'<div class="finding scale"><h4>Out of domain: four of six unseen benchmarks pass; code is the exception</h4><p><b>Widen the unseen sample (P0).</b> Rule written in v14.1: Qwen positive and outside the floor under both controls within strings on at least four of six unseen benchmarks &rarr; domain-general with named exceptions; two or fewer &rarr; specific to the studied benchmark types. <b>Met: four of six</b> (cnn_dailymail, scientific_papers, ultrachat, dialogsum). Both code benchmarks fail &mdash; mbpp reverses, humaneval is negative at its floor &mdash; so code is the named exception. The raw sign is domain-dependent; the within-string association is what generalises. Gemma&rsquo;s raw inversion holds on all six. Section 15.</p></div>\n'
 f'<div class="finding scale"><h4>The v14 drop is complete, and its pass stands</h4><p><b>Complete the unseen-benchmark drop (P0).</b> Rule: if dialogsum stays outside its measured floor with at least 9 of 10 configurations positive, the letter of the v13 rule stands. Measured floor &plusmn;{QW["dialogsum"]["floor"]:.4f}; value {f3(QW["dialogsum"]["wS"],4)}; nine of ten positive. <b>Stands.</b> The {N["REPRO_ROWS"]} rows shared with v14 are identical. Section 15.</p></div>\n'
 f'<div class="finding scale"><h4>HC carries information simpler router statistics do not &mdash; met in three of four cells</h4><p><b>Last-layer baselines (P0).</b> Rule: at least half of Qwen&rsquo;s within-string r(HC, IG | CE<sub>small</sub>) survives all baselines in three or more of four tier-A cells. <b>Met</b>: {pct(qa["competition_math"]["ret"])}, {pct(qa["lambada"]["ret"])}, {pct(qa["race"]["ret"])}; fineweb-edu {pct(qa["fineweb-edu"]["ret"])}. Only competition_math clears the line by more than its interval; lambada and race clear it at the point estimate. The baselines are themselves as predictive of IG as HC, so the finding is distinctness, not superiority. dialogsum&rsquo;s out-of-domain pass is absorbed ({pct(N["DS_RET"])} retained). The hidden-state-norm baseline was not in the drop. Section 16.</p></div>\n'
 '<div class="finding null"><h4>Not run</h4><p>The layer sweep, the format-token decomposition, the shared-expert ablation, the Qwen3-8B partner and lambada&rsquo;s top tokens under every null are carried forward (section 27).</p></div>\n')
s = s[:i] + changes + s[j:]

# ------------------------------------------------------------ findings register
a = s.index('<ul class="so-far">'); b = s.index('</ul>', a)
items = re.findall(r'<li>.*?</li>(?=<li>|$)', s[a + len('<ul class="so-far">'):b], flags=re.S)
assert len(items) == 15
def li(tier, claim, since, ev):
    return (f'<li><span class="tier tier-{tier}">{tier}</span><div class="body"><div class="claim">{claim}'
            f'<span class="since">since {since}</span></div><div class="ev">{ev}</div></div></li>')
items[1] = items[1].replace('shrinks as the compute gap closes.</div>',
    f'shrinks as the compute gap closes. <b>v15:</b> distinct from simpler router statistics &mdash; after entropy, maximum probability and logit gap are partialled out, {pct(qa["competition_math"]["ret"])}, {pct(qa["lambada"]["ret"])}, {pct(qa["race"]["ret"])} and {pct(qa["fineweb-edu"]["ret"])} remain, every interval excluding zero.</div>')
assert 'v15:' in items[1]
items[2] = li('B', '<b>Beyond the studied benchmarks, Qwen&rsquo;s scale-specific association is domain-general, with code as the exception.</b>', 'v15',
    f'Pre-registered rule met on four of six unseen benchmarks, measured floors: cnn_dailymail {f3(QW["cnn_dailymail"]["wS"])}, scientific_papers {f3(QW["scientific_papers"]["wS"])}, ultrachat {f3(QW["ultrachat"]["wS"])}, dialogsum {f3(QW["dialogsum"]["wS"])} (within strings | CE<sub>small</sub>). Fails on mbpp ({f3(QW["mbpp"]["wS"])}) and humaneval ({f3(QW["humaneval"]["wS"])}). Raw sign domain-dependent (3 positive, 2 negative, 1 undetermined). Held at B: two passes are small, dialogsum&rsquo;s does not survive the router baselines, and no tiers were computed out of domain. Replaces the v14 &lsquo;mixed&rsquo; finding.')
new_base = li('B', '<b>HC carries last-layer information that simpler router statistics do not</b> &mdash; but they predict Intelligence Gain about as well as it does.', 'v15',
    f'Pre-registered rule met in three of four Qwen tier-A cells (&ge; 50% retained after router entropy, maximum probability and logit gap together); remainder excludes zero in all four. Only competition_math clears 50% beyond its interval. Each baseline&rsquo;s own within-string association with IG is of HC&rsquo;s size (race: maximum probability {f3(BC[("Qwen","race")]["mp_ig"])} vs HC {f3(BC[("Qwen","race")]["ref"])}). Hidden-state norm not yet tested.')
items[4] = items[4].replace('</div></div></li>', '') if False else items[4]
items[4] = re.sub(r'<div class="ev">.*?</div>', lambda m: f'<div class="ev">Mean -0.0449 on the studied grid; every estimator, ranks, both halves, within token strings. <b>v15:</b> negative on all six unseen benchmarks in every configuration, against measured floors (cnn_dailymail {f3(GW["cnn_dailymail"]["raw"])}, scientific_papers {f3(GW["scientific_papers"]["raw"])}, ultrachat {f3(GW["ultrachat"]["raw"])}, dialogsum {f3(GW["dialogsum"]["raw"])}, mbpp {f3(GW["mbpp"]["raw"])}, humaneval {f3(GW["humaneval"]["raw"])}). Not absorbed by router baselines in race ({pct(BC[("gemma","race")]["ret"])}).</div>', items[4], count=1, flags=re.S)
items[5] = re.sub(r'<div class="ev">.*?</div>', lambda m: f'<div class="ev">race survives both controls within strings (-0.031), and the router baselines. <b>v15:</b> out of domain, strictly negative within strings | CE<sub>small</sub> on cnn_dailymail ({f3(GW["cnn_dailymail"]["wS"])}) and humaneval ({f3(GW["humaneval"]["wS"])}); positive on dialogsum ({f3(GW["dialogsum"]["wS"])}); null on scientific_papers; split on mbpp and ultrachat. Cause open.</div>', items[5], count=1, flags=re.S)
items[10] = re.sub(r'<div class="ev">.*?</div>', lambda m: f'<div class="ev">Median nominal n / ESS = 2.61 on the studied grid. <b>v15:</b> measured out of domain, 1.0&ndash;5.5&times; (median {N["DEFL_MED"]:.1f}&times;).</div>', items[10], count=1, flags=re.S)
items = items[:3] + [new_base] + items[3:]
s = s[:a + len('<ul class="so-far">')] + ''.join(items) + s[b:]
# intro paragraph before the list
i = s.index('<h3>Main findings so far</h3>'); j = s.index('<ul class="so-far">', i)
s = s[:i] + ('<h3>Main findings so far</h3>\n<p>What the project holds after this report. Every result is a statement about the routing of each large model&rsquo;s <b>last MoE layer</b> (v14.1). v15 widens the Qwen claim beyond the nine studied benchmarks, with code excepted, and adds one finding: HC&rsquo;s association is not reducible to simpler statistics of the same router. Nothing is withdrawn.</p>') + s[j:]

# ------------------------------------------------------------ checks register
rep('<p>Recorded as each check ran. One new check this cycle; none outstanding.</p>',
    '<p>Recorded as each check ran. Three new checks this cycle. One planned component, the hidden-state-norm baseline, is missing from the drop and is marked partial.</p>')
old_ood = re.search(r'<tr><td><strong>Out-of-domain confirmation</strong>.*?</tr>', s, flags=re.S).group(0)
new_rows = (
 '<tr><td><strong>Out-of-domain confirmation</strong><span class="since">v14</span></td><td>Whether the Qwen result appears on benchmarks the series had never used, with predictions fixed in advance</td><td><span class="status status-done">done</span></td>'
 f'<td><b>Widened in v15</b> to six unseen benchmarks with measured floors. Qwen meets the prediction on four (cnn_dailymail, scientific_papers, ultrachat, dialogsum) and fails on both code sets (mbpp {f3(QW["mbpp"]["wS"])}, humaneval {f3(QW["humaneval"]["wS"])}): domain-general with code excepted, by the v14.1 rule. Gemma&rsquo;s raw r negative on all six, every configuration. 120/120 rows.</td></tr>'
 '<tr><td><strong>Measured ESS out of domain</strong><span class="since">v15</span></td><td>Whether the v14 floors, inferred from bootstrap widths, were right</td><td><span class="status status-done">done</span></td>'
 f'<td><b>New (v15).</b> Measured by the 26_09_03 autocorrelation method for all twelve unseen cells: deflation 1.0&ndash;5.5&times; (median {N["DEFL_MED"]:.1f}&times;). The v14 inference was within 8% for Qwen; dialogsum&rsquo;s floor &plusmn;{QW["dialogsum"]["floor"]:.4f} against &plusmn;0.0067 inferred. The {N["REPRO_ROWS"]} rows shared with v14 are identical, so v14&rsquo;s numbers stand.</td></tr>'
 '<tr><td><strong>Last-layer router baselines</strong><span class="since">v15</span></td><td>Whether HC adds anything beyond simpler statistics of the same router</td><td><span class="status status-partial">partial</span></td>'
 f'<td><b>New (v15).</b> Router entropy, maximum probability, top-1/top-2 logit gap; within-string r(HC, IG | CE<sub>small</sub>) partialled on each and all. Qwen tier-A retention {pct(qa["competition_math"]["ret"])}, {pct(qa["lambada"]["ret"])}, {pct(qa["race"]["ret"])}, {pct(qa["fineweb-edu"]["ret"])}: rule met in 3 of 4, remainder outside zero in all four. Baselines predict IG about as strongly as HC. <b>Partial:</b> the hidden-state norm was not computed, and ultrachat, mbpp and humaneval were not run.</td></tr>')
s = s.replace(old_ood, new_rows, 1)

# ------------------------------------------------------------ family notes
qn = (f'<div class="finding scale"><h4>v15: out of domain, and against simpler router signals</h4><p>On six unseen benchmarks with measured floors, Qwen&rsquo;s within-string scale-specific association is positive under both controls on news, scientific papers, chat and dialogue, and negative on both code sets (section 15). Partialling router entropy, maximum probability and logit gap from the same router keeps {pct(qa["competition_math"]["ret"])}, {pct(qa["lambada"]["ret"])}, {pct(qa["race"]["ret"])} and {pct(qa["fineweb-edu"]["ret"])} of it in the four tier-A cells; each of those statistics predicts IG about as strongly on its own (section 16).</p></div>\n')
gn = (f'<div class="finding gemma"><h4>v15: the raw inversion on six unseen benchmarks</h4><p>Negative on all six in every configuration, from {f3(GW["scientific_papers"]["raw"])} (scientific_papers) to {f3(GW["humaneval"]["raw"])} (humaneval, the strongest Gemma cell measured). The within-string scale-specific part is strictly negative on cnn_dailymail and humaneval and positive on dialogsum. Router baselines do not absorb the race inversion ({pct(BC[("gemma","race")]["ret"])} retained). Section 15&ndash;16.</p></div>\n')
for sid, txt in (('qwen', qn), ('gemma', gn)):
    i = s.index(f'<section id="{sid}">'); j = s.index('<div class="card">', i)
    s = s[:j] + txt + s[j:]

# ------------------------------------------------------------ assessment
i = s.index('<div class="assess">'); j = s.index('</section>', i)
assess = ('<div class="assess"><div class="col strong"><h4>Holds up</h4><ul>'
 '<li><b>On the nine studied benchmarks</b>, Qwen&rsquo;s &lsquo;where scale helps&rsquo; holds at every level tested: both difficulty controls, within token strings, every configuration, held-out halves, across compute ratios &mdash; and, new in v15, after simpler statistics of the same router are partialled out.</li>'
 '<li><b>Out of domain, Qwen&rsquo;s scale-specific association generalises</b> to news, scientific text, chat and dialogue: four of six unseen benchmarks against measured floors, meeting a rule written before the data.</li>'
 '<li><b>Gemma&rsquo;s raw inversion is the most general result in the series</b>: negative on all nine studied and all six unseen benchmarks, every configuration.</li>'
 '<li><b>HC tracks token difficulty in all four families</b>; 31 and 28 of 36 cells outside the floor.</li>'
 '<li><b>The method</b>: three pre-registered rules this cycle, all applied as written, verdicts recomputed from source tables, exact reproduction of the previous drop.</li></ul></div>'
 '<div class="col weak"><h4>Not yet established</h4><ul>'
 '<li><b>Code.</b> Both code benchmarks fail for Qwen, and Gemma&rsquo;s scale-specific part is also negative on both. Two benchmarks, no mechanism.</li>'
 '<li><b>How much of HC is unique.</b> Half or more survives the baselines in three cells, but only competition_math clears that line beyond its interval; the hidden-state norm is untested.</li>'
 '<li><b>HC is not the strongest last-layer predictor.</b> Maximum router probability and the logit gap predict IG about as well; HC is complementary, not better.</li>'
 '<li><b>One layer.</b> Every result is last-MoE-layer routing; no earlier layer has been measured.</li>'
 '<li><b>Small out-of-domain passes.</b> ultrachat and dialogsum pass by less than one floor width, and dialogsum&rsquo;s is absorbed by the baselines.</li>'
 '<li><b>Gemma&rsquo;s scale-specific part</b> changes sign across unseen benchmarks; cause open. The extreme tail is null-dependent and the lambada 4B reversal unexplained.</li></ul></div>'
 '<div class="col open"><h4>Open</h4><ul>'
 '<li><b>Why does code reverse, in both families?</b> Indentation and whitespace tokens, identifiers, or something about code as a domain?</li>'
 '<li><b>Where in the network does the relationship live?</b> The layer sweep is now the largest scope caveat left.</li>'
 '<li><b>Is the raw-sign reversal on dialogue and chat a format-token effect?</b></li>'
 '<li><b>Does HC add to the hidden-state norm, and to all four baselines in a joint model?</b></li>'
 '<li><b>Is Gemma/race&rsquo;s inversion carried by the shared expert?</b></li></ul></div></div>')
s = s[:i] + assess + s[j:]

# ------------------------------------------------------------ next experiments
i = s.index('<section id="next">'); j = s.index('<section id="tables">')
h2 = re.match(r'<section id="next"><h2>[^<]*</h2>', s[i:]).group(0)
def exp(title, pr, cost, q, d, st, rule):
    return (f'<div class="exp"><h4>{title}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd>'
            f'<dt>Design</dt><dd>{d}</dd><dt>Statistic</dt><dd>{st}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>\n')
nxt = (h2 + '<h3>What happened to v14.1&rsquo;s list</h3>\n<div class="table-wrap"><table><thead><tr><th>v14.1 experiment</th><th>Status</th><th>Outcome</th></tr></thead><tbody>'
 '<tr><td>Complete the unseen-benchmark drop (P0)</td><td><b>Done</b></td><td>Measured floors; dialogsum&rsquo;s pass stands; v14 rows reproduce exactly.</td></tr>'
 '<tr><td>Widen the unseen sample (P0)</td><td><b>Done</b></td><td>Four of six pass: domain-general with code excepted.</td></tr>'
 '<tr><td>Last-layer baselines (P0)</td><td><b>Done, partial</b></td><td>Rule met in 3 of 4 cells; hidden-state norm not computed.</td></tr>'
 '<tr><td>Layer sweep (P1)</td><td>Not run</td><td>Carried, now P0.</td></tr>'
 '<tr><td>What drives the negative raw sign out of domain (P1)</td><td>Not run</td><td>Carried; now covers ultrachat too.</td></tr>'
 '<tr><td>Shared-expert ablation in Gemma (P1)</td><td>Not run</td><td>Carried.</td></tr>'
 '<tr><td>A larger partner and lambada&rsquo;s tokens under every null (P2)</td><td>Not run</td><td>Carried.</td></tr></tbody></table></div>\n<h3>Proposed</h3>\n'
 + exp('Layer sweep', 'P0', '~1 day, existing forward passes if router logits were cached',
   'Every result is a statement about the last MoE layer, the one feeding the output head. Does the relationship live there only, or throughout the network?',
   'HC from every MoE layer&rsquo;s router (five estimators, two variants) for Qwen on its four tier-A benchmarks plus cnn_dailymail and scientific_papers, and Gemma on race. Raw r(HC, IG) and both difficulty controls within strings, per layer.',
   'The within-string r(HC, IG | CE<sub>small</sub>) as a function of depth, against each cell&rsquo;s floor.',
   'If the scale-specific partial is outside the floor only in the last few layers, the finding is stated as a property of late routing. If it appears from the middle layers on, the claim widens to routing in general and the last-layer choice stops being a caveat.')
 + exp('Complete the baselines', 'P0', 'a few hours, same forward passes',
   'The hidden-state norm was planned and not computed, and three unseen cells were not run. Separately: HC and the simpler statistics predict IG about equally well &mdash; what does a joint model gain from HC?',
   'Add the norm of the hidden state entering the last MoE layer. Rerun the partials with all four baselines on Qwen&rsquo;s four tier-A cells, Gemma/race, and all six unseen cells. Add a within-string regression of IG on the four baselines with and without HC.',
   'Retention after all four baselines; the increase in within-string R&sup2; from adding HC, with its passage-bootstrap interval.',
   'If at least half survives all four baselines in three of four tier-A cells, the v15 finding stands as stated. If below a quarter in two or more, it is restated in terms of the absorbing baseline. If HC&rsquo;s R&sup2; increment excludes zero in every tier-A cell, a paper can state that HC adds to the best simple router statistics, not only to each one.')
 + exp('Why code reverses', 'P1', 'a few hours, existing token series',
   'Both code benchmarks fail for Qwen, and Gemma&rsquo;s scale-specific part is negative on both too. Is that one token class, or code as a domain?',
   'Classify mbpp and humaneval tokens: indentation and whitespace, newlines, identifiers, keywords, operators and punctuation, string literals. Compute the within-string r(HC, IG | CE<sub>small</sub>) per class, both families, and the full-benchmark value with each class removed.',
   'Per-class partials and their contributions to the benchmark value.',
   'If removing one class turns Qwen positive on both code sets, code is reported as a token-class exception. If the negative part is spread across classes, code is reported as a domain where last-layer routing anomalies mark tokens the large model handles worse.')
 + exp('What drives the negative raw sign out of domain', 'P1', 'a few hours, existing token series',
   'On dialogsum and ultrachat the within-string association is positive and the raw one negative or null. Which token types carry the between-string part?',
   'Decompose r(HC, IG) into between- and within-string parts for dialogsum, ultrachat and mbpp. Rank strings by contribution to the between part; check speaker tags, role markers, newlines and indentation. Recompute raw r with them removed.',
   'Between-string contribution by token string; raw r with format tokens excluded.',
   'If removing format tokens turns Qwen&rsquo;s raw r positive on dialogsum and ultrachat, the reversal is a formatting effect. If it stays negative, it is about content.')
 + exp('Shared-expert ablation in Gemma&rsquo;s large model', 'P1', '~half a day, existing models',
   'Is Gemma/race&rsquo;s scale-specific inversion carried by the always-active shared expert?',
   'Scale the last MoE layer&rsquo;s shared-expert output by 1, 0.5 and 0 and recompute CE<sub>big</sub> on race and competition_math (HC is unchanged by construction). Second arm: ablate the shared experts of all layers and recompute HC too.',
   'r(HC, IG | CE<sub>small</sub>) at each setting; the large model&rsquo;s mean CE at each setting.',
   'If the race inversion shrinks by half or more at 0.5 while mean CE rises by less than 10%, the shared expert is implicated. If unchanged at 0, it is ruled out.')
 + exp('A larger partner and lambada&rsquo;s tokens under every null', 'P2', '~half a day',
   'Carried from v13: does a Qwen3-8B partner reverse other benchmarks, and are lambada&rsquo;s near-exclusive strings at the top under every null?',
   'Add Qwen3-8B as a fourth small partner and rerun the two-sided control on the nine studied benchmarks; list lambada&rsquo;s top-ventile tokens under each of the ten configurations.',
   'r(HC, IG | CE<sub>small</sub>) per benchmark with the 8B partner; share of the two strings in lambada&rsquo;s top ventile per configuration.',
   'If three or more benchmarks turn negative with the 8B partner, the active-compute account is supported. If those two strings keep at least half their GMM share under every null, the lambada description holds across nulls.')
 + '</section>\n')
s = s[:i] + nxt + s[j:]

# ------------------------------------------------------------ old section 14: pointer
rep('<section id="ood"><h2>14. Out of domain: two unseen benchmarks</h2>',
    '<section id="ood"><h2>14. Out of domain: two unseen benchmarks</h2><div class="caveat"><b>Superseded in part (v15).</b> The floors in this section were inferred; section 15 measures them, reproduces every number here exactly, and widens the sample to six benchmarks. The reading below is v14&rsquo;s and is kept as the record of that cycle.</div>')
rep('<title>HC-MoE v14.1', '<title>HC-MoE v15') if '<title>HC-MoE v14.1' in s else None
open('/home/claude/v15/HC_MoE_Report_v15.html', 'w', encoding='utf-8').write(s)
print('ok', len(s))

# ------------------------------------------------------------ changed-this-cycle table and data-gaps note
s = open('/home/claude/v15/HC_MoE_Report_v15.html', encoding='utf-8').read()
i = s.index('<h3>Changed this cycle</h3>'); j = s.index('<h3>Checks done</h3>', i)
chg = ('<h3>Changed this cycle</h3>\n<div class="table-wrap"><table><thead><tr><th>Claim</th><th>Was</th><th>From</th><th>Now</th></tr></thead><tbody>'
 '<tr><td>The Qwen result generalises beyond the studied benchmarks</td><td>Mixed (B)</td><td>v14</td><td>Domain-general with code excepted (B): four of six unseen benchmarks pass a pre-registered rule against measured floors.</td></tr>'
 '<tr><td>HC adds to simpler last-layer router statistics</td><td>untested</td><td>&mdash;</td><td>New (B): rule met in three of four Qwen tier-A cells; the baselines predict IG about as well as HC.</td></tr>'
 '<tr><td>Qwen: HC marks the tokens where scale helps (studied benchmarks)</td><td>A</td><td>v11</td><td>Stays A; now also shown distinct from router entropy, maximum probability and logit gap.</td></tr>'
 '<tr><td>Gemma&rsquo;s raw inversion</td><td>A</td><td>v5</td><td>Strengthened: negative on all six unseen benchmarks, every configuration.</td></tr>'
 '<tr><td>Gemma&rsquo;s scale-specific inversion</td><td>B</td><td>v11</td><td>Stays B: strictly negative out of domain on cnn_dailymail and humaneval, positive on dialogsum.</td></tr>'
 '<tr><td>Unseen-benchmark floors</td><td>inferred from CI widths</td><td>v14</td><td>Measured; the v14 inference held to within 8% for Qwen.</td></tr></tbody></table></div>\n')
s = s[:i] + chg + s[j:]
i = s.index('<div class="caveat"><b>Data gaps in this drop.</b>'); j = s.index('</div>', i) + 6
s = s[:i] + ('<div class="caveat"><b>Data gaps in this drop.</b> The baseline experiment omits the hidden-state norm, the fourth planned baseline, and was not run on ultrachat, mbpp or humaneval. The unseen-benchmark cells are graded by the pre-registered rule against measured floors; ARF and tiers were not computed for them. The v14 gap is closed: token counts, measured ESS and raw intervals are now present, and the rerun reproduces v14&rsquo;s Gemma numbers exactly.</div>') + s[j:]
s = s.replace('<title>HC-MoE v15: Out of Domain, at the Last MoE Layer', '<title>HC-MoE v15: Beyond the Studied Benchmarks', 1)
open('/home/claude/v15/HC_MoE_Report_v15.html', 'w', encoding='utf-8').write(s)
print('stage3 ok')
