from common import *

c = CELL
q5 = ['fineweb-edu', 'lambada', 'race', 'cnn_dailymail', 'scientific_papers']
F = N['dose_fam']; DB = N['dose_bm']
qd = D['Qwen']['cells']
def rng(vals): vals = [v for v in vals if v is not None]; return f'{f3(min(vals))} to {f3(max(vals))}'
race_mid_b = [qd['race'][str(l)]['b'] for l in [5, 10, 11, 12, 17, 22, 23, 24, 29]]
fw_mid_b = [qd['fineweb-edu'][str(l)]['b'] for l in [5, 11, 17, 23, 29, 35]]
race_mid_s = [qd['race'][str(l)]['s'] for l in [10, 11, 12, 22, 23, 24, 29, 34, 35, 36]]
cpq_s = N['cp']['Qwen|ce_mean_decile|mean_part_s_c']; cpq_b = N['cp']['Qwen|ce_mean_decile|mean_part_b_c']
cpg_s = N['cp']['gemma|ce_mean_decile|mean_part_s_c']; cpg_b = N['cp']['gemma|ce_mean_decile|mean_part_b_c']
cpq_ss = N['cp']['Qwen|ce_small_decile|mean_part_s_c']
gd = D['gemma']['cells']; ed = D['ernie']['cells']; ld = D['liquid']['cells']
E18 = ESEL[18]; E27 = ESEL[27]
lp = LPOS
GAB = {(r['dataset'], r['stats_mode'], r['alpha']): r for r in N['gab']}
UEL = {(r['model'], r['benchmark']): r for r in N['unseen_el']}
cpe_s = N['cp']['ernie|ce_mean_decile|mean_part_s_c']; cpe_b = N['cp']['ernie|ce_mean_decile|mean_part_b_c']
cpl_s = N['cp']['liquid|ce_mean_decile|mean_part_s_c']; cpl_b = N['cp']['liquid|ce_mean_decile|mean_part_b_c']
def gab(d, a, k, mode='alpha_self_stats'): return GAB[(d, mode, a)][k]
shr = lambda d, k, mode='alpha_self_stats': GAB[(d, mode, 0.75)][k]
pos_cells = [(m, b) for (m, b), x in CELL.items() if x['c_cnt'] >= 9]
bl_cells = {'Qwen': 'Qwen_L47_', 'ernie': 'ernie_L27_', 'liquid': 'liquid_L23_'}
bl_tested = [(m, b, bl_cells[m] + b) for m, b in pos_cells if m in bl_cells and (bl_cells[m] + b) in B]
bl_pass = [(m, b) for m, b, k in bl_tested if B[k]['R3s'] >= 0.5 and B[k]['R3b'] >= 0.5]
bl_fail = [(m, b) for m, b, k in bl_tested if not (B[k]['R3s'] >= 0.5 and B[k]['R3b'] >= 0.5)]
cross = N['ca_cross']
a_pos = [(m, b) for (m, b), x in CELL.items() if x['a_cnt'] >= 9]
a_only = [(m, b) for (m, b) in a_pos if CELL[(m, b)]['c_cnt'] < 9]
c_only = [(m, b) for (m, b), x in CELL.items() if x['c_cnt'] >= 9 and x['a_cnt'] < 9]
fs_cells = [(m, b) for (m, b), x in CELL.items() if x['floor_sens_c']]
def shrinks(d, a, mode):
    if a == 1.0: return '&ndash;'
    base_s = abs(GAB[(d, mode, 1.0)]['mean_r_part_ces']); fl_ = GAB[(d, mode, 1.0)]['floor']
    ss = f'{GAB[(d, mode, a)]["inversion_shrink_pct_ces"]:.0f}%' if base_s > fl_ else 'n/a'
    return f'{ss} / {GAB[(d, mode, a)]["inversion_shrink_pct_ceb"]:.0f}%'
DIF = {(r['model'], r['bm']): r for r in N['diff']}; DS = N['diff_sum']
QT = {(r['dataset'], r['layer']): r for r in N['qtrans']}
CD = {(r['model'], r['benchmark'], r['stratum']): r for r in N['codeD']}
GL = {(r['layer'], r['benchmark'], r['ablation_mode'], r['k_dropped']): r for r in N['gl']}
GA = {(r['benchmark'], r['layer']): r for r in N['gaudit']}
bl4_pass = [(m, b) for m, b, k in bl_tested if B[k]['R4s'] is not None and B[k]['R4s'] >= 0.5 and B[k]['R4b'] >= 0.5]
bl4_fail = [(m, b) for m, b, k in bl_tested if (m, b) not in bl4_pass]
STL = {'ALL_TOKENS': 'all tokens', 'CLASS_NEWLINE': 'newlines only', 'LOCO_NO_NEWLINE': 'without newlines', 'CLASS_IDENTIFIER': 'identifiers only', 'LOCO_NO_IDENTIFIER': 'without identifiers', 'ALL_1ST_OCC': 'first occurrences', 'ALL_REPEAT_OCC': 'repeat occurrences'}
lab = lambda mb: f'{FAMN[mb[0]]} {SHORT[mb[1]]}'

# ------------------------------------------------------------------ figures
specs = [fig_cells(), fig_dose()]
_labs = MAIN + UNSEEN
_sd = bar_chart('f_diff', [SHORT[b] for b in _labs], {FAMN[f]: [DIF[(f, b)]['c_within_s'] for b in _labs] for f in ['Qwen', 'ernie', 'liquid', 'gemma']},
                null_band=FLOOR, colors=[FAMC['Qwen'], FAMC['ernie'], FAMC['liquid'], FAMC['gemma']], y_title='within-string r(HC, CE_small)', height=360)
for d_ in _sd['data']['datasets']:
    if d_.get('label','').startswith('null envelope'): d_['label']='typical floor (\u00b10.0128)'
specs.append(_sd)
for d_ in specs[0]['data']['datasets']:
    if d_.get('label','').startswith('null envelope'): d_['label']='typical floor (\u00b10.0128)'
specs.append(line_gaps('f_dose_bm', [f'{p} ({RAT[p]}×)' for p in P],
              {SHORT[b]: DB['ts'][b] for b in ['fineweb-edu', 'race', 'lambada', 'competition_math', 'structured-wikipedia']} | {'family mean': F['ts']},
              colors=['#60a5fa', '#f59e0b', '#a78bfa', '#94a3b8', '#2dd4bf', '#f1f5f9'], x_title='Qwen3 dense partner (active-parameter ratio)',
              y_title='two-sided strength (r)', null_band=FLOOR, dashed=('family mean',)))
bl = ['Qwen_L47_fineweb-edu', 'Qwen_L47_lambada', 'Qwen_L47_race', 'Qwen_L47_cnn_dailymail', 'Qwen_L47_scientific_papers', 'ernie_L27_race', 'ernie_L27_fineweb-edu',
      'ernie_L27_tinystories', 'ernie_L27_cnn_dailymail', 'ernie_L27_mbpp', 'liquid_L23_race', 'liquid_L23_pg19', 'liquid_L23_lambada', 'gemma_L29_race', 'gemma_L29_humaneval']
blab = ['Qwen fineweb', 'Qwen lambada', 'Qwen race', 'Qwen cnn', 'Qwen sci', 'ERNIE race', 'ERNIE fineweb', 'ERNIE tinystories', 'ERNIE cnn', 'ERNIE mbpp',
        'LFM2 race', 'LFM2 pg19', 'LFM2 lambada', 'Gemma race (inv.)', 'Gemma humaneval (inv.)']
sp = bar_chart('f_base', blab, {'r | CE_small, 3 statistics': [100 * B[k]['R3s'] for k in bl], 'r | CE_big, 3 statistics': [100 * B[k]['R3b'] for k in bl],
               'r | CE_small, + norm': [100 * B[k]['R4s'] for k in bl], 'r | CE_big, + norm': [100 * B[k]['R4b'] for k in bl]},
               colors=['rgba(245,158,11,0.45)', 'rgba(96,165,250,0.45)', '#f59e0b', '#60a5fa'], y_title='% of the partial retained', height=380)
sp['data']['datasets'].append(dict(type='line', label='50% rule', data=[50] * len(bl), borderColor='rgba(226,232,240,0.8)', borderDash=[5, 4], borderWidth=1.5, pointRadius=0, fill=False, order=99))
specs.append(sp)
specs.append(fig_qdepth())
Lq = D['Qwen']['layers']
specs.append(line_gaps('f_qrace', [f'L{l}' for l in Lq], {'race r | CE_small': [qd['race'][str(l)]['s'] for l in Lq], 'race r | CE_big': [qd['race'][str(l)]['b'] for l in Lq],
              'fineweb r | CE_small': [qd['fineweb-edu'].get(str(l), {}).get('s') for l in Lq], 'fineweb r | CE_big': [qd['fineweb-edu'].get(str(l), {}).get('b') for l in Lq]},
              colors=['#f59e0b', '#f59e0b', '#60a5fa', '#60a5fa'], dashed=('race r | CE_big', 'fineweb r | CE_big'), x_title='Qwen3-30B-A3B MoE layer', y_title='within-string partial r', null_band=FLOOR))
Le, se = ts_series('ernie', ['race', 'fineweb-edu', 'pg19', 'tinystories', 'lambada', 'competition_math', 'structured-wikipedia'])
specs.append(line_gaps('f_edepth', [f'L{l}' for l in Le], se, colors=['#f59e0b', '#60a5fa', '#2dd4bf', '#a78bfa', '#f472b6', '#94a3b8', '#fde68a'],
              x_title='ERNIE-4.5-21B-A3B MoE layer', y_title='two-sided strength (r)', null_band=FLOOR))
Lg, sg = ts_series('gemma', ['race', 'fineweb-edu', 'competition_math', 'lambada'])
specs.append(line_gaps('f_gdepth', [f'L{l}{"*" if D["gemma"]["att"][str(l)] == "full_attention" else ""}' for l in Lg], sg, colors=['#f59e0b', '#60a5fa', '#94a3b8', '#a78bfa'],
              x_title='gemma-4-26B-A4B MoE layer (* = full-attention layer)', y_title='two-sided strength (r)', null_band=FLOOR))
Ll, sl = ts_series('liquid', ['pg19', 'race', 'lambada', 'tinystories', 'competition_math', 'structured-wikipedia'])
specs.append(line_gaps('f_ldepth', [f'L{l}' for l in Ll], sl, colors=['#2dd4bf', '#f59e0b', '#a78bfa', '#60a5fa', '#94a3b8', '#f87171'],
              x_title='LFM2-8B-A1B MoE layer', y_title='two-sided strength (r)', null_band=FLOOR))
AL = [1.0, 0.75, 0.5]
specs.append(line_gaps('f_abl', ['\u03b1 = 1', '\u03b1 = 0.75', '\u03b1 = 0.5'],
              {f'{SHORT[d]} r | CE_{k}': [gab(d, a, 'mean_r_part_ce' + ('s' if k == 'small' else 'b')) for a in AL] for d in ['race', 'fineweb-edu', 'competition_math'] for k in ['small', 'big']},
              colors=['#f59e0b', '#f59e0b', '#60a5fa', '#60a5fa', '#94a3b8', '#94a3b8'],
              dashed=tuple(f'{SHORT[d]} r | CE_big' for d in ['race', 'fineweb-edu', 'competition_math']),
              x_title='shared-expert output scale in every Gemma layer (HC recomputed from the ablated model)', y_title='within-string partial r', null_band=FLOOR))
dec = [f'D{i}' for i in range(1, 11)]
specs.append(line_gaps('f_code', dec, {'code, r | CE_small': cpq_s['code'], 'prose, r | CE_small': cpq_s['prose'], 'code, r | CE_big': cpq_b['code'], 'prose, r | CE_big': cpq_b['prose']},
              colors=['#f87171', '#60a5fa', '#f87171', '#60a5fa'], dashed=('code, r | CE_big', 'prose, r | CE_big'), x_title='decile of mean loss of the two models (easy → hard), Qwen',
              y_title='within-decile partial r', null_band=FLOOR))

# ------------------------------------------------------------------ section 1 registers
findings = [
 li('A', '<b>In Qwen3, HC marks the tokens where scale helps, beyond either model&rsquo;s difficulty</b> &mdash; on fineweb-edu, lambada and race, and on unseen news and scientific text.',
    'Within token strings, both partials beyond the floor in 10/10 configurations in all five cells: ' +
    '; '.join(f'{SHORT[b]} {f3(c[("Qwen", b)]["c_s"])} / {f3(c[("Qwen", b)]["c_b"])}' for b in q5) +
    ' (r | CE<sub>small</sub> / r | CE<sub>big</sub>). Each replicates in both held-out halves (9&ndash;10/10); survives winsorising, ranks, removing the strongest expert and partialling p<sub>min</sub>; holds under both floor methods and under the old scaling (Arm A, 10/10).', 'since v11 &middot; re-established under C in v20'),
 li('B', '<b>The same scale-specific association appears in ERNIE and LFM2 on specific benchmarks; race is positive in three of four families.</b>',
    f'ERNIE (layer 27): race {cnt_html(cnt("ernie","race"))}, fineweb {cnt_html(cnt("ernie","fineweb-edu"))}, tinystories {cnt_html(cnt("ernie","tinystories"))}, and out of domain cnn {cnt_html(cnt("ernie","cnn_dailymail"))} and mbpp {cnt_html(cnt("ernie","mbpp"))} (sci {cnt("ernie","scientific_papers")}/10) &mdash; but ERNIE&rsquo;s mbpp signal is carried entirely by newline tokens (v21). LFM2 (layer 23): race {cnt_html(cnt("liquid","race"))}, pg19 {cnt_html(cnt("liquid","pg19"))}, lambada {cnt_html(cnt("liquid","lambada"))}, but none of its six unseen benchmarks. The pre-registered rule (unseen prose passes in at least one of the two families) is met through ERNIE cnn. Held at B: other benchmarks go the other way on r | CE<sub>small</sub> (ERNIE lambada, hellaswag, dialogsum; LFM2 comp_math), LFM2 does not generalise, and LFM2 race and pg19 and ERNIE cnn fail the router-baseline rule.', 'new in v20 &middot; out of domain v20.5'),
 li('B', '<b>The association shrinks as the dense partner closes the compute gap; the family mean reaches the floor at 8B, but race and fineweb persist.</b>',
    f'Qwen3-30B-A3B fixed, partner 0.6B &rarr; 8B, family means over nine benchmarks: r | CE<sub>small</sub> {" &rarr; ".join(f3(v) for v in F["part_s_c"])}; r | CE<sub>big</sub> {" &rarr; ".join(f3(v) for v in F["part_b_c"])}. Cells passing both partials: 3, 3, 2, 2. fineweb ({f3(DB["part_s_c"]["fineweb-edu"][3])} / {f3(DB["part_b_c"]["fineweb-edu"][3])}) and race ({f3(DB["part_s_c"]["race"][3])} / {f3(DB["part_b_c"]["race"][3])}) stay 10/10 with the 8B partner, which has 2.4&times; more active parameters than the MoE model.', 'since v8 &middot; restated v20'),
 li('B', '<b>HC is distinct from router confidence, but in most positive cells its scale signal is shared with the hidden-state norm.</b>',
    f'Three router statistics (entropy, top-expert probability, logit gap) leave &ge;50% of both partials in {len(bl_pass)} of {len(bl_tested)} positive last-layer cells. Adding the hidden-state norm entering the layer, measured for every cell in v21, leaves &ge;50% of both in only {len(bl4_pass)}: {", ".join(lab(x) for x in bl4_pass)}. In Qwen the norm takes most of what the router statistics leave: fineweb keeps {pct(B["Qwen_L47_fineweb-edu"]["R4s"])} / {pct(B["Qwen_L47_fineweb-edu"]["R4b"])}, cnn {pct(B["Qwen_L47_cnn_dailymail"]["R4s"])} / {pct(B["Qwen_L47_cnn_dailymail"]["R4b"])}, race {pct(B["Qwen_L47_race"]["R4s"])} / {pct(B["Qwen_L47_race"]["R4b"])}, sci {pct(B["Qwen_L47_scientific_papers"]["R4s"])} / {pct(B["Qwen_L47_scientific_papers"]["R4b"])}, lambada {pct(B["Qwen_L47_lambada"]["R4s"])} / {pct(B["Qwen_L47_lambada"]["R4b"])}. Gemma&rsquo;s inversion is untouched (78&ndash;101% in all three cells). The pre-registered four-statistic rule is not met; the norm is the main shared component.', 'since v15 &middot; restated v21'),
 li('B', '<b>In Qwen, the scale-specific signal builds up gradually through the last third of the network &mdash; except on lambada.</b>',
    f'With every layer from 35 to 42 now measured, r | CE<sub>big</sub> crosses zero gradually: for race at L37 ({f3(QT[("race",37)]["mean_part_b_c"])}), passing from L38 ({QT[("race",38)]["pos_configs_both"]}/10) and 10/10 at L40&ndash;41; for fineweb only at L40&ndash;41 (10/10 at L41). Under the pre-registered rule this is a depth trend, not an abrupt late regime. At layers 5&ndash;29, r | CE<sub>small</sub> is positive ({rng(race_mid_s)} for race) but r | CE<sub>big</sub> is negative ({rng(race_mid_b)}), shrinking toward zero at L34&ndash;36, so the earlier layers track the small model&rsquo;s difficulty, not the gap. lambada passes 9&ndash;10/10 at every layer from 5 to 40 (and at 42, 44, 47), and not because of the final word: context tokens alone pass 10/10 at L11&ndash;35. Pooling the last 2&ndash;5 layers does not beat layer 47 alone.', 'new in v20'),
 li('B', '<b>In ERNIE, the association is stronger in the middle layers, but there it is no longer distinct from simpler router statistics.</b>',
    f'Layers 14&ndash;22 pass both partials 10/10 on race, fineweb, pg19 and tinystories (comp_math, lambada and struct-wiki at L20&ndash;22). The layer chosen on half A of the passages (L18) beats the last layer on half B (6 robust cells vs 3; two-sided score {E18["half_b_two_sided_score"]:.3f} vs {E27["half_b_two_sided_score"]:.3f}). The gain is in r | CE<sub>small</sub> (race {f3(ed["race"]["18"]["s"])} vs {f3(ed["race"]["27"]["s"])} at L27); r | CE<sub>big</sub> is about the same ({f3(ed["race"]["18"]["b"])} vs {f3(ed["race"]["27"]["b"])}). At L18&ndash;20 the three router statistics remove r | CE<sub>big</sub> entirely (retention {pct(min(B[k]["R3b"] for k in B if k.startswith("ernie_L1") or k.startswith("ernie_L20")))} to {pct(max(B[k]["R3b"] for k in B if k.startswith("ernie_L1") or k.startswith("ernie_L20")))}), while at L27 they keep 62&ndash;89%.', 'new in v20'),
 li('A', '<b>Gemma 4 is never positive at its last layer, and robustly inverted in three cells.</b>',
    f'Both partials negative in humaneval {cnt_html(cnt("gemma","humaneval"))}, race {cnt_html(cnt("gemma","race"))} and cnn {cnt_html(cnt("gemma","cnn_dailymail"))}; fineweb, mbpp, mmlu and struct-wiki 7&ndash;8/10 negative; no cell positive. race and humaneval replicate in both halves ({H["gemma|race"]["bothn"]}/10, {H["gemma|humaneval"]["bothn"]}/10), survive the robustness battery, and keep 94% of the inversion after all four router statistics.', 'since v5 &middot; restated v20'),
 li('B', '<b>Gemma&rsquo;s shared experts do not explain its inversion.</b>',
    f'Scaling every layer&rsquo;s shared expert with HC recomputed from the ablated model: at &alpha; = 0.75 (CE +{gab("race",0.75,"ce_big_rise_pct"):.1f}% on race, +{gab("fineweb-edu",0.75,"ce_big_rise_pct"):.1f}% fineweb, +{gab("competition_math",0.75,"ce_big_rise_pct"):.1f}% comp_math) the race inversion shrinks {shr("race","inversion_shrink_pct_ces"):.0f}% / {shr("race","inversion_shrink_pct_ceb"):.0f}% (r | CE<sub>small</sub> / r | CE<sub>big</sub>), fineweb&rsquo;s r | CE<sub>big</sub> {shr("fineweb-edu","inversion_shrink_pct_ceb"):.0f}%, and competition_math&rsquo;s grows. The pre-registered rule (&ge;50% within +10% CE) is not met under either standardization mode. The inversion disappears only at &alpha; = 0.5, where loss rises 30&ndash;82%.', 'new in v20.5'),
 li('B', '<b>Gemma&rsquo;s inversion builds up with depth after a brief positive layer.</b>',
    f'Positive under both partials only at L7 (race and fineweb 10/10) and fineweb at L8&ndash;9. Negative from mid depth: race &minus;10/10 at L18, &minus;5/10 at L22&ndash;26; all three benchmarks &minus;10/10 at L27; nothing at L28; race &minus;9/10 at L29. The pattern does not follow attention type (L23 and L29 are full-attention, L27 sliding-window). v21: the L27 peak is spread across experts &mdash; dropping the 10 experts that most often set the minimum p-value, or the 10 most negatively correlated, leaves it &minus;10/10 on all three benchmarks; an empirical-rank null agrees; 126&ndash;128 of 128 experts&rsquo; standardized scores are negatively related to IG at L27; the standardization statistics are unremarkable.', 'new in v20 &middot; L27 audited v21'),
 li('B', '<b>In Qwen and Gemma, code is where HC&rsquo;s scale signal breaks down; ERNIE&rsquo;s positive code cell is a newline-token effect.</b>',
    f'At matched mean loss of the two models, code sits below prose in {cpq_s["below"]}/10 deciles on r | CE<sub>small</sub> (mean gap {f3(cpq_s["gap"])}) and {cpq_b["below"]}/10 on r | CE<sub>big</sub> ({f3(cpq_b["gap"])}) in Qwen; {cpg_s["below"]}/10 and {cpg_b["below"]}/10 in Gemma. On r | CE<sub>big</sub> the gap is largest at the hardest decile ({f3(cpq_b["top"])}); on r | CE<sub>small</sub> it is present at every level of difficulty. Qwen&rsquo;s code cells fail the two-sided test: r | CE<sub>small</sub> is negative (humaneval {f3(c[("Qwen","humaneval")]["c_s"])}, mbpp {f3(c[("Qwen","mbpp")]["c_s"])}) while r | CE<sub>big</sub> is positive. ERNIE&rsquo;s positive mbpp cell is not code-as-such: its {CD[("ernie","mbpp","CLASS_NEWLINE")]["share_pct"]:.0f}% newline tokens carry {f3(CD[("ernie","mbpp","CLASS_NEWLINE")]["mean_r_part_ces"])} / {f3(CD[("ernie","mbpp","CLASS_NEWLINE")]["mean_r_part_ceb"])}, and without them mbpp drops to {f3(CD[("ernie","mbpp","LOCO_NO_NEWLINE")]["mean_r_part_ces"])} / {f3(CD[("ernie","mbpp","LOCO_NO_NEWLINE")]["mean_r_part_ceb"])} (0/10). Neither ERNIE ({cpe_s["below"]}/10) nor LFM2 ({cpl_s["below"]}/10) shows a consistent code-below-prose gap. In Gemma the code inversion is strongest in identifiers (humaneval {f3(CD[("gemma","humaneval","CLASS_IDENTIFIER")]["mean_r_part_ces"])} / {f3(CD[("gemma","humaneval","CLASS_IDENTIFIER")]["mean_r_part_ceb"])}, &minus;10/10) but survives their removal.', 'since v16 &middot; restated v21'),
 li('B', '<b>HC tracks token difficulty in every family; the family-specific sign holds within token strings, not in the raw correlation.</b>',
    f'Under C, {DS["c_raw_b"]["beyond"]} and {DS["c_raw_s"]["beyond"]} of 36 studied cells are beyond the floor against CE<sub>big</sub> and CE<sub>small</sub> (the old scale reproduces v14&rsquo;s 31 and 28 exactly), but only {DS["c_raw_b"]["fam"]} and {DS["c_raw_s"]["fam"]} carry the v14 family sign: Qwen&rsquo;s fineweb and pg19 and ERNIE&rsquo;s fineweb and race are robustly negative. The pre-registered rule (&ge;25 of 36 with the family sign) is not met, so the finding stays at B. Within token strings the pattern is clean: r(HC, CE<sub>small</sub>) is beyond the floor with the family sign in {DS["c_within_s"]["fam"]} of 36 cells, and all {DS["c_within_s"]["robfam"]} cells that pass 9/10 configurations have the family sign (none the opposite): positive in Qwen, ERNIE and LFM2, negative in Gemma.', 'since v3 &middot; restated v21'),
]
def chk(name, status, what, note):
    cls = {'done': 'status-done', 'partial': 'status-partial', 'not run': 'status-notrun'}[status]
    return [f'{name}', f'<span class="status {cls}">{status}</span>', what, note]
checks = [
 chk('Standardization audit', 'done', 'Which reference each cell was standardized against, and with what scale', 'Original pipeline: own-benchmark median/MAD but MAD scaled twice; silent wikitext fallback at 219 of 234 intermediate-layer cells. Fallback now raises an error.'),
 chk('Arm A reproduces the original', 'done', 'Same code, old scale flag', '72 of 72 overlapping last-layer configurations match the original HC within 5&times;10<sup>&minus;6</sup>.'),
 chk('Depth run = last-layer run', 'done', 'One evaluation function for every layer', 'Identical at the exit layer in Qwen, Gemma and ERNIE.'),
 chk('Two floor methods', 'done', 'ESS from CE<sub>small</sub> (lag 20) vs from IG (v15 ACF)', f'Conservative maximum used. Verdict class changes in {N["floor_sens_c"]} of 60 cells under C ({", ".join(lab(x) for x in fs_cells)}), {N["floor_sens_a"]} under A.'),
 chk('Held-out halves, last layer', 'done', 'Both partials in each half separately', '60 cells. Every robust positive cell replicates in both halves (LFM2 race 8/10); LFM2 comp_math&rsquo;s inversion does not (3/10).'),
 chk('Partner sweep under C', 'done', '4 partners &times; 9 benchmarks &times; 10 configurations', 'fineweb on the 207k-token aligned set for all partners (0.6B also on 477,782 tokens: identical verdict).'),
 chk('Router baselines under C', 'done', 'Entropy, top probability, logit gap; + hidden-state norm', 'Norm now measured for every cell (v21), Qwen fineweb on all 477,782 tokens. 4 of 13 positive cells keep &ge;50% of both partials.'),
 chk('Depth profiles, 10 configurations', 'done', 'Qwen 23 layers, Gemma 18, ERNIE 15, LFM2 8', 'Every layer under the same standardization and the two-sided rule.'),
 chk('ERNIE layer selection out of sample', 'done', 'Choose on half A, test on half B', 'L18 chosen; beats L27 on half B.'),
 chk('Code vs prose under C', 'done', 'Matched deciles, both partials, all four families', 'Qwen and Gemma: code below prose on both partials. ERNIE and LFM2: no consistent gap.'),
 chk('lambada position control', 'done', 'Partial on position; context tokens only; final word only', 'Context tokens carry the effect.'),
 chk('Pooled multi-layer HC, held out', 'done', 'Qwen last 2/3/5 layers; ERNIE L16&ndash;20', 'No gain for Qwen; ERNIE gains on r | CE<sub>small</sub> only.'),
 chk('Robustness battery under C', 'done', 'Winsorised, ranks, p<sub>min</sub>, drop strongest expert', '16 cells; every robust cell passes.'),
 chk('Gemma shared-expert ablation', 'done', 'All-layer &alpha; = 1, 0.75, 0.5; HC from the ablated forward pass', 'Logits and HC differ by &alpha; (audit). Inversion shrinks &le;27% within the CE budget: rule not met.'),
 chk('ERNIE and LFM2 on unseen benchmarks', 'done', 'Six unseen benchmarks at the last layer', 'ERNIE cnn and mbpp pass (sci 8/10); LFM2 none. Pre-registered rule met through ERNIE cnn.'),
 chk('Difficulty correlations under C', 'done', 'r(HC, CE<sub>big</sub>), r(HC, CE<sub>small</sub>), raw and within strings, 60 cells', f'Raw: {DS["c_raw_b"]["beyond"]}/{DS["c_raw_s"]["beyond"]} of 36 beyond floor, {DS["c_raw_b"]["fam"]}/{DS["c_raw_s"]["fam"]} with family sign (rule &ge;25 not met). Within strings vs CE<sub>small</sub>: {DS["c_within_s"]["fam"]}/36 with family sign. Arm A reproduces v14 (31, 28).'),
 chk('Qwen layers 35&ndash;42, every layer', 'done', 'Onset of the late signal', 'Gradual: race from L37&ndash;38, fineweb at L40&ndash;41.'),
 chk('Code dissection, four families', 'done', 'Token classes, leave-one-class-out, first occurrences', 'ERNIE mbpp is carried by newline tokens; Gemma&rsquo;s code inversion is strongest in identifiers but survives without them.'),
 chk('Gemma L26&ndash;29 expert audit', 'done', 'Leave-k-experts-out, empirical-rank null, per-expert standardization', 'L27 inversion distributed across experts; not one or two.'),
]
chk_rows = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in checks)

# ------------------------------------------------------------------ what changed table
changes = [
 ['Standardization', 'Own-benchmark median/MAD, MAD scaled twice (&ldquo;Convention A&rdquo;)', 'Same reference, scale fixed (&ldquo;Convention C&rdquo;); A kept as a sensitivity arm'],
 ['Qwen tier-A cells', 'comp_math, lambada, fineweb, race', f'lambada, fineweb, race (comp_math {cnt_html(cnt("Qwen","competition_math"))} under C, 10/10 under A)'],
 ['Unseen benchmarks (Qwen)', '4 of 6 pass (cnn, sci, ultrachat, dialogsum)', f'2 of 6 (cnn, sci); dialogsum {cnt_html(cnt("Qwen","dialogsum"))}, ultrachat {cnt_html(cnt("Qwen","ultrachat"))} under either floor method; under A they pass only with the v15 floor'],
 ['ERNIE and LFM2', 'No consistent family-level association (raw sign)', 'Two-sided positive cells in both (race in each); LFM2 comp_math a robust inversion'],
 ['Dose-response', 'r | CE<sub>small</sub> +0.044 &rarr; +0.006', f'{f3(F["part_s_c"][0])} &rarr; {f3(F["part_s_c"][3])}; race and fineweb persist at 8B'],
 ['Router baselines', 'Four baselines keep 77% / 49% / 18% (comp_math / lambada / race)', f'Three baselines keep &ge;50% in 10 of 13 cells; with the norm: lambada {pct(B["Qwen_L47_lambada"]["R4s"])}, race {pct(B["Qwen_L47_race"]["R4s"])}'],
 ['Layers', 'Last MoE layer only', 'Depth profiles in all four families, one standardization'],
 ['Gemma shared expert', '41% of race inversion removed at &alpha; = 0.75 (v18)', f'Retired (wikitext-standardized HC). Rerun with recomputed HC (v20.5): {shr("race","inversion_shrink_pct_ces"):.0f}% / {shr("race","inversion_shrink_pct_ceb"):.0f}% on race at &alpha; = 0.75; rule not met'],
 ['Out of domain, ERNIE and LFM2', 'Not run', f'ERNIE cnn {cnt("ernie","cnn_dailymail")}/10 and mbpp {cnt("ernie","mbpp")}/10 pass; LFM2 0 of 6 (v20.5)'],
 ['Code vs prose', '10/10 deciles below prose, r | CE<sub>small</sub> (Qwen)', f'Qwen {cpq_s["below"]}/10 and {cpq_b["below"]}/10 on the two partials at matched mean loss; Gemma 10/10; ERNIE and LFM2 no consistent gap, and ERNIE mbpp is positive'],
]

# ------------------------------------------------------------------ last-layer table
def cell_row(m, b):
    x = c[(m, b)]
    return [FAMN[m], SHORT[b], 'unseen' if b in UNSEEN else 'studied', f'{x["fl"]:.4f}', f3(x['c_s']), f3(x['c_b']), cnt_html(x['c_cnt']), cnt_html(x['a_cnt'])]
ll_rows = [cell_row(m, b) for m in ['Qwen', 'ernie', 'liquid', 'gemma'] for b in MAIN + UNSEEN if (m, b) in c]

# ------------------------------------------------------------------ baseline table
def brow(k, lab):
    x = B[k]
    return [lab, f3(x['ref_s']), f3(x['ref_b']), pct(x['R3s']), pct(x['R3b']), pct(x['R4s']) if x['R4s'] is not None else '&ndash;', pct(x['R4b']) if x['R4b'] is not None else '&ndash;']
bkeys = [('Qwen_L47_fineweb-edu', 'Qwen fineweb'), ('Qwen_L47_fineweb-edu_max512_hnorm', 'Qwen fineweb (512-token subset)'), ('Qwen_L47_lambada', 'Qwen lambada'), ('Qwen_L47_race', 'Qwen race'),
         ('Qwen_L47_cnn_dailymail', 'Qwen cnn'), ('Qwen_L47_scientific_papers', 'Qwen sci'), ('Qwen_L47_competition_math', 'Qwen comp_math'), ('ernie_L27_race', 'ERNIE race (L27)'),
         ('ernie_L27_fineweb-edu', 'ERNIE fineweb (L27)'), ('ernie_L27_tinystories', 'ERNIE tinystories (L27)'), ('ernie_L18_race', 'ERNIE race (L18)'), ('ernie_L19_race', 'ERNIE race (L19)'),
         ('ernie_L20_race', 'ERNIE race (L20)'), ('ernie_L18_fineweb-edu', 'ERNIE fineweb (L18)'), ('ernie_L19_fineweb-edu', 'ERNIE fineweb (L19)'), ('ernie_L20_fineweb-edu', 'ERNIE fineweb (L20)'),
         ('ernie_L27_cnn_dailymail', 'ERNIE cnn (L27)'), ('ernie_L27_scientific_papers', 'ERNIE sci (L27)'), ('ernie_L27_mbpp', 'ERNIE mbpp (L27)'), ('liquid_L23_race', 'LFM2 race'), ('liquid_L23_pg19', 'LFM2 pg19'), ('liquid_L23_lambada', 'LFM2 lambada'), ('gemma_L29_race', 'Gemma race'), ('gemma_L29_cnn_dailymail', 'Gemma cnn'), ('gemma_L29_humaneval', 'Gemma humaneval')]

# ------------------------------------------------------------------ robustness table
rob_rows = []
for (m, l, d), r in ROB.items():
    rob_rows.append([f'{FAMN[m]} L{l}', SHORT[d], f3(r['base_s']), f3(r['base_b']), f'{r["winsor_pass"]}/10', f'{r["rank_pass"]}/10', f'{r["no_top1_pass"]}/10', f3(r['ctrl_min_p_s']), f3(r['ctrl_min_p_b'])])

# ------------------------------------------------------------------ depth tables (counts)
def dtable(fam, bms):
    L = D[fam]['layers']
    head_ = ['benchmark'] + [f'L{l}' for l in L]
    rows = []
    for b in bms:
        row = [SHORT[b]]
        for l in L:
            v = dcnt(fam, b, l)
            row.append('' if v is None else (f'<b style="color:#86efac">{v}</b>' if v >= 9 else (f'<b style="color:#fca5a5">{v}</b>' if v <= -9 else str(v))))
        rows.append(row)
    return tbl(head_, rows, num_from=1)

# ------------------------------------------------------------------ pooled + lambada tables
pool_rows = []
for k, r in POOL.items():
    pool_rows.append([FAMN[r['model']], r['block_name'].replace('_', ' '), SHORT[r['dataset']], f3(r['best_single_s_split_b']), f3(r['best_single_b_split_b']),
                      f3(r['mean_pool_s_split_b']), f3(r['mean_pool_b_split_b']), f'{r["mean_pool_pass_split_b_configs"]}/10'])
lam_rows = []
for (m, l), r in LPOS.items():
    lam_rows.append([f'{FAMN[m]} L{l}', f3(r['mean_part_s_base']), f3(r['mean_part_b_base']), f'{r["ret_s_pos_mean_pct"]:.0f}% / {r["ret_b_pos_mean_pct"]:.0f}%',
                     f'{r["pos_ctrl_pass_configs"]}/10', f'{r["context_only_pass_configs"]}/10'])
esel_rows = [[f'L{l}', f'{r["half_a_two_sided_score"]:.3f}', r['half_a_rob_pos_cells'], f'{r["half_b_two_sided_score"]:.3f}', r['half_b_rob_pos_cells'], r['full_rob_pos_cells']] for l, r in sorted(ESEL.items())]
dose_rows = [[SHORT[b]] + [cnt_html(N['dose_cnt'][b][p]) + f' <span style="color:#93a3b8">({f3(DB["part_s_c"][b][i])} / {f3(DB["part_b_c"][b][i])})</span>' for i, p in enumerate(P)] for b in MAIN]

def exp(title, pr, cost, q, design, stat, rule):
    return (f'<div class="exp"><h4>{title}</h4><div class="meta">{pr} &middot; {cost}</div><dl><dt>Question</dt><dd>{q}</dd><dt>Design</dt><dd>{design}</dd>'
            f'<dt>Statistic</dt><dd>{stat}</dd><dt>Decision rule</dt><dd>{rule}</dd></dl></div>')

abstract = (f'<b>v21 answers the five follow-ups from v20.5.</b> (1) Under the corrected scale HC still tracks difficulty in every family, and the family-specific sign holds within token strings but not in the raw correlation. (2) With the hidden-state norm measured everywhere, it absorbs most of the positive scale signal in Qwen and ERNIE: only Qwen and LFM2 lambada and ERNIE tinystories and mbpp keep half of both partials, while Gemma&rsquo;s inversion is untouched. (3) In Qwen the signal builds up gradually from layer 37 to 41. (4) ERNIE&rsquo;s positive code result is a newline-token effect. (5) Gemma&rsquo;s layer-27 inversion is spread across experts. <b>From v20.5:</b> ERNIE&rsquo;s positive result extends out of domain (news and code pass 10/10; the code cell has since been traced to newline tokens), LFM2&rsquo;s does not, and the Gemma shared-expert ablation &mdash; now with HC recomputed from the ablated model &mdash; does not explain Gemma&rsquo;s inversion within the loss budget. <b>The v20 base:</b> The original standardization scaled the MAD twice and silently fell back to wikitext statistics at intermediate layers; '
 f'v20 fixes the scale (&ldquo;Convention C&rdquo;), keeps the old one as a sensitivity arm that reproduces every original number, and regrades all last-layer cells (60 in v20.5) with a two-sided rule (both partials beyond the floor, same sign, &ge;9/10 configurations). '
 f'<b>Qwen3&rsquo;s scale-specific association survives on fineweb, lambada, race, news and scientific text</b> (10/10 everywhere, both held-out halves, every robustness check); competition_math needed the old scale, and dialogue and chat fail under the corrected one. '
 f'<b>The same association appears in ERNIE and LFM2</b> on race and a few other benchmarks, so it is no longer a one-family result; Gemma is never positive and robustly inverted on humaneval, race and news. '
 f'The dose-response holds (family r | CE<sub>small</sub> {f3(F["part_s_c"][0])} &rarr; {f3(F["part_s_c"][3])}), but race and fineweb persist with the 8B partner. '
 f'The first consistent depth profiles show that Qwen&rsquo;s effect builds up through the last third of the network (lambada excepted), ERNIE&rsquo;s is strongest at mid depth but there is absorbed by simple router statistics, and Gemma inverts from mid depth after one positive layer.')

nav = '<nav><span class="brand">HC-MoE v21</span><a href="#goals">Status</a><a href="#changed">What changed</a><a href="#setup">Setup</a><a href="#difficulty">Difficulty</a><a href="#last">Last layer</a><a href="#dose">Dose-response</a><a href="#base">Baselines</a><a href="#depth">Depth</a><a href="#families">Families</a><a href="#gemma-abl">Ablation</a><a href="#code">Code</a><a href="#health">Health</a><a href="#assess">Assessment</a><a href="#next">Next</a><a href="#tables">Tables</a></nav>'

body = f'''<body>
{nav}
<main>
<header class="report-header"><h1>HC-MoE v21: What Carries the Signal</h1>
<div class="subtitle">Higher Criticism on MoE routing p-values versus token difficulty and Intelligence Gain under the corrected standardization &mdash; difficulty re-established, the hidden-state norm measured everywhere, the Qwen onset resolved, ERNIE&rsquo;s code cell and Gemma&rsquo;s layer 27 dissected</div>
<div class="date">November 3, 2026 &middot; v21 &middot; drops reports/v21_experiments/v21 + reports/26_10_29/v20</div>
<div class="abstract">{abstract}</div></header>

<section id="goals"><h2>1. Research goals &amp; where the study stands</h2>
<p>Two questions: does HC over the large model&rsquo;s routing p-values track <b>token difficulty</b> (CE<sub>big</sub>, CE<sub>small</sub>), and does it track <b>Intelligence Gain</b> (IG = CE<sub>small</sub> &minus; CE<sub>big</sub>) beyond difficulty &mdash; the &ldquo;scale-specific&rdquo; test, r(HC, IG) with either loss held fixed, within the same token string. A claim needs both partials beyond the floor with the same sign in at least 9 of 10 configurations.</p>
<h3>Main findings so far</h3>
<div class="tier-legend"><span><span class="tier tier-A">A</span> holds under every check run</span><span><span class="tier tier-B">B</span> holds, with a named sensitivity or a limited sample</span></div>
<ul class="so-far">{''.join(findings)}</ul>
<h3>Checks done</h3>
<div class="table-wrap"><table class="checks"><thead><tr><th>Check</th><th>Status</th><th>What it tests</th><th>Result</th></tr></thead><tbody>{chk_rows}</tbody></table></div>
</section>

<section id="changed"><h2>2. What changed in v20</h2>
<p>Revalidating Qwen&rsquo;s layer 47 exposed two problems in the original HC pipeline. <b>(1) Double MAD scaling.</b> The per-expert MAD was stored already multiplied by 1.4826 and divided by 1.4826 again at standardization, so test z-scores were 1.48&times; narrower than the scale the null estimators were fitted on, and the p-values were not calibrated. <b>(2) Silent fallback.</b> Each benchmark was standardized against its own per-expert median and MAD only when those statistics existed; otherwise the code fell back to wikitext. Every last-layer cell had its own statistics, so the main grid was internally consistent, but 219 of 234 intermediate-layer cells had fallen back, which is why the first layer sweep disagreed with the last layer.</p>
<div class="grid-2">
<div class="card"><h3 style="margin-top:0">Three conventions</h3><ul class="plain">
<li><b>A</b> &mdash; own-benchmark median/MAD, MAD scaled twice. Every report v1&ndash;v19. Kept as a sensitivity arm.</li>
<li><b>B</b> &mdash; wikitext median/MAD. HC saturates on shifted domains (mean HC 12&ndash;17 on competition_math) and measures vocabulary shift, not token anomaly. Every B-based result is retired.</li>
<li><b>C</b> &mdash; own-benchmark median/MAD, scale fixed. Primary from v20. This is a per-benchmark recentring of the null, the idea behind Efron&rsquo;s empirical null.</li></ul></div>
<div class="card"><h3 style="margin-top:0">Acceptance tests (passed)</h3><ul class="plain">
<li>One function computes HC and both partials for every arm and layer, with one flag for the scale.</li>
<li>With the flag on A it reproduces the original last-layer numbers in all 72 overlapping configurations (|&Delta;| &le; 5&times;10<sup>&minus;6</sup>), and the 10-configuration means of v15&ndash;v19.</li>
<li>The depth runs equal the last-layer grid at the exit layer, cell for cell.</li>
<li>Every verdict recomputes from the records with the two-sided rule.</li></ul></div></div>
<div class="alert"><strong>v21 adds:</strong> r(HC, CE) for all 60 cells under both scales; the hidden-state norm for every router-baseline cell; every Qwen layer from 35 to 42; a token-class dissection of all eight code cells; and an expert-level audit of Gemma layers 26&ndash;29. <strong>v20.5 added:</strong> ERNIE and LFM2 on the six unseen benchmarks (60 last-layer cells instead of 48; the 48 earlier cells reproduce exactly), router baselines for ERNIE&rsquo;s unseen cells, code vs prose for all four families, and the Gemma all-layer shared-expert ablation with HC recomputed from the ablated forward pass.</div>
<h3>What v20 and v20.5 revise</h3>
{tbl(['Claim', 'v19 (Convention A)', 'v20 / v20.5 (Convention C)'], changes)}
</section>

<section id="setup"><h2>3. Setup</h2>
{tbl(['Family', 'Large (MoE)', 'Small (dense)', 'Routing in the last MoE layer', 'Active ratio'], [['Qwen', 'Qwen3-30B-A3B (48 MoE layers)', 'Qwen3-0.6B', '128 experts, top-8, no shared', '5.5&times;'], ['ERNIE', 'ERNIE-4.5-21B-A3B (layer 27 last)', 'ERNIE-4.5-0.3B', '64 experts, top-6, 2 shared', '10&times;'], ['Gemma', 'gemma-4-26B-A4B (30 layers)', 'gemma-4-E2B', '128 experts, top-8, 1 shared', '1.7&times;'], ['Liquid', 'LFM2-8B-A1B (layer 23 last)', 'LFM2-700M', '32 experts, top-4', '2.0&times;']], num_from=4)}
<ul class="plain">
<li><b>Grid.</b> 9 studied benchmarks and 6 unseen benchmarks (cnn_dailymail, scientific_papers, ultrachat, dialogsum, humaneval, mbpp) for every family: 60 last-layer cells. Five null estimators (GMM, JohnsonSU, KDE, StudentT, StudentTSkewed) &times; two HC variants = 10 configurations per cell.</li>
<li><b>Scale-specific test.</b> Within token strings (tokens with &ge;5 occurrences grouped by string, rarer ones pooled), the partial correlations r(HC, IG | CE<sub>small</sub>) and r(HC, IG | CE<sub>big</sub>). Holding CE<sub>small</sub> fixed, IG varies only through the big model&rsquo;s loss; holding CE<sub>big</sub> fixed, only through the small model&rsquo;s. A token-level signal that is positive on both is about the gap, not about difficulty.</li>
<li><b>Two-sided strength.</b> For charts, each configuration contributes the weaker of the two partials when they agree in sign and zero when they disagree; this is averaged over configurations.</li>
<li><b>Floors.</b> 1.96/&radic;ESS, with ESS estimated two ways (from the CE<sub>small</sub> series at lag 20, and from IG with the v15 ACF method). v20 uses the larger floor per cell; {N["floor_sens_c"]} of 60 cells ({", ".join(lab(x) for x in fs_cells)}) change verdict class between the methods under C.</li>
<li><b>Inference.</b> Passage-level block bootstrap (1,000 resamples), held-out passage halves, decision rules written before each experiment.</li></ul>
</section>

<section id="difficulty"><h2>4. Question 1 under Convention C: does HC track difficulty?</h2>
<p>Recomputed for all 60 last-layer cells, raw and within token strings, under both scales. With the old scale the v14 counts reproduce exactly (31 and 28 of 36 studied cells beyond the floor), so the change below is the scale fix, not the code. Pre-registered rule: restore the finding to A if &ge;25 of 36 studied cells pass with the v14 family signs (positive for Qwen, ERNIE and LFM2, negative for Gemma).</p>
{card(1, 'Within token strings, HC tracks the small model&rsquo;s loss with a family-specific sign', 'f_diff', 'Within-string r(HC, CE<sub>small</sub>), mean of ten configurations, last layer. Gemma (red) is negative almost everywhere, the other three mostly positive. Dashed: typical floor.', h=360)}
{tbl(['Statistic', 'Scale', 'beyond floor', 'with family sign', 'robust (&ge;9/10)', 'robust, family sign', 'robust, opposite sign'], [[st, sc, DS[k]['beyond'], DS[k]['fam'], DS[k]['rob'], DS[k]['robfam'], DS[k]['robopp']] for st, sc, k in [('raw r(HC, CE<sub>big</sub>)', 'C', 'c_raw_b'), ('raw r(HC, CE<sub>small</sub>)', 'C', 'c_raw_s'), ('within-string r(HC, CE<sub>big</sub>)', 'C', 'c_within_b'), ('within-string r(HC, CE<sub>small</sub>)', 'C', 'c_within_s'), ('raw r(HC, CE<sub>big</sub>)', 'A', 'a_raw_b'), ('raw r(HC, CE<sub>small</sub>)', 'A', 'a_raw_s')]], num_from=2)}
<p><b>Reading.</b> HC still tracks difficulty in every family: {DS["c_raw_b"]["beyond"]} and {DS["c_raw_s"]["beyond"]} of 36 studied cells are beyond the floor under C. But the raw sign is no longer family-wide: Qwen fineweb ({f3(DIF[("Qwen","fineweb-edu")]["c_raw_b"])}) and pg19 ({f3(DIF[("Qwen","pg19")]["c_raw_b"])}) and ERNIE fineweb ({f3(DIF[("ernie","fineweb-edu")]["c_raw_b"])}) and race ({f3(DIF[("ernie","race")]["c_raw_b"])}) are robustly negative against CE<sub>big</sub>. The rule is therefore not met ({DS["c_raw_b"]["fam"]} and {DS["c_raw_s"]["fam"]} of 36), and the finding stays at B. Within token strings, where token identity no longer drives the comparison, the family pattern is clean: against CE<sub>small</sub> {DS["c_within_s"]["fam"]} of 36 cells have the family sign and every one of the {DS["c_within_s"]["rob"]} robust cells agrees with it. Family means of within-string r(HC, CE<sub>small</sub>): Qwen {f3(N["diff_fam"]["Qwen"]["c_within_s"])}, ERNIE {f3(N["diff_fam"]["ernie"]["c_within_s"])}, LFM2 {f3(N["diff_fam"]["liquid"]["c_within_s"])}, Gemma {f3(N["diff_fam"]["gemma"]["c_within_s"])}.</p>
</section>

<section id="last"><h2>5. The last-layer grid under Convention C</h2>
<div class="grid-4">
<div class="stat-box"><div class="val">{len(robust("Qwen"))}</div><div class="lbl">Qwen cells positive on both partials</div></div>
<div class="stat-box"><div class="val">{len(robust("ernie")) + len(robust("liquid"))}</div><div class="lbl">ERNIE + LFM2 cells positive</div></div>
<div class="stat-box"><div class="val">{len(robust("gemma", sign=-1)) + len(robust("liquid", sign=-1))}</div><div class="lbl">robust inversions (Gemma {len(robust("gemma", sign=-1))}, LFM2 {len(robust("liquid", sign=-1))})</div></div>
<div class="stat-box"><div class="val">{cross["pos"]["pos"]}/{sum(cross["pos"].values())}</div><div class="lbl">A-positive cells still positive under C</div></div></div>
{card(2, 'Two-sided strength in every last-layer cell', 'f_cells', 'Mean over ten configurations of the weaker partial when both agree in sign (zero otherwise). Dashed: the typical floor (&plusmn;0.0128; each cell&rsquo;s own floor is in the table in section 15). Qwen is positive across prose; ERNIE on a subset that includes news and mbpp; LFM2 on three studied benchmarks only; Gemma negative throughout.', h=380)}
<p><b>Convention A vs C.</b> Of the {len(a_pos)} cells positive under A, {cross["pos"]["pos"]} stay positive under C, and both A inversions stay inverted. Needed the old scale: {", ".join(f"{lab(x)} (A {CELL[x]['a_cnt']}, C {CELL[x]['c_cnt']})" for x in a_only)}. Pass only under C: {", ".join(f"{lab(x)} (C {CELL[x]['c_cnt']}, A {CELL[x]['a_cnt']})" for x in c_only)}, plus the Gemma inversions on race and cnn (&minus;9 vs &minus;8). Differences of one configuration are better read as insensitive to the scale. Qwen dialogsum and ultrachat fail under C with either floor method (5/10 and 3&ndash;5/10); under A they pass only with the v15 floor.</p>
<div class="finding scale"><h4>Qwen: five cells hold under every check</h4><p>{"; ".join(f"<b>{SHORT[b]}</b> {f3(c[('Qwen', b)]['c_s'])} / {f3(c[('Qwen', b)]['c_b'])}, halves {H[f'Qwen|{b}']['both']}/10" for b in q5)}. Effects are r = 0.04&ndash;0.14 &mdash; small, as before &mdash; but every one of these cells is 10/10 on both partials, in both halves, under both scales and both floor methods.</p></div>
<div class="finding null"><h4>What dropped out</h4><p>competition_math (r | CE<sub>small</sub> {f3(c[("Qwen","competition_math")]["c_s"])}, just above its 0.0128 floor in half the configurations) and the two conversational unseen benchmarks: dialogsum {f3(c[("Qwen","dialogsum")]["c_s"])} / {f3(c[("Qwen","dialogsum")]["c_b"])} and ultrachat {f3(c[("Qwen","ultrachat")]["c_s"])} / {f3(c[("Qwen","ultrachat")]["c_b"])}. In each, r | CE<sub>big</sub> is healthy but r | CE<sub>small</sub> is near zero: HC tracks where the small model struggles, not specifically where the big one succeeds. The v15 claim &ldquo;4 of 6 unseen benchmarks pass&rdquo; becomes 2 of 6.</p></div>
<h3>ERNIE and LFM2 out of domain (v20.5)</h3>
{tbl(['Family', 'Benchmark', 'Domain', 'Floor', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'both partials', 'held-out halves'], [[FAMN[m], SHORT[b], UEL[(m, b)]['domain'], f'{CELL[(m, b)]["fl"]:.4f}', f3(CELL[(m, b)]['c_s']), f3(CELL[(m, b)]['c_b']), cnt_html(CELL[(m, b)]['c_cnt']), f'{H[f"{m}|{b}"]["both"]}/10'] for m in ['ernie', 'liquid'] for b in UNSEEN], num_from=3)}
<p>The rule written before the run &mdash; the cross-family claim stands if unseen prose (cnn or sci) passes in at least one of the two families &mdash; is met: ERNIE cnn passes 10/10 in both halves ({f3(c[("ernie","cnn_dailymail")]["c_s"])} / {f3(c[("ernie","cnn_dailymail")]["c_b"])}); sci is {cnt("ernie","scientific_papers")}/10 (9/10 with the v15 floor). LFM2 passes nothing out of domain: its r | CE<sub>big</sub> is positive almost everywhere but r | CE<sub>small</sub> is near zero or negative, the pattern of a difficulty signal rather than a gap signal. The surprise is ERNIE <b>mbpp</b> ({f3(c[("ernie","mbpp")]["c_s"])} / {f3(c[("ernie","mbpp")]["c_b"])}), ERNIE&rsquo;s strongest last-layer cell; it replicates 10/10 in both halves and keeps {pct(B["ernie_L27_mbpp"]["R3s"])} / {pct(B["ernie_L27_mbpp"]["R3b"])} after the router statistics. humaneval is {cnt("ernie","humaneval")}/10. Dialogue fails in both families.</p>
</section>

<section id="dose"><h2>6. Dose-response under Convention C</h2>
<p>Qwen3-30B-A3B fixed; the dense partner grows from 0.6B to 8B (active-parameter ratio 5.5&times; &rarr; 0.41&times;). fineweb uses the 207k-token aligned set for every partner; with the full 477,782 tokens at 0.6B the verdict is the same.</p>
{card(3, 'Family means as the dense partner grows', 'f_dose', 'Means over nine studied benchmarks and ten configurations. Both partials fall by about four fifths; the family mean is inside the typical floor at 8B.')}
{card(4, 'Per benchmark: race and fineweb persist', 'f_dose_bm', 'Two-sided strength per benchmark. lambada collapses at 4B; competition_math and structured-wikipedia go to zero; fineweb and race stay 10/10 with the 8B partner. Dashed white: family mean.')}
<p>Robust cells: {", ".join(f"{p}: {', '.join(SHORT[b] for b in N['dose_robust'][p])}" for p in P)}. r | CE<sub>small</sub> falls strictly monotonically in {N["dose_mono_s"]} of 9 benchmarks, r | CE<sub>big</sub> in {N["dose_mono_b"]}. The v19 reading &mdash; &ldquo;goes to zero as the compute gap closes&rdquo; &mdash; holds for the family mean but not for race and fineweb, where HC still marks a scale benefit when the dense partner has 2.4&times; the MoE model&rsquo;s active parameters.</p>
</section>

<section id="base"><h2>7. Is HC more than router confidence?</h2>
<p>Each partial is recomputed after also partialling out three router statistics from the same layer (entropy, top-expert probability, logit gap), and then also the hidden-state norm entering the layer (measured for every cell from v21; Qwen fineweb on all 477,782 tokens). Rule written before the run: at least 50% of both partials retained in at least half the primary cells. Retention is the ratio of 10-configuration means.</p>
{card(5, 'Retention after the router statistics and the hidden-state norm', 'f_base', 'Faint bars: percentage of r | CE<sub>small</sub> (amber) and r | CE<sub>big</sub> (blue) left after entropy, top probability and logit gap. Solid bars: also after the hidden-state norm. Dashed: the 50% rule. In Qwen the norm takes most of what is left; LFM2 and Gemma are unaffected by it.', h=380)}
{tbl(['Cell', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'kept | CE<sub>small</sub>, 3 stats', 'kept | CE<sub>big</sub>, 3 stats', 'kept | CE<sub>small</sub>, + norm', 'kept | CE<sub>big</sub>, + norm'], [brow(k, l) for k, l in bkeys], num_from=1)}
<p><b>Reading.</b> With three router statistics the rule is met: {len(bl_pass)} of {len(bl_tested)} positive last-layer cells keep at least half of both partials ({", ".join(lab(x) for x in bl_fail)} do not). <b>With the hidden-state norm added, it is not:</b> only {", ".join(lab(x) for x in bl4_pass)} keep half of both ({len(bl4_pass)} of {len(bl_tested)}). The norm does the work in Qwen &mdash; it removes most of what the router statistics leave in fineweb, cnn, race and sci, and spares only lambada &mdash; and partly in ERNIE (race {pct(B["ernie_L27_race"]["R4s"])} / {pct(B["ernie_L27_race"]["R4b"])}, fineweb {pct(B["ernie_L27_fineweb-edu"]["R4s"])} / {pct(B["ernie_L27_fineweb-edu"]["R4b"])}); ERNIE cnn and LFM2&rsquo;s race and pg19 already fail on the router statistics, not the norm. Gemma&rsquo;s inversion keeps 78&ndash;101% throughout. HC&rsquo;s added within-string IG variance stays 0.01&ndash;0.4% (1&ndash;4% only for r | CE<sub>big</sub> in Gemma humaneval and Qwen lambada). The paper has to say this plainly: beyond router confidence, most of HC&rsquo;s positive scale signal is shared with the norm of the residual stream entering the last MoE layer. Whether the norm is a confound or the mechanism (tokens whose representation grows larger are routed more anomalously) is the next question (section 14).</p>
</section>

<section id="depth"><h2>8. Depth profiles</h2>
<p>All layers now use the same standardization and 10 configurations. Tables show configurations passing both partials (negative = both partials negative); charts show two-sided strength.</p>
<h3>Qwen3-30B-A3B</h3>
{card(6, 'Qwen: the scale-specific signal builds up through the last third', 'f_qdepth', 'Two-sided strength by layer, now with every layer from 35 to 42. race turns positive from layer 37&ndash;38, fineweb at 40&ndash;41; lambada is positive from layer 5; competition_math is intermittent.')}
{card(7, 'Why the early layers fail: the two partials disagree', 'f_qrace', 'Solid: r | CE<sub>small</sub>; dashed: r | CE<sub>big</sub>. Before layer 37, HC is high where the small model struggles (r | CE<sub>small</sub> &gt; 0) but also where the big model struggles (r | CE<sub>big</sub> &lt; 0): it tracks difficulty, not the gap.')}
{dtable('Qwen', ['race', 'fineweb-edu', 'lambada', 'competition_math'])}
<p><b>The onset (v21).</b> {tbl(['Layer'] + [f'L{l}' for l in range(35, 43)], [[f'{SHORT[d]} {"r | CE<sub>big</sub>" if k == "mean_part_b_c" else "both partials"}'] + [(f3(QT[(d, l)][k]) if k == 'mean_part_b_c' else f'{QT[(d, l)]["pos_configs_both"]}/10') for l in range(35, 43)] for d in ['race', 'fineweb-edu'] for k in ['mean_part_b_c', 'pos']], num_from=1)}
r | CE<sub>big</sub> rises steadily rather than jumping: race crosses zero at layer 37, passes at 38 (9/10), dips to 8/10 at 39 and passes 10/10 from 40; fineweb crosses at 40 and passes at 41. Under the rule written for this experiment, a gradual onset is described as a depth trend through the last third of the network, not as a separate late regime. lambada passes 10/10 at every layer from 35 to 39 (9/10 at 40, 8/10 at 41).</p>
<p>Pooling the last 2, 3 or 5 layers never beats layer 47 on held-out passages: on race, r | CE<sub>big</sub> falls from {f3(POOL[("Qwen","last_2_layers_46_47","race")]["best_single_b_split_b"])} (best single layer) to {f3(POOL[("Qwen","last_2_layers_46_47","race")]["mean_pool_b_split_b"])} pooled over 46&ndash;47, and layer 47 is the best single layer chosen on half A in 27&ndash;32 of 40 configurations. <b>lambada</b> is the exception: it passes from layer 5, and the effect is not the final word &mdash; partialling out position keeps 92&ndash;95% of r | CE<sub>small</sub> at L11&ndash;35, and context tokens alone pass 10/10 there (section 12).</p>
<h3>ERNIE-4.5-21B-A3B</h3>
{card(8, 'ERNIE: strongest at mid depth', 'f_edepth', 'Two-sided strength by layer for seven benchmarks (hellaswag and mmlu are null throughout). race, fineweb, pg19 and tinystories pass from layer 13&ndash;14 to 22.')}
{dtable('ernie', MAIN)}
{tbl(['Layer', 'Half A score', 'Half A robust cells', 'Half B score', 'Half B robust cells', 'Full sample robust cells'], esel_rows, num_from=1)}
<p>Chosen on half A, layer 18 beats layer 27 on half B ({E18["half_b_rob_pos_cells"]} vs {E27["half_b_rob_pos_cells"]} robust cells), so the mid-depth advantage is not a selection artifact. Two qualifications decide how it can be used. The gain is almost entirely in r | CE<sub>small</sub> (race {f3(ed["race"]["18"]["s"])} at L18 vs {f3(ed["race"]["27"]["s"])} at L27), while r | CE<sub>big</sub> is similar ({f3(ed["race"]["18"]["b"])} vs {f3(ed["race"]["27"]["b"])}); and at L18&ndash;20 the three router statistics remove r | CE<sub>big</sub> completely (table in section 7). Pooling L16&ndash;20 raises held-out r | CE<sub>small</sub> on race to {f3(POOL[("ernie","mid_block_16_20","race")]["mean_pool_s_split_b"])} but leaves r | CE<sub>big</sub> at {f3(POOL[("ernie","mid_block_16_20","race")]["mean_pool_b_split_b"])}.</p>
<h3>gemma-4-26B-A4B</h3>
{card(9, 'Gemma: one positive layer, then inversion', 'f_gdepth', 'Two-sided strength by layer (* = full-attention layer). Positive at layer 7, negative from layer 18; layer 27 is the most negative and layer 28 null. lambada was profiled only from layer 20.')}
{dtable('gemma', ['race', 'fineweb-edu', 'competition_math', 'lambada'])}
<p><b>Layer 27 audited (v21).</b> The rule was to treat L27 as an outlier if one or two experts carry it. They do not:</p>
{tbl(['Benchmark', 'all 128 experts', 'drop top-10 p<sub>min</sub> experts', 'drop 10 most negative experts', 'empirical-rank null', 'experts negative of 128 (s / b)', 'top-5 p<sub>min</sub> share'], [[SHORT[d]] + [f'{f3(GL[(27, d, m_, k_)]["mean_r_part_ces"])} / {f3(GL[(27, d, m_, k_)]["mean_r_part_ceb"])}' for m_, k_ in [('FULL_128_EXPERTS', 0), ('DROP_TOP_TRIGGER_EXPERTS', 10), ('DROP_MOST_NEG_CORR_EXPERTS', 10), ('EMPIRICAL_RANK', 0)]] + [f'{int(GA[(d, 27)]["neg_s"])} / {int(GA[(d, 27)]["neg_b"])}', f'{GA[(d, 27)]["top5"]:.0f}%'] for d in ['race', 'fineweb-edu', 'competition_math']], num_from=1)}
<p>Dropping the experts that most often set the smallest p-value, or those most negatively correlated with IG, leaves the inversion &minus;10/10 on all three benchmarks; an empirical-rank null (no parametric fit) gives the same partials; almost every expert&rsquo;s standardized score is negatively related to IG at L27 (126&ndash;128 of 128 on r | CE<sub>small</sub>, against 75&ndash;94 at L28); and the per-expert standardization statistics are unremarkable (median MAD ratio to wikitext {GA[("race", 27)]["mad_med"]:.2f}&ndash;{GA[("competition_math", 27)]["mad_med"]:.2f}). L27 is a genuine, distributed late-layer peak of the inversion. L28 is null because its two partials disagree (r | CE<sub>small</sub> {f3(GL[(28, "race", "FULL_128_EXPERTS", 0)]["mean_r_part_ces"])}, r | CE<sub>big</sub> {f3(GL[(28, "race", "FULL_128_EXPERTS", 0)]["mean_r_part_ceb"])} on race), not because the routing goes quiet.</p>
<h3>LFM2-8B-A1B</h3>
{card(10, 'LFM2: late and benchmark-specific', 'f_ldepth', 'Two-sided strength by layer. pg19 passes from layer 11, race only at the last layer; competition_math and structured-wikipedia are negative; layer 2 is negative on most benchmarks.')}
{dtable('liquid', MAIN)}
</section>

<section id="families"><h2>9. Family by family</h2>
<div class="finding"><h4>Qwen3 &mdash; the core result, now localised</h4><p>Five cells pass under every check at the last layer, the effect shrinks with the compute gap but survives on race and fineweb with an 8B partner, and on race and fineweb it builds up through layers 37&ndash;41. In v21 the hidden-state norm takes most of what the router statistics leave in every headline cell except lambada. Earlier layers carry a difficulty signal of the wrong shape for scale (positive with CE<sub>small</sub> held fixed, negative with CE<sub>big</sub> held fixed). lambada is the exception at every depth. competition_math needs the old scale to pass, and dialogue and chat fail under the corrected one; none of the three should be claimed.</p></div>
<div class="finding ernie"><h4>ERNIE-4.5 &mdash; from &ldquo;null&rdquo; to positive on race, fineweb, tinystories and news</h4><p>v19&rsquo;s null was about the raw correlation&rsquo;s sign, which cancels across benchmarks. Under the two-sided test, race, fineweb and tinystories pass at layer 27 and replicate in both halves, and out of domain so do cnn and mbpp; lambada, hellaswag and dialogsum fail (r | CE<sub>small</sub> negative, r | CE<sub>big</sub> positive or null). ERNIE&rsquo;s positive mbpp cell turns out to be carried by newline tokens (v21), so it is a formatting effect rather than evidence that code carries the signal. The middle layers carry more, but there the router statistics explain the CE<sub>big</sub> side.</p></div>
<div class="finding liquid"><h4>LFM2 &mdash; positive on race, pg19, lambada; nothing out of domain</h4><p>pg19 {f3(c[("liquid","pg19")]["c_s"])} / {f3(c[("liquid","pg19")]["c_b"])} and race {f3(c[("liquid","race")]["c_s"])} / {f3(c[("liquid","race")]["c_b"])} pass at the last layer; competition_math is &minus;10/10 on the full sample but does not replicate in both halves ({H["liquid|competition_math"]["bothn"]}/10). None of the six unseen benchmarks passes. Weakest family against the router baselines: race keeps {pct(B["liquid_L23_race"]["R3s"])} of r | CE<sub>small</sub>, pg19 {pct(B["liquid_L23_pg19"]["R3b"])} of r | CE<sub>big</sub>.</p></div>
<div class="finding gemma"><h4>Gemma 4 &mdash; the inverted family</h4><p>No positive cell at the last layer; robust inversion on humaneval, race and news, and 7&ndash;8/10 on four more. The inversion is distinct from the router statistics (94% kept) and builds up with depth. <b>The shared experts are not the explanation</b> (section 10): with HC recomputed from the ablated model, the inversion shrinks by at most a quarter within the loss budget.</p></div>
</section>

<section id="gemma-abl"><h2>10. Gemma: the shared-expert ablation, redone</h2>
<p>Every Gemma layer&rsquo;s shared-expert output is scaled by &alpha;, the model is rerun, and both the losses and the last-layer routing are taken from the ablated forward pass (the v20 rerun had reused the unablated routing). The audit confirms the routing changed: last-layer logits move by up to {gab("race",0.75,"max_logit_diff_vs_alpha1"):.1f} at &alpha; = 0.75 and the mean HC shifts. HC is standardized two ways &mdash; against each &alpha;&rsquo;s own statistics, and against the &alpha; = 1 statistics &mdash; with the same conclusion. fineweb uses a {gab("fineweb-edu",1.0,"n_tokens"):,}-token subset. Rule written before the run: the shared experts explain the inversion if both partials shrink by &ge;50% while CE rises no more than 10%.</p>
{card(11, 'The inversion barely moves within the loss budget', 'f_abl', 'Both partials at &alpha; = 1, 0.75 and 0.5 (solid r | CE<sub>small</sub>, dashed r | CE<sub>big</sub>). Loss rise at &alpha; = 0.75: race +3.4%, fineweb +3.9%, competition_math +9.4% (all inside the budget); at &alpha; = 0.5: +30%, +35%, +82% (outside).')}
{tbl(['Benchmark', 'Shared-expert scale', 'CE rise', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'both partials', 'shrink (s / b)', 'shrink, &alpha;=1 stats'], [[SHORT[d], a, f'+{gab(d,a,"ce_big_rise_pct"):.1f}%', f3(gab(d,a,"mean_r_part_ces")), f3(gab(d,a,"mean_r_part_ceb")), cnt_html(N["gab_cnt"][f"{d}|alpha_self_stats|{a}"]), shrinks(d, a, 'alpha_self_stats'), shrinks(d, a, 'fixed_alpha1_stats')] for d in ['race', 'fineweb-edu', 'competition_math'] for a in [1.0, 0.75, 0.5]], num_from=1)}
<p><b>Verdict: rule not met.</b> Inside the budget the race inversion shrinks {shr("race","inversion_shrink_pct_ces"):.0f}% on r | CE<sub>small</sub> and {shr("race","inversion_shrink_pct_ceb"):.0f}% on r | CE<sub>big</sub> and stays &minus;9/10; fineweb&rsquo;s r | CE<sub>big</sub> moves {shr("fineweb-edu","inversion_shrink_pct_ceb"):.0f}%; competition_math&rsquo;s inversion grows. Only at &alpha; = 0.5, where the model loses 30&ndash;82% in loss, does the inversion disappear &mdash; too degraded a model to attribute it to the shared expert. (Shrink on r | CE<sub>small</sub> is shown as n/a where its &alpha; = 1 value is inside the floor, as for fineweb and competition_math.) The shared experts may contribute a little; they do not explain the inversion, which leaves it a property of Gemma&rsquo;s routed experts or its training.</p>
</section>

<section id="code"><h2>11. Code against prose</h2>
{card(12, 'Code vs prose at matched loss (Qwen)', 'f_code', 'Tokens binned by the mean loss of the two models; within each decile, both partials for code (mbpp, humaneval; red) and prose (cnn, sci; blue). Solid: r | CE<sub>small</sub>; dashed: r | CE<sub>big</sub>.')}
<p>Binned by mean loss of the two models, Qwen&rsquo;s code is below prose in {cpq_s["below"]}/10 deciles on r | CE<sub>small</sub> (mean gap {f3(cpq_s["gap"])}) and {cpq_b["below"]}/10 on r | CE<sub>big</sub> ({f3(cpq_b["gap"])}); Gemma {cpg_s["below"]}/10 and {cpg_b["below"]}/10. Binned by CE<sub>small</sub> alone, the v19 pattern reproduces for r | CE<sub>small</sub> ({N["cp"]["Qwen|ce_small_decile|mean_part_s_c"]["below"]}/10, gap {f3(N["cp"]["Qwen|ce_small_decile|mean_part_s_c"]["gap"])}) but r | CE<sub>big</sub> is mixed ({N["cp"]["Qwen|ce_small_decile|mean_part_b_c"]["below"]}/10). On r | CE<sub>big</sub> the difference is largest among the hardest tokens ({f3(cpq_b["top"])}): on hard prose HC marks the most gain, on hard code none; on r | CE<sub>small</sub> code is below prose at every level of difficulty.</p>
<p><b>The other two families (v20.5).</b> ERNIE shows no code-below-prose gap: {cpe_s["below"]}/10 deciles on r | CE<sub>small</sub> (mean gap {f3(cpe_s["gap"])}) and {cpe_b["below"]}/10 on r | CE<sub>big</sub> ({f3(cpe_b["gap"])}), consistent with mbpp being its strongest positive cell. LFM2&rsquo;s gaps are small and inconsistent ({cpl_s["below"]}/10, {f3(cpl_s["gap"])}; {cpl_b["below"]}/10, {f3(cpl_b["gap"])}). The code break is therefore a property of Qwen and Gemma, not of code as such.</p>
<h3>What carries the code cells (v21)</h3>
{tbl(['Family', 'Benchmark', 'Tokens', 'share', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'both partials'], [[FAMN[m], b, STL[st], f'{CD[(m, b, st)]["share_pct"]:.0f}%', f3(CD[(m, b, st)]["mean_r_part_ces"]), f3(CD[(m, b, st)]["mean_r_part_ceb"]), (f'+{CD[(m, b, st)]["n_pos_both_cell_floor_10"]}/10' if CD[(m, b, st)]["n_pos_both_cell_floor_10"] > CD[(m, b, st)]["n_neg_both_cell_floor_10"] else (f'&minus;{CD[(m, b, st)]["n_neg_both_cell_floor_10"]}/10' if CD[(m, b, st)]["n_neg_both_cell_floor_10"] else '0/10'))] for m, b, st in [('ernie', 'mbpp', 'ALL_TOKENS'), ('ernie', 'mbpp', 'CLASS_NEWLINE'), ('ernie', 'mbpp', 'LOCO_NO_NEWLINE'), ('ernie', 'mbpp', 'CLASS_IDENTIFIER'), ('ernie', 'mbpp', 'ALL_1ST_OCC'), ('ernie', 'humaneval', 'ALL_TOKENS'), ('ernie', 'humaneval', 'CLASS_NEWLINE'), ('gemma', 'humaneval', 'ALL_TOKENS'), ('gemma', 'humaneval', 'CLASS_IDENTIFIER'), ('gemma', 'humaneval', 'LOCO_NO_IDENTIFIER'), ('gemma', 'mbpp', 'CLASS_IDENTIFIER'), ('Qwen', 'mbpp', 'CLASS_IDENTIFIER'), ('Qwen', 'mbpp', 'LOCO_NO_IDENTIFIER')]], num_from=3)}
<p>The rule for ERNIE mbpp &mdash; name a token class if removing it takes away &ge;50% of both partials &mdash; is met by <b>newline tokens</b>: {CD[("ernie","mbpp","CLASS_NEWLINE")]["share_pct"]:.0f}% of mbpp&rsquo;s tokens, with within-class partials of {f3(CD[("ernie","mbpp","CLASS_NEWLINE")]["mean_r_part_ces"])} / {f3(CD[("ernie","mbpp","CLASS_NEWLINE")]["mean_r_part_ceb"])}; without them the cell is null. Newlines also pass on their own in humaneval ({f3(CD[("ernie","humaneval","CLASS_NEWLINE")]["mean_r_part_ces"])} / {f3(CD[("ernie","humaneval","CLASS_NEWLINE")]["mean_r_part_ceb"])}, 10/10), where the full cell does not. One loose end: first occurrences alone also pass on mbpp ({f3(CD[("ernie","mbpp","ALL_1ST_OCC")]["mean_r_part_ces"])} / {f3(CD[("ernie","mbpp","ALL_1ST_OCC")]["mean_r_part_ceb"])}), which the newline account does not explain, since most newlines are repeats; v22 should regrade first occurrences with newlines removed. ERNIE&rsquo;s code result is a formatting-token effect (where a line ends), not a property of code tokens, so it no longer counts as out-of-domain generalisation. In Gemma the code inversion is strongest in identifiers (humaneval {f3(CD[("gemma","humaneval","CLASS_IDENTIFIER")]["mean_r_part_ces"])} / {f3(CD[("gemma","humaneval","CLASS_IDENTIFIER")]["mean_r_part_ceb"])}) but survives their removal; in Qwen identifiers carry the negative r | CE<sub>small</sub> on mbpp, as in v16&ndash;v18.</p>
</section>

<section id="health"><h2>12. Statistical health</h2>
<h3>Robustness battery (Convention C)</h3>
{tbl(['Cell', 'Dataset', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'Winsorised', 'Ranks', 'Drop strongest expert', '| CE<sub>small</sub> after p<sub>min</sub>', '| CE<sub>big</sub> after p<sub>min</sub>'], rob_rows, num_from=2)}
<p>Every robust cell passes winsorising, ranks and removing the strongest expert in 9&ndash;10 of 10 configurations, and keeps both partials beyond its floor after partialling the single smallest p-value: the signal is multi-expert and not carried by outliers. Qwen competition_math fails all three, consistent with its demotion.</p>
<h3>lambada position control</h3>
{tbl(['Layer', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'kept after position (s / b)', 'passes after position', 'context tokens only'], lam_rows, num_from=1)}
<h3>Pooled HC on held-out passages</h3>
{tbl(['Family', 'Block', 'Dataset', 'best single | CE<sub>small</sub>', 'best single | CE<sub>big</sub>', 'pooled | CE<sub>small</sub>', 'pooled | CE<sub>big</sub>', 'pooled passes'], pool_rows, num_from=3)}
<div class="caveat"><strong>Multiplicity.</strong> 60 last-layer cells, 9 benchmarks &times; up to 19 layers per family, and 10 configurations that are highly redundant (worth roughly one independent test for multiplicity, ten for robustness). The paper&rsquo;s claims rest on the five Qwen cells and the cross-family race result, each pre-specified and replicating in both halves (with the stricter half-sample floor, &radic;2 &times; the cell floor, the Qwen cells still pass 9&ndash;10/10 in both halves); depth findings are descriptive, and the one layer selection (ERNIE) was validated out of sample.</div>
</section>

<section id="assess"><h2>13. Assessment</h2>
<div class="assess">
<div class="col strong"><h4>Strong</h4><ul>
<li>A pipeline that reproduces every original number with one flag and fixes the bug with another; all tables recomputed from records.</li>
<li>Five Qwen cells pass a strict two-sided rule in 10/10 configurations, both held-out halves, both floor methods, both scales and the full robustness battery.</li>
<li>The effect is no longer one family: race passes in Qwen, ERNIE and LFM2, and ERNIE generalises to unseen news.</li>
<li>Every open anomaly from v20 has an answer: Qwen&rsquo;s onset is gradual, ERNIE&rsquo;s code cell is newline tokens, Gemma&rsquo;s layer 27 is distributed.</li>
<li>A pre-registered intervention (shared-expert ablation, HC recomputed) cleanly rejects one explanation of Gemma&rsquo;s inversion.</li>
<li>A dose-response with the large model fixed, now with both partials.</li>
<li>Depth profiles that place the Qwen effect in the last third of the network and explain why earlier layers fail.</li></ul></div>
<div class="col weak"><h4>Weak</h4><ul>
<li><b>The hidden-state norm absorbs most of the positive scale signal</b> in Qwen and ERNIE; only lambada (Qwen, LFM2), ERNIE tinystories and newline-driven mbpp keep half of both partials. Effects are small (r 0.02&ndash;0.14).</li>
<li>Several earlier claims do not survive the fix: competition_math (needs the old scale), dialogue and chat (fail under C).</li>
<li>ERNIE and LFM2 are positive only on some benchmarks and negative on others; LFM2 passes nothing out of domain.</li>
<li>ERNIE&rsquo;s mid-depth gain is one-sided and absorbed by router statistics.</li>
<li>The raw difficulty sign is benchmark-dependent within families; the family pattern holds only within token strings.</li></ul></div>
<div class="col open"><h4>Open</h4><ul>
<li>Why Gemma inverts &mdash; not the shared experts; the routed experts or training remain.</li>
<li>Whether the hidden-state norm is a confound or the mechanism behind HC&rsquo;s scale signal.</li>
<li>Why race and fineweb persist with the 8B partner while lambada collapses at 4B.</li>
<li>Why newline tokens in ERNIE carry such a strong scale signal (r &asymp; 0.5&ndash;0.65 within the class).</li></ul></div></div>
</section>

<section id="next"><h2>14. Next experiments</h2>
{exp('The hidden-state norm on its own', 'P0', 'hours, existing data', 'Does the norm itself pass the two-sided test where HC does, and does HC keep anything once only the norm is partialled out?', 'For every positive cell: both within-string partials of the norm with IG; HC partialling the norm alone (without the router statistics); the same at Qwen layers 35&ndash;47.', 'Two-sided counts for the norm; retention of HC after the norm alone.', 'If the norm passes wherever HC passes and HC keeps &lt;50% after it, describe HC&rsquo;s scale signal as a norm effect in the paper; if HC keeps &ge;50% in most cells, the overlap is with the router statistics jointly, not the norm.')}
{exp('HC within norm strata', 'P0', 'hours, existing data', 'The router reads an RMS-normalised input, so the norm cannot inflate the router logits directly. Is the overlap then a co-occurrence &mdash; the same tokens (e.g. those with very large residual activations) are both anomalously routed and where scale helps?', 'Split each positive cell into deciles of the hidden-state norm; compute both within-string partials inside each decile; list the token types in the top norm decile.', 'Number of deciles passing both partials; token composition of the top decile.', 'If the association holds inside most norm deciles, the norm is a correlated covariate and HC carries information of its own; if it lives only in the top decile, report it as a property of high-norm tokens.')}
{exp('Format-token control everywhere', 'P1', 'hours', 'Does any other headline cell depend on newline, whitespace or other format tokens, as ERNIE mbpp does?', 'Drop newline, indentation and special tokens from all 60 cells; regrade, including ERNIE mbpp first occurrences without newlines.', 'Two-sided counts before and after.', 'A cell keeps its verdict only if it still passes 9/10 without format tokens.')}
{exp('Raw difficulty sign by benchmark', 'P2', 'hours', 'Why are Qwen fineweb and pg19 and ERNIE fineweb and race negative in raw r(HC, CE) but not within strings?', 'Decompose raw r into between-string and within-string parts per cell.', 'Share of the raw correlation explained by token identity.', 'If the between-string part flips the sign, report difficulty only within strings.')}
</section>

<section id="tables"><h2>15. Tables</h2>
<h3>Every last-layer cell (Convention C primary; conservative floor)</h3>
{tbl(['Family', 'Benchmark', 'Grid', 'Floor', 'r | CE<sub>small</sub>', 'r | CE<sub>big</sub>', 'C: both partials', 'A: both partials'], ll_rows, num_from=3)}
<h3>Dose-response per benchmark (two-sided count; r | CE<sub>small</sub> / r | CE<sub>big</sub>)</h3>
{tbl(['Benchmark'] + [f'{p} ({RAT[p]}&times;)' for p in P], dose_rows, num_from=1)}
<p style="font-size:0.85rem;color:#64748b">Sources: reports/v21_experiments/v21 (difficulty correlations, router baselines with the norm for every cell, Qwen L35&ndash;42, code dissection, Gemma L26&ndash;29 audit); reports/26_10_29/v20, updated (60-cell last-layer records, 60-cell floors, ERNIE/LFM2 unseen, Gemma ablation with recomputed HC, baselines, code vs prose for four families; partner sweep, baselines, depth with 10 configurations, held-out splits, code vs prose, ERNIE layer selection, lambada position, pooled HC, robustness, Gemma ablation audit) . All numbers recomputed from the records; verdicts use the conservative floor.</p>
</section>
</main>
<script>
const CHART_SPECS = {json.dumps(specs)};
{BOOT}'''
save('/home/claude/v21/HC_MoE_Report_v21.html', head('HC-MoE Report v21') + body)
