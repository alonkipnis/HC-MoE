import os
import json
import pandas as pd
import numpy as np

# Load simulation
sim_path = 'reports/26_08_25/summary_n10000_rep10000_seed42.json'
with open(sim_path, 'r') as f:
    sim_data = json.load(f)

sim_cfg = sim_data['experiment_config']
sim_stats = sim_data['summary_statistics']
null_std = sim_stats['std']
null_crit_05 = 1.96 * null_std
null_crit_01 = 2.576 * null_std
null_max = max(abs(sim_stats['max']), abs(sim_stats['min']))

# Load models
csv_files = {
    'Qwen': 'reports/26_08_25/correlation_Qwen_summary_combined_per_expert_small.csv',
    'ERNIE': 'reports/26_08_25/correlation_ernie_summary_combined_per_expert_small.csv',
    'Gemma': 'reports/26_08_25/correlation_gemma_summary_combined_per_expert_small.csv',
    'Liquid': 'reports/26_08_25/correlation_liquid_summary_combined_per_expert_small.csv'
}

dfs = {k: pd.read_csv(v) for k, v in csv_files.items()}
all_df = pd.concat([df.assign(Model_Family=k) for k, df in dfs.items()], ignore_index=True)

# Read inlined Chart.js
with open('reports/HC_MoE_Report_26_08_11.html', 'r') as f:
    old_lines = f.readlines()

chartjs_code = ''.join(old_lines[7:31])

# Data calculations for tables
datasets_order = ['lambada', 'competition_math', 'structured-wikipedia', 'fineweb-edu', 'tinystories', 'race', 'mmlu', 'pg19', 'hellaswag']
models_order = ['Qwen', 'ERNIE', 'Gemma', 'Liquid']

# Precompute dataset table rows
dataset_table_rows = []
for d in datasets_order:
    row = {'Dataset': d}
    for m in models_order:
        sub = dfs[m][dfs[m]['Dataset']==d]
        diff_mean = sub['Pearson_Max_Per_Row_Diff'].mean()
        diff_q90 = sub['Pearson_Max_Per_Row_Diff_Q90'].mean()
        sig_diff = (sub['Pearson_Max_Per_Row_Diff'].abs() > null_crit_05).sum()
        sig_q90 = (sub['Pearson_Max_Per_Row_Diff_Q90'].abs() > null_crit_05).sum()
        row[f'{m}_Full'] = diff_mean
        row[f'{m}_Full_Sig'] = sig_diff
        row[f'{m}_Q90'] = diff_q90
        row[f'{m}_Q90_Sig'] = sig_q90
    dataset_table_rows.append(row)

# Precompute estimator table rows
estimators_order = ['GMM', 'GLM', 'StudentT', 'StudentTSkewed', 'KDE', 'JohnsonSU']
est_table_rows = []
for e in estimators_order:
    row = {'Estimator': e}
    for m in models_order:
        sub = dfs[m][dfs[m]['Estimator']==e]
        diff_mean = sub['Pearson_Max_Per_Row_Diff'].mean()
        diff_q90 = sub['Pearson_Max_Per_Row_Diff_Q90'].mean()
        sig_diff = (sub['Pearson_Max_Per_Row_Diff'].abs() > null_crit_05).sum()
        row[f'{m}_Full'] = diff_mean
        row[f'{m}_Q90'] = diff_q90
        row[f'{m}_Sig'] = sig_diff
    est_table_rows.append(row)

# Precompute quantile progression table rows
quant_cols = [
    ('Full Layers', 'Pearson_Max_Per_Row_Diff'),
    ('Q50 Subset', 'Pearson_Max_Per_Row_Diff_Q50'),
    ('Q75 Subset', 'Pearson_Max_Per_Row_Diff_Q75'),
    ('Q85 Subset', 'Pearson_Max_Per_Row_Diff_Q85'),
    ('Q90 Subset', 'Pearson_Max_Per_Row_Diff_Q90'),
    ('Q95 Subset', 'Pearson_Max_Per_Row_Diff_Q95')
]
quant_table_rows = []
for label, col in quant_cols:
    row = {'Subset': label}
    for m in models_order:
        vals = dfs[m][col].dropna()
        mean_val = vals.mean()
        sig_cnt = (vals.abs() > null_crit_05).sum()
        ext_cnt = (vals.abs() > null_max).sum()
        row[f'{m}_Mean'] = mean_val
        row[f'{m}_Sig'] = sig_cnt
        row[f'{m}_Ext'] = ext_cnt
    quant_table_rows.append(row)

# Precompute multi-expert table rows
multi_table_rows = []
for m in models_order:
    df = dfs[m]
    hc_diff = df['Pearson_Max_Per_Row_Diff'].mean()
    inv_diff = df['Pearson_Inverse_Min_P_Diff'].mean()
    prm_diff = df['Pearson_HC_Pmin_rm_Diff'].mean()
    part_diff = df['Partial_Pearson_Max_Per_Row_Diff'].mean()
    hc_sig = (df['Pearson_Max_Per_Row_Diff'].abs() > null_crit_05).sum()
    inv_sig = (df['Pearson_Inverse_Min_P_Diff'].abs() > null_crit_05).sum()
    prm_sig = (df['Pearson_HC_Pmin_rm_Diff'].abs() > null_crit_05).sum()
    part_sig = (df['Partial_Pearson_Max_Per_Row_Diff'].abs() > null_crit_05).sum()
    multi_table_rows.append({
        'Model': m,
        'HC': hc_diff, 'HC_Sig': hc_sig,
        'Inv': inv_diff, 'Inv_Sig': inv_sig,
        'Prm': prm_diff, 'Prm_Sig': prm_sig,
        'Part': part_diff, 'Part_Sig': part_sig
    })

# Format helper
def fmt_corr(val, sig_level=0):
    badge = ''
    if abs(val) > null_max:
        badge = ' <span class="badge badge-ext">p<0.0001</span>'
    elif abs(val) > null_crit_01:
        badge = ' <span class="badge badge-high">p<0.01</span>'
    elif abs(val) > null_crit_05:
        badge = ' <span class="badge badge-sig">p<0.05</span>'
    
    color_style = 'color:#16a34a;font-weight:600;' if val > 0.0193 else ('color:#dc2626;font-weight:600;' if val < -0.0193 else 'color:#64748b;')
    return f'<span style="{color_style}">{val:+.4f}</span>{badge}'

print('Data tables prepared successfully!')
