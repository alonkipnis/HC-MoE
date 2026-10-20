import os
import json
import pandas as pd
import numpy as np

# 1. Load Simulation
sim_path = 'reports/26_08_25/summary_n10000_rep10000_seed42.json'
with open(sim_path, 'r') as f:
    sim_data = json.load(f)

sim_cfg = sim_data['experiment_config']
sim_stats = sim_data['summary_statistics']
null_std = sim_stats['std']
null_crit_05 = 1.96 * null_std
null_crit_01 = 2.576 * null_std
null_max = max(abs(sim_stats['max']), abs(sim_stats['min']))

# 2. Load Models
csv_files = {
    'Qwen': 'reports/26_08_25/correlation_Qwen_summary_combined_per_expert_small.csv',
    'ERNIE': 'reports/26_08_25/correlation_ernie_summary_combined_per_expert_small.csv',
    'Gemma': 'reports/26_08_25/correlation_gemma_summary_combined_per_expert_small.csv',
    'Liquid': 'reports/26_08_25/correlation_liquid_summary_combined_per_expert_small.csv'
}

dfs = {k: pd.read_csv(v) for k, v in csv_files.items()}
all_df = pd.concat([df.assign(Model_Family=k) for k, df in dfs.items()], ignore_index=True)

# 3. Read inlined Chart.js from previous report
with open('reports/HC_MoE_Report_26_08_11.html', 'r') as f:
    old_lines = f.readlines()

chartjs_code = ''.join(old_lines[7:31])

# 4. Generate Chart Data Payloads
datasets_order = ['lambada', 'competition_math', 'structured-wikipedia', 'fineweb-edu', 'tinystories', 'race', 'mmlu', 'pg19', 'hellaswag']
models_order = ['Qwen', 'ERNIE', 'Gemma', 'Liquid']

# Fig 1: 36 bars (GMM, per_expert_standard)
fig1_labels = []
fig1_big = []
fig1_small = []
fig1_diff = []

dataset_short_names = {
    'lambada': 'lambada',
    'competition_math': 'comp_math',
    'structured-wikipedia': 'struct_wiki',
    'fineweb-edu': 'fineweb_edu',
    'tinystories': 'tinystories',
    'race': 'race',
    'mmlu': 'mmlu',
    'pg19': 'pg19',
    'hellaswag': 'hellaswag'
}

for m in models_order:
    df = dfs[m]
    sub = df[(df['Estimator']=='GMM') & (df['HC_File']=='hc_values_per_expert_standard_1_new.parquet')]
    for d in datasets_order:
        row = sub[sub['Dataset']==d]
        if len(row) > 0:
            fig1_labels.append([m, dataset_short_names.get(d, d)])
            fig1_big.append(round(float(row['Pearson_Max_Per_Row_Big'].iloc[0]), 4))
            fig1_small.append(round(float(row['Pearson_Max_Per_Row_Small'].iloc[0]), 4))
            fig1_diff.append(round(float(row['Pearson_Max_Per_Row_Diff'].iloc[0]), 4))

# Fig 2: Qwen lambada estimator robustness
q_lam = dfs['Qwen'][(dfs['Qwen']['Dataset']=='lambada') & (dfs['Qwen']['HC_File']=='hc_values_per_expert_standard_1_new.parquet')]
fig2_estimators = q_lam['Estimator'].tolist()
fig2_big = [round(float(x), 4) for x in q_lam['Pearson_Max_Per_Row_Big']]
fig2_small = [round(float(x), 4) for x in q_lam['Pearson_Max_Per_Row_Small']]
fig2_diff = [round(float(x), 4) for x in q_lam['Pearson_Max_Per_Row_Diff']]

# Fig 3: Layer progression by model
subsets = ['Full', 'Q50', 'Q75', 'Q85', 'Q90', 'Q95']
cols = ['Pearson_Max_Per_Row_Diff', 'Pearson_Max_Per_Row_Diff_Q50', 'Pearson_Max_Per_Row_Diff_Q75', 'Pearson_Max_Per_Row_Diff_Q85', 'Pearson_Max_Per_Row_Diff_Q90', 'Pearson_Max_Per_Row_Diff_Q95']
fig3_qwen = [round(float(dfs['Qwen'][c].mean()), 4) for c in cols]
fig3_ernie = [round(float(dfs['ERNIE'][c].mean()), 4) for c in cols]
fig3_gemma = [round(float(dfs['Gemma'][c].mean()), 4) for c in cols]
fig3_liquid = [round(float(dfs['Liquid'][c].mean()), 4) for c in cols]

# Fig 4: Raw vs Partial Diff across 6 estimators for Qwen/lambada
fig4_raw = [round(float(x), 4) for x in q_lam['Pearson_Max_Per_Row_Diff']]
fig4_part = [round(float(x), 4) for x in q_lam['Partial_Pearson_Max_Per_Row_Diff']]

# Fig 5: HC vs InvMinP vs Pmin-rm Diff means across 4 models
fig5_hc = [round(float(dfs[m]['Pearson_Max_Per_Row_Diff'].mean()), 4) for m in models_order]
fig5_inv = [round(float(dfs[m]['Pearson_Inverse_Min_P_Diff'].mean()), 4) for m in models_order]
fig5_prm = [round(float(dfs[m]['Pearson_HC_Pmin_rm_Diff'].mean()), 4) for m in models_order]
fig5_part = [round(float(dfs[m]['Partial_Pearson_Max_Per_Row_Diff'].mean()), 4) for m in models_order]

# Fig 6: Dataset IG ranking for all 4 models
fig6_datasets = ['lambada', 'comp_math', 'structured-wiki', 'fineweb-edu', 'tinystories', 'race', 'mmlu', 'pg19', 'hellaswag']
fig6_qwen_diff = [round(float(dfs['Qwen'][dfs['Qwen']['Dataset']==d]['Pearson_Max_Per_Row_Diff'].mean()), 4) for d in datasets_order]
fig6_ernie_diff = [round(float(dfs['ERNIE'][dfs['ERNIE']['Dataset']==d]['Pearson_Max_Per_Row_Diff'].mean()), 4) for d in datasets_order]
fig6_gemma_diff = [round(float(dfs['Gemma'][dfs['Gemma']['Dataset']==d]['Pearson_Max_Per_Row_Diff'].mean()), 4) for d in datasets_order]
fig6_liquid_diff = [round(float(dfs['Liquid'][dfs['Liquid']['Dataset']==d]['Pearson_Max_Per_Row_Diff'].mean()), 4) for d in datasets_order]

# Fig 7: Q90 IG landscape across 9 datasets x 4 models
fig7_qwen_q90 = [round(float(dfs['Qwen'][dfs['Qwen']['Dataset']==d]['Pearson_Max_Per_Row_Diff_Q90'].mean()), 4) for d in datasets_order]
fig7_ernie_q90 = [round(float(dfs['ERNIE'][dfs['ERNIE']['Dataset']==d]['Pearson_Max_Per_Row_Diff_Q90'].mean()), 4) for d in datasets_order]
fig7_gemma_q90 = [round(float(dfs['Gemma'][dfs['Gemma']['Dataset']==d]['Pearson_Max_Per_Row_Diff_Q90'].mean()), 4) for d in datasets_order]
fig7_liquid_q90 = [round(float(dfs['Liquid'][dfs['Liquid']['Dataset']==d]['Pearson_Max_Per_Row_Diff_Q90'].mean()), 4) for d in datasets_order]

print('All data payloads precomputed successfully!')
