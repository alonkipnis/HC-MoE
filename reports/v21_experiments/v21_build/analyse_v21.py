import pandas as pd, numpy as np, json
V='/mnt/user-data/uploads/HC-MoE-Alon/reports/v21_experiments/v21/'
N=json.load(open('/home/claude/v21/n205.json'))
r3=lambda x: None if x is None or (isinstance(x,float) and np.isnan(x)) else round(float(x),5)
def ts(s,b): return np.where((s>0)&(b>0),np.minimum(s,b),np.where((s<0)&(b<0),np.maximum(s,b),0.0))
# ---- baselines (replace)
Bs=pd.read_csv(V+'p0_4_router_baselines_convention_c_all_configs.csv')
g=Bs.groupby('cell_label').agg(model=('model','first'),ds=('dataset','first'),layer=('layer','first'),prim=('is_p0_4_primary','first'),hn=('has_hnorm','first'),
   ref_s=('r_ref_s','mean'),ref_b=('r_ref_b','mean'),r3s=('r_3_s','mean'),r3b=('r_3_b','mean'),r4s=('r_4_s','mean'),r4b=('r_4_b','mean'),d3=('delta_r2_3_s','mean'),d4s=('delta_r2_4_s','mean'),d4b=('delta_r2_4_b','mean'))
for k in ['3s','3b','4s','4b']: g['R'+k]=g['r'+k]/g['ref_'+k[1]]
N['base']={i:{k:(r3(v) if isinstance(v,(float,np.floating)) else (bool(v) if isinstance(v,(bool,np.bool_)) else (int(v) if isinstance(v,np.integer) else v))) for k,v in r.items()} for i,r in g.iterrows()}
# ---- Qwen depth (replace with 36-41 included)
D=pd.read_csv(V+'p1_7_qwen_depth_with_36_41_all_configs.csv'); D['ts']=ts(D.r_part_ces,D.r_part_ceb)
D['P']=(D.r_part_ces>D.floor)&(D.r_part_ceb>D.floor); D['Nn']=(D.r_part_ces<-D.floor)&(D.r_part_ceb<-D.floor)
out={}
for (d,l),x in D.groupby(['dataset','layer']):
    p=int(x.P.sum()); n=int(x.Nn.sum())
    out.setdefault(d,{})[int(l)]=dict(cnt=p if p>=n else -n,s=r3(x.r_part_ces.mean()),b=r3(x.r_part_ceb.mean()),ts=r3(x.ts.mean()),fl=r3(x.floor.iloc[0]))
att=pd.read_csv(V+'p1_7_qwen_depth_with_36_41_summary.csv').groupby('layer').attention_type.first().to_dict()
old=N['depth']['Qwen']['cells']
chk=max(abs(out[d][int(l)]['s']-v['s']) for d in old for l,v in old[d].items())
print('Qwen depth reproduction max |ds|',chk)
N['depth']['Qwen']=dict(cells=out,layers=sorted(int(l) for l in D.layer.unique()),att={int(k):v for k,v in att.items()})
T=pd.read_csv(V+'p1_7_qwen_transition_36_41_summary.csv'); N['qtrans']=T.round(5).to_dict('records')
# ---- difficulty
A=pd.read_csv(V+'p0_6_difficulty_correlations_60_cells_all_configs.csv')
sg={'Qwen':1,'ernie':1,'liquid':1,'gemma':-1}
rows=[]
for (m,b,gr),x in A.groupby(['model','benchmark','grid'],sort=False):
    fl=x.floor_conservative.iloc[0]; d=dict(model=m,bm=b,grid=gr,fl=r3(fl))
    for arm in 'ca':
        for st in ['raw','within']:
            for ce,cn in [('ce_big','b'),('ce_small','s')]:
                v=x[f'{arm}_{st}_r_{ce}']; p=int((v>fl).sum()); n=int((v<-fl).sum())
                d[f'{arm}_{st}_{cn}']=r3(v.mean()); d[f'{arm}_{st}_{cn}_cnt']=p if p>=n else -n
    rows.append(d)
Df=pd.DataFrame(rows); Df['sg']=Df.model.map(sg)
N['diff']=Df.drop(columns=['sg']).to_dict('records')
st=Df[Df.grid=='studied']; summ={}
for arm in 'ca':
    for s_ in ['raw','within']:
        for cn in 'bs':
            k=f'{arm}_{s_}_{cn}'
            summ[k]=dict(beyond=int((st[k].abs()>st.fl).sum()),fam=int(((st[k]*st.sg)>st.fl).sum()),rob=int((st[k+'_cnt'].abs()>=9).sum()),robfam=int(((st[k+'_cnt']*st.sg)>=9).sum()),robopp=int(((st[k+'_cnt']*st.sg)<=-9).sum()))
N['diff_sum']=summ; print(json.dumps(summ,indent=0)[:900])
N['diff_fam']={m:{k:r3(st[st.model==m][k].mean()) for k in ['c_raw_b','c_raw_s','c_within_b','c_within_s']} for m in sg}
# ---- code dissection
C=pd.read_csv(V+'p1_8_code_dissection_convention_c_summary.csv')
N['codeD']=C[['model','benchmark','analysis_group','stratum','share_pct','stratum_floor','mean_r_part_ces','mean_r_part_ceb','n_pos_both_cell_floor_10','n_neg_both_cell_floor_10']].round(5).to_dict('records')
# ---- gemma L26-29
G=pd.read_csv(V+'p2_4_gemma_l26_29_leave_k_out_summary.csv'); N['gl']=G.round(5).to_dict('records')
E=pd.read_csv(V+'p2_4_gemma_l26_29_expert_audit.csv'); E=E[E.estimator=='GMM']
ea=E.groupby(['benchmark','layer']).apply(lambda x: pd.Series({'mad_med':x.mad_ratio_bench_over_wiki.median(),'top5':x.p_min_win_share_pct.nlargest(5).sum(),'neg_s':int((x.z_partial_r_ces<0).sum()),'neg_b':int((x.z_partial_r_ceb<0).sum())})).reset_index()
N['gaudit']=ea.round(4).to_dict('records'); print(ea)
json.dump(N,open('/home/claude/v21/n21.json','w'),indent=0,default=lambda o: o.item() if hasattr(o,'item') else str(o)); print('saved')
