import pandas as pd, numpy as np, json
V='/mnt/user-data/uploads/HC-MoE-Alon/reports/26_10_29/v20/'
R25=V+'authoritative_last_layer_60_cells_records.csv'
N={}
r3=lambda x: None if x is None or (isinstance(x,float) and np.isnan(x)) else round(float(x),5)
# ---------- last layer regrade with conservative floor ----------
R=pd.read_csv(R25)
F=pd.read_csv(V+'p0_1_noise_floors_regraded_60_cells.csv')
R=R.merge(F[['model','benchmark','floor_ces_lag20','floor_ig_v15']],on=['model','benchmark'])
R['fl']=R[['floor_ces_lag20','floor_ig_v15']].max(axis=1)
def ts(s,b): # two-sided strength per config
    return np.where((s>0)&(b>0),np.minimum(s,b),np.where((s<0)&(b<0),np.maximum(s,b),0.0))
cells=[]
for (m,bm),x in R.groupby(['model','benchmark'],sort=False):
    d=dict(model=m,bm=bm,grid=x.grid.iloc[0],fl=x.fl.iloc[0],f1=x.floor_ces_lag20.iloc[0],f2=x.floor_ig_v15.iloc[0])
    for a in 'ca':
        s=x[a+'_within_r_ce_small'];b=x[a+'_within_r_ce_big']
        for tag,f in (('',x.fl),('_f1',x.floor_ces_lag20),('_f2',x.floor_ig_v15)):
            p=int(((s>f)&(b>f)).sum()); n=int(((s<-f)&(b<-f)).sum())
            d[a+'_cnt'+tag]=p if p>=n else -n
        d[a+'_s']=s.mean(); d[a+'_b']=b.mean(); d[a+'_ts']=ts(s,b).mean(); d[a+'_raw']=x[a+'_raw_r'].mean()
    cells.append(d)
C=pd.DataFrame(cells)
def cls(c): return 'pos' if c>=9 else ('neg' if c<=-9 else 'none')
for a in 'ca':
    C[a+'_cls']=C[a+'_cnt'].map(cls)
C['floor_sens_c']=(C.c_cnt_f1.map(cls)!=C.c_cnt_f2.map(cls))
C['floor_sens_a']=(C.a_cnt_f1.map(cls)!=C.a_cnt_f2.map(cls))
N['cells']=[{k:(r3(v) if isinstance(v,(float,np.floating)) else (int(v) if isinstance(v,(np.integer,)) else (bool(v) if isinstance(v,np.bool_) else v))) for k,v in r.items()} for r in C.to_dict('records')]
N['floor_sens_c']=int(C.floor_sens_c.sum()); N['floor_sens_a']=int(C.floor_sens_a.sum())
N['ca_cross']=pd.crosstab(C.c_cls,C.a_cls).to_dict()
print(C[['model','bm','c_cnt','a_cnt','c_s','c_b','c_ts','floor_sens_c','floor_sens_a']].round(4).to_string())
print(pd.crosstab(C.c_cls,C.a_cls)); print('floor sens',N['floor_sens_c'],N['floor_sens_a'])
# ---------- held-out ----------
H={}
for (m,bm),x in R.groupby(['model','benchmark']):
    f=x.fl*np.sqrt(2)
    pa=((x.c_split_a_r_part_ces>f)&(x.c_split_a_r_part_ceb>f)); pb=((x.c_split_b_r_part_ces>f)&(x.c_split_b_r_part_ceb>f))
    na=((x.c_split_a_r_part_ces<-f)&(x.c_split_a_r_part_ceb<-f)); nb=((x.c_split_b_r_part_ces<-f)&(x.c_split_b_r_part_ceb<-f))
    pa1=((x.c_split_a_r_part_ces>x.fl)&(x.c_split_a_r_part_ceb>x.fl)); pb1=((x.c_split_b_r_part_ces>x.fl)&(x.c_split_b_r_part_ceb>x.fl))
    H[f'{m}|{bm}']=dict(a=int(pa.sum()),b=int(pb.sum()),both=int((pa&pb).sum()),bothn=int((na&nb).sum()),both1=int((pa1&pb1).sum()),bothn1=int((((x.c_split_a_r_part_ces<-x.fl)&(x.c_split_a_r_part_ceb<-x.fl))&((x.c_split_b_r_part_ces<-x.fl)&(x.c_split_b_r_part_ceb<-x.fl))).sum()),
        sa=r3(x.c_split_a_r_part_ces.mean()),sb=r3(x.c_split_b_r_part_ces.mean()),ba=r3(x.c_split_a_r_part_ceb.mean()),bb=r3(x.c_split_b_r_part_ceb.mean()))
N['held']=H
# ---------- partner sweep ----------
A=pd.read_csv(V+'p0_3_partner_sweep_convention_c_all_configs.csv')
B=A[~((A.dataset=='fineweb-edu')&(A.partner=='0.6B'))].copy(); B.loc[B.partner=='0.6B_aligned207k','partner']='0.6B'
P=['0.6B','1.7B','4B','8B']
B['ts']=ts(B.part_s_c,B.part_b_c)
m=B.groupby(['dataset','partner'])[['r_raw','part_s_c','part_b_c','ts']].mean()
fam=m.groupby('partner').mean().loc[P]
N['dose_fam']={c:[r3(v) for v in fam[c]] for c in fam.columns}
N['dose_bm']={c:{d:[r3(v) for v in m[c].unstack().loc[d,P]] for d in m.index.levels[0]} for c in ['part_s_c','part_b_c','ts']}
S=pd.read_csv(V+'p0_3_partner_sweep_convention_c_summary.csv')
cnt={}
for (d,p),x in B.groupby(['dataset','partner']):
    f=np.maximum(x.floor_ces,x.floor_ig); pz=int(((x.part_s_c>f)&(x.part_b_c>f)).sum()); nz=int(((x.part_s_c<-f)&(x.part_b_c<-f)).sum())
    cnt.setdefault(d,{})[p]=pz if pz>=nz else -nz
N['dose_cnt']=cnt
N['dose_robust']={p:sorted([d for d in cnt if cnt[d][p]>=9]) for p in P}
print(fam.round(4)); print(N['dose_robust'])
mono=(m.part_s_c.unstack()[P].diff(axis=1).iloc[:,1:]<0).all(axis=1); N['dose_mono_s']=int(mono.sum())
mono2=(m.part_b_c.unstack()[P].diff(axis=1).iloc[:,1:]<0).all(axis=1); N['dose_mono_b']=int(mono2.sum())
print('mono',N['dose_mono_s'],N['dose_mono_b'])
# ---------- baselines ----------
Bs=pd.read_csv(V+'p0_4_router_baselines_convention_c_all_configs.csv')
g=Bs.groupby('cell_label').agg(model=('model','first'),ds=('dataset','first'),layer=('layer','first'),prim=('is_p0_4_primary','first'),hn=('has_hnorm','first'),
   ref_s=('r_ref_s','mean'),ref_b=('r_ref_b','mean'),r3s=('r_3_s','mean'),r3b=('r_3_b','mean'),r4s=('r_4_s','mean'),r4b=('r_4_b','mean'),d3=('delta_r2_3_s','mean'),d4s=('delta_r2_4_s','mean'),d4b=('delta_r2_4_b','mean'))
for k in ['3s','3b','4s','4b']: g['R'+k]=g['r'+k]/g['ref_'+k[1]]
N['base']={i:{k:(r3(v) if isinstance(v,(float,np.floating)) else (bool(v) if isinstance(v,(bool,np.bool_)) else (int(v) if isinstance(v,np.integer) else v))) for k,v in r.items()} for i,r in g.iterrows()}
print(g[['R3s','R3b','R4s','R4b','d3','d4s','d4b']].round(3))
# ---------- depth ----------
DF={'Qwen':'p0_5_qwen_depth_10configs_all.csv','gemma':'p0_5_p1_4_gemma_depth_10configs_all.csv','ernie':'p0_5_p1_5_ernie_depth_10configs_all.csv','liquid':'p1_6_liquid_depth_10configs_all.csv'}
SF={'Qwen':'p0_5_qwen_depth_10configs_summary.csv','gemma':'p0_5_p1_4_gemma_depth_10configs_summary.csv','ernie':'p0_5_p1_5_ernie_depth_10configs_summary.csv','liquid':'p1_6_liquid_depth_10configs_summary.csv'}
N['depth']={}
for fm,fn in DF.items():
    D=pd.read_csv(V+fn); D['ts']=ts(D.r_part_ces,D.r_part_ceb)
    D['P']=(D.r_part_ces>D.floor)&(D.r_part_ceb>D.floor); D['Nn']=(D.r_part_ces<-D.floor)&(D.r_part_ceb<-D.floor)
    out={}
    for (d,l),x in D.groupby(['dataset','layer']):
        p=int(x.P.sum()); n=int(x.Nn.sum())
        out.setdefault(d,{})[int(l)]=dict(cnt=p if p>=n else -n,s=r3(x.r_part_ces.mean()),b=r3(x.r_part_ceb.mean()),ts=r3(x.ts.mean()),fl=r3(x.floor.iloc[0]))
    att=pd.read_csv(V+SF[fm]).groupby('layer').attention_type.first().to_dict()
    N['depth'][fm]=dict(cells=out,layers=sorted(int(l) for l in D.layer.unique()),att={int(k):v for k,v in att.items()})
# ---------- ernie layer selection ----------
E=pd.read_csv(V+'p1_5_ernie_layer_selection_split_half.csv'); N['esel']=E.round(5).to_dict('records')
# ---------- lambada position ----------
Lp=pd.read_csv(V+'p2_1_lambada_depth_position_summary.csv'); N['lpos']=Lp.round(5).to_dict('records')
# ---------- pooled ----------
Po=pd.read_csv(V+'p2_2_pooled_hc_heldout_summary.csv'); N['pool']=Po.round(5).to_dict('records')
Pa=pd.read_csv(V+'p2_2_pooled_hc_heldout_all_configs.csv'); N['pool_best']={f'{k[0]}|{k[1]}':{str(a):int(b) for a,b in v.items()} for k,v in Pa.groupby(['model','block_name']).best_single_l_on_a.agg(lambda s:s.value_counts().to_dict()).items()}
# ---------- robustness ----------
Ro=pd.read_csv(V+'p2_3_robustness_checks_summary.csv'); N['rob']=Ro.round(5).to_dict('records')
# ---------- code vs prose ----------
Cp=pd.read_csv(V+'p1_3_code_vs_prose_convention_c_summary.csv'); Cp['mod']=np.where(Cp.dataset.isin(['mbpp','humaneval']),'code','prose')
N['cp']={}
for mdl in ['Qwen','gemma','ernie','liquid']:
  for bsch in ['ce_small_decile','ce_mean_decile']:
    x=Cp[(Cp.model==mdl)&(Cp.bin_scheme==bsch)]
    for c in ['mean_part_s_c','mean_part_b_c']:
      gg=x.groupby(['decile','mod'])[c].mean().unstack()
      N['cp'][f'{mdl}|{bsch}|{c}']=dict(code=[r3(v) for v in gg['code']],prose=[r3(v) for v in gg['prose']],below=int((gg['code']<gg['prose']).sum()),gap=r3((gg['code']-gg['prose']).mean()),
        top=r3((gg['code']-gg['prose']).iloc[-1]))
    N['cp'][f'{mdl}|{bsch}|ce']=[r3(v) for v in x.groupby('decile').mean_ce_small.mean()] if bsch=='ce_small_decile' else [r3(v) for v in x.assign(cm=(x.mean_ce_small+x.mean_ce_big)/2).groupby('decile').cm.mean()]
# ---------- gemma ablation ----------
Ga=pd.read_csv(V+'p0_2_gemma_ablation_hc_audit.csv'); N['gab']=Ga.round(5).to_dict('records')
Gc=pd.read_csv(V+'p0_2_gemma_ablation_all_configs.csv')
Gc['P']=(Gc.r_part_ces>Gc.floor)&(Gc.r_part_ceb>Gc.floor); Gc['Nn']=(Gc.r_part_ces<-Gc.floor)&(Gc.r_part_ceb<-Gc.floor)
N['gab_cnt']={f'{d}|{m}|{a}':(int(x.P.sum()) if x.P.sum()>=x.Nn.sum() else -int(x.Nn.sum())) for (d,m,a),x in Gc.groupby(['dataset','stats_mode','alpha'])}
U=pd.read_csv(V+'p1_1_ernie_liquid_unseen_6_summary.csv'); N['unseen_el']=U.round(5).to_dict('records')
json.dump(N,open('/home/claude/v205/n205.json','w'),indent=0,default=lambda o: o.item() if hasattr(o,'item') else str(o))
print('saved')
