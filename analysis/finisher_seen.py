# Pre-registered test in docs/finisher-seen-FRA.md. Run: uv run --directory <repo> python analysis/finisher_seen.py
import json,itertools,numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf,warnings
warnings.filterwarnings('ignore')
R='/home/morris/Projects/mtga-draft-companion/'
cr=json.load(open(R+'data/Sets/FRA_PremierDraft_All_Data.json'))['card_ratings']
gi=lambda c:(c['deck_colors'].get('All Decks') or {}).get('gihwr',0) or 0
nl={c['name']:c for c in cr.values() if 'Land' not in c['types']}
v=np.array([gi(c) for c in cr.values() if 'Land' not in c['types'] and gi(c)>0]);mu,sd=v.mean(),v.std(ddof=1);FC=mu+sd
fin={n for n,c in nl.items() if c['cmc']>=5 and gi(c)>=FC}
print(f'mean {mu:.2f} sd {sd:.2f} finisher>={FC:.2f}; finishers {len(fin)}')
pairs=[set(p) for p in itertools.combinations('WUBRG',2)]; pn=[''.join(sorted(p)) for p in pairs]
inpair=lambda n,p:set(nl[n]['colors'])<=p
# final pools (pool_ + final pick) from fra_step1 output; pair = best top-23 GIH WR (as fra_step2)
d=pd.read_csv(R+'analysis/out/final_pools_FRA.csv.gz')
d['trials']=d.event_match_wins+d.event_match_losses;d=d[d.trials>0].reset_index(drop=True)
pc=[c for c in d.columns if c.startswith('pool_') and c[5:] in nl];names=[c[5:] for c in pc]
P=d[pc].values.astype(int);g=np.array([gi(nl[n]) or mu-sd for n in names]);isf=np.array([n in fin for n in names])
def top23(row,mask,excl=False):
    m=mask&(row>0)&(~isf if excl else True);idx=np.where(m)[0];idx=idx[np.argsort(-g[idx])]
    s=np.concatenate([[g[i]]*row[i] for i in idx]) if len(idx) else np.array([])
    return s[:23].mean() if len(s) else np.nan
masks=[np.array([inpair(n,p) for n in names]) for p in pairs]
best=[];qual=[]
for row in P:
    q=[top23(row,m) for m in masks];b=int(np.nanargmax(q));best.append(b);qual.append(top23(row,masks[b],True))
d['pair']=[pn[b] for b in best];d['quality']=qual
# seen / taken among finishers, picks 1-28 (pack_number 0,1)
fc=sorted(fin);cols=['pack_card_'+n for n in fc]
h=pd.read_csv(R+'data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz',nrows=0).columns
print('finisher cols missing',[c for c in cols if c not in h])
cols=[c for c in cols if c in h];fc=[c[10:] for c in cols]
seen={};took={}
for ch in pd.read_csv(R+'data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz',usecols=['draft_id','pack_number','pick']+cols,chunksize=200000):
    ch=ch[ch.pack_number<=1]
    S=pd.DataFrame((ch[cols].values>0),columns=fc);S['d']=ch.draft_id.values
    T=pd.DataFrame({n:(ch.pick.values==n) for n in fc});T['d']=ch.draft_id.values
    for k,(s,t) in enumerate(((S,seen),(T,took))):
        for did,r in s.groupby('d').any().iterrows(): t[did]=t[did]|r if did in t else r
SN=pd.DataFrame(seen).T[fc];TK=pd.DataFrame(took).T[fc]
print('drafts with picks<=28',len(SN))
d=d[d.draft_id.isin(SN.index)].reset_index(drop=True)
SN=SN.loc[d.draft_id].values;TK=TK.loc[d.draft_id].values
ip=np.array([[inpair(n,set(p)) for n in fc] for p in d.pair])
d['seen']=(SN&ip).sum(1);d['took']=(TK&ip&SN).sum(1)
print('n drafts',len(d),'; seen 0/1/2+ %:',[round(x*100,1) for x in ((d.seen==0).mean(),(d.seen==1).mean(),(d.seen>=2).mean())])
e=d[d.seen>=2].copy();N=len(e);p0=(e.took==0).mean();se=np.sqrt(p0*(1-p0)/N)
print(f'2+ seen: N={N}; mean taken {e.took.mean():.2f}; P(took none)={p0*100:.1f}% [{(p0-1.96*se)*100:.1f},{(p0+1.96*se)*100:.1f}]')
e['y']=e.event_match_wins/e.trials;e['t1']=(e.took>=1).astype(int)
m=smf.glm('y ~ t1+quality+C(pair)+C(user_n_games_bucket)',e,family=sm.families.Binomial(),freq_weights=e.trials).fit(cov_type='HC1')
k=.55*.45*100;lo,hi=m.conf_int().loc['t1']
print(f'G = {m.params.t1*k:.2f} pts [{lo*k:.2f},{hi*k:.2f}] p={m.pvalues.t1:.3g}; share took 1+ {e.t1.mean()*100:.1f}%')
G,L=m.params.t1*k,lo*k
print('VERDICT:','PROPOSE TEST' if p0>=.30 and G>=1.5 and L>0 else 'STOP (for good)' if p0<.15 or G<.5 else 'IN BETWEEN (stop)')
