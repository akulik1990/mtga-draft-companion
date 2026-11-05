import pandas as pd, numpy as np, itertools, statsmodels.api as sm, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
d=pd.read_csv('analysis/out/final_pools_FRA.csv.gz')
cards=pd.read_csv('data/tables/FRA/2_cards.csv').drop_duplicates('name').set_index('name')
pc=[c for c in d.columns if c.startswith('pool_')]
names=[c[5:] for c in pc]
un=[n for n in names if n not in cards.index]; print('unmatched',un)
ok=[n for n in names if n in cards.index]
print('meta vals', d.user_n_games_bucket.value_counts().sort_index().to_dict(), d['rank'].value_counts().to_dict())
print('wlcheck',d.event_match_wins.max(),d.event_match_losses.max(), 'zero trials',((d.event_match_wins+d.event_match_losses)==0).sum())
c=cards.loc[ok]
P=d[['pool_'+n for n in ok]].values.astype(int)
g=c.gihwr.fillna(0).values.astype(float)
nl=(c.land==0).values
has=nl&(g>0)
mu,sd=g[has].mean(),g[has].std()
print('gihwr mean sd',mu,sd)
gf=np.where(g>0,g,mu-sd)
cols=['W','U','B','R','G']
cf=np.stack([c['col_'+x].values for x in cols],1).astype(int)
pairs=list(itertools.combinations(range(5),2))
cmc=c.cmc.values; rem=(c.role_removal.fillna(0).values>0); dual=(c.land_multi.fillna(0).values==1)&(c.land.values==1)
fin_ok=(cmc>=5)&(g>=mu+sd)
order=np.argsort(-gf)
res=[]
for row in P:
    bestm=-1
    for pi,(a,b) in enumerate(pairs):
        mask=np.ones(5,bool);mask[[a,b]]=False
        cast=nl&(cf[:,mask].sum(1)==0)
        idx=[i for i in order if cast[i] and row[i]>0]
        sel=[];
        for i in idx:
            sel+= [i]*row[i]
            if len(sel)>=23: break
        sel=np.array(sel[:23],dtype=int)
        if len(sel)==0: continue
        m=gf[sel].mean()
        if m>bestm: bestm,bs,bp,bmask=m,sel,pi,cast
        bm_=mask if m==bestm else None
    a,b=pairs[bp]; mask=np.ones(5,bool);mask[[a,b]]=False
    dc=sum(row[i] for i in np.where(dual&(cf[:,mask].sum(1)==0))[0])
    res.append((bestm,rem[bs].sum(),fin_ok[bs].sum(),dc,bp,len(bs)))
r=pd.DataFrame(res,columns=['quality','removal','finishers','duals','pair','n'],index=d.index)
d=pd.concat([d[['draft_id','rank','event_match_wins','event_match_losses','user_n_games_bucket','user_game_win_rate_bucket']],r],axis=1)
d['trials']=d.event_match_wins+d.event_match_losses
d=d[d.trials>0].copy()
print('deck sizes <23',(d.n<23).sum())
d.to_csv('analysis/out/draft_features_FRA.csv.gz',index=False)
print(d[['quality','removal','finishers','duals']].describe().round(2))
print('share removal<=4',(d.removal<=4).mean().round(3),'>=9',(d.removal>=9).mean().round(3))
print('corr quality',d[['quality','removal','finishers','duals']].corr().round(3).iloc[0].to_dict())
d['rem']=d.removal.clip(3,10);d['fin']=d.finishers.clip(0,4);d['dul']=d.duals.clip(0,6)
d['rbin']=pd.cut(d.removal,[-1,4,6,8,99],labels=['le4','5-6','7-8','9+'])
d['rbin']=pd.Categorical(d.rbin.astype(str),['5-6','le4','7-8','9+'])
d['y']=d.event_match_wins/d.trials
k=0.55*0.45*100
def fit(df,f):
    return smf.glm('y ~ '+f,df,family=sm.families.Binomial(),freq_weights=df.trials).fit(cov_type='HC1')
base='quality+C(pair)+C(user_n_games_bucket)+C(rank)'
def rep(m,tag,df):
    out=[]
    for v in m.params.index:
        if v in('rem','fin','dul') or v.startswith('C(rbin'):
            lo,hi=m.conf_int().loc[v]
            out.append(f'{tag} {v:22s} {m.params[v]*k:6.3f} [{lo*k:6.3f},{hi*k:6.3f}] p={m.pvalues[v]:.3g}')
    print('\n'.join(out))
def run(df,label):
    print('==',label,'n drafts',len(df),'trials',int(df.trials.sum()))
    m0=fit(df,base+'+rem+fin+dul');m1=fit(df,base+'+rem+fin+dul+C(user_game_win_rate_bucket)')
    rep(m0,'M0',df);rep(m1,'M1',df)
    b0=fit(df,base+'+C(rbin)+fin+dul');b1=fit(df,base+'+C(rbin)+fin+dul+C(user_game_win_rate_bucket)')
    rep(b0,'M0bins',df);rep(b1,'M1bins',df)
run(d,'all')
big=d[d.user_n_games_bucket>=100]
run(big,'ngames>=100') if len(big) else print('no bucket>=100')
