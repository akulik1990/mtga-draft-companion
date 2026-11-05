import pickle,gzip,json,numpy as np,pandas as pd,statsmodels.api as sm
R='/mnt/projects/mtga-draft-companion/data/'
cards=pd.read_csv(R+'tables/FRA/2_cards.csv').set_index('name')
cardd=cards.to_dict('index')
games=pickle.load(open(R+'Temp/context_games_FRA.pkl','rb'))
SB=pickle.load(open(R+'Temp/sideboards_FRA.pkl','rb'))
decks=json.load(gzip.open(R+'tables/FRA/1_decks.json.gz','rt'))
first={};order=[];W={};G={}
for x in games:
    k=(x[0],x[1],x[2])
    if k not in first: order.append(k);first[k]=x;W[k]=0;G[k]=0
    elif x[3]<first[k][3]: first[k]=x
    W[k]+=int(x[10]);G[k]+=1
assert len(order)==len(decks) and all((k[0],k[1])==(d['format'],d['draft']) for k,d in zip(order,decks))
cols='WUBRG'
def isrm(n):
    c=cardd.get(n); return c is not None and not c['land'] and c['role_removal']>0 and c['gihwr']>0
def castable(n,lane):
    c=cardd[n]; return all((c['col_'+x]==0) or x in lane for x in cols)
rows=[]
for k,d in zip(order,decks):
    f=first[k]
    if f[5] not in ('100','500') or not 2<=len(d['lane'])<=3: continue
    lane=d['lane']+d['splash']; side=SB.get(k,{})
    pl=[cardd[n]['gihwr'] for n,c in d['counts'] for _ in range(c) if isrm(n)] if isinstance(d['counts'],list) else [cardd[n]['gihwr'] for n,c in d['counts'].items() for _ in range(c) if isrm(n)]
    items=d['counts'].items() if isinstance(d['counts'],dict) else d['counts']
    spells=[(n,c) for n,c in items if n in cardd and not cardd[n]['land']]
    avg=np.average([cardd[n]['gihwr'] for n,c in spells if cardd[n]['gihwr']>0],weights=[c for n,c in spells if cardd[n]['gihwr']>0])
    lf=[cardd[n]['gihwr'] for n in side if isrm(n) and castable(n,lane)]
    rows.append(dict(fmt=k[0],draft=k[1],lane=d['lane'],skill=float(f[4]),rank=f[6],gb=f[5],w=W[k],n=G[k],nrm=len(pl),
        worst=min(pl) if pl else np.nan,best=max(lf) if lf else np.nan,nleft=len(lf),avg=avg))
df=pd.DataFrame(rows);print(len(df))
df['gap']=df.best-df.worst
sub=df[(df.nrm.between(3,5))&(df.nleft>0)].copy()
print('low-removal with left',len(sub),'upgrade',(sub.gap>0).sum())
sub['up']=(sub.gap>0).astype(int)
def wr(g): return 100*g.w.sum()/g.n.sum()
sub['gapbin']=pd.cut(sub.gap,[-99,-5,0,2,5,99])
print(sub.groupby(['nrm','gapbin'],observed=True).apply(lambda g:pd.Series(dict(n=len(g),wr=wr(g),skill=g.skill.mean(),worst=g.worst.mean(),avg=g.avg.mean()))).round(2))
print(sub.groupby('up').apply(lambda g:pd.Series(dict(n=len(g),wr=wr(g),skill=g.skill.mean(),worst=g.worst.mean(),best=g.best.mean(),avg=g.avg.mean()))).round(2))
print('corr gap~worst',sub[['gap','worst','best','skill','avg']].corr().round(2))
sub['wt']=pd.qcut(sub.worst,3,labels=['weak','mid','strong'])
print(sub.groupby(['wt','up'],observed=True).apply(lambda g:pd.Series(dict(n=len(g),wr=wr(g),skill=g.skill.mean(),avg=g.avg.mean()))).round(2))
# regression
def fit(formula,data,label):
    import statsmodels.formula.api as smf
    d=data.copy(); d['p']=d.w/d.n
    m=smf.glm(formula,d,family=sm.families.Binomial(),freq_weights=d.n).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.draft)[0]})
    b,s=m.params['up'],m.bse['up']; print(f"{label}: up {24.5*b:+.2f} SE {24.5*s:.2f}")
base='p ~ up + C(nrm) + C(fmt)+C(rank)+C(gb)+skill+C(lane)'
fit(base,sub,'basic (no card quality)')
fit(base+'+avg',sub,'+ deck mean GIH WR')
fit(base+'+avg+worst',sub,'+ deck mean GIH WR + worst played')
fit(base+'+avg+worst+best',sub,'+ worst + best (gap split)')
# terciles
for t,g in sub.groupby('wt',observed=True):
    fit('p ~ up + C(nrm)+C(fmt)+C(rank)+skill+C(lane)+avg',g,f'tercile {t}')
# placebo: gap defined with random best-left
rng=np.random.default_rng(0)
print('--- extra')
print(sub.groupby('up').nleft.mean())
fit(base+'+avg+worst+C(nleft.clip(upper=4))',sub,'+avg+worst+nleft')
fit(base+'+avg+worst+best+C(nleft.clip(upper=4))',sub,'+avg+worst+best+nleft')
big=sub[sub.gap>2].copy(); 
# dose-response: continuous gap among up decks
u=sub[sub.up==1].copy(); u['p']=u.w/u.n
import statsmodels.formula.api as smf
m=smf.glm('p ~ gap + worst+avg+C(nrm)+C(fmt)+C(rank)+skill',u,family=sm.families.Binomial(),freq_weights=u.n).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.draft)[0]})
print('dose among upgrade decks, per GIH point of gap',24.5*m.params['gap'],24.5*m.bse['gap'])
# placebo: shuffle best within nrm
rng=np.random.default_rng(1); res=[]
for _ in range(100):
    s=sub.copy(); s['best']=s.groupby('nrm').best.transform(lambda x: rng.permutation(x.values)); s['up']=(s.best>s.worst).astype(int)
    d=s.copy();d['p']=d.w/d.n
    m=smf.glm('p ~ up + C(nrm)+C(fmt)+C(rank)+skill+avg+worst',d,family=sm.families.Binomial(),freq_weights=d.n).fit()
    res.append(24.5*m.params['up'])
print('placebo shuffled best, mean/sd',np.mean(res),np.std(res))
