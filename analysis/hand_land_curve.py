import pandas as pd, numpy as np, statsmodels.api as sm
from scipy.stats import hypergeom
B='/mnt/projects/mtga-draft-companion/analysis/'
out=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.append(s)
d=pd.read_csv(B+'out/hand_games_FRA.csv.gz')
d['deck']=d.draft_id+'_'+d.build_index.astype(str)
n0=len(d)
d['on_play']=d.on_play.astype(int)
d=d[d.groupby('deck').deck.transform('size')>=4].copy()
P('rows',n0,'-> decks>=4 games',len(d),'decks',d.deck.nunique(),'drafts',d.draft_id.nunique())
CAV=("CAVEATS: (a) mull=0 hands were KEPT (conditional on keeping); (b) Arena Bo1 hand smoothing may apply, hand dist not hypergeometric; "
"(c) value of the extra spell in a 16-land deck NOT priced; (d) no data near 10-14 lands; (e) 17Lands tracker users only.")
P(CAV)
LB=['le1','l2','l4','l5','ge6']
def bins(x):
    return pd.DataFrame({'le1':(x<=1),'l2':(x==2),'l4':(x==4),'l5':(x==5),'ge6':(x>=6)}).astype(float)
def fit(df,xcols,label):
    X=df[xcols].copy(); X=X.loc[:,X.sum()>0] if True else X
    dropped=[c for c in xcols if c not in X.columns]
    y=df.won.astype(float)
    g=df.deck
    Xd=X-X.groupby(g).transform('mean'); yd=y-y.groupby(g).transform('mean')
    m=sm.OLS(yd,Xd).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.draft_id)[0]})
    return m,dropped
def curve(df,hl,label,extra):
    D=df.copy()
    bb=bins(hl.loc[D.index]); D=pd.concat([D,bb],axis=1)
    D['opp1']=(D.opp_num_mulligans==1).astype(float); D['opp2']=(D.opp_num_mulligans>=2).astype(float)
    xc=LB+['on_play','opp1','opp2']+extra
    m,dr=fit(D,xc,label)
    P(f'\n== {label} == n={len(D)} decks={D.deck.nunique()} (empty bins dropped: {dr})')
    P('bin  games  coef_pts  CI95_lo  CI95_hi')
    ci=m.conf_int()
    for k in LB+['on_play','opp1','opp2']+extra:
        if k in m.params.index:
            c=m.params[k]*100;l,h=ci.loc[k]*100
            cnt=int(D[k].sum()) if k in D else ''
            P(f'{k:8s} {cnt:7} {c:7.2f} {l:7.2f} {h:7.2f}')
        else: P(f'{k:8s} {int(D[k].sum()):7} (empty)')
    P('ref bin = 3 lands; base games:',int((hl.loc[D.index]==3).sum()))
    return m
K=d[d.num_mulligans==0]
m1=curve(K,K.hand_lands,'FIT1 kept hands, hand_lands',[])
Da=d.copy(); Da['mul1']=(Da.num_mulligans==1).astype(float); Da['mul2']=(Da.num_mulligans>=2).astype(float)
m2=curve(Da,Da.hand_lands,'FIT2 all games, hand_lands + mulligan dummies',['mul1','mul2'])
curve(K,K.hand_lands+K.hand_lc,'FIT1b kept hands, effective lands (hand_lands+hand_lc)',[])

# D16
def binshare(x):
    b=bins(x); return b.mean()
def hyp(K_):
    p=hypergeom(40,K_,7); pr=[p.cdf(1),p.pmf(2),p.pmf(4),p.pmf(5),p.sf(5)]
    return pd.Series(pr,index=LB)
def d16(mask16,mask17,lab):
    k16=K[mask16(K)];k17=K[mask17(K)]
    dP=(binshare(k16.hand_lands)-binshare(k17.hand_lands))
    return dP,len(k16),len(k17)
def report(dP,lab,extra=''):
    idx=[k for k in LB if k in m1.params.index]
    w=m1.params[idx]; V=m1.cov_params().loc[idx,idx]; dp=dP[idx]
    D=float((dp*w).sum())*100; se=float(np.sqrt(dp.values@V.values@dp.values))*100
    P(f'{lab}: D16={D:.2f} pts [{D-1.96*se:.2f}, {D+1.96*se:.2f}] {extra}')
    return D,D-1.96*se,D+1.96*se
P('\n== D16 (win-rate pts/game, 16 vs 17 lands; delta method, cluster cov of w(k), P treated as fixed) ==')
res={}
for lab,f in [('all decks',lambda x:x.deck_lc>=0),('deck_lc>=2',lambda x:x.deck_lc>=2)]:
    dP,n16,n17=d16(lambda x:(x.deck_lands==16)&f(x),lambda x:(x.deck_lands==17)&f(x),lab)
    P(f'[{lab}] kept games n16={n16} n17={n17}')
    P(' P16-P17 empirical:',dP.round(4).to_dict())
    res[lab]=report(dP,f'EMPIRICAL (primary) {lab}')
    h=hyp(16)-hyp(17)
    P(' hypergeometric UNCONDITIONAL (sensitivity) P16-P17:',h.round(4).to_dict())
    report(h,f'HYPERGEOMETRIC unconditional (sensitivity) {lab}')
# descriptive
P('\n== Descriptive: mulligan rate, kept hands share <=2 or >=5 lands ==')
def desc(g,lab):
    r=g.groupby(lab).apply(lambda x:pd.Series({'games':len(x),'mull_rate':(x.num_mulligans>=1).mean(),
        'kept_n':(x.num_mulligans==0).sum(),
        'kept_le2_or_ge5':((x.num_mulligans==0)&((x.hand_lands<=2)|(x.hand_lands>=5))).sum()/max((x.num_mulligans==0).sum(),1)}),include_groups=False)
    P(r.round(4).to_string())
dd=d[d.deck_lands.isin([15,16,17,18])]
P('by deck_lands (decks>=4 games)');desc(dd,'deck_lands')
d2=d[d.deck_lands.isin([16,17])].copy(); d2['lc_b']=d2.deck_lc.clip(upper=2)
P('by deck_lands x deck_lc (0,1,2=2+)');desc(d2,['deck_lands','lc_b'])
D,lo,hi=res['all decks']
v=('fewer lands costs; keep 17' if (D<=-0.5 and hi<0) else 'fewer lands helps in opening hands; propose allowing 16 for pools with 2+ landcyclers' if (D>=0.3 and lo>0) else 'land count is a wash in opening hands; no change')
P(f'\nVERDICT (all decks, empirical): D16={D:.2f} [{lo:.2f},{hi:.2f}] -> {v}')
open(B+'out/hand_land_curve.txt','w').write('\n'.join(out))
