import pandas as pd, numpy as np, statsmodels.api as sm, io, sys
d=pd.read_csv('out/hand_games_FRA.csv.gz')
d['deck']=d.draft_id+'_'+d.build_index.astype(str)
d=d[d.num_mulligans==0].copy()
d=d[d.groupby('deck').won.transform('size')>=4].copy()
out=io.StringIO()
def P(*a): print(*a,file=out)
P(f"games {len(d)}, decks {d.deck.nunique()}, drafts {d.draft_id.nunique()}; on_play dtype {d.on_play.dtype}; median games/deck {d.groupby('deck').size().median()}")
def dm(df,lands='lands'):
    X=pd.DataFrame(index=df.index)
    l=df.hand_lands+(df.hand_lc if lands=='eff' else 0)
    X['lands<=1']=(l<=1);X['lands=2']=(l==2);X['lands=4']=(l==4);X['lands=5']=(l==5);X['lands>=6']=(l>=6)
    c=df.hand_cheap
    X['cheap=0']=(c==0);X['cheap=1']=(c==1);X['cheap>=3']=(c>=3)
    X['removal=1']=(df.hand_removal==1);X['removal>=2']=(df.hand_removal>=2)
    X['big=1']=(df.hand_big==1);X['big>=2']=(df.hand_big>=2)
    X['on_play']=df.on_play;X['opp_mull>=1']=(df.opp_num_mulligans>=1)
    return X.astype(float)
def fit(df,X=None,lands='lands'):
    df=df[df.groupby('deck').won.transform('size')>=2]
    X=(dm(df,lands) if X is None else X.loc[df.index])
    X=X.loc[:,X.std()>0]
    g=df.deck
    Xd=X-X.groupby(g).transform('mean'); y=df.won-df.groupby('deck').won.transform('mean')
    r=sm.OLS(y,Xd).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.draft_id)[0]})
    n=X.sum()
    t=pd.DataFrame({'pts':r.params*100,'lo':r.conf_int()[0]*100,'hi':r.conf_int()[1]*100,'se':r.bse*100,'n':n})
    return t,r,len(df)
def show(t,title):
    P('\n'+title);P(t[['pts','lo','hi','n']].round(2).to_string())
m,_,N=fit(d); show(m,f'MAIN (N={N}; refs: lands=3, cheap=2 (modal), removal=0, big=0, opp_mull=0; on_play = play vs draw)')
def cmp(ta,tb,na,nb,label):
    P(f'\n{label}: {na} vs {nb}')
    rows=[]
    for k in ta.index.intersection(tb.index):
        df_=ta.se[k]**2+tb.se[k]**2; z=(ta.pts[k]-tb.pts[k])/np.sqrt(df_)
        diff=ta.pts[k]-tb.pts[k];h=1.96*np.sqrt(df_)
        rows.append((k,ta.pts[k],tb.pts[k],diff,diff-h,diff+h,z))
    r=pd.DataFrame(rows,columns=['term',na,nb,'diff','lo','hi','z']).set_index('term').round(2)
    P(r[r.z.abs()>2].to_string() if (r.z.abs()>2).any() else '(no |z|>2)')
    P(f'terms compared {len(r)}; expect ~{0.05*len(r):.1f} by chance')
pl,_,npl=fit(d[d.on_play]);dr,_,ndr=fit(d[~d.on_play])
P(f'\n(a) play N={npl}, draw N={ndr}')
cmp(pl,dr,'play','draw','(a) split by on_play')
band=pd.cut(d.user_game_win_rate_bucket,[-1,0.45-1e-9,0.6-1e-9,2],labels=['low','mid','high'])
P(f'\n(b) skill bands (bucket includes own game; NaN dropped {band.isna().sum()}): {band.value_counts().to_dict()}')
bt={b:fit(d[band==b])[0] for b in ['low','mid','high']}
for a,b in [('low','mid'),('low','high'),('mid','high')]: cmp(bt[a],bt[b],a,b,'(b)')
e,_,_=fit(d,lands='eff'); show(e,'(c) effective lands (hand_lands+hand_lc) bins')
# notable checks
P('\nNOTABLE CHECKS (main model)')
for k in ['removal=1','removal>=2']: P(f'{k}: {m.pts[k]:.2f} [{m.lo[k]:.2f},{m.hi[k]:.2f}] pts vs 0; per card for 2+: {m.pts[k]/ (2 if k=="removal>=2" else 1):.2f}')
def alt(df):
    X=dm(df); X=X.drop(columns=['lands=2','lands<=1','cheap=0','cheap=1','cheap>=3'])
    l=df.hand_lands
    X['lands<=2']=(l<=2).astype(float); X['cheap>=2']=(df.hand_cheap>=2).astype(float)
    X['lands>=4']=X.pop('lands=4')*0+(l>=4).astype(float); X=X.drop(columns=['lands=5','lands>=6'])
    return X
ap,_,_=fit(d[d.on_play],alt(d[d.on_play])); ad,_,_=fit(d[~d.on_play],alt(d[~d.on_play]))
P('\nplay-vs-draw contrasts (alt model: lands<=2 vs 3 with lands>=4 separate; cheap>=2 vs 0/1)')
for k in ['cheap>=2','lands<=2']:
    diff=ap.pts[k]-ad.pts[k];se=np.hypot(ap.se[k],ad.se[k])
    P(f'{k}: play {ap.pts[k]:.2f} [{ap.lo[k]:.2f},{ap.hi[k]:.2f}], draw {ad.pts[k]:.2f} [{ad.lo[k]:.2f},{ad.hi[k]:.2f}], diff {diff:.2f} [{diff-1.96*se:.2f},{diff+1.96*se:.2f}] z={diff/se:.2f}')
P('\nCAVEATS: num_mulligans=0 = kept hands (player-selected); Arena Bo1 hand smoothing, hands not hypergeometric; within-deck association quasi-random except keep decisions; tracker users only. Skill bucket includes the game itself; mid/high gap (0.595-0.60) empty at 0.01 resolution. Subgroup SEs treated independent (split samples share no games, but share clusters/decks in skill splits only partly).')
open('out/hand_spells.txt','w').write(out.getvalue());print(out.getvalue())
