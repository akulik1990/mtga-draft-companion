import pandas as pd, numpy as np, time
t=time.time()
f='data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz'
h=pd.read_csv(f,nrows=0).columns
meta=[c for c in ['draft_id','rank','event_match_wins','event_match_losses','pack_number','pick_number','pick','user_n_games_bucket','user_game_win_rate_bucket'] if c in h]
print('missing meta',[c for c in ['user_n_games_bucket','user_game_win_rate_bucket','rank'] if c not in h])
pool=[c for c in h if c.startswith('pool_')]
best={};n=0
for ch in pd.read_csv(f,usecols=meta+pool,chunksize=200000):
    n+=len(ch)
    ch['k']=ch.pack_number*100+ch.pick_number
    i=ch.groupby('draft_id').k.idxmax()
    for r in ch.loc[i].itertuples(index=False): pass
    sub=ch.loc[i]
    for did,row in zip(sub.draft_id,sub.to_dict('records')):
        if did not in best or row['k']>best[did]['k']: best[did]=row
df=pd.DataFrame(best.values())
for i,r in df.iterrows(): pass
pk='pool_'+df['pick']
for c in pool:
    df[c]=df[c]+(pk==c).astype(int)
print('unmatched pick',(~pk.isin(pool)).sum())
df.to_csv('analysis/out/final_pools_FRA.csv.gz',index=False)
print('rows',n,'drafts',len(df),'sec',round(time.time()-t),'maxpack/pick',df.pack_number.max(),df.pick_number.max(),df[pool].sum(axis=1).describe().to_dict())
print(df[['rank','user_n_games_bucket','user_game_win_rate_bucket']].head(), df.rank.unique()[:10])
