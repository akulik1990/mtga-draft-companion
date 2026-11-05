import re,json,itertools,collections,numpy as np,pandas as pd
R='/mnt/projects/mtga-draft-companion/'
cr=json.load(open(R+'data/Sets/FRA_PremierDraft_All_Data.json'))['card_ratings']
cc=pd.read_csv(R+'data/tables/FRA/2_cards.csv')
rem=dict(zip(cc.name,cc.role_removal>0))
def gi(c,k='All Decks'):
    return (c['deck_colors'].get(k) or {}).get('gihwr',0) or 0
nl={g:c for g,c in cr.items() if 'Land' not in c['types']}
v=np.array([gi(c) for c in nl.values() if gi(c)>0]); mu,sd=v.mean(),v.std(ddof=1)
FC=mu+sd; NC=mu+.5*sd
print(f'mean {mu:.2f} sd {sd:.2f} finisher>={FC:.2f} near>={NC:.2f}')
steps={}
for l in open(R+'data/Temp/live/FRA-quickdraft-2026-11-02.Player.log',errors='ignore'):
    if '\\"DraftPack\\"' not in l: continue
    m=re.search(r'"Payload":"(\{.*?\})"',l)
    if not m: continue
    p=json.loads(m.group(1).encode().decode('unicode_escape'))
    if p.get('DraftStatus')!='PickNext': continue
    steps.setdefault((p['PackNumber'],p['PickNumber']),p)
keys=sorted(steps); print('steps',len(keys),'last pack size',len(steps[keys[-1]]['DraftPack']))
S=[steps[k] for k in keys]
picks=[]
for i,p in enumerate(S):
    if i+1<len(S):
        d=collections.Counter(map(str,S[i+1]['PickedCards']))-collections.Counter(map(str,p['PickedCards']))
        picks.append(list(d)[0] if len(d)==1 else None)
    else:
        picks.append(str(p['DraftPack'][0]) if len(p['DraftPack'])==1 else None)
print('unresolved picks',picks.count(None))
pool=[str(x) for x in S[-1]['PickedCards']]+([picks[-1]] if picks[-1] else [])
print('pool size',len(pool))
miss=set()
seen=collections.defaultdict(lambda:{'n':0,'first':None})
for i,p in enumerate(S):
    for g in map(str,p['DraftPack']):
        if g not in cr: miss.add(g); continue
        s=seen[g]; s['n']+=1
        if s['first'] is None: s['first']=i
print('ungrouped ids in packs',sorted(miss),'in pool',[g for g in pool if g not in cr])
taken=set(g for g in picks if g)|set(pool)
def cast(g,pr): return all(c in pr for c in cr[g]['colors'])
def info(g): c=cr[g]; return f"{c['name']} (cmc{c['cmc']}, {gi(c):.1f}, first p{seen[g]['first']}, {'TAKEN' if g in pool else 'passed'})"
def lane(pr):
    ws=sorted([gi(cr[g]) for g in pool if g in cr and g in nl and cast(g,pr)],reverse=True)[:23]
    ws+= [45.0]*(23-len(ws)); return np.mean(ws)
rows=[];det={}
for pr in map(''.join,itertools.combinations('WUBRG',2)):
    sg=[g for g in seen if g in nl and cast(g,pr)]
    fin=[g for g in sg if cr[g]['cmc']>=5 and gi(cr[g])>=FC]
    near=[g for g in sg if cr[g]['cmc']>=5 and NC<=gi(cr[g])<FC]
    rm=[g for g in sg if rem.get(cr[g]['name'])]
    f28=[g for g in fin if seen[g]['first']<=28]
    rows.append(dict(pair=pr,fin=len(fin),fin_by28=len(f28),fin_taken=sum(g in pool for g in fin),near=len(near),
      rem_names=len({cr[g]['name'] for g in rm}),rem_copies=sum(seen[g]['n'] for g in rm),rem_taken=sum(g in pool for g in rm),lane=round(lane(pr),2)))
    det[pr]=(fin,near)
T=pd.DataFrame(rows);print(T.to_string(index=False))
ub=T[T.pair=='UB'].lane.iloc[0]
ok=T[(T.pair!='UB')&(T.fin_by28>=2)&((T.lane-ub).abs()<=1.0)]
a=T[(T.pair!='UB')&(T.fin_by28>=2)]
print('UB lane',ub,'; (a)+(b):',ok.pair.tolist(),'; (a) only:',a.pair.tolist())
print('\nFinishers/near seen (any pair):')
allf=set(g for f,n in det.values() for g in f+n)
for g in sorted(allf,key=lambda g:seen[g]['first']):
    i=seen[g]['first']; print(' ',info(g),'| pair-castable',[p for p in det if g in det[p][0]+det[p][1]],'| his pick then:',cr[picks[i]]['name'] if picks[i] in cr else picks[i], gi(cr[picks[i]]) if picks[i] in cr else '')
f=pd.read_csv(R+'analysis/out/draft_features_FRA.csv.gz');f=f[f.n==23];print('\nn==23 drafts',len(f))
for c in['finishers','removal']:
    print(c,'>=1',round((f[c]>=1).mean()*100,1),'>=2',round((f[c]>=2).mean()*100,1),'==0',round((f[c]==0).mean()*100,1),'mean',round(f[c].mean(),2),'; >=4 removal' if c=='removal' else '',round((f[c]>=4).mean()*100,1) if c=='removal' else '')
print('\nLyra check:',[(g,cr[g]['name'],seen[g]['n'],seen[g]['first'],g in pool) for g in cr if cr[g]['name'].startswith('Lyra')])
