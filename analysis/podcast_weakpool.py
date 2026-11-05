"""Pre-registered test: do extra colours pay more in weak pools? (LR ep. 224). Run:
nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_weakpool.py
"""
import gzip, itertools, json, os, sys
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import norm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0.55
cr = json.load(open(f'{R}/data/Sets/FRA_PremierDraft_All_Data.json'))['card_ratings']
card = {}
for c in cr.values():
    g = c['deck_colors']['All Decks']['gihwr']
    n = c['name'].split(' // ')[0]
    if n not in card or c.get('isprimarycard', 1):
        card[n] = dict(col=frozenset(c['colors']), g=g if g > 0 else np.nan, land='Land' in c['types'])

decks = pd.DataFrame(json.load(gzip.open(f'{R}/analysis/out/decks_FRA.json.gz')))
pools = pd.read_csv(f'{R}/analysis/out/final_pools_FRA.csv.gz')
pc = [c for c in pools.columns if c.startswith('pool_')]
print('decks', len(decks), 'pools', len(pools), 'formats', decks.format.value_counts().to_dict())
pools = pools.drop_duplicates('draft_id').set_index('draft_id')

pairs = [frozenset(p) for p in itertools.combinations('WUBRG', 2)]
spell = {c[5:]: card[c[5:].split(' // ')[0]] for c in pc if c[5:].split(' // ')[0] in card and not card[c[5:].split(' // ')[0]]['land']}
unk = [c for c in pc if c[5:].split(' // ')[0] not in card]
print('pool cols unmatched to ratings', len(unk), unk[:5])
names = list(spell); cols = [spell[n]['col'] for n in names]; gih = np.array([spell[n]['g'] for n in names])
ok = ~np.isnan(gih)  # ponytail: unrated cards (gihwr 0) ignored for strength

def strength(row):
    best = np.nan
    for p in pairs:
        m = np.array([c <= p for c in cols]) & ok & (row > 0)
        v = np.repeat(gih[m], row[m])
        if len(v) >= 23:
            s = np.sort(v)[-23:].mean(); best = s if np.isnan(best) or s > best else best
    return best

P = pools[['pool_' + n for n in names]].to_numpy()
ps = pd.Series([strength(r) for r in P], index=pools.index, name='pool_strength')

def deckfeat(d):
    cnt = {k.split(' // ')[0]: v for k, v in d['counts'].items()}
    sp = {k: v for k, v in cnt.items() if k in card and not card[k]['land']}
    cols_ = set().union(*[card[k]['col'] for k in sp]) if sp else set()
    q = [(card[k]['g'], v) for k, v in sp.items() if not np.isnan(card[k]['g'])]
    qual = sum(g * v for g, v in q) / sum(v for _, v in q)
    clean = d['splash'] == '' and len(d['lane']) == 2  # deck type from splash/lane fields only (per-card colours misclassify hybrids etc.)
    return clean, qual

f = decks.apply(deckfeat, axis=1, result_type='expand'); decks['clean'] = f[0]; decks['quality'] = f[1]
prem = decks[decks.format == 'PremierDraft']
print('funnel: all', len(decks), '| Premier', len(prem), '| Premier with splash', round((prem.splash != '').mean(), 3), '| Premier lane>=3', int((prem.lane.str.len() >= 3).sum()))
decks = decks.join(ps, on='draft'); print('funnel: with pool strength', decks.pool_strength.notna().sum())
decks = decks.dropna(subset=['pool_strength', 'quality', 'skill']); print('funnel: + quality/skill', len(decks))
decks = decks[decks.deck_games > 0].copy(); print('funnel: + games>0', len(decks))
decks['extended'] = (~decks.clean).astype(int)
decks['q'] = pd.qcut(decks.pool_strength, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
decks['loss'] = decks.deck_games - decks.deck_wins
decks['gb'] = decks.games_bucket.astype(str)
print('n analysed', len(decks), 'extended share', round(decks.extended.mean(), 3), 'games', int(decks.deck_games.sum()))
print(decks.groupby('q', observed=True).agg(n=('extended', 'size'), ext=('extended', 'mean'), games=('deck_games', 'sum'),
      wr=('deck_wins', 'sum'), strength=('pool_strength', 'mean')).assign(wr=lambda x: x.wr / x.games).to_string())

def fit(ctrl_quality):
    rhs = 'extended*C(q) + skill + C(gb) + C(lane)' + (' + quality' if ctrl_quality else '')
    return smf.glm(f'deck_wins + loss ~ {rhs}'.replace('deck_wins + loss', 'I(deck_wins)+I(loss)'), data=decks,
                   family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(decks.draft)[0]})

def contrast(m, w):
    L = np.zeros(len(m.params)); 
    for k, v in w.items(): L[list(m.params.index).index(k)] = v
    b = L @ m.params.values; se = np.sqrt(L @ m.cov_params().values @ L); return b, se

def pts(b, se):  # log-odds effect -> win-rate points at 55%
    f = lambda x: 100 * (expit(logit(BASE) + x) - BASE)
    return f(b), f(b - 1.96 * se), f(b + 1.96 * se), 2 * norm.sf(abs(b / se))

out = {}
for label, cq in [('without deck quality control (primary)', False), ('with deck quality control', True)]:
    m = fit(cq); rows = []; res = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        w = {'extended': 1}
        if q != 'Q1': w[f'extended:C(q)[T.{q}]'] = 1
        res[q] = contrast(m, w); rows.append((q,) + pts(*res[q]))
    d = {'extended': 0, 'extended:C(q)[T.Q4]': -1}  # Q1 minus Q4
    b, se = contrast(m, {'extended:C(q)[T.Q4]': -1}); inter = pts(b, se)
    out[label] = (rows, inter)
    print('\n' + label); print(pd.DataFrame(rows, columns=['q', 'pts', 'lo', 'hi', 'p']).round(3).to_string(index=False))
    print('Q1 minus Q4 (pts, lo, hi, p):', [round(x, 3) for x in inter])

rows, inter = out['without deck quality control (primary)']
p1 = rows[0]; sig_better_q1 = p1[4] < .05 and p1[1] > 0
if sig_better_q1 and inter[3] < .05 and inter[0] > 0: v = 'propose an engine test: weak-pool mode (earlier fixers / splash in weak pools)'
elif all(r[4] < .05 and r[1] < 0 for r in rows): v = 'no change, adding colours does not pay'
else: v = 'no evidence, no change'
print('\nVERDICT:', v)
json.dump(dict(n=len(decks), verdict=v, results={k: dict(rows=r, inter=i) for k, (r, i) in out.items()}),
          open(f'{R}/analysis/out/weakpool_FRA.json', 'w'), default=float, indent=1)
