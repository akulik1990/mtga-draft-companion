"""Do outcomes side with v2 or top players on flagged cards? Output: analysis/out/trophy_cards.txt"""
import gzip, json, collections, numpy as np
R = 'analysis/out/'
D = json.load(gzip.open(R + 'decks_FRA.json.gz'))
OVER = ["Murmuring Volume","Stingerquill Annex","Archive Arbiter","Rescue Girl, First Responder","Konstrari Annex","Twinned Vision","Geist of Saint Thalia","Ruric Thar, Biomagus","Keeper of the Quiet Hour","Fatehold Chronologist","Bestial Incursion","Tetsuko Umezawa, Fugitive"]
UNDER = ["Precise Redaction","Extrapolate the Impossible","Kiora of Salt and Sand","Edgar, Ancient Bloodlord","Mabel, Valley Hero","Koth, the Geomancer","Transformative Commons","Warrior's Blades","Stingerquill Annex","Restore with Empathy","Winter, Team Player","Cruel Calculations"]
FLAG = list(dict.fromkeys(OVER + UNDER))
GROUPS = {'all': lambda s: s >= 0, 'good': lambda s: s >= 0.60, 'average': lambda s: (s >= 0.52) & (s <= 0.58), 'weak': lambda s: s < 0.52}
rng = np.random.default_rng(20261106)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

def prep(fmt_prem):
    ds = [d for d in D if (d['format'] == 'PremierDraft') == fmt_prem and d['deck_games'] > 0]
    cards = collections.Counter()
    for d in ds:
        for c, n in d['counts'].items(): cards[c] += 1
    cs = [c for c in cards]
    return ds, cards
def mats(ds, cardlist):
    idx = {c: i for i, c in enumerate(cardlist)}
    X = np.zeros((len(ds), len(cardlist)), bool)
    for r, d in enumerate(ds):
        for c, n in d['counts'].items():
            if c in idx and n > 0: X[r, idx[c]] = True
    return X
def logor(x, t):
    a = (x & t).sum(); b = (x & ~t).sum(); c = (~x & t).sum(); d = (~x & ~t).sum()
    a, b, c, d = [v + 0.5 for v in (a, b, c, d)]
    lo = np.log(a * d / (b * c)); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return lo, se

# ---- Premier
ds, freq = prep(True)
BASIC = {'Plains','Island','Swamp','Mountain','Forest'}
cand = [c for c in freq if c not in BASIC and c not in FLAG and freq[c] >= 100]
allc = FLAG + cand
X = mats(ds, allc)
skill = np.array([d['skill'] for d in ds]); wins = np.array([d['deck_wins'] for d in ds], float)
games = np.array([d['deck_games'] for d in ds], float); pred = np.array([d['pred'] for d in ds])
lane = np.array([d['lane'] for d in ds]); draft = np.array([d['draft'] for d in ds])
troph = wins >= 7
P(f"Premier decks {len(ds)}; trophies {troph.sum()}; flagged-card frequency (decks): " + ', '.join(f"{c}={X[:,i].sum()}" for i, c in enumerate(FLAG)))
fl = {c: i for i, c in enumerate(FLAG)}
# placebo: for each of 12 slots, a random candidate among the 15 nearest in frequency to the matched flagged card
cf = np.array([X[:, i].sum() for i in range(len(allc))])
def placebo_set():
    s = []
    for c in OVER:  # 12 slots matched to the OVER list frequencies
        f = cf[fl[c]]
        near = sorted(range(len(FLAG), len(allc)), key=lambda j: abs(cf[j] - f))[:15]
        s.append(int(rng.choice([j for j in near if j not in s] )))
    return s
PL = [placebo_set() for _ in range(200)]
used = sorted({j for s in PL for j in s})
P("Placebo: 200 sets of 12 cards (seed 20261106), each slot random among 15 non-flagged cards nearest in deck frequency to one overrated-list card.")
P("Skill groups: good>=0.60, average 0.52-0.58, weak<0.52 (0.58-0.60 excluded from the three).")

def wr_effects(mask, cols):
    """FWL: y=wins/games ~ card + lane + skill dummies + pred; WLS weights=games; cluster by draft."""
    idx = np.where(mask)[0]
    y = (wins / games)[idx]; w = games[idx]
    L = np.unique(lane[idx]); S = np.unique(np.round(skill[idx] / 0.02))
    cols_ = [np.ones(len(idx)), pred[idx]]
    cols_ += [(lane[idx] == l).astype(float) for l in L[1:]]
    cols_ += [(np.round(skill[idx] / 0.02) == s).astype(float) for s in S[1:]]
    Z = np.column_stack(cols_); sw = np.sqrt(w)
    Q, _ = np.linalg.qr(Z * sw[:, None])
    ys = y * sw; yr = ys - Q @ (Q.T @ ys)
    g = np.unique(draft[idx], return_inverse=True)[1]; G = g.max() + 1
    res = {}
    for j in cols:
        xs = X[idx, j].astype(float) * sw
        if xs.sum() < 1e-9 or X[idx, j].sum() < 30: res[j] = (np.nan, np.nan); continue
        xr = xs - Q @ (Q.T @ xs)
        b = (xr @ yr) / (xr @ xr)
        e = yr - b * xr
        sc = np.zeros(G); np.add.at(sc, g, xr * e)
        se = np.sqrt((sc ** 2).sum() * G / (G - 1)) / (xr @ xr)
        res[j] = (b * 100, se * 100)
    return res

LOR = {}; WR = {}
for gname, f in GROUPS.items():
    m = f(skill)
    # trophy OR
    LOR[gname] = {}
    for j in list(range(len(FLAG))) + used:
        LOR[gname][j] = logor(X[m, j], troph[m])
    WR[gname] = wr_effects(m, list(range(len(FLAG))) + used)
    P(f"group {gname}: decks {m.sum()}, trophies {(troph&m).sum()}, drafts {len(set(draft[m]))}")

P("\n=== Per card (Premier). OR = trophy-deck odds of containing card vs other decks, [95% CI]; WR = win-rate points beyond `pred`, lane, skill; (z)")
for gname in GROUPS:
    P(f"\n-- {gname}")
    P("| Card | list | decks | OR [CI] | WR eff (z) |")
    P("|---|---|---|---|---|")
    m = GROUPS[gname](skill)
    for c in FLAG:
        j = fl[c]; lo, se = LOR[gname][j]; b, s = WR[gname][j]
        lst = ('O' if c in OVER else '') + ('U' if c in UNDER else '')
        wr = f"{b:+.2f} ({b/s:+.1f})" if s == s else 'n/a(<30)'
        P(f"| {c} | {lst} | {X[m,j].sum()} | {np.exp(lo):.2f} [{np.exp(lo-1.96*se):.2f},{np.exp(lo+1.96*se):.2f}] | {wr} |")

def boot_mean(v, B=4000):
    v = np.array(v); v = v[~np.isnan(v)]
    return v.mean(), [np.mean(rng.choice(v, len(v))) for _ in range(B)], len(v)
P("\n=== List means over cards (Stingerquill Annex is on both lists), vs placebo")
for metric, name in (('lor', 'mean log-OR (trophy presence)'), ('wr', 'mean WR effect, pts')):
    P(f"\n{name}")
    P("| group | overrated | underrated | placebo mean [2.5,97.5 of 200 sets] | over-minus-under [boot CI over cards] |")
    P("|---|---|---|---|---|")
    for gname in GROUPS:
        def val(j): return LOR[gname][j][0] if metric == 'lor' else WR[gname][j][0]
        vo = [val(fl[c]) for c in OVER]; vu = [val(fl[c]) for c in UNDER]
        mo, bo, no = boot_mean(vo); mu, bu, nu = boot_mean(vu)
        pl = [np.nanmean([val(j) for j in s]) for s in PL]
        d = np.array(bo) - np.array(bu)
        P(f"| {gname} | {mo:+.3f} (n{no}) | {mu:+.3f} (n{nu}) | {np.mean(pl):+.3f} [{np.percentile(pl,2.5):+.3f},{np.percentile(pl,97.5):+.3f}] | {mo-mu:+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] |")
    if metric == 'lor': pass
# good-minus-weak for list means
P("\nGood-minus-weak (list mean WR effect, bootstrap over cards)")
for nm, L in (('overrated', OVER), ('underrated', UNDER)):
    a = np.array([WR['good'][fl[c]][0] for c in L]); b = np.array([WR['weak'][fl[c]][0] for c in L])
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    bs = [ (lambda i: a[i].mean() - b[i].mean())(rng.integers(0, len(a), len(a))) for _ in range(4000)]
    P(f"{nm}: {a.mean()-b.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] n{len(a)}")

# ---- other formats check: trophy OR, all skills
ds2 = [d for d in D if d['format'] != 'PremierDraft']
P(f"\n=== Check: non-Premier formats ({collections.Counter(d['format'] for d in ds2)}), trophy OR all skills")
if ds2:
    X2 = mats(ds2, FLAG); t2 = np.array([d['deck_wins'] >= 7 for d in ds2])
    P(f"decks {len(ds2)}, trophies {t2.sum()}")
    lo = {c: logor(X2[:, i], t2) for i, c in enumerate(FLAG)}
    for c in FLAG: P(f"{c}: n={X2[:,fl[c]].sum()} OR {np.exp(lo[c][0]):.2f} [{np.exp(lo[c][0]-1.96*lo[c][1]):.2f},{np.exp(lo[c][0]+1.96*lo[c][1]):.2f}]")
    P(f"mean logOR overrated {np.mean([lo[c][0] for c in OVER]):+.3f}, underrated {np.mean([lo[c][0] for c in UNDER]):+.3f}")
open(R + 'trophy_cards.txt', 'w').write('\n'.join(out) + '\n')
