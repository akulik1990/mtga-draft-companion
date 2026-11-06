"""Video claims: (1) a Way's value rises with the number of OTHER empower-Jace cards in the deck (threshold ~5-7 for Healer);
(2) Ways stack badly past ~3. Rules and design choices in docs/video-test-empower-FRA.md.
Deck level: binomial GLM on deck wins (SEs clustered by draft), Premier decks. Within deck: tools/within_deck.py's method (deck-demeaned
linear probability, cards and lands in hand held fixed, bootstrap over decks) reimplemented here; tools/within_deck.py is NOT run or imported.
Replication: drafts split by parity of sorted draft id.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_empower.py"""
import gzip, json, os, re, sys
import numpy as np, pandas as pd, statsmodels.api as sm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{R}/analysis/out"; os.makedirs(OUT, exist_ok=True)
BASE = 0.55; K = 100 * BASE * (1 - BASE)  # WR points per logit unit at 55%
BOOT = 300; NPLAC = 40
rng = np.random.default_rng(0)
log = open(f"{OUT}/video_empower_log.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# ---------------- cards
raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
by = {}
for v in raw.values():
    n = v["name"].split(" // ")[0]
    if n in by and not int(v.get("isprimarycard", 1)): continue
    by[n] = v
gih = {n: v["deck_colors"]["All Decks"]["gihwr"] for n, v in by.items()}
island = {n for n, v in by.items() if "Land" in v["types"]}
WAYS = sorted(n for n in by if n.startswith("Way of the "))
EMP = sorted(n for n, v in by.items() if n not in WAYS and n not in island and "empower jace" in (v.get("oracle_text") or "").lower())
P("Ways", len(WAYS), WAYS); P("Empower (non-Way, non-land)", len(EMP), EMP)
json.dump(dict(ways=WAYS, empower=EMP), open(f"{OUT}/video_empower_lists.json", "w"), indent=1)
WS, ES = set(WAYS), set(EMP)
NAMED = ["Way of the Healer", "Way of the Mind Sculptor", "Way of the Paradox"]

# ---------------- deck level
decks = [d for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")) if d["format"] == "PremierDraft" and d["deck_games"] >= 1]
rows = []
for d in decks:
    c = {n.split(" // ")[0]: k for n, k in d["counts"].items()}
    sp = [(gih[n], k) for n, k in c.items() if n in gih and n not in island and gih[n] > 0]
    g = np.repeat([a for a, k in sp], [k for a, k in sp])
    r = dict(draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"], gb=d["games_bucket"],
             w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"], mean_g=g.mean(), top10=np.sort(g)[-10:].mean(),
             E=sum(c.get(n, 0) for n in ES), W=sum(c.get(n, 0) for n in WS), Wd=sum(1 for n in WS if c.get(n, 0)))
    for n in NAMED: r[n] = int(c.get(n, 0) > 0)
    rows.append(r)
df = pd.DataFrame(rows)
ids = sorted(df.draft.unique()); half = {x: i % 2 for i, x in enumerate(ids)}; df["half"] = df.draft.map(half)
P("Premier decks", len(df), "drafts", len(ids), "games", int(df.w.sum() + df.l.sum()))
P("E distribution", df.E.value_counts().sort_index().to_dict()); P("W distribution", df.W.value_counts().sort_index().to_dict())
EB = ["0-2", "3-4", "5-7", "8+"]
df["eb"] = pd.cut(df.E, [-1, 2, 4, 7, 99], labels=EB).astype(str)
df["wb"] = np.select([df.W == 0, df.W == 1, df.W == 2, df.W == 3], ["0", "1", "2", "3"], "4+")
for n in NAMED: df[n + "_eb"] = df.eb

def glm(d, terms, extra=()):
    """terms: dict name -> column array (design columns already built); returns fitted result."""
    X = pd.DataFrame(terms, index=d.index)
    ctr = pd.get_dummies(d[["lane"]].astype(str)).astype(float); ctr = ctr.iloc[:, 1:]
    g = pd.get_dummies(d.gb.astype(str), prefix="gb").astype(float).iloc[:, 1:]
    X = pd.concat([X, ctr, g, d[["mean_g", "top10", "skill", "splash"]]], axis=1)
    X = sm.add_constant(X.astype(float))
    y = np.column_stack([d.w, d.l])
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(d.draft)[0]))

def pts(res, names, w=None):
    """combine coefficient list (optionally signed weights) into WR points + 95% CI via the covariance"""
    w = np.ones(len(names)) if w is None else np.array(w, float)
    b = sum(wi * res.params[n] for wi, n in zip(w, names))
    V = res.cov_params().loc[names, names].values; se = float(np.sqrt(w @ V @ w))
    return K * b, K * (b - 1.96 * se), K * (b + 1.96 * se), 2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(b / se)))

out = []  # (analysis, subset, label, n, pts, lo, hi, p)
def rec(a, s, lab, n, t): out.append((a, s, lab, n) + tuple(t)); P(f"{a:28} {s:6} {lab:34} n={n:6} {t[0]:+6.2f} [{t[1]:+6.2f},{t[2]:+6.2f}] p={t[3]:.3f}")

subsets = dict(all=df, h0=df[df.half == 0], h1=df[df.half == 1])
# A. any Way x E bucket: cell effect vs (no Way, E 0-2) ... use W1 main effect + W1*bucket, E bucket mains
for s, d in subsets.items():
    d = d.copy(); d["W1"] = (d.W >= 1).astype(float)
    T = {"W1": d.W1}
    for b in EB[1:]: T["E" + b] = (d.eb == b).astype(float)
    for b in EB: T["WxE" + b] = d.W1 * (d.eb == b)   # WxE0-2 is the W1 effect at low density
    T.pop("W1")
    res = glm(d, T)
    for b in EB:
        n = int(((d.W >= 1) & (d.eb == b)).sum())
        rec("A Way present vs none", s, f"E {b}: Way(s) vs none (n with Way)", n, pts(res, ["WxE" + b]))
    for b in EB[1:]:
        rec("A interaction vs E 0-2", s, f"E {b} minus E 0-2", int(((d.W >= 1) & (d.eb == b)).sum()), pts(res, ["WxE" + b, "WxE0-2"], [1, -1]))
    # continuous: W1 x E (E centred at 4) slope per extra empower card
    d["Ec"] = d.E - 4
    res = glm(d, {"Ec": d.Ec, "W1xEc": (d.W >= 1) * d.Ec, "W1": (d.W >= 1).astype(float)})
    rec("A slope per empower card", s, "Way present x E (extra WR pts per card)", int((d.W >= 1).sum()), pts(res, ["W1xEc"]))
    rec("A slope per empower card", s, "E main effect, no Way", int((d.W == 0).sum()), pts(res, ["Ec"]))

# B. per Way: present x E bucket (low 0-4 vs high 5+, and 4 buckets in the pooled sample)
for nm in NAMED:
    for s, d in subsets.items():
        d = d.copy(); has = d[nm].astype(float)
        T = {"has": has}
        for b in EB[1:]: T["E" + b] = (d.eb == b).astype(float)
        for b in EB[1:]: T["hasxE" + b] = has * (d.eb == b)
        res = glm(d, T)
        rec("B " + nm[11:], s, "E 0-2: has vs not", int(((has == 1) & (d.E <= 2)).sum()), pts(res, ["has"]))
        for b in EB[1:]:
            rec("B " + nm[11:], s, f"E {b}: has vs not", int(((has == 1) & (d.eb == b)).sum()), pts(res, ["has", "hasxE" + b], [1, 1]))
        lowhigh = (d.E >= 5).astype(float)
        res = glm(d, {"has": has, "hi": lowhigh, "hasxhi": has * lowhigh})
        rec("B " + nm[11:], s, "interaction: E>=5 minus E<=4", int(((has == 1) & (d.E >= 5)).sum()), pts(res, ["hasxhi"]))

# C. Way count dummies (ref 0), quality controlled; E also controlled
for s, d in subsets.items():
    d = d.copy(); T = {}
    for b in ["1", "2", "3", "4+"]: T["W" + b] = (d.wb == b).astype(float)
    T["Ec"] = d.E - 4
    res = glm(d, T)
    for b in ["1", "2", "3", "4+"]: rec("C Way count vs 0", s, f"{b} Ways", int((d.wb == b).sum()), pts(res, ["W" + b]))
    rec("C marginal Way", s, "2 minus 1", int((d.wb == "2").sum()), pts(res, ["W2", "W1"], [1, -1]))
    rec("C marginal Way", s, "3 minus 2", int((d.wb == "3").sum()), pts(res, ["W3", "W2"], [1, -1]))
    rec("C marginal Way", s, "4+ minus 3", int((d.wb == "4+").sum()), pts(res, ["W4+", "W3"], [1, -1]))
    # per-Way slope split at 3: linear in min(W,3) and max(W-3,0)
    res = glm(d, {"lin": np.minimum(d.W, 3).astype(float), "over": np.maximum(d.W - 3, 0).astype(float), "Ec": d.E - 4})
    rec("C slope", s, "per Way, 0 to 3", int((d.W >= 1).sum()), pts(res, ["lin"]))
    rec("C slope", s, "per Way beyond 3", int((d.W > 3).sum()), pts(res, ["over"]))

# ---------------- within deck
P("loading games ...")
cols_keep = set(WAYS) | set(EMP)
hdr = pd.read_csv(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", nrows=0).columns.tolist()
spells = sorted(n for n in by if n not in island and f"opening_hand_{n}" in hdr)
oh = [f"opening_hand_{n}" for n in by if f"opening_hand_{n}" in hdr]; dr = [f"drawn_{n}" for n in by if f"drawn_{n}" in hdr]
landn = [n for n in island if f"opening_hand_{n}" in hdr]
use = ["draft_id", "won"] + oh + dr + [f"deck_{n}" for n in spells]
parts = []
for ch in pd.read_csv(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", usecols=use, chunksize=40000):
    H = ch[[f"opening_hand_{n}" for n in spells]].fillna(0).values.astype(np.int8) + ch[[f"drawn_{n}" for n in spells]].fillna(0).values.astype(np.int8)
    Lh = sum(ch[f"opening_hand_{n}"].fillna(0).values + ch[f"drawn_{n}"].fillna(0).values for n in landn)
    Dk = (ch[[f"deck_{n}" for n in spells]].fillna(0).values > 0)
    parts.append((ch.draft_id.values, ch.won.values.astype(float), H, Lh.astype(np.int8), Dk))
did = np.concatenate([p[0] for p in parts]); won = np.concatenate([p[1] for p in parts])
H = np.vstack([p[2] for p in parts]); Lh = np.concatenate([p[3] for p in parts]); Dk = np.vstack([p[4] for p in parts]); del parts
sidx = {n: i for i, n in enumerate(spells)}
P("games", len(won), "spell cols", len(spells))
wi = [sidx[n] for n in WAYS if n in sidx]; ei = [sidx[n] for n in EMP if n in sidx]
nW = H[:, wi].sum(1); nE = H[:, ei].sum(1); ncard = H.sum(1) + Lh
dW = Dk[:, wi].sum(1); dE = Dk[:, ei].sum(1)       # deck-level distinct counts (copies not in this file; distinct names)
udr = np.array(sorted(set(did))); hmap = {x: i % 2 for i, x in enumerate(udr)}
gh = np.array([hmap[x] for x in did])           # game-data halves: parity of sorted game-data draft ids
bidx = pd.read_csv(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", usecols=["build_index"]).build_index.values
codes, _ = pd.factorize(pd.Series(did) + "|" + pd.Series(bidx.astype(str)))   # deck = draft x build
dcode, _ = pd.factorize(did)                    # cluster = draft

def demean(X, g):
    """subtract group means (g integer codes)"""
    n = g.max() + 1; cnt = np.bincount(g, minlength=n).astype(float)
    out_ = X.astype(float).copy()
    for j in range(X.shape[1]): out_[:, j] -= (np.bincount(g, X[:, j], minlength=n) / np.maximum(cnt, 1))[g]
    return out_

def wfit(X, y, g, mask, boot=0, keep=None):
    """within-deck OLS on rows[mask]: deck-demeaned, returns coefficients (*100) for the first `keep` columns; bootstrap by deck."""
    m = np.where(mask)[0]; gg = pd.factorize(g[m])[0]
    cnt = np.bincount(gg); ok = cnt[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]; cl = pd.factorize(dcode[m])[0]
    A = demean(np.column_stack([X[m], y[m]]), gg)
    Xd, yd = A[:, :-1], A[:, -1]
    if Xd[:, 0].std() == 0 or len(np.unique(gg)) < 50: return None
    def solve(Xd, yd): return 100 * np.linalg.lstsq(Xd, yd, rcond=None)[0][:keep or Xd.shape[1]]
    est = solve(Xd, yd)
    if not boot: return est, None, gg.max() + 1
    order = np.argsort(cl, kind="stable"); starts = np.r_[0, np.cumsum(np.bincount(cl))]
    nc = cl.max() + 1; ng = gg.max() + 1; Xs, ys = Xd[order], yd[order]
    seg = [(starts[i], starts[i + 1]) for i in range(nc)]
    # bootstrap over drafts via sufficient statistics (X'X, X'y) per draft: fast
    XtX = np.array([Xs[a:b].T @ Xs[a:b] for a, b in seg]); Xty = np.array([Xs[a:b].T @ ys[a:b] for a, b in seg])
    bs = []
    for _ in range(boot):
        i = rng.integers(0, nc, nc)
        try: bs.append(100 * np.linalg.solve(XtX[i].sum(0), Xty[i].sum(0))[:keep or Xd.shape[1]])
        except np.linalg.LinAlgError: pass
    global LASTBS; LASTBS = np.array(bs)
    return est, np.std(bs, axis=0), ng

wrows = []
def wrec(a, s, lab, n, est, se, ref=None):
    z = None
    wrows.append((a, s, lab, n, est, se) + (() if ref is None else ref));
    P(f"{a:30} {s:6} {lab:40} decks={n:5} {est:+6.2f} ±{se:.2f}" + ("" if ref is None else f"  placebo {ref[0]:+.2f}±{ref[1]:.2f} z {(est-ref[0])/np.hypot(se, ref[1]):+.1f}"))

# W1. Way (any) in hand: slope of non-Way empower cards in hand; vs placebo sets of ordinary spells same size, held as often
Hs = H.astype(np.int16)
rate = {}
def placebo_set(inw_mask, k_rate_pool):
    pass
for s, hm in (("all", np.ones(len(won), bool)), ("h0", gh == 0), ("h1", gh == 1)):
    for lab, wcols in [("any Way", wi)] + [(n[11:], [sidx[n]]) for n in NAMED]:
        inhand = (H[:, wcols].sum(1) > 0) & hm
        # nE excludes other Ways; for named Way the other Ways are not in E
        X = np.column_stack([nE, ncard, Lh]).astype(float)
        r = wfit(X, won, codes, inhand, boot=BOOT, keep=1)
        if r is None: continue
        est, se, ng = r
        # placebo: random ordinary-spell sets, same size, deck-presence rate within 0.5-2x of empower cards
        pool_rate = Dk[inhand].mean(0); er = pool_rate[ei].mean()
        pool = [j for j in range(len(spells)) if j not in ei and j not in wi and 0.5 * er <= pool_rate[j] <= 2 * er]
        nul = []
        for _ in range(NPLAC):
            fk = rng.choice(pool, len(ei), replace=False)
            x = wfit(np.column_stack([H[:, fk].sum(1), ncard, Lh]).astype(float), won, codes, inhand)
            if x is not None: nul.append(x[0][0])
        wrec("W1 empower in hand | Way in hand", s, lab, ng, est[0], se[0], (float(np.mean(nul)), float(np.std(nul))))

# W2. interaction within deck: all games in decks that run >=1 Way (per Way: decks that run it): empower_in_hand x Way_in_hand
for s, hm in (("all", np.ones(len(won), bool)), ("h0", gh == 0), ("h1", gh == 1)):
    for lab, wcols in [("any Way", wi)] + [(n[11:], [sidx[n]]) for n in NAMED]:
        indeck = (Dk[:, wcols].sum(1) > 0) & hm
        wh = (H[:, wcols].sum(1) > 0).astype(float)
        X = np.column_stack([nE * wh, nE, wh, ncard, Lh]).astype(float)
        r = wfit(X, won, codes, indeck, boot=BOOT, keep=2)
        if r is None: continue
        est, se, ng = r
        er = Dk[indeck].mean(0)[ei].mean()
        pool = [j for j in range(len(spells)) if j not in ei and j not in wi and 0.5 * er <= Dk[indeck].mean(0)[j] <= 2 * er]
        nul = []
        for _ in range(NPLAC):
            fk = rng.choice(pool, len(ei), replace=False); pc = H[:, fk].sum(1)
            x = wfit(np.column_stack([pc * wh, pc, wh, ncard, Lh]).astype(float), won, codes, indeck)
            if x is not None: nul.append(x[0][0])
        wrec("W2 interaction E_hand x Way_hand", s, lab, ng, est[0], se[0], (float(np.mean(nul)), float(np.std(nul))))
        wrec("W2 (E_hand slope w/o Way in hand)", s, lab, ng, est[1], se[1])

# W3. deck density split: Way in hand, slope of empower in hand in decks with E>=5 vs E<=4 (distinct empower cards in deck)
for s, hm in (("all", np.ones(len(won), bool)), ("h0", gh == 0), ("h1", gh == 1)):
    for lab, wcols in [("any Way", wi)] + [(n[11:], [sidx[n]]) for n in NAMED]:
        for dn, dm in (("deck E<=4", dE <= 4), ("deck E>=5", dE >= 5)):
            inhand = (H[:, wcols].sum(1) > 0) & hm & dm
            r = wfit(np.column_stack([nE, ncard, Lh]).astype(float), won, codes, inhand, boot=BOOT, keep=1)
            if r is None: continue
            wrec("W3 empower in hand | Way in hand", s, f"{lab}, {dn}", r[2], r[0][0], r[1][0])
        # W3b: Way in hand vs not, by deck density (games in decks with that Way), cards/lands in hand fixed
        for dn, dm in (("deck E<=4", dE <= 4), ("deck E>=5", dE >= 5)):
            indeck = (Dk[:, wcols].sum(1) > 0) & hm & dm
            wh = (H[:, wcols].sum(1) > 0).astype(float)
            r = wfit(np.column_stack([wh, ncard, Lh]).astype(float), won, codes, indeck, boot=BOOT, keep=1)
            if r is None: continue
            wrec("W3b Way in hand vs not", s, f"{lab}, {dn}", r[2], r[0][0], r[1][0])

# W4. Ways in hand: 1, 2, 3+ vs 0 within deck, decks that run >=3 distinct Ways
for s, hm in (("all", np.ones(len(won), bool)), ("h0", gh == 0), ("h1", gh == 1)):
    indeck = (dW >= 3) & hm
    X = np.column_stack([nW == 1, nW == 2, nW >= 3, ncard, Lh]).astype(float)
    r = wfit(X, won, codes, indeck, boot=BOOT, keep=3)
    if r is None: continue
    est, se, ng = r
    for i, lab in enumerate(["1 Way in hand", "2 Ways in hand", "3+ Ways in hand"]):
        wrec("W4 Ways in hand vs 0 (decks >=3 Ways)", s, lab, ng, est[i], se[i])
    for lab, a_, b_ in (("2 minus 1", 1, 0), ("3+ minus 2", 2, 1)):
        wrec("W4 marginal Way in hand", s, lab, ng, est[a_] - est[b_], float(np.std(LASTBS[:, a_] - LASTBS[:, b_])))

pd.DataFrame(out, columns="analysis subset label n pts lo hi p".split()).to_csv(f"{OUT}/video_empower_deck.csv", index=False)
pd.DataFrame([(r + (None, None))[:6] + (r[6:8] if len(r) > 6 else (None, None)) for r in wrows], columns="analysis subset label n est se placebo_mean placebo_sd".split()).to_csv(f"{OUT}/video_empower_within.csv", index=False)
P("done")
