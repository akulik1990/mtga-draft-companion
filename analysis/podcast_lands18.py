"""B1 + B7: 18 lands vs 17 (and cantrips). Rule (summary B1, as D16): >= +0.3 pts/game propose, <= -0.5 keep 17, else wash.
(a) D18 with the opening-hand method of hand_land_curve.py (deck FE on kept hands, empirical hand-land distributions priced).
(b) deck GLM 18 vs 17 as land_count_check.py, strata by number of cmc>=5 spells (0-2 / 3+).
(c) cantrip stratum: cmc<=2 spells whose oracle text has "draw a card", scry or surveil; 16 vs 17 and 18 vs 17.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_lands18.py"""
import gzip, json, re
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import hypergeom

R = Path(__file__).resolve().parents[1]; out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---------- (a) opening-hand method
d = pd.read_csv(R / "analysis/out/hand_games_FRA.csv.gz")
d["deck"] = d.draft_id + "_" + d.build_index.astype(str); d["on_play"] = d.on_play.astype(int)
d = d[d.groupby("deck").deck.transform("size") >= 4].copy()
K = d[d.num_mulligans == 0].copy()
LB = ["le1", "l2", "l4", "l5", "ge6"]
bins = lambda x: pd.DataFrame({"le1": x <= 1, "l2": x == 2, "l4": x == 4, "l5": x == 5, "ge6": x >= 6}).astype(float)
B = bins(K.hand_lands); K = pd.concat([K, B], axis=1)
K["opp1"] = (K.opp_num_mulligans == 1).astype(float); K["opp2"] = (K.opp_num_mulligans >= 2).astype(float)
xc = [c for c in LB if K[c].sum() > 0] + ["on_play", "opp1", "opp2"]
g = K.deck; X = K[xc] - K[xc].groupby(g).transform("mean"); y = K.won.astype(float); y = y - y.groupby(g).transform("mean")
m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(K.draft_id)[0]})
idx = [c for c in LB if c in m.params.index]; w = m.params[idx]; V = m.cov_params().loc[idx, idx]
def D(dP, lab, extra=""):
    dp = dP[idx]; v = float((dp * w).sum()) * 100; se = float(np.sqrt(dp.values @ V.values @ dp.values)) * 100
    P(f"{lab}: D = {v:+.2f} pts/game [{v-1.96*se:+.2f}, {v+1.96*se:+.2f}] {extra}"); return v
P(f"(a) kept hands {len(K)}, decks {K.deck.nunique()}; hand-land weights vs 3 lands (pts): " + ", ".join(f"{k} {w[k]*100:.2f}" for k in idx))
hyp = lambda n: pd.Series(list(map(float, [hypergeom(40, n, 7).cdf(1), hypergeom(40, n, 7).pmf(2), hypergeom(40, n, 7).pmf(4),
                                          hypergeom(40, n, 7).pmf(5), hypergeom(40, n, 7).sf(5)])), index=LB)
masks = {"all decks": K.deck_lands >= 0, "deck_big>=3": K.deck_big >= 3, "deck_big<=2": K.deck_big <= 2, "deck_lc>=1": K.deck_lc >= 1}
for lab, f in masks.items():
    for a, b in ((18, 17), (16, 17)):
        ka, kb = K[(K.deck_lands == a) & f], K[(K.deck_lands == b) & f]
        if len(ka) < 300: P(f"[{lab}] {a} vs {b}: kept games {len(ka)} vs {len(kb)} (too few {a}-land)"); continue
        dP = B.loc[ka.index].mean() - B.loc[kb.index].mean()
        D(dP, f"[{lab}] D{a} vs {b} (empirical; kept games {len(ka)} / {len(kb)})")
        if lab == "all decks": D(hyp(a) - hyp(b), f"[{lab}] D{a} vs {b} hypergeometric (sensitivity)")
n18 = d[d.deck_lands == 18]; P(f"18-land decks (4+ games): {n18.deck.nunique()}, games {len(n18)}; with deck_big>=3: {n18[n18.deck_big>=3].deck.nunique()}")
P("note: D18 prices the hand distribution only; the spell lost for the 18th land (and flood late) is not priced.")

# ---------- (b), (c) deck GLM
by = {c["name"].split(" // ")[0]: c for c in json.load(open(R / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
BASIC = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
def is_cantrip(c):
    t = (c.get("oracle_text") or "").lower()
    return "Land" not in c["types"] and c["cmc"] <= 2 and bool(re.search(r"draw a card|scry|surveil", t))
cant = sorted(n for n, c in by.items() if is_cantrip(c))
P(f"\n(c) derived cantrip list ({len(cant)}): " + "; ".join(cant))
def build(skill_min):
    rows = []
    for dk in json.load(gzip.open(R / "analysis/out/decks_FRA.json.gz", "rt")):
        if dk["skill"] < skill_min or len(dk["lane"]) != 2 or dk["deck_games"] < 1: continue
        lands = n = big = ct = 0; q = []
        for name, k in dk["counts"].items():
            c = by.get(name.split(" // ")[0]); n += k
            if name in BASIC: lands += k; continue
            if not c: continue
            if "Land" in c["types"]: lands += k; continue
            if c["cmc"] >= 5: big += k
            if is_cantrip(c): ct += k
            gg = (c["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
            if gg: q += [gg] * k
        if n != 40 or lands not in (16, 17, 18) or not q: continue
        rows.append(dict(draft=dk["draft"], lane=dk["lane"], skill=dk["skill"], gb=dk["games_bucket"], w=dk["deck_wins"], g=dk["deck_games"],
                         lands=lands, big=big, ct=ct, quality=float(np.mean(q))))
    return pd.DataFrame(rows)
def glm(s, var, lab, alpha):
    s = s.copy(); s["x"] = (s.lands == var).astype(int); s["y"] = s.w / s.g; nx = int(s.x.sum())
    if nx < 30 or (s.lands == 17).sum() < 30: P(f"  {lab:28} {var} vs 17: n {nx} (too few)"); return
    mm = smf.glm("y ~ x + quality + skill + C(gb) + C(lane)", s, family=sm.families.Binomial(), freq_weights=s.g).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s.draft)[0]})
    b, se, p = mm.params["x"], mm.bse["x"], mm.pvalues["x"]; f = 0.55 * 0.45 * 100
    P(f"  {lab:28} {var} vs 17: {b*f:+.2f} pts [{(b-1.96*se)*f:+.2f}, {(b+1.96*se)*f:+.2f}] p {p:.3f}; n {nx} vs {int((s.lands==17).sum())}")
for smin in (0.60, 0.0):
    df = build(smin)
    P(f"\n(b) deck GLM, skill >= {smin}: decks {len(df)}; lands 16/17/18: " + "/".join(str(int((df.lands == k).sum())) for k in (16, 17, 18)))
    for lab, s in (("all", df), ("cmc>=5 spells 0-2", df[df.big <= 2]), ("cmc>=5 spells 3+", df[df.big >= 3])):
        glm(s[s.lands != 16], 18, lab, 0.05)
    P(f"(c) cantrip strata, skill >= {smin}")
    for lab, s in (("cantrips 0", df[df.ct == 0]), ("cantrips 1", df[df.ct == 1]), ("cantrips 2+", df[df.ct >= 2]), ("cantrips 1+", df[df.ct >= 1])):
        glm(s[s.lands != 18], 16, lab, 0.05); glm(s[s.lands != 16], 18, lab, 0.05)
    if smin == 0.60: df[df.lands == 18].big.value_counts().sort_index().pipe(lambda v: P("  18-land decks by cmc>=5 count:", v.to_dict()))
(R / "analysis/out/podcast_lands18.txt").write_text("\n".join(out))
