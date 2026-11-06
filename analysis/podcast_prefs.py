"""Podcast P1/P2/P3 (summary.md E item 1): do the learnt FRA per-card preferences (fitted.json prefs, strength z learnt from
top drafters' picks) carry a gold discount (P1), a flat-card preference (P3) or a late-value term (P2a) beyond GIH WR?
Run: nix shell nixpkgs#python3Packages.{pandas,numpy,statsmodels,scipy} -c python3 analysis/podcast_prefs.py
Writes data/tables/podcast_prefs_FRA.csv (card table). No engine run."""
import json, os
import numpy as np, pandas as pd, statsmodels.formula.api as smf

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = "WU UB BR RG WG WB UR BG WR UG".split()
K = 200  # shrinkage pseudo-samples for a pair's GIH WR towards the card's All Decks GIH WR
cr = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
fit = json.load(open(f"{R}/companion/advisor_v2/sets/FRA/fitted.json"))["prefs"]

cards = {}
for v in cr.values():
    n = v["name"].split(" // ")[0]; dc = v["deck_colors"]; a = dc["All Decks"]
    if n in cards and a["samples"] <= cards[n]["samples"]: continue
    m = a["gihwr"]; sh = []
    for p in PAIRS:
        e = dc.get(p)
        if e and e["samples"] > 0 and e["gihwr"] > 0:
            sh.append((e["samples"] * e["gihwr"] + K * m) / (e["samples"] + K))
        else:
            sh.append(m)
    cards[n] = dict(name=n, samples=a["samples"], gih=m, oh=a["ohwr"], gd=a["gdwr"], cmc=v["cmc"],
                    land="Land" in v["types"], gold=int(len(v["colors"]) >= 2), colourless=int(not v["colors"] and "Land" not in v["types"]),
                    rarity=v["rarity"], spread=float(np.std(sh, ddof=1)))
df = pd.DataFrame(cards.values())
for w in ("early", "all"):
    df[w] = [fit[w].get(n, np.nan) for n in df.name]
df = df[(df.gih > 0) & (df.oh > 0) & (df.gd > 0)].copy()
df["band"] = df.cmc.clip(1, 6)
df["late"] = df.gd - df.oh
df["late"] -= df.groupby("band").late.transform("mean")  # centred within cmc band
df["rarity"] = pd.Categorical(df.rarity, ["common", "uncommon", "rare", "mythic"])
def z(c, base): df[c + "_z"] = (df[c] - base[c].mean()) / base[c].std()
base = df[df.early.notna() & ~df.land]
for c in ("gih", "spread", "late"): z(c, base)
base = df[df.early.notna() & ~df.land]
df.to_csv(f"{R}/data/tables/podcast_prefs_FRA.csv", index=False)
print(f"cards in card_ratings {len(cards)}; with prefs: early {df.early.notna().sum()}, all {df['all'].notna().sum()}; lands among them {df[df.early.notna()].land.sum()}")
print(f"gold share (non-land, with early pref): {base.gold.mean():.3f}; gold n {base.gold.sum()}, colourless n {base.colourless.sum()}")
print("corr gold~spread_z", np.corrcoef(base.gold, base.spread_z)[0, 1].round(3), " late~gih", np.corrcoef(base.late_z, base.gih_z)[0, 1].round(3))

F = "~ gih_z + gold + colourless + C(rarity) + spread_z + late_z"
def fit_(y, sub, label, f=F, wt=None):
    d = sub[sub[y].notna()]
    m = (smf.wls(y + f, d, weights=d.samples) if wt else smf.ols(y + f, d)).fit(cov_type="HC3")
    ci = m.conf_int()
    print(f"\n## {label}  y={y} n={int(m.nobs)} R2={m.rsquared:.3f}")
    for t in m.params.index[1:]:
        print(f"  {t:28s} {m.params[t]:+.3f}  SE {m.bse[t]:.3f}  CI [{ci.loc[t,0]:+.3f}, {ci.loc[t,1]:+.3f}]  p={m.pvalues[t]:.3f}")
nl = df[~df.land]
for y in ("early", "all"):
    fit_(y, nl, "main, no lands")
    fit_(y, df, "with lands (+ land flag)", F + " + land")
    fit_(y, nl, "WLS by samples", wt=1)
    fit_(y, nl[nl.samples >= 2000], "samples >= 2000")
    fit_(y, nl, "no spread term (gold alone)", "~ gih_z + gold + colourless + C(rarity) + late_z")
    fit_(y, nl, "no gold term (spread alone)", "~ gih_z + colourless + C(rarity) + spread_z + late_z")
print("\nsample quantiles", nl.samples.quantile([0, .1, .25, .5, .75, 1]).round(0).tolist(), " n>=2000:", (nl.samples >= 2000).sum())

# ---- harness part: gold share of the human pick, P1 picks 1-5 (train drafts). v2's top-1 identity needs a per-pick log
# (analysis/replay_log.py output); none exists under data/Temp, the *.base.json files hold only ranks, so v2's share is not computed.
import gzip
gold = {}
for v in cr.values(): gold[v["name"].split(" // ")[0]] = int(len(v["colors"]) >= 2)
land = {v["name"].split(" // ")[0] for v in cr.values() if "Land" in v["types"]}
dr = json.load(gzip.open(f"{R}/data/Temp/replay/FRA.top600.train.drafts.json.gz"))
print("\n## P1 picks 1-5, train drafts:", len(dr), "drafts; pref SD (early, non-land):", round(base.early.std(), 3))
for lo, hi in ((1, 5), (1, 1), (6, 10)):
    h, p, nb = [], [], 0
    for d in dr:
        for pk, pi, name, pack, pool in d["picks"]:
            if pk == 1 and lo <= pi <= hi and name in gold and name not in land:
                cs = [c for c in set(pack) if c in gold and c not in land]
                h.append(gold[name]); p.append(np.mean([gold[c] for c in cs]))
    h, p = np.array(h), np.array(p); n = len(h); se = np.sqrt(h.var(ddof=1) / n)
    print(f"  picks {lo}-{hi}: n={n}  human gold share {h.mean():.3f} (SE {se:.3f})  mean gold share of pack {p.mean():.3f}  diff {h.mean()-p.mean():+.3f}")
