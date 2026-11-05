"""Pre-registered test: do decks short of two-drops underperform beyond card quality, and did
their drafters pass playable two-drops late? Rule in docs/podcast-test-twodrops-FRA.md.
Run: nix shell python3 with pandas numpy statsmodels scipy; python3 analysis/podcast_twodrop.py"""
import gzip, json, os
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0.55
BUCKETS = ["0-2", "3", "4", "5", "6+"]; REF = "5"

# ---- cards: name (front face) -> (is_creature, cmc, colors, gihwr); one row per name, most samples wins
cr = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
cards = {}
for v in cr.values():
    n = v["name"].split(" // ")[0]; a = v["deck_colors"]["All Decks"]
    if n not in cards or a["samples"] > cards[n]["s"]:
        cards[n] = dict(s=a["samples"], land="Land" in v["types"], cre="Creature" in v["types"],
                        cmc=v["cmc"], col=set(v["colors"]), g=a["gihwr"])

# ---- Part A
decks = json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz"))
rows = []
for i, d in enumerate(decks):
    sp = [(cards[n], k) for n, k in d["counts"].items() if n in cards and not cards[n]["land"] and cards[n]["g"] > 0]
    g = np.repeat([c["g"] for c, k in sp], [k for c, k in sp])
    td = sum(k for c, k in sp if c["cre"] and c["cmc"] == 2)
    rows.append(dict(i=i, draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"],
                     gb=d["games_bucket"], w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"],
                     mean_g=g.mean(), top10=np.sort(g)[-10:].mean(), td=td))
df = pd.DataFrame(rows)
df["bucket"] = np.select([df.td <= 2, df.td == 3, df.td == 4, df.td == 5], ["0-2", "3", "4", "5"], "6+")
n_tot = len(df); print("decks", n_tot, "drafts", df.draft.nunique())
print(df.bucket.value_counts().reindex(BUCKETS).to_string())
y = np.column_stack([df.w, df.l])
X = pd.get_dummies(df[["bucket", "lane", "gb"]].astype(str)).astype(float)
X = X.drop(columns=[f"bucket_{REF}"])
X = X.drop(columns=[X.columns[X.columns.str.startswith("lane_")][0], "gb_1"])
X = sm.add_constant(pd.concat([X, df[["mean_g", "top10", "skill", "splash"]]], axis=1))
res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds=dict(groups=pd.factorize(df.draft)[0]))
lg = np.log(BASE / (1 - BASE)); ip = lambda x: 1 / (1 + np.exp(-x))
A = []
for b in BUCKETS:
    if b == REF: A.append((b, int((df.bucket == b).sum()), 0, 0, 0, np.nan)); continue
    c = f"bucket_{b}"; bt, se, p = res.params[c], res.bse[c], res.pvalues[c]
    A.append((b, int((df.bucket == b).sum()), 100 * (ip(lg + bt) - BASE), 100 * (ip(lg + bt - 1.96 * se) - BASE),
              100 * (ip(lg + bt + 1.96 * se) - BASE), p))
tA = pd.DataFrame(A, columns=["two_drops", "n_decks", "pts_vs_5", "ci_lo", "ci_hi", "p"])
print(tA.round(3).to_string(index=False))
short = [r.two_drops for r in tA.itertuples() if r.p < 0.05 and r.pts_vs_5 < 0]
print("significant shortfall buckets:", short)
for c in ["mean_g", "top10", "skill"]: print(c, round(res.params[c], 4), "p", round(res.pvalues[c], 4))

# ---- Part B (only if A shows a shortfall in a low bucket)
tB = None
low = [b for b in short if b in ("0-2", "3", "4")]
if low:
    td2 = {n: c for n, c in cards.items() if c["cre"] and c["cmc"] == 2 and not c["land"] and c["g"] > 0}
    cols = [f"pack_card_{n}" for n in td2]
    hdr = pd.read_csv(f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz", nrows=0).columns
    cols = [c for c in cols if c in hdr]; print("two-drop pack columns", len(cols), "of", len(td2))
    want = set(df.draft); parts = []
    for ch in pd.read_csv(f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz",
                          usecols=["draft_id", "pack_number", "pick_number", "pick"] + cols, chunksize=200000):
        ch = ch[ch.draft_id.isin(want)]
        ch = ch[ch.pack_number * 14 + ch.pick_number >= 15]; parts.append(ch)
    pk = pd.concat(parts); print("late pick rows", len(pk))
    # lane median GIH of two-drops actually played (copy-weighted)
    med = {}
    for lane, gdf in df.groupby("lane"):
        v = []
        for d in (decks[i] for i in gdf.i):
            v += [td2[n]["g"] for n, k in d["counts"].items() if n in td2 for _ in range(k)]
        med[lane] = np.median(v)
    def passed_by_draft(lane):
        ok = [c for c in cols if td2[c[10:]]["col"] <= set(lane) and td2[c[10:]]["g"] >= med[lane]]
        pass_ = pk[ok].values > 0
        picked = np.stack([(pk["pick"] == c[10:]).values for c in ok], 1)
        hit = (pass_ & ~picked).any(1)
        return pd.Series(hit, index=pk.index).groupby(pk.draft_id).any()
    cache = {l: passed_by_draft(l) for l in df.lane.unique()}
    df["passed"] = [bool(cache[l].get(d, False)) for l, d in zip(df.lane, df.draft)]
    df = df[df.draft.isin(set(pk.draft_id))]
    def prop(s):
        k, n = int(s.sum()), len(s); lo, hi = sm.stats.proportion_confint(k, n, method="wilson"); return k, n, k / n, lo, hi
    out = []
    for b in low + [REF]:
        out.append((b,) + prop(df.passed[df.bucket == b]))
    tB = pd.DataFrame(out, columns=["bucket", "k", "n", "share", "ci_lo", "ci_hi"])
    print(tB.round(4).to_string(index=False))
    rk, rn = tB.k.iloc[-1], tB.n.iloc[-1]
    for r in tB.iloc[:-1].itertuples():
        z, p = sm.stats.proportions_ztest([r.k, rk], [r.n, rn]); print(r.bucket, "vs ref: diff", round(r.share - rk / rn, 4), "z", round(z, 2), "p", round(p, 4))
        
os.makedirs(f"{R}/analysis/out", exist_ok=True)
tA.to_csv(f"{R}/analysis/out/twodrop_partA.csv", index=False)
if tB is not None: tB.to_csv(f"{R}/analysis/out/twodrop_partB.csv", index=False)
