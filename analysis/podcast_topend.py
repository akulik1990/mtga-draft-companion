"""B2 top-end saturation, FRA. Deck GLM on wins/losses (binomial, SEs clustered by draft); dummies for the count of non-land
spells with cmc >= 5 (0-1, 2 = reference, 3, 4, 5+) plus a separate count of cmc >= 7. Controls as podcast_twodrop.py.
Replication: same model on two halves of the drafts (draft_id hash parity). Rule in docs/podcast-test-topend-FRA.md.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_topend.py"""
import gzip, json, os, zlib
import numpy as np, pandas as pd, statsmodels.api as sm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0.55; BUCKETS = ["0-1", "2", "3", "4", "5+"]; REF = "2"
cr = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
cards = {}
for v in cr.values():
    n = v["name"].split(" // ")[0]; a = v["deck_colors"]["All Decks"]
    if n not in cards or a["samples"] > cards[n]["s"]:
        cards[n] = dict(s=a["samples"], land="Land" in v["types"], cmc=v["cmc"], g=a["gihwr"])
decks = json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt"))
rows = []
for d in decks:
    sp = [(cards[n], k) for n, k in d["counts"].items() if n in cards and not cards[n]["land"] and cards[n]["g"] > 0]
    g = np.repeat([c["g"] for c, k in sp], [k for c, k in sp])
    rows.append(dict(draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"], gb=d["games_bucket"],
                     w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"], mean_g=g.mean(), top10=np.sort(g)[-10:].mean(),
                     big=sum(k for c, k in sp if c["cmc"] >= 5), c7=sum(k for c, k in sp if c["cmc"] >= 7)))
df = pd.DataFrame(rows)
df["bucket"] = np.select([df.big <= 1, df.big == 2, df.big == 3, df.big == 4], BUCKETS[:4], "5+")
lg = np.log(BASE / (1 - BASE)); ip = lambda x: 1 / (1 + np.exp(-x)); pts = lambda b: 100 * (ip(lg + b) - BASE)

def fit(s):
    X = pd.get_dummies(s[["bucket", "lane", "gb"]].astype(str)).astype(float).drop(columns=f"bucket_{REF}")
    X = X.drop(columns=[X.columns[X.columns.str.startswith("lane_")][0], X.columns[X.columns.str.startswith("gb_")][0]])
    X = sm.add_constant(pd.concat([X, s[["c7", "mean_g", "top10", "skill", "splash"]]], axis=1))
    return sm.GLM(np.column_stack([s.w, s.l]), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds=dict(groups=pd.factorize(s.draft)[0]))

def table(s, label):
    res = fit(s); out = []
    print(f"\n== {label}: decks {len(s)}, drafts {s.draft.nunique()}")
    for b in BUCKETS:
        n = int((s.bucket == b).sum())
        if b == REF: out.append((b, n, 0, 0, 0, np.nan)); continue
        c = f"bucket_{b}"; bt, se = res.params[c], res.bse[c]
        out.append((b, n, pts(bt), pts(bt - 1.96 * se), pts(bt + 1.96 * se), res.pvalues[c]))
    t = pd.DataFrame(out, columns=["cmc5plus_count", "n_decks", "pts_vs_2", "ci_lo", "ci_hi", "p"]); print(t.round(3).to_string(index=False))
    # adjacent steps (marginal card), Wald on coefficient differences
    nm = {"2": None, "3": "bucket_3", "4": "bucket_4", "5+": "bucket_5+", "0-1": "bucket_0-1"}
    P = res.params; V = res.cov_params()
    for a, b in [("0-1", "2"), ("2", "3"), ("3", "4"), ("4", "5+")]:
        v = pd.Series(0.0, index=P.index)
        if nm[b]: v[nm[b]] += 1
        if nm[a]: v[nm[a]] -= 1
        d = float(v @ P); se = float(np.sqrt(v @ V @ v)); print(f"  step {a} -> {b}: {pts(d):+.2f} pts [{pts(d-1.96*se):+.2f}, {pts(d+1.96*se):+.2f}]")
    b, se = res.params["c7"], res.bse["c7"]
    print(f"  per cmc>=7 card (beyond bucket): {pts(b):+.2f} pts [{pts(b-1.96*se):+.2f}, {pts(b+1.96*se):+.2f}] p {res.pvalues['c7']:.3f}; decks with >=1: {(s.c7>=1).sum()}")
    t["half"] = label; return t

print("distribution of cmc>=5 count:", df.big.value_counts().sort_index().to_dict(), "| cmc>=7:", df.c7.value_counts().sort_index().to_dict())
half = df.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
T = [table(df, "all"), table(df[half == 0], "half A"), table(df[half == 1], "half B")]
os.makedirs(f"{R}/analysis/out", exist_ok=True)
pd.concat(T).to_csv(f"{R}/analysis/out/topend_partA.csv", index=False)
