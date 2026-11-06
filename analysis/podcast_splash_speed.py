"""B9: does splashing / three colours pay less in fast (low mean cmc) decks? Design of podcast_weakpool.py with deck mean spell cmc
tercile in place of pool-strength quartile. PremierDraft decks only. extended = splash non-empty or 3+ colour lane; clean = 2-colour lane,
no splash. Secondary split: splash-only (2-colour lane + splash) vs clean. Effects in win-rate points at 55%, SEs clustered by draft.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_splash_speed.py"""
import gzip, json, os
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import norm
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0.55
card = {}
for c in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
    n = c["name"].split(" // ")[0]; g = c["deck_colors"]["All Decks"]["gihwr"]
    if n not in card or c.get("isprimarycard", 1):
        card[n] = dict(cmc=c["cmc"], g=g if g > 0 else np.nan, land="Land" in c["types"])
rows = []
for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
    if d["format"] != "PremierDraft" or d["deck_games"] < 1:
        continue
    sp = [(card[k.split(" // ")[0]], v) for k, v in d["counts"].items() if k.split(" // ")[0] in card and not card[k.split(" // ")[0]]["land"]]
    if not sp: continue
    n = sum(v for _, v in sp); gs = [(c["g"], v) for c, v in sp if not np.isnan(c["g"])]
    rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=str(d["games_bucket"]), w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"],
                     cmc=sum(c["cmc"] * v for c, v in sp) / n, quality=sum(g * v for g, v in gs) / sum(v for _, v in gs),
                     ext=int(d["splash"] != "" or len(d["lane"]) >= 3), splash_only=int(d["splash"] != "" and len(d["lane"]) == 2),
                     clean=int(d["splash"] == "" and len(d["lane"]) == 2)))
df = pd.DataFrame(rows).dropna()
df["t"] = pd.qcut(df.cmc, 3, labels=["T1", "T2", "T3"])
print("decks", len(df), "extended share", round(df.ext.mean(), 3), "games", int((df.w + df.l).sum()))
print(df.groupby("t", observed=True).agg(n=("ext", "size"), cmc=("cmc", "mean"), lo=("cmc", "min"), hi=("cmc", "max"), ext=("ext", "mean"),
      splash_only=("splash_only", "mean"), wr=("w", "sum"), g=("l", "sum")).assign(wr=lambda x: x.wr / (x.wr + x.g)).drop(columns="g").round(3).to_string())

def pts(b, se):
    f = lambda x: 100 * (expit(logit(BASE) + x) - BASE)
    return round(f(b), 2), round(f(b - 1.96 * se), 2), round(f(b + 1.96 * se), 2), round(2 * norm.sf(abs(b / se)), 4)
def run(flag, sub, qc):
    d = sub.copy()
    m = smf.glm("I(w)+I(l) ~ x*C(t) + skill + C(gb) + C(lane)" + (" + quality" if qc else ""), d.assign(x=d[flag]), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
    ix = list(m.params.index); out = {}
    for t in ["T1", "T2", "T3"]:
        L = np.zeros(len(ix)); L[ix.index("x")] = 1
        if t != "T1": L[ix.index(f"x:C(t)[T.{t}]")] = 1
        out[t] = pts(L @ m.params.values, np.sqrt(L @ m.cov_params().values @ L))
    L = np.zeros(len(ix)); L[ix.index("x:C(t)[T.T3]")] = 1
    out["T1-T3"] = pts(-(L @ m.params.values), np.sqrt(L @ m.cov_params().values @ L))
    return out
res = {}
for label, flag, sub, qc in [("extended vs clean (primary)", "ext", df, False), ("extended vs clean, + deck quality", "ext", df, True),
                             ("splash-only vs clean", "splash_only", df[(df.clean == 1) | (df.splash_only == 1)], False),
                             ("splash-only vs clean, + deck quality", "splash_only", df[(df.clean == 1) | (df.splash_only == 1)], True)]:
    res[label] = run(flag, sub, qc)
    print("\n" + label + f" (n {len(sub)}): pts, lo, hi, p"); [print(" ", k, v) for k, v in res[label].items()]
json.dump(res, open(f"{R}/analysis/out/splash_speed_FRA.json", "w"), indent=1)
