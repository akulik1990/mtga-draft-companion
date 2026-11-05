"""15/16 vs 17 lands by landcyclers and duals in the deck (Akshin: "15-16 lands with better land cards and landcyclers").
Rule (fixed 2026-11-05 before the run): 60%+ players' two-colour decks with 15-17 lands and 40 cards; binomial GLM on game wins,
SEs clustered by draft; controls: mean All Decks GIH WR of the deck's spells, lane, skill, games bucket. Strata: landcyclers 0 / 1 / 2+
x on-colour duals 0-1 / 2+. In a stratum where 16 (or 15) is not significantly worse than 17 (p < 0.05 two-sided, Bonferroni over
the strata), the builder may run that count for such decks; otherwise 17 stays.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels])' -c python3 analysis/land_count_check.py"""
import gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
by = {c["name"].split(" // ")[0]: c for c in json.load(open(R / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
BASIC = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
rows = []
for d in json.load(gzip.open(R / "analysis/out/decks_FRA.json.gz", "rt")):
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["deck_games"] < 1:
        continue
    lands = cyc = duals = n = 0; q = []
    for name, k in d["counts"].items():
        c = by.get(name.split(" // ")[0]); n += k
        if name in BASIC:
            lands += k; continue
        if not c:
            continue
        if "Land" in c["types"]:
            lands += k
            if len(c.get("colors") or []) >= 2 and set(c["colors"]) <= set(d["lane"]):
                duals += k
            continue
        if "landcycling" in (c.get("oracle_text") or "").lower():
            cyc += k
        g = (c["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
        if g:
            q += [g] * k
    if n != 40 or lands not in (15, 16, 17) or not q:
        continue
    rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], w=d["deck_wins"], g=d["deck_games"],
                     lands=lands, cyc=min(cyc, 2), duals=min(duals, 2) // 2, quality=float(np.mean(q))))
df = pd.DataFrame(rows)
df["l15"], df["l16"] = (df.lands == 15).astype(int), (df.lands == 16).astype(int)
print(f"decks {len(df)}; lands 15/16/17: {(df.lands == 15).sum()}/{(df.lands == 16).sum()}/{(df.lands == 17).sum()}")
strata = [(c, u) for c in (0, 1, 2) for u in (0, 1)]
alpha = 0.05 / len(strata)
for c, u in strata:
    s = df[(df.cyc == c) & (df.duals == u)].copy()
    lab = f"landcyclers {c if c < 2 else '2+'}, duals {'2+' if u else '0-1'}"
    if s.l16.sum() < 30:
        print(f"{lab:32} n {len(s):5}: too few 16-land decks ({s.l16.sum()})"); continue
    s["y"] = s.w / s.g
    m = smf.glm("y ~ l15 + l16 + quality + skill + C(gb) + C(lane)", s, family=sm.families.Binomial(), freq_weights=s.g).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s.draft)[0]})
    out = []
    for v in ("l16", "l15"):
        if s[v].sum() < 30:
            out.append(f"{v[1:]}: n<30"); continue
        b, se, p = m.params[v], m.bse[v], m.pvalues[v]
        pts = 100 * b * 0.55 * 0.45
        verdict = "worse" if (p < alpha and b < 0) else "not worse"
        out.append(f"{v[1:]} vs 17 {pts:+.2f} pts [{100*(b-1.96*se)*.2475:+.2f}, {100*(b+1.96*se)*.2475:+.2f}] p {p:.3f} ({verdict}, n {s[v].sum()})")
    print(f"{lab:32} n {len(s):5}: " + " | ".join(out))
print(f"Bonferroni alpha per stratum {alpha:.4f}")
