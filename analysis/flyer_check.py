"""Do decks with more flying creatures win more than their cards' own GIH WR says? (Akshin: "flyer creature > normal creature")
Rule (fixed 2026-11-05 before the run): 60%+ players' two-colour decks, binomial GLM on game wins, SEs clustered by draft; controls: mean
lane GIH WR of the spells, creature count, lane, skill, games bucket. If the flyer coefficient is significantly positive (p < 0.05), the
builder gets that many win-rate points per flying creature; otherwise no bonus.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels])' -c python3 analysis/flyer_check.py"""
import gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
by = {c["name"].split(" // ")[0]: c for c in json.load(open(R / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
fly = lambda c: "Creature" in c["types"] and re.search(r"(^|\n)Flying|\bFlying\b(?! creatures)", c.get("oracle_text") or "") is not None
rows = []
for d in json.load(gzip.open(R / "analysis/out/decks_FRA.json.gz", "rt")):
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["deck_games"] < 1:
        continue
    q, cr, fl = [], 0, 0
    for n, k in d["counts"].items():
        c = by.get(n.split(" // ")[0])
        if not c or "Land" in c["types"]:
            continue
        g = (c["deck_colors"].get(d["lane"]) or {}).get("gihwr") or (c["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
        if g:
            q += [g] * k
        cr += k * ("Creature" in c["types"]); fl += k * fly(c)
    if len(q) >= 20:
        rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], y=d["deck_wins"] / d["deck_games"],
                         g=d["deck_games"], quality=float(np.mean(q)), creatures=cr, flyers=fl))
df = pd.DataFrame(rows)
print(f"decks {len(df)}; flyers per deck mean {df.flyers.mean():.2f} (sd {df.flyers.std():.2f}), creatures {df.creatures.mean():.2f}")
m = smf.glm("y ~ flyers + creatures + quality + skill + C(gb) + C(lane)", df, family=sm.families.Binomial(), freq_weights=df.g).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
b, se, p = m.params["flyers"], m.bse["flyers"], m.pvalues["flyers"]
k = 100 * 0.25
print(f"per flying creature (same creature count, same card quality): {k*b:+.2f} WR points [{k*(b-1.96*se):+.2f}, {k*(b+1.96*se):+.2f}] p {p:.3g}"
      f" -> bonus {k*b if (p < 0.05 and b > 0) else 0.0:.2f}")
