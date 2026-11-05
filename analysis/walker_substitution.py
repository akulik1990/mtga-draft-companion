"""Do top players' decks run fewer creatures per planeswalker or Jace-token maker ("empower Jace": Protege's Awakening, Mindseeker Oculus, ...)?
Rule (fixed 2026-11-05 before the run): within-pair OLS of creature count on the count of such cards, 60%+ players' two-colour decks, SEs
clustered by draft. Creature credit = -coefficient only if the 95% CI is entirely below 0; otherwise none.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels])' -c python3 analysis/walker_substitution.py"""
import gzip, json, re
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
by = {c["name"].split(" // ")[0]: c for c in json.load(open(R / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
WALKER = {n for n, c in by.items() if "Planeswalker" in c["types"] or re.search(r"empower jace", c.get("oracle_text") or "", re.I)}
WALKER = {n for n in WALKER if "Creature" not in by[n]["types"]}
print("walker/empower non-creatures:", len(WALKER), sorted(WALKER)[:14])
rows = []
for d in json.load(gzip.open(R / "analysis/out/decks_FRA.json.gz", "rt")):
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["splash"]:
        continue
    sp = cr = w = 0
    for n, k in d["counts"].items():
        c = by.get(n.split(" // ")[0])
        if not c or "Land" in c["types"]:
            continue
        sp += k; cr += k * ("Creature" in c["types"]); w += k * (n.split(" // ")[0] in WALKER)
    if 22 <= sp <= 24:
        rows.append(dict(draft=d["draft"], lane=d["lane"], spells=sp, creatures=cr, walkers=w))
df = pd.DataFrame(rows)
print(f"decks {len(df)}, walkers per deck {df.walkers.mean():.2f}, share 1+ {100 * (df.walkers > 0).mean():.0f}%")
m = smf.ols("creatures ~ walkers + C(lane) + spells", df).fit(cov_type="cluster", cov_kwds={"groups": df.draft})
b, lo, hi = m.params["walkers"], *m.conf_int().loc["walkers"]
print(f"creatures per walker/empower card {b:+.3f} [{lo:+.3f}, {hi:+.3f}] p {m.pvalues['walkers']:.3g} -> credit {(-b if hi < 0 else 0.0):.2f}")
