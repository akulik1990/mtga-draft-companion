"""Do top players' decks run fewer creatures / removal per cheap tempo spell (Unsummon, Icy Reception, Countersculpt, Fatehold Charm)?
Rule (fixed 2026-11-05 before the run): within-pair OLS of count on tempo count, 60%+ players' two-colour decks, SEs clustered by draft.
A builder floor (creatures, removal) gets tempo credit = -coefficient only if its 95% CI excludes 0; otherwise no credit.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels])' -c python3 analysis/tempo_substitution.py"""
import gzip, json
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
cr = json.load(open(R / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
by = {c["name"].split(" // ")[0]: c for c in cr.values()}
TEMPO = {"Unsummon", "Icy Reception", "Countersculpt", "Fatehold Charm"}
REMOVAL = {n for n, c in by.items() if "removal" in (c.get("tags") or [])}
rows = []
for d in json.load(gzip.open(R / "analysis/out/decks_FRA.json.gz", "rt")):
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["splash"]:
        continue
    cnt = {}
    for n, k in d["counts"].items():
        c = by.get(n.split(" // ")[0])
        if not c or "Land" in c["types"]:
            continue
        f = n.split(" // ")[0]
        cnt["spells"] = cnt.get("spells", 0) + k
        cnt["creatures"] = cnt.get("creatures", 0) + k * ("Creature" in c["types"])
        cnt["tempo"] = cnt.get("tempo", 0) + k * (f in TEMPO)
        cnt["removal"] = cnt.get("removal", 0) + k * (f in REMOVAL)
    if 22 <= cnt.get("spells", 0) <= 24:
        rows.append(dict(draft=d["draft"], lane=d["lane"], **cnt))
df = pd.DataFrame(rows)
print(f"decks {len(df)}, drafts {df.draft.nunique()}, tempo per deck mean {df.tempo.mean():.2f}, share with 1+ {100 * (df.tempo > 0).mean():.0f}%")
print("REMOVAL tag source: card_ratings 'tags' field;", len(REMOVAL), "cards tagged removal")
for y in ("creatures", "removal"):
    m = smf.ols(f"{y} ~ tempo + C(lane) + spells", df).fit(cov_type="cluster", cov_kwds={"groups": df.draft})
    b, lo, hi = m.params["tempo"], *m.conf_int().loc["tempo"]
    credit = -b if (hi < 0 or lo > 0) else 0.0
    print(f"{y:10} per tempo card {b:+.3f} [{lo:+.3f}, {hi:+.3f}] p {m.pvalues['tempo']:.3g} -> credit {credit:.2f}")
