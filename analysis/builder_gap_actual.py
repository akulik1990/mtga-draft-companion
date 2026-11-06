"""Part 2(c): do real results match the deck model for decks with 2+ splash colours? FRA, analysis/out/decks_FRA.json.gz (100+ games players).

PRE-REGISTERED (written 2026-11-06 before any outcome was looked at; the deck model's predictions were not looked at either):
  Population: two-colour-lane decks (len(lane)==2), >= 1 game, one row per deck (draft, build) with deck_wins / deck_games.
  Prediction: tools/deck_model.py "deckmodel" (the mock yardstick) on the deck's 40 with main = lane, splash = splash colours, at the typical
              context (skill 0.56, 100 games, on the play half the time, no mulligans): the deck's win rate for an average player.
  Primary: actual minus predicted win rate of decks with 2+ splash colours minus the same for decks with exactly 1 splash colour,
           skill controlled. Model: binomial GLM, y = wins/games weighted by games, offset logit(prediction), skill-bucket dummies
           (0.02 buckets) + category dummies; SEs clustered by draft; no game-count term (docs/foundations-check-FRA.md).
           Effect in WR points = 100 * pbar * (1 - pbar) * coefficient. Two-sided z test, alpha 0.05, one primary so no multiplicity correction.
  Reported with it: 95% CI, the smallest effect ruled out (|estimate| + 1.96 SE on the harmful side), skill groups good (>= 0.60) / average
           (0.52-0.58) / weak (< 0.52) and good minus weak with CI, halves (draft hash parity).
  Secondary (labelled, not adjusted): free calibration slope on the logit; deckmodel_top3 as predictor; Premier only; four-or-five-colour decks
           (3-colour lane + splash) against 3-colour lane with no splash; clean pairs against 1 splash colour (what the model says a splash costs).
  Leakage note: the deck model was refit on all games (incl. these decks), so predictions are in-sample. The primary contrast 2+ vs 1 is not
  something the model can absorb (its splash term is binary, colours term counts main colours only), so in-sample fit biases it only through
  the card coefficients, towards zero.
Output: analysis/out/builder_gap_actual.txt"""

import gzip
import hashlib
import json
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis" / "out"
lines = []


def P(s=""):
    print(s, flush=True)
    lines.append(s)


# predictions: analysis/builder_gap_predict.py (project env); here only the regression (nix env with statsmodels)
DF = pd.read_csv(REPO / "data" / "Temp" / "gap1106" / "decks_pred.csv.gz").rename(columns={"games": "g"})
DF["y"] = DF.wins / DF.g
DF["nl"] = DF.lane.str.len()
DF["ns"] = DF.S.fillna("").str.len()
DF["fmt"] = DF["fmt"]
P(f"decks {len(DF)}; card total per deck mean {DF['size'].mean():.1f} (40 expected); lane lengths {DF.nl.value_counts().to_dict()}; splash colours {DF.ns.value_counts().to_dict()}")
DF["half"] = DF.draft.map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 2)
DF["lo"] = np.log(DF.p / (100 - DF.p))
DF["lo3"] = np.log(DF.p3 / (100 - DF.p3))
DF["sk"] = DF.skill.astype(str)


def group(df, k):
    s = df.skill
    return {"all": df, "good": df[s >= .6 - 1e-9], "avg": df[(s >= .52 - 1e-9) & (s <= .58 + 1e-9)], "weak": df[s < .52 - 1e-9]}[k]


def fit(df, rhs, off="lo"):
    G = pd.factorize(df.draft)[0]
    return smf.glm("y ~ " + rhs, df, family=sm.families.Binomial(), freq_weights=df.g, offset=df[off]).fit(cov_type="cluster", cov_kwds={"groups": G})


def contrast(m, a, b, K):
    """coefficient a minus b (names of C(cat) levels, baseline level has coefficient 0), in WR points."""
    names = list(m.params.index)
    v = np.zeros(len(names))
    for t, c in ((a, 1), (b, -1)):
        if t in names:
            v[names.index(t)] += c
    est = float(v @ m.params.values)
    se = float(np.sqrt(v @ m.cov_params().values @ v))
    return K * est, K * se


def fmt(e):
    est, se = e
    return f"{est:+.2f} ({se:.2f}) [{est - 1.96 * se:+.2f}, {est + 1.96 * se:+.2f}] z {est / se:+.2f}"


def analyse(df, cats, a, b, label, off="lo", slope=False):
    """cats: function row -> category label; a, b the compared categories."""
    d = df.copy()
    d["cat"] = d.apply(cats, axis=1)
    d = d[d.cat.isin(["x"] + [c for c in d.cat.unique()])]
    pbar = np.average(d.y, weights=d.g)
    K = 100 * pbar * (1 - pbar)
    base = [c for c in sorted(d.cat.unique()) if c not in (a, b)] or [b]
    # baseline = b so that the coefficient on a is a minus b
    rhs = f"C(cat, Treatment('{b}')) + C(sk)" + (f" + {off}" if slope else "")
    m = fit(d, rhs, off)
    name = f"C(cat, Treatment('{b}'))[T.{a}]"
    names = list(m.params.index)
    est = K * float(m.params[name])
    se = K * float(m.bse[name])
    n_a, n_b = int((d.cat == a).sum()), int((d.cat == b).sum())
    return est, se, n_a, n_b, m


two = DF[DF.nl == 2]
cat2 = lambda r: "s0" if r.ns == 0 else "s1" if r.ns == 1 else "s2"
P(f"\ntwo-colour-lane decks {len(two)}: clean {int((two.ns == 0).sum())}, 1 splash colour {int((two.ns == 1).sum())}, 2+ {int((two.ns >= 2).sum())}")
P("\n=== calibration of the yardstick (all two-colour decks), by category: mean predicted vs mean actual win rate (games-weighted)")
for lab, mask in (("clean", two.ns == 0), ("1 splash", two.ns == 1), ("2+ splash", two.ns >= 2)):
    s = two[mask]
    P(f"  {lab:10s} decks {len(s):6d} games {int(s.g.sum()):7d}  actual {100 * np.average(s.y, weights=s.g):.2f}  predicted {np.average(s.p, weights=s.g):.2f}  skill {np.average(s.skill, weights=s.g):.3f}")

P("\n=== PRIMARY: (actual - predicted) for 2+ splash colours minus 1 splash colour, skill controlled, WR points (SE) [95% CI] z")
res = {}
for g in ("all", "good", "avg", "weak"):
    d = group(two, g)
    est, se, na, nb, m = analyse(d, cat2, "s2", "s1", g)
    res[g] = (est, se)
    P(f"  {g:5s} {fmt((est, se))}   decks s2 {na}, s1 {nb}")
dg = res["good"][0] - res["weak"][0]
dse = (res["good"][1] ** 2 + res["weak"][1] ** 2) ** .5
P(f"  good minus weak {fmt((dg, dse))}")
est, se = res["all"]
P(f"  smallest harmful effect ruled out (all players): {min(est - 1.96 * se, 0):+.2f} is the CI's lower end, i.e. a deficit worse than {abs(est - 1.96 * se):.2f} is excluded; "
  f"a surplus above {est + 1.96 * se:+.2f} is excluded")
P("  halves (draft hash parity): " + "; ".join(fmt(analyse(two[two.half == h], cat2, "s2", "s1", "h")[:2]) for h in (0, 1)))

P("\n--- secondary (not adjusted)")
for lab, kw in (("free calibration slope", dict(slope=True)), ("deckmodel_top3 offset", dict(off="lo3"))):
    P(f"  {lab}: " + "; ".join(f"{g} " + fmt(analyse(group(two, g), cat2, "s2", "s1", g, **kw)[:2]) for g in ("all", "good", "weak")))
prem = two[two.fmt == "PremierDraft"]
P(f"  Premier only ({len(prem)} decks): " + "; ".join(f"{g} " + fmt(analyse(group(prem, g), cat2, "s2", "s1", g)[:2]) for g in ("all", "good", "weak")))
P("  by format (all players): " + "; ".join(f"{f} " + fmt(analyse(two[two.fmt == f], cat2, "s2", "s1", f)[:2]) + f" (n s2 {int(((two.fmt == f) & (two.ns >= 2)).sum())})" for f in two.fmt.unique()))
P("  clean minus 1 splash colour (what the model's binary splash term mis-prices): " + "; ".join(f"{g} " + fmt(analyse(group(two, g), cat2, "s0", "s1", g)[:2]) for g in ("all", "good", "avg", "weak")))

three = DF[DF.nl == 3]
cat3 = lambda r: "t0" if r.ns == 0 else "t1"
P(f"\n=== four-or-five colours: 3-colour lane + splash vs 3-colour lane clean ({len(three)} decks, with splash {int((three.ns > 0).sum())}): actual - predicted")
for g in ("all", "good", "weak"):
    d = group(three, g)
    if (d.ns > 0).sum() > 5:
        est, se, na, nb, m = analyse(d, cat3, "t1", "t0", g)
        P(f"  {g:5s} {fmt((est, se))}   decks with splash {na}, clean {nb}")
allc = DF.copy()
allc["tot"] = allc.nl + allc.ns
catn = lambda r: "c4p" if r.nl + r.ns >= 4 else "c3m"
P("  total colours >= 4 (2+2/3 or 3+1/2) vs <= 3, lane length controlled by cat only: " + "; ".join(f"{g} " + fmt(analyse(group(allc, g), catn, "c4p", "c3m", g)[:2]) for g in ("all", "good", "weak")))

P("\n=== how often (share of decks, by skill group)")
for g in ("all", "good", "avg", "weak"):
    d = group(DF, g)
    t = d.nl + d.ns
    P(f"  {g:5s} decks {len(d):6d}  2+ splash colours (of 2-colour-lane decks) {100 * ((d.nl == 2) & (d.ns >= 2)).sum() / max(1, (d.nl == 2).sum()):.1f}%   "
      f"4-5 colours total (of all) {100 * (t >= 4).mean():.1f}%   3-colour lane {100 * (d.nl == 3).mean():.1f}%   1 splash colour in 2-lane {100 * ((d.nl == 2) & (d.ns == 1)).sum() / max(1, (d.nl == 2).sum()):.1f}%")
(OUT / "builder_gap_actual.txt").write_text("\n".join(lines) + "\n")
