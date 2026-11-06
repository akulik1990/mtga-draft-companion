"""Step A check: does the version 2 deck model close the 2+ vs 1 splash-colour residual? Calibration by skill group. nix env (statsmodels).

PRE-REGISTERED (written before any version 2 prediction was compared with outcomes):
  Design = analysis/builder_gap_actual.py primary: two-colour-lane decks of 100+ game players, binomial GLM on wins/games (freq weights = games),
  offset logit(prediction), skill-bucket dummies, no game-count term, SEs clustered by draft, effect in WR points = 100*pbar*(1-pbar)*coef.
  Contrast: 2+ splash colours minus exactly 1. Skill groups good (>= 0.60), average (0.52-0.58), weak (< 0.52), good minus weak with CI.
  Predictors: p1 (stored deckmodel; reproduces -2.82), p2 (version 2 fitted on all games).
  p2 is IN-SAMPLE: the splash2p term is fitted on these decks' games, so a residual near 0 is partly by construction. The real test is the
  held-out slice: Premier decks of the 20% of drafts (hash < 52) the train-only models never saw: p1t (version 1 refit on train drafts)
  vs p2t (version 2 on train drafts). Smallest effect the held-out slice rules out is stated from its CI.
  Calibration: per skill group, games-weighted actual minus predicted win rate (WR points) and the free logit slope, held-out slice.
Output: analysis/out/deckmodel_v2_resid.txt"""

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
lines = []


def P(s=""):
    print(s, flush=True)
    lines.append(s)


DF = pd.read_csv(REPO / "data" / "Temp" / "dm1106" / "decks_pred_v2.csv.gz").rename(columns={"games": "g"})
DF["y"] = DF.wins / DF.g
DF["nl"] = DF.lane.str.len()
DF["ns"] = DF.S.fillna("").str.len()
DF["sk"] = DF.skill.astype(str)
for k in ("p1", "p3", "p1t", "p1t3", "p2", "p23", "p2t", "p2t3", "p2s", "p2s3", "p2st", "p2st3"):
    DF["lo_" + k + "k"] = np.log(DF[k + "k"] / (100 - DF[k + "k"]))
for k in ("p1", "p3", "p1t", "p1t3", "p2", "p23", "p2t", "p2t3", "p2s", "p2s3", "p2st", "p2st3"):
    DF["lo_" + k] = np.log(DF[k] / (100 - DF[k]))
DF["cat"] = np.where(DF.ns == 0, "s0", np.where(DF.ns == 1, "s1", "s2"))


def group(df, k):
    s = df.skill
    return {"all": df, "good": df[s >= .6 - 1e-9], "avg": df[(s >= .52 - 1e-9) & (s <= .58 + 1e-9)], "weak": df[s < .52 - 1e-9]}[k]


def glm(df, rhs, off):
    G = pd.factorize(df.draft)[0]
    return smf.glm("y ~ " + rhs, df, family=sm.families.Binomial(), freq_weights=df.g, offset=df[off]).fit(cov_type="cluster", cov_kwds={"groups": G})


def contrast(df, off, a="s2", b="s1"):
    d = df[df.cat.isin([a, b])]
    if (d.cat == a).sum() < 5:
        return None
    pbar = np.average(d.y, weights=d.g)
    K = 100 * pbar * (1 - pbar)
    m = glm(d, f"C(cat, Treatment('{b}')) + C(sk)", off)
    n = f"C(cat, Treatment('{b}'))[T.{a}]"
    return K * float(m.params[n]), K * float(m.bse[n]), int((d.cat == a).sum()), int((d.cat == b).sum())


def fmt(e):
    if e is None:
        return "n/a"
    est, se, na, nb = e
    return f"{est:+.2f} ({se:.2f}) [{est - 1.96 * se:+.2f}, {est + 1.96 * se:+.2f}] n {na}/{nb}"


def block(title, df, preds):
    P(f"\n=== {title}: 2+ splash colours minus 1 (actual - predicted), WR points (SE) [95% CI] n s2/s1")
    for p in preds:
        res = {g: contrast(group(df, g), "lo_" + p) for g in ("all", "good", "avg", "weak")}
        P(f"  {p:5s} " + " | ".join(f"{g} {fmt(r)}" for g, r in res.items()))
        if res["good"] and res["weak"]:
            d = res["good"][0] - res["weak"][0]
            se = (res["good"][1] ** 2 + res["weak"][1] ** 2) ** .5
            P(f"        good minus weak {d:+.2f} ({se:.2f}) [{d - 1.96 * se:+.2f}, {d + 1.96 * se:+.2f}]")


two = DF[DF.nl == 2]
P(f"decks {len(DF)}; two-colour lane {len(two)}; held-out Premier decks {int(DF.test.sum())} (two-colour {int(two.test.sum())})")
block("ALL two-colour decks, all formats (p2* / p23 in-sample on Premier)", two, ["p1", "p3", "p2", "p23", "p2s", "p2s3"])
tst = two[two.test == 1]
block("HELD-OUT Premier drafts only (train-only fits)", tst, ["p1t", "p2t", "p2st", "p1t3", "p2t3", "p2st3"])
trn = two[(two.test == 0) & (two.fmt == "PremierDraft")]
block("TRAIN Premier drafts (in-sample for p2t)", trn, ["p1t", "p2t", "p2st"])
oth = two[two.fmt != "PremierDraft"]
block("Trad + Pick-Two (never in the fit) with the all-games fits", oth, ["p1", "p2", "p2s"])

P("\n=== calibration by skill group, held-out Premier drafts (all decks, lane length 2 or 3): games-weighted actual - predicted at the deck's own skill bucket (WR points), free logit slope (1 = calibrated)")
ho = DF[DF.test == 1]
for g in ("all", "good", "avg", "weak"):
    d = group(ho, g)
    out = []
    for p in ("p1tk", "p2tk", "p2stk"):  # predictions at the deck's own skill bucket
        a, pr = np.average(d.y, weights=d.g) * 100, np.average(d[p], weights=d.g)
        m = smf.glm("y ~ lo_" + p, d, family=sm.families.Binomial(), freq_weights=d.g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
        out.append(f"{p}: actual {a:.2f} pred {pr:.2f} diff {a - pr:+.2f}, slope {m.params['lo_' + p]:.2f} ({m.bse['lo_' + p]:.2f})")
    P(f"  {g:5s} decks {len(d):5d} games {int(d.g.sum()):6d}  " + "  |  ".join(out))
P("\n  by category, held-out two-colour decks, actual - predicted (WR points):")
for lab in ("s0", "s1", "s2"):
    d = tst[tst.cat == lab]
    P(f"  {lab}: decks {len(d)}  " + "  ".join(f"{p} {np.average(d.y, weights=d.g) * 100 - np.average(d[p], weights=d.g):+.2f}" for p in ("p1t", "p2t", "p2st")))
(REPO / "analysis" / "out" / "deckmodel_v2_resid.txt").write_text("\n".join(lines) + "\n")
