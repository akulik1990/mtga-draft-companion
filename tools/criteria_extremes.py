"""Good decks against bad ones inside the same kind of player: trophy decks vs decks that went out with 2 wins or fewer.

    uv run python tools/criteria_extremes.py FRA [MIN_SKILL] [complete]      # about 1 min; default MIN_SKILL 0.60

Premier decks of players with 100+ games whose skill bucket is MIN_SKILL or higher (good drafters: a bad result is
informative, a good one is mostly skill + deck). Trophy = 7 wins; failed = 3 losses with 2 wins or fewer (0-3, 1-3, 2-3).
Everything else is left out. "complete": only decks whose observed match numbers run 1..k without a gap (Premier matches are one
game each, so their win-loss record is exact; 42% of Premier decks have gaps and 27% do not start at match 1: games are missing from the public data). Questions: how well do card quality, structure and both separate the two (out-of-fold AUC, 2-fold
by draft, bootstrap interval), which fields differ (standardised difference, trophy minus failed), and how likely each result is
for a deck of a given game win rate (so how much of either is luck).
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import GROUP_OF, build  # noqa: E402


def auc(score, y):
    r = rankdata(score)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def complete_chains(set_code):
    """{(draft_id, build_index)} of Premier decks whose match numbers run 1..k with no gap."""
    import csv
    import gzip

    seen = {}
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            seen.setdefault((row["draft_id"], row["build_index"]), set()).add(int(row["match_number"]))
    return {k for k, v in seen.items() if max(v) == len(v)}


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    min_skill = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
    D0 = load_data(set_code, "all")
    rows = D0["rows"]
    wins, games = D0["w"], D0["n"]
    losses = games - wins
    trophy = np.array([r["trophy"] > 0 for r in rows]) & (wins >= 7)
    fail = (losses >= 3) & (wins <= 2)
    prem = np.array([r["format"] == "PremierDraft" for r in rows])
    skill = np.array([r["skill"] for r in rows])
    only_complete = len(sys.argv) > 3 and sys.argv[3] == "complete"
    chain = complete_chains(set_code) if only_complete else None
    ok = np.array([(x[1], x[2]) in chain for x in D0["fg"]]) if only_complete else np.ones(len(rows), bool)
    prem = prem & ok
    sel = prem & (skill >= min_skill - 1e-9) & (trophy | fail)
    print("restricted to decks with an unbroken match chain" if only_complete else "all decks (records may have missing games)")
    print(f"Premier, 100+ games, skill >= {min_skill}: {int((prem & (skill >= min_skill - 1e-9)).sum())} decks; trophy {int((prem & (skill >= min_skill - 1e-9) & trophy).sum())}, "
          f"failed (<=2 wins, 3 losses) {int((prem & (skill >= min_skill - 1e-9) & fail).sum())}; compared: {int(sel.sum())}", flush=True)
    print("\nchance of each result by a deck's game win rate p (7 wins before 3 losses / 3 losses with <= 2 wins):")
    for p in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        q = 1 - p
        t = sum(comb(6 + k, k) * p ** 7 * q ** k for k in range(3))
        f = sum(comb(2 + w, w) * q ** 3 * p ** w for w in range(3))
        print(f"  p {p:.2f}: trophy {100 * t:4.1f}%  failed {100 * f:4.1f}%")
    D = subset(D0, sel)
    y = trophy[sel].astype(float)
    D["w"], D["n"] = y, np.ones(len(y))
    sets, structure, quality, cols, base, combos, F = build(D)
    Xq, Xs = sets["quality"], np.hstack([sets["colour"], sets["structure"]])
    for label, Xc in (("skill bucket + rank + games as controls", context(D, skill=True)), ("no controls (intercept only)", np.ones((len(y), 1)))):
        print(f"\n[{label}]")
        res = {}
        for name, Xd in (("card quality", Xq), ("structure", Xs), ("quality + structure", np.hstack([Xq, Xs])), ("none (controls only)", np.zeros((len(y), 0)))):
            lam = choose_lam(Xc, Xd, D) if Xd.shape[1] else 0
            z, zd = crossfit(Xc, Xd, D, lam)
            res[name] = (z, zd)
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(200):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(y):
                    bs.append(auc(z[i], y[i]))
            print(f"  {name:24} AUC of the full score {auc(z, y):.3f} (95% {np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}); "
                  f"deck part alone {auc(zd, y) if Xd.shape[1] else float('nan'):.3f}   ({100 * y.mean():.0f}% trophies)", flush=True)
    print("\nstandardised difference, trophy minus failed (positive = more in trophy decks), top 22 by size; SE about "
          f"{np.sqrt(1 / y.sum() + 1 / (1 - y).sum()):.2f}")
    names = quality + structure
    X = np.column_stack([cols[f] for f in names])
    sd = X.std(0)
    sd[sd == 0] = 1
    d = (X[y == 1].mean(0) - X[y == 0].mean(0)) / sd
    for j in np.argsort(-np.abs(d))[:22]:
        print(f"  {names[j]:24} {('quality' if names[j] in quality else GROUP_OF.get(names[j], 'other')):10} {d[j]:+.2f}   trophy {X[y == 1, j].mean():7.2f}  failed {X[y == 0, j].mean():7.2f}")
    print("\nthe quantities we looked at (mean per deck):")
    for f, lab in (("role_removal", "removal"), ("role_creatures", "creatures"), ("cr_late", "late creatures"), ("finishers", "finishers"), ("lands", "lands"),
                   ("land_multi", "dual lands"), ("flow", "card flow"), ("weak", "weak cards (GIH WR 1 SD under)"), ("strong", "strong cards (1 SD over)"), ("gih_mean", "mean GIH WR of spells")):
        v = np.asarray(cols[f] if f in cols else (F[f] if f in F else [r[f] for r in D["rows"]]), float)
        print(f"  {lab:30} trophy {v[y == 1].mean():6.2f}  failed {v[y == 0].mean():6.2f}  diff {v[y == 1].mean() - v[y == 0].mean():+.2f} (SE {np.sqrt(v[y == 1].var() / y.sum() + v[y == 0].var() / (1 - y).sum()):.2f})")


if __name__ == "__main__":
    main()
