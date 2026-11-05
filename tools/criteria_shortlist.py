"""Criteria search, part 4: is the structure signal in a few fields? A short list chosen inside each training half.

    uv run python tools/criteria_shortlist.py FRA all|first     # about 1 min; writes <data>/tables/SET/10_shortlist_<outcome>.csv

In each fold: fit quality + colour + all structure fields on the training half, keep the k structure fields with the largest
standardised coefficient, refit quality + colour + those k on the training half, score the other half. Gain over quality + colour
in log-loss per game (SE clustered by draft). The penalty per list is chosen on validation slices, as everywhere. Also two named lists
(chosen after seeing the permutation importances, so optimistic): removal + finishers + lands + dual lands, and the current 11 categories.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build  # noqa: E402


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "all"
    D = load_data(set_code, outcome)
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    Q = np.hstack([sets["quality"], sets["colour"]])
    S = sets["structure"]
    lam_q = choose_lam(Xc, Q, D)
    z_base, _ = crossfit(Xc, Q, D, lam_q)
    lam_s = choose_lam(Xc, np.hstack([Q, S]), D)
    rows, chosen = [], {}
    b_all = fit(np.hstack([Xc, Q, S]), D["w"], D["n"], lams(Xc.shape[1], Q.shape[1] + S.shape[1], lam_s), np.ones(len(D["w"]), bool))
    rank_all = np.argsort(-np.abs(b_all[Xc.shape[1] + Q.shape[1]:]))
    for k in (2, 4, 6, 8, 12, 20, 40, 133):
        lam_k = choose_lam(Xc, np.hstack([Q, S[:, rank_all[:k]]]), D)  # the penalty only; the fields are chosen per fold below
        z = np.zeros(len(D["w"]))
        for h in (0, 1):
            tr, te = D["half"] != h, D["half"] == h
            X = np.hstack([Xc, Q, S])
            b = fit(X, D["w"], D["n"], lams(Xc.shape[1], X.shape[1] - Xc.shape[1], lam_s), tr)
            bs = np.abs(b[Xc.shape[1] + Q.shape[1]:])
            pick = np.argsort(-bs)[:k]
            for j in pick:
                chosen.setdefault(k, {}).setdefault(structure[j], 0)
                chosen[k][structure[j]] += 1
            Xk = np.hstack([Xc, Q, S[:, pick]])
            bk = fit(Xk, D["w"], D["n"], lams(Xc.shape[1], Xk.shape[1] - Xc.shape[1], lam_k), tr)
            z[te] = Xk[te] @ bk
        g, se = gain(z_base, z, D)
        rows.append(dict(model=f"top {k} structure fields (chosen in each training half)", gain_over_quality_colour=round(g, 5), se=round(se, 5), z=round(g / se, 1)))
        print(f"  top {k:3} fields: gain over quality + colour {g:+.5f} (SE {se:.5f}, z {g / se:+.1f})", flush=True)
    both = [f for f, c in chosen[12].items() if c == 2]
    print("  top 12 picked by both folds:", ", ".join(both))
    col = lambda names: np.column_stack([standardise(np.asarray(cols[f] if f in cols else F[f], float)[:, None]) for f in names])
    named = {"removal + finishers + lands + dual lands": ["role_removal", "finishers", "lands", "land_multi"],
             "the 11 current categories": ["role_creatures", "role_removal", "role_card_advantage", "role_evasion", "role_mana", "1", "2", "3", "4", "5", "6+"]}
    for label, fl in named.items():
        Xd = np.hstack([Q, col(fl)])
        z, _ = crossfit(Xc, Xd, D, choose_lam(Xc, Xd, D))
        g, se = gain(z_base, z, D)
        rows.append(dict(model=label + " (named after the fact)", gain_over_quality_colour=round(g, 5), se=round(se, 5), z=round(g / se, 1)))
        print(f"  {label:44}: {g:+.5f} (SE {se:.5f}, z {g / se:+.1f})", flush=True)
    rows.append(dict(model="fields picked by both folds at k=12", gain_over_quality_colour="", se="", z="", fields=" ".join(both)))
    path = tables.path(set_code, f"10_shortlist_{outcome}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
