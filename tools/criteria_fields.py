"""Criteria search, part 3: effect curves of the quantities that matter, one at a time, everything else controlled.

    uv run python tools/criteria_fields.py FRA all|first     # about 1 min; writes <data>/tables/SET/10_fields_<outcome>.csv

For each quantity: decks are put in bins of its value (the commonest bin is the reference); the model holds context, card
quality, colour combination and every other structure field (those that define the quantity are left out), and gives each bin's
win-rate effect in points (logit x 24.5, ridge-shrunk) with a cluster-robust (by draft) SE. "controlled" is that model;
"total" holds only context, quality and colour combination (the quantity with everything that goes with it).
QUALITY_PLUS=1 adds the deck's mean, weakest and strongest card GIH WR to the quality control (writes 10_fields_<outcome>_qplus.csv).
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build  # noqa: E402

KEYS = [  # label, source, kind, lo, hi, fields left out of the background
    ("removal spells", "removal", "int", 3, 9, {"role_removal", "rm_cheap", "rm_late"}),
    ("creatures", "creatures", "int", 11, 18, {"role_creatures", "creature", "cr_early", "cr_late", "cr_big", "cr_blockers", "cr_late_share", "noncreature"}),
    ("late creatures (cmc 4+)", "cr_late", "int", 2, 8, {"cr_late", "cr_big", "cr_late_share"}),
    ("early creatures (cmc <= 3)", "cr_early", "int", 6, 12, {"cr_early", "cr_blockers"}),
    ("card flow spells", "flow", "int", 7, 14, {"role_card_advantage", "flow_cheap", "draw", "scry", "surveil", "loot", "cycling"}),
    ("lands", "lands", "int", 15, 19, {"lands", "spells"}),
    ("multicolour (dual) lands", "land_multi", "int", 0, 6, {"land_multi", "land_mono"}),
    ("finishers (cmc 5+, GIH WR 1 SD over)", "finishers", "int", 0, 3, {"finishers"}),
    ("mana spells (ramp, fixing)", "mana", "int", 0, 6, {"role_mana", "makes_mana", "tag_fixing_ramp", "treasure"}),
    ("total toughness (quintiles)", "toughness", "quint", 0, 0, {"toughness", "cr_toughness"}),
    ("average mana value (quintiles)", "avg_mv", "quint", 0, 0, {"avg_mv"}),
    ("cheap removal (cmc <= 2)", "rm_cheap", "int", 0, 5, {"rm_cheap"}),
]


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "all"
    D = load_data(set_code, outcome)
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    cols = dict(cols)
    for k, v in F.items():
        cols.setdefault(k, v)
    import os

    plus = os.environ.get("QUALITY_PLUS") == "1"  # also control the deck's mean / weakest / strongest card GIH WR (card-level, leaks mildly)
    Q = np.hstack([sets["quality_plus" if plus else "quality"], sets["colour"]])
    S = sets["structure"]
    out = []
    ones = np.ones(len(D["w"]), bool)
    for label, src, kind, lo, hi, left_out in KEYS:
        v = np.asarray(cols[src], float)
        if kind == "int":
            b_ = np.clip(np.round(v), lo, hi).astype(int) - lo
            names = [f"<= {lo}" if i == 0 else (f">= {hi}" if i == hi - lo else f"{lo + i}") for i in range(hi - lo + 1)]
        else:
            cut = np.percentile(v, [20, 40, 60, 80])
            b_ = np.digitize(v, cut)
            names = [f"Q{i + 1} (<= {np.percentile(v, 20 * (i + 1)):.1f})" if i < 4 else f"Q5 (> {cut[3]:.1f})" for i in range(5)]
        cnt = np.bincount(b_, minlength=len(names))
        ref = int(np.argmax(cnt))
        use = [i for i in range(len(names)) if i != ref and cnt[i] >= 30]
        B = np.column_stack([(b_ == i).astype(float) for i in use])
        keep = [i for i, f in enumerate(structure) if f not in left_out]
        for variant, bg in (("controlled", np.hstack([Q, S[:, keep]])), ("total", Q)):
            lam_bg = choose_lam(Xc, bg, D, grid=(300, 1000, 3000))
            Xall = np.hstack([Xc, bg, B])
            lam = lams(Xc.shape[1], Xall.shape[1] - Xc.shape[1], lam_bg)
            lam[Xc.shape[1] + bg.shape[1]:] = 10.0
            b = fit(Xall, D["w"], D["n"], lam, ones)
            se = np.sqrt(np.diag(sandwich(Xall, D["w"], D["n"], b, lam, D["draft"])))
            k0 = Xc.shape[1] + bg.shape[1]
            eff = {i: (PTS * b[k0 + j], PTS * se[k0 + j]) for j, i in enumerate(use)}
            for i in range(len(names)):
                e, s = eff.get(i, (0.0, 0.0))
                out.append(dict(quantity=label, variant=variant, bin=names[i], decks=int(cnt[i]), reference=int(i == ref), points=round(e, 2), se=round(s, 2),
                                z=round(e / s, 1) if s else ""))
            if variant == "controlled" or True:
                print(f"{label:38} {variant:10} (ref {names[ref]}): " + "  ".join(
                    f"{names[i]}:{eff[i][0]:+.1f}{'*' if abs(eff[i][0] / eff[i][1]) >= 2.5 else ''}" if i in eff else f"{names[i]}:0" for i in range(len(names))), flush=True)
    path = tables.path(set_code, f"10_fields_{outcome}{'_qplus' if plus else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
