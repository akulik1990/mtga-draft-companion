"""Do good decks run 17 lands or fewer? Land count by skill band, with what travels with it.

    uv run python tools/lands_check.py FRA      # about 1 min; writes <data>/tables/SET/12_lands.csv

Decks as in tools/skill_contrast.py pass A (Premier + Pick-Two, 50+ games, 2-3 base colours, skill bands low <0.45 / mid 0.45-0.59 /
high 0.60+). Part 1: shares of decks by land count and what else differs (average mana value, landcyclers, mana spells, cheap spells).
Part 2: effect of the land count on win rate (points vs 17, other fields held fixed, ridge, cluster-robust SE by draft), for lands as
counted and for lands + landcycling cards ("effective lands"). Associations: players choose the count with the rest of the deck.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build, write  # noqa: E402

BANDS = ("low <0.45", "mid 0.45-0.59", "high 0.60+", "all")


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    D0 = load_data(set_code, "all", buckets=("50", "100", "500"), formats=("PremierDraft", "PickTwoDraft"))
    sk = np.array([d["skill"] if d["skill"] is not None else np.nan for d in D0["decks"]], float)
    D = subset(D0, ~np.isnan(sk))
    sk = sk[~np.isnan(sk)]
    band = np.where(sk < 0.45, 0, np.where(sk < 0.595, 1, 2))
    rows = D["rows"]
    g = lambda f: np.array([r[f] for r in rows], float)
    lands, lc, mv, mana, cheap = g("lands"), g("landcycling"), g("avg_mv"), g("role_mana"), g("1") + g("2")
    trophy = g("trophy") > 0
    eff = lands + lc
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    Q, S = np.hstack([sets["quality"], sets["colour"]]), sets["structure"]
    out = []
    print(f"{len(lands)} decks; mean lands {lands.mean():.2f}, landcycling cards {lc.mean():.2f}, effective {eff.mean():.2f}\n")
    print("part 1: share of decks by land count (as counted), and what travels with it")
    print(f"{'band':14} " + " ".join(f"{c:>7}" for c in ("<=15", "16", "17", "18", ">=19")) + "   | trophy decks share: " + " ".join(f"{c:>6}" for c in ("<=15", "16", "17", "18+")))
    for b in range(3):
        m = band == b
        sh = [np.mean(lands[m] <= 15), np.mean(lands[m] == 16), np.mean(lands[m] == 17), np.mean(lands[m] == 18), np.mean(lands[m] >= 19)]
        t = trophy & m
        ts = [np.mean(lands[t] <= 15), np.mean(lands[t] == 16), np.mean(lands[t] == 17), np.mean(lands[t] >= 18)]
        print(f"{BANDS[b]:14} " + " ".join(f"{100 * x:6.1f}%" for x in sh) + f"   | ({int(t.sum())} trophy) " + " ".join(f"{100 * x:5.1f}%" for x in ts))
        out.append(dict(part="share", band=BANDS[b], le15=round(100 * sh[0], 1), n16=round(100 * sh[1], 1), n17=round(100 * sh[2], 1), n18=round(100 * sh[3], 1), ge19=round(100 * sh[4], 1)))
    print(f"\n{'lands':>6} {'decks':>6} {'avg mana value':>14} {'landcyclers':>11} {'mana spells':>11} {'1-2 drops':>9} {'effective lands':>15} {'WR (all bands)':>14}")
    for v in (15, 16, 17, 18, 19):
        m = (lands <= v) if v == 15 else ((lands >= v) if v == 19 else (lands == v))
        print(f"{v:>6} {m.sum():6d} {mv[m].mean():14.2f} {lc[m].mean():11.2f} {mana[m].mean():11.2f} {cheap[m].mean():9.2f} {eff[m].mean():15.2f} {100 * D['w'][m].sum() / D['n'][m].sum():13.1f}%")

    print("\npart 2: win-rate effect vs 17 (points; * |z| >= 2.5; decks in brackets). controlled = quality, colours and every other structure field held")
    for label, src, left_out in (("lands", lands, {"lands", "spells"}), ("lands + landcyclers", eff, {"lands", "spells", "cycling"})):
        b_all = np.clip(np.round(src), 15, 19).astype(int) - 15
        names = ["<=15", "16", "17", "18", ">=19"]
        keep = [i for i, f in enumerate(structure) if f not in left_out]
        for b in range(4):
            m = np.ones(len(lands), bool) if b == 3 else band == b
            Dm = subset(D, m)
            cnt = np.bincount(b_all[m], minlength=5)
            use = [i for i in (0, 1, 3, 4) if cnt[i] >= 30]
            B = np.column_stack([(b_all[m] == i).astype(float) for i in use])
            bg = np.hstack([Q[m], S[m][:, keep]])
            l2 = choose_lam(Xc[m], bg, Dm, grid=(300, 1000, 3000))
            Xall = np.hstack([Xc[m], bg, B])
            kc = Xc.shape[1]
            lam = lams(kc, Xall.shape[1] - kc, l2)
            lam[kc + bg.shape[1]:] = 10.0
            bb = fit(Xall, Dm["w"], Dm["n"], lam, np.ones(m.sum(), bool))
            se = np.sqrt(np.diag(sandwich(Xall, Dm["w"], Dm["n"], bb, lam, Dm["draft"])))
            eff_ = {i: (PTS * bb[kc + bg.shape[1] + j], PTS * se[kc + bg.shape[1] + j]) for j, i in enumerate(use)}
            for i in range(5):
                e, s_ = eff_.get(i, (0.0, 0.0))
                out.append(dict(part="effect", counted=label, band=BANDS[b], lands=names[i], decks=int(cnt[i]), points=round(e, 2), se=round(s_, 2)))
            print(f"  {label:20} {BANDS[b]:14} " + "  ".join((f"{names[i]}:{eff_[i][0]:+.1f}{'*' if abs(eff_[i][0] / eff_[i][1]) >= 2.5 else ''}({cnt[i]})" if i in eff_ else f"{names[i]}:0({cnt[i]})") for i in range(5)), flush=True)
    write(tables.path(set_code, "12_lands.csv"), out)


if __name__ == "__main__":
    main()
