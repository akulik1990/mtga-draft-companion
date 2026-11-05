"""Good players against bad players: how different are their decks, and how much of a bad result is the deck?

    uv run python tools/skill_contrast.py FRA      # about 5-8 min (three passes); writes <data>/tables/SET/11_skill_*.csv

Decks of Premier and Pick-Two players with 50+ games and 2-3 base colours, split by the player's skill bucket (a win rate, so partly
an outcome: players with 50-99 games have this deck's games in it at about 10-20%, the 100+ pass checks that): low < 0.45,
mid 0.45-0.59, high 0.60+. Passes: A Premier + Pick-Two 50+ games (main), B Premier only, C Premier + Pick-Two 100+ games.
Sections: 1 bands; 2 how different the decks are (fields, tails, can a deck alone tell the bands apart); 3 how much of the band gap in
win rate the deck explains (one model on all bands, deck part by band) and whether the same rules hold in each band (fit in one band,
scored in another); 4 effect curves inside each band; 5 good cards in bad hands (quality tercile x structure tercile per band);
6 complete records: share of 0-3 (Premier) / 0-2 (Pick-Two) decks per band and whether the deck separates them from trophy decks.
Scores are cross-fitted by draft half as in tools/criteria_search.py; effects in win-rate points (logit x 24.5).
"""

import csv
import gzip
import sys
import collections
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import GROUP_OF, build, write  # noqa: E402

BANDS = ("low <0.45", "mid 0.45-0.59", "high 0.60+")
PASSES = [("A Premier + Pick-Two, 50+ games", ("PremierDraft", "PickTwoDraft"), ("50", "100", "500"), True),
          ("B Premier only, 50+ games", ("PremierDraft",), ("50", "100", "500"), False),
          ("C Premier + Pick-Two, 100+ games", ("PremierDraft", "PickTwoDraft"), ("100", "500"), False)]
CURVES = [("removal", "role_removal", 3, 9, {"role_removal", "rm_cheap", "rm_late"}),
          ("finishers", "finishers", 0, 3, {"finishers"}),
          ("lands", "lands", 15, 19, {"lands", "spells"}),
          ("dual lands", "land_multi", 0, 6, {"land_multi", "land_mono"}),
          ("creatures", "role_creatures", 11, 18, {"role_creatures", "creature", "cr_early", "cr_late", "cr_big", "cr_blockers", "cr_late_share", "noncreature"})]
OUT = collections.defaultdict(list)


def auc(score, y):
    r = rankdata(score)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def auc_ci(score, y, reps=200):
    rng = np.random.default_rng(0)
    a = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            a.append(auc(score[i], y[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def wr_cell(D, m):
    w, n = D["w"][m], D["n"][m]
    if n.sum() == 0:
        return float("nan"), float("nan")
    p = w.sum() / n.sum()
    r = w - n * p
    return 100 * p, 100 * float(np.sqrt((r ** 2).sum())) / n.sum()


def calibrate(z0, zd, D):
    """Slope on a deck score added to a baseline logit z0 (1 = the score means what it says) and the held-out gain."""
    n = len(z0)
    A, B = np.column_stack([np.ones(n), z0]), np.column_stack([np.ones(n), z0, zd])
    lA, lB = np.array([0, 1e-3]), np.array([0, 1e-3, 1e-3])

    def cf(M, lam):
        z = np.zeros(n)
        for h in (0, 1):
            b = fit(M, D["w"], D["n"], lam, D["half"] != h)
            z[D["half"] == h] = M[D["half"] == h] @ b
        return z

    g, gse = gain(cf(A, lA), cf(B, lB), D)
    b = fit(B, D["w"], D["n"], lB, np.ones(n, bool))
    se = np.sqrt(np.diag(sandwich(B, D["w"], D["n"], b, lB, D["draft"])))
    return float(b[2]), float(se[2]), g, gse


def records(set_code, formats):
    out = {}
    for fmt, win_end, loss_end in (("PremierDraft", 7, 3), ("PickTwoDraft", 4, 2)):
        if fmt not in formats:
            continue
        decks = collections.defaultdict(dict)
        with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.{fmt}.csv.gz", "rt", newline="") as f:
            r = csv.reader(f)
            ix = {k: i for i, k in enumerate(next(r))}
            a, b, m, w = ix["draft_id"], ix["build_index"], ix["match_number"], ix["won"]
            for row in r:
                decks[(row[a], row[b])][int(row[m])] = row[w] == "True"
        for (d, b), ms in decks.items():
            wins = sum(ms.values())
            losses = len(ms) - wins
            out[(fmt, d, b)] = (len(ms) == max(ms) and (wins == win_end or losses == loss_end), wins, losses)
    return out


def analyse(set_code, label, formats, games, rec, verbose):
    print(f"\n{'=' * 110}\n{label}", flush=True)
    D0 = load_data(set_code, "all", buckets=games, formats=formats)
    sk = np.array([d["skill"] if d["skill"] is not None else np.nan for d in D0["decks"]], float)
    D = subset(D0, ~np.isnan(sk))
    sk = sk[~np.isnan(sk)]
    band = np.where(sk < 0.45, 0, np.where(sk < 0.595, 1, 2))
    N, fmt = len(band), np.array([x[0] for x in D["fg"]])
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    Xq, Xs = sets["quality"], np.hstack([sets["colour"], sets["structure"]])
    Xd = np.hstack([Xq, Xs])
    kc, kq = Xc.shape[1], Xq.shape[1]
    tag = lambda **kw: dict(kw, **{"pass": label})

    # 1 bands, and the pooled model's parts
    lam = choose_lam(Xc, Xd, D)
    X = np.hstack([Xc, Xd])
    lamv = lams(kc, Xd.shape[1], lam)
    zc, zq, zs = np.zeros(N), np.zeros(N), np.zeros(N)
    for h in (0, 1):
        b = fit(X, D["w"], D["n"], lamv, D["half"] != h)
        te = D["half"] == h
        zc[te], zq[te], zs[te] = Xc[te] @ b[:kc], Xq[te] @ b[kc:kc + kq], Xs[te] @ b[kc + kq:]
    print(f"{N} decks, {int(D['n'].sum())} games; pooled model on all three bands (skill bucket, rank, games, format as context), lam {lam}")
    print(f"\n{'band':14} {'decks':>6} {'games/deck':>10} {'win rate':>9} | vs the average deck in the pass, points: {'context':>8} {'quality':>8} {'structure+colour':>17}")
    for b_ in range(3):
        m = band == b_
        wr, se = wr_cell(D, m)
        row = tag(band=BANDS[b_], decks=int(m.sum()), games=int(D["n"][m].sum()), games_per_deck=round(float(D["n"][m].mean()), 2), win_rate=round(wr, 2),
                  context_pts=round(PTS * (zc[m].mean() - zc.mean()), 2), quality_pts=round(PTS * (zq[m].mean() - zq.mean()), 2),
                  structure_pts=round(PTS * (zs[m].mean() - zs.mean()), 2))
        OUT["bands"].append(row)
        print(f"{BANDS[b_]:14} {row['decks']:6d} {row['games_per_deck']:10.2f} {wr:8.1f}% | {'':35}{row['context_pts']:+8.1f} {row['quality_pts']:+8.1f} {row['structure_pts']:+17.1f}")
    lo, hi = band == 0, band == 2
    gap, gse = wr_cell(D, hi)[0] - wr_cell(D, lo)[0], np.hypot(wr_cell(D, hi)[1], wr_cell(D, lo)[1])
    dq, dqse = PTS * (zq[hi].mean() - zq[lo].mean()), PTS * np.hypot(zq[hi].std() / np.sqrt(hi.sum()), zq[lo].std() / np.sqrt(lo.sum()))
    ds, dsse = PTS * (zs[hi].mean() - zs[lo].mean()), PTS * np.hypot(zs[hi].std() / np.sqrt(hi.sum()), zs[lo].std() / np.sqrt(lo.sum()))
    print(f"\nwin rate gap high - low: {gap:.1f} points (SE {gse:.1f}). A low-band player with the average high-band deck, by the model fitted within skill:")
    print(f"  quality {dq:+.1f} (SE {dqse:.1f}), structure + colour {ds:+.1f} (SE {dsse:.1f}), deck total {dq + ds:+.1f} = {100 * (dq + ds) / gap:.0f}% of the gap")
    OUT["shift"].append(tag(gap=round(gap, 2), gap_se=round(gse, 2), quality=round(dq, 2), quality_se=round(dqse, 2), structure=round(ds, 2), structure_se=round(dsse, 2),
                            deck_share_of_gap=round((dq + ds) / gap, 3)))
    print(f"  spread of the deck part inside each band (SD, points): " + ", ".join(f"{BANDS[b_]} {PTS * (zq + zs)[band == b_].std():.1f}" for b_ in range(3)))

    # 2 how different are the decks
    cell = np.array([f"{a}|{b}" for a, b in zip(fmt, base)])
    _, inv = np.unique(cell, return_inverse=True)
    names = quality + structure
    rows = []
    for f in names:
        v = np.asarray(cols[f], float)
        r = v - (np.bincount(inv, weights=v) / np.bincount(inv))[inv]
        sd = r.std() or 1.0
        d = (r[hi].mean() - r[lo].mean()) / sd
        rows.append(tag(field=f, group="quality" if f in quality else GROUP_OF.get(f, "other"), mean_low=round(float(v[lo].mean()), 2), mean_mid=round(float(v[band == 1].mean()), 2),
                        mean_high=round(float(v[hi].mean()), 2), sd_diff=round(float(d), 3), se=round(float(np.sqrt(1 / lo.sum() + 1 / hi.sum())), 3)))
    OUT["fields"] += rows
    if verbose:
        print(f"\nfields that separate high from low band (same format and colour combination; difference in SD units, SE {rows[0]['se']}), top 22:")
        for r in sorted(rows, key=lambda r: -abs(r["sd_diff"]))[:22]:
            print(f"  {r['field']:22} {r['group']:10} low {r['mean_low']:7.2f}  mid {r['mean_mid']:7.2f}  high {r['mean_high']:7.2f}   {r['sd_diff']:+.2f}")
    nb = np.array([r["n_base"] for r in D["rows"]])
    flags = [("removal <= 3", cols["role_removal"] <= 3), ("removal <= 4", cols["role_removal"] <= 4), ("removal >= 8", cols["role_removal"] >= 8),
             ("creatures <= 11", cols["role_creatures"] <= 11), ("creatures >= 18", cols["role_creatures"] >= 18), ("late creatures <= 2", cols["cr_late"] <= 2),
             ("lands <= 15", cols["lands"] <= 15), ("lands >= 18", cols["lands"] >= 18), ("dual lands >= 5", cols["land_multi"] >= 5),
             ("no finishers", cols["finishers"] == 0), ("3+ finishers", cols["finishers"] >= 3), ("2+ weak cards (GIH 1 SD under)", F["weak"] >= 2),
             ("no strong cards (GIH 1 SD over)", F["strong"] == 0), ("3 base colours", nb == 3), ("card flow <= 8", F["flow"] <= 8)]
    if verbose:
        print(f"\n{'share of decks':34} {'low':>7} {'mid':>7} {'high':>7}   low - high (SE)")
    for name, fl in flags:
        p = [float(np.mean(fl[band == b_])) for b_ in range(3)]
        se = np.sqrt(p[0] * (1 - p[0]) / lo.sum() + p[2] * (1 - p[2]) / hi.sum())
        OUT["flags"].append(tag(flag=name, low=round(100 * p[0], 1), mid=round(100 * p[1], 1), high=round(100 * p[2], 1), low_minus_high=round(100 * (p[0] - p[2]), 1), se=round(100 * se, 1)))
        if verbose:
            print(f"{name:34} {100 * p[0]:6.1f}% {100 * p[1]:6.1f}% {100 * p[2]:6.1f}%   {100 * (p[0] - p[2]):+5.1f} ({100 * se:.1f})")
    m2 = band != 1
    D2 = subset(D, m2)
    D2["w"], D2["n"] = (band[m2] == 2).astype(float), np.ones(m2.sum())
    Xc2 = np.column_stack([np.ones(m2.sum()), (fmt[m2] == "PickTwoDraft").astype(float)])
    print("\ncan the deck alone tell a high-band deck from a low-band deck? AUC of the deck score (0.5 = no, 1 = always), out of fold:")
    sep = {}
    for nm, Xm in (("colour combination only", sets["colour"][m2]), ("card quality", Xq[m2]), ("structure + colour", Xs[m2]), ("quality + structure + colour", Xd[m2])):
        l2 = choose_lam(Xc2, Xm, D2)
        _, zd = crossfit(Xc2, Xm, D2, l2)
        a, (c0, c1) = auc(zd, D2["w"]), auc_ci(zd, D2["w"])
        sep[nm] = a
        OUT["separation"].append(tag(model=nm, auc=round(a, 3), lo=round(c0, 3), hi=round(c1, 3)))
        print(f"  {nm:30} {a:.3f} ({c0:.3f}-{c1:.3f})")

    # 3 does the same rule hold in each band
    print("\nwithin-band fits (held out by draft half): gain over the band's own context model, and the deck part's spread")
    z0b, coef = {}, {}
    for b_ in range(3):
        m = band == b_
        Dm = subset(D, m)
        z0b[b_], _ = crossfit(Xc[m], np.zeros((m.sum(), 0)), Dm, 0)
        for nm, Xm in (("quality", Xq[m]), ("structure + colour", Xs[m]), ("quality + structure + colour", Xd[m])):
            l2 = choose_lam(Xc[m], Xm, Dm)
            z, zd = crossfit(Xc[m], Xm, Dm, l2)
            g, se = gain(z0b[b_], z, Dm)
            OUT["within"].append(tag(band=BANDS[b_], model=nm, gain=round(g, 5), se=round(se, 5), spread_sd_points=round(float(PTS * zd.std()), 2)))
            print(f"  {BANDS[b_]:14} {nm:30} gain {g:+.5f} (SE {se:.5f})  spread {PTS * zd.std():.1f} pts")
        Xm = np.hstack([Xc[m], Xd[m]])
        coef[b_] = fit(Xm, Dm["w"], Dm["n"], lams(kc, Xd.shape[1], choose_lam(Xc[m], Xd[m], Dm)), np.ones(m.sum(), bool))[kc:]
    print("\nrules fitted in one band, applied to another (slope 1 = the deck score means the same; 0 = it does not carry over):")
    for s_, d_ in ((1, 0), (2, 0), (0, 2), (1, 2), (0, 1), (2, 1)):
        m = band == d_
        Dm = subset(D, m)
        sl, se, g, gse = calibrate(z0b[d_], Xd[m] @ coef[s_], Dm)
        OUT["transfer"].append(tag(fitted_in=BANDS[s_], applied_to=BANDS[d_], slope=round(sl, 2), slope_se=round(se, 2), gain=round(g, 5), gain_se=round(gse, 5)))
        print(f"  fitted in {BANDS[s_]:14} applied to {BANDS[d_]:14} slope {sl:.2f} (SE {se:.2f})  gain {g:+.5f} (SE {gse:.5f})")

    # 4 curves inside each band
    Q, S = np.hstack([Xq, sets["colour"]]), sets["structure"]
    if verbose:
        print("\neffect curves inside each band (points vs the commonest value, other fields held; * |z| >= 2.5; decks in brackets):")
        for qn, src, lo_, hi_, left_out in CURVES:
            v = np.asarray(cols[src], float)
            b_all = np.clip(np.round(v), lo_, hi_).astype(int) - lo_
            labels = [f"<={lo_}" if i == 0 else (f">={hi_}" if i == hi_ - lo_ else str(lo_ + i)) for i in range(hi_ - lo_ + 1)]
            keep = [i for i, f in enumerate(structure) if f not in left_out]
            for b_ in range(3):
                m = band == b_
                Dm = subset(D, m)
                cnt = np.bincount(b_all[m], minlength=len(labels))
                ref = int(np.argmax(cnt))
                use = [i for i in range(len(labels)) if i != ref and cnt[i] >= 30]
                B = np.column_stack([(b_all[m] == i).astype(float) for i in use])
                bg = np.hstack([Q[m], S[m][:, keep]])
                l2 = choose_lam(Xc[m], bg, Dm, grid=(300, 1000, 3000))
                Xall = np.hstack([Xc[m], bg, B])
                lam_ = lams(kc, Xall.shape[1] - kc, l2)
                lam_[kc + bg.shape[1]:] = 10.0
                bb = fit(Xall, Dm["w"], Dm["n"], lam_, np.ones(m.sum(), bool))
                se_ = np.sqrt(np.diag(sandwich(Xall, Dm["w"], Dm["n"], bb, lam_, Dm["draft"])))
                eff = {i: (PTS * bb[kc + bg.shape[1] + j], PTS * se_[kc + bg.shape[1] + j]) for j, i in enumerate(use)}
                for i in range(len(labels)):
                    e, s_ = eff.get(i, (0.0, 0.0))
                    OUT["curves"].append(tag(quantity=qn, band=BANDS[b_], value=labels[i], decks=int(cnt[i]), reference=int(i == ref), points=round(e, 2), se=round(s_, 2)))
                print(f"  {qn:11} {BANDS[b_]:14} (ref {labels[ref]}): " + "  ".join(
                    f"{labels[i]}:{eff[i][0]:+.1f}{'*' if abs(eff[i][0] / eff[i][1]) >= 2.5 else ''}({cnt[i]})" if i in eff else f"{labels[i]}:0({cnt[i]})" for i in range(len(labels))), flush=True)

    # 5 good cards in bad hands
    tq = np.digitize(zq, np.percentile(zq, [100 / 3, 200 / 3]))
    ts = np.digitize(zs, np.percentile(zs, [100 / 3, 200 / 3]))
    print("\nwin rate (%) by band and card-quality tercile (terciles over all decks of the pass); high - low quality in points:")
    for b_ in range(3):
        c = [wr_cell(D, (band == b_) & (tq == q)) for q in range(3)]
        d, se = c[2][0] - c[0][0], np.hypot(c[2][1], c[0][1])
        OUT["goodcards"].append(tag(band=BANDS[b_], kind="quality tercile", low=round(c[0][0], 2), mid=round(c[1][0], 2), high=round(c[2][0], 2), high_minus_low=round(d, 2), se=round(se, 2)))
        print(f"  {BANDS[b_]:14} " + "  ".join(f"{c[q][0]:5.1f} ({c[q][1]:.1f}; {int(((band == b_) & (tq == q)).sum())} decks)" for q in range(3)) + f"   high - low {d:+.1f} (SE {se:.1f})")
    for b_ in (0, 2):
        print(f"\n{BANDS[b_]} only: win rate by structure tercile (rows) and quality tercile (columns), deck counts in brackets")
        for j, lab in enumerate(("structure low", "structure mid", "structure high")):
            c = [wr_cell(D, (band == b_) & (ts == j) & (tq == q)) for q in range(3)]
            cn = [int(((band == b_) & (ts == j) & (tq == q)).sum()) for q in range(3)]
            for q in range(3):
                OUT["goodcards"].append(tag(band=BANDS[b_], kind="structure x quality tercile", structure=lab, quality=("low", "mid", "high")[q], decks=cn[q], win_rate=round(c[q][0], 2), se=round(c[q][1], 2)))
            print(f"  {lab:14} " + "  ".join(f"{c[q][0]:5.1f} ({c[q][1]:.1f}; {cn[q]})" for q in range(3)))
        m = band == b_
        Dm = subset(D, m)
        t = ts[m]
        M = np.column_stack([np.ones(m.sum()), zc[m], t == 1, t == 2] + [zq[m] * (t == j) for j in range(3)]).astype(float)
        lamq = np.array([0] + [1e-3] * 6)
        bb = fit(M, Dm["w"], Dm["n"], lamq, np.ones(m.sum(), bool))
        se_ = np.sqrt(np.diag(sandwich(M, Dm["w"], Dm["n"], bb, lamq, Dm["draft"])))
        OUT["goodcards"].append(tag(band=BANDS[b_], kind="quality slope by structure tercile", low=round(float(bb[4]), 2), mid=round(float(bb[5]), 2), high=round(float(bb[6]), 2),
                                    se_low=round(float(se_[4]), 2), se_mid=round(float(se_[5]), 2), se_high=round(float(se_[6]), 2)))
        print(f"  slope of the quality score (1 = as the model says) in structure low / mid / high: {bb[4]:.2f} ({se_[4]:.2f}) / {bb[5]:.2f} ({se_[5]:.2f}) / {bb[6]:.2f} ({se_[6]:.2f})")
    topq = tq == 2
    print("\ndecks with top-tercile card quality, low band against high band (what a bad player builds with good cards):")
    print(f"  decks: {int((lo & topq).sum())} low, {int((hi & topq).sum())} high; structure score (points vs average) low {PTS * (zs[lo & topq].mean() - zs.mean()):+.1f}, high {PTS * (zs[hi & topq].mean() - zs.mean()):+.1f}")
    for name, fl in flags:
        a, b2 = float(np.mean(fl[lo & topq])), float(np.mean(fl[hi & topq]))
        se = np.sqrt(a * (1 - a) / max(1, (lo & topq).sum()) + b2 * (1 - b2) / (hi & topq).sum())
        OUT["goodcards"].append(tag(band="top-quality decks", kind="flag share", flag=name, low=round(100 * a, 1), high=round(100 * b2, 1), low_minus_high=round(100 * (a - b2), 1), se=round(100 * se, 1)))
        if verbose and abs(a - b2) > 2 * se:
            print(f"  {name:34} low {100 * a:5.1f}%  high {100 * b2:5.1f}%  ({100 * (a - b2):+.1f}, SE {100 * se:.1f})")

    # 6 complete records
    if rec:
        R = [rec.get((x[0], x[1], x[2])) for x in D["fg"]]
        comp = np.array([bool(r and r[0]) for r in R])
        fw = np.array([r[1] if r else -1 for r in R])
        win_end = np.where(fmt == "PremierDraft", 7, 4)
        bad, good = comp & (fw == 0), comp & (fw == win_end)
        print("\ncomplete records (Premier 7 wins or 3 losses, Pick-Two 4 wins or 2 losses, match numbers 1..k all present):")
        print(f"  {'band':14} {'decks':>6} {'complete':>9} {'0-3/0-2 of complete':>20} {'trophy of complete':>19}")
        for b_ in range(3):
            m = band == b_
            c = comp & m
            OUT["records"].append(tag(band=BANDS[b_], decks=int(m.sum()), complete_share=round(float(c.sum() / m.sum()), 3), bad_share_of_complete=round(float((bad & m).sum() / c.sum()), 3),
                                      trophy_share_of_complete=round(float((good & m).sum() / c.sum()), 3), complete_decks=int(c.sum()), bad_decks=int((bad & m).sum()), trophy_decks=int((good & m).sum())))
            print(f"  {BANDS[b_]:14} {m.sum():6d} {100 * c.sum() / m.sum():8.0f}% {100 * (bad & m).sum() / c.sum():19.1f}% {100 * (good & m).sum() / c.sum():18.1f}%   ({int((bad & m).sum())} bad, {int((good & m).sum())} trophy)")
        print("  AUC of the deck score, 0-3/0-2 decks against trophy decks and against the other complete decks, by band (bootstrap 95% in brackets):")
        for b_ in range(3):
            m = band == b_
            for nm, sc in (("quality", zq), ("structure + colour", zs), ("quality + structure", zq + zs)):
                row = tag(band=BANDS[b_], score=nm)
                txt = []
                for vs, other in (("vs trophy", good), ("vs other complete", comp & ~bad)):
                    mm = m & (bad | other)
                    y = bad[mm].astype(float)
                    if y.sum() < 20 or (1 - y).sum() < 20:
                        continue
                    a, (c0, c1) = auc(sc[mm], y), auc_ci(sc[mm], y)
                    gap_pts = PTS * (sc[mm & other].mean() - sc[mm & bad].mean())
                    row[vs] = round(a, 3)
                    row[vs + " lo"], row[vs + " hi"], row[vs + " n"], row[vs + " points"] = round(c0, 3), round(c1, 3), f"{int((mm & bad).sum())}/{int((mm & other).sum())}", round(gap_pts, 2)
                    txt.append(f"{vs} {a:.3f} ({c0:.3f}-{c1:.3f}), n {int((mm & bad).sum())}/{int((mm & other).sum())}, deck score {gap_pts:+.1f} points")
                OUT["record_auc"].append(row)
                if txt:
                    print(f"    {BANDS[b_]:14} {nm:20} " + "; ".join(txt))
        mm = bad | good
        if verbose:
            print("  tail shares, 0-3/0-2 decks against trophy decks (all bands pooled; bands differ in make-up of both groups):")
            for name, fl in flags:
                a, b2 = float(np.mean(fl[bad])), float(np.mean(fl[good]))
                se = np.sqrt(a * (1 - a) / bad.sum() + b2 * (1 - b2) / good.sum())
                OUT["record_flags"].append(tag(flag=name, bad=round(100 * a, 1), trophy=round(100 * b2, 1), diff=round(100 * (a - b2), 1), se=round(100 * se, 1), bad_decks=int(bad.sum()), trophy_decks=int(good.sum())))
                print(f"    {name:34} 0-3 {100 * a:5.1f}%  trophy {100 * b2:5.1f}%  ({100 * (a - b2):+.1f}, SE {100 * se:.1f})")


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    rec = records(set_code, ("PremierDraft", "PickTwoDraft"))
    for label, formats, games, verbose in PASSES:
        analyse(set_code, label, formats, games, rec, verbose)
    for name, rows in OUT.items():
        write(tables.path(set_code, f"11_skill_{name}.csv"), rows)
    print("\nwrote " + ", ".join(f"11_skill_{n}.csv" for n in OUT))


if __name__ == "__main__":
    main()
