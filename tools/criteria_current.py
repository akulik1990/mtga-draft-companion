"""Criteria search, part 2: the current min / best / max criteria against the data, failure points and combinations.

    uv run python tools/criteria_current.py FRA all|first      # about 2-4 min; writes <data>/tables/SET/10_current_<outcome>_*.csv

Decks: the clean design of tools/criteria_lib.py, restricted to base colour combinations with a fitted `structure` section.
Everything is judged against quality + colour combination + context (so only what the structure adds).
  _models.csv     how much of the structure gain the current criteria carry: the builder's own structure_dev (cards off structure),
                  its parts, the 11 categories linear / in bins around best, plus derived splits, against all fields
  _compensation.csv  the replacement rule: win rate against shortfall below best and against the part the replacements cover
  _bins.csv       effect of each category by distance from best (points of win rate, cluster-robust SE), all categories jointly
  _flags.csv      failure flags (bottom fifth within the colour combination) beyond the linear trend, and pairs of flags together
Effects are in win-rate points (logit x 24.5), ridge-shrunk, so they read slightly small.
"""

import csv
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import DERIVED, GROUP_OF, build  # noqa: E402

CATS = ("creatures", "removal", "card_advantage", "evasion", "mana", "1", "2", "3", "4", "5", "6+")
COLS = dict(creatures="role_creatures", removal="role_removal", card_advantage="role_card_advantage", evasion="role_evasion", mana="role_mana")
LINEAR_OUT = set(COLS.values()) | {"1", "2", "3", "4", "5", "6+"}


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, restval="")
        w.writeheader()
        w.writerows(rows)


def current_features(D, set_code):
    from companion.advisor_v2 import config, deckbuilder
    import companion.advisor_v2.engine  # noqa: F401

    config.activate(set_code)
    slack, ratio = deckbuilder.COMP_SLACK, deckbuilder.COMP_RATIO
    n = len(D["rows"])
    X = {k: np.array([r[COLS.get(k, k)] for r in D["rows"]], float) for k in CATS}
    out = {k: np.zeros(n) for k in ("dev", "below_min", "above_max", "short", "covered", "uncovered", "off_best")}
    pos = {k: np.zeros(n) for k in CATS}
    short_k, unc_k, cov_k = {k: np.zeros(n) for k in CATS}, {k: np.zeros(n) for k in CATS}, {k: np.zeros(n) for k in CATS}
    has = np.zeros(n, bool)
    for i, r in enumerate(D["rows"]):
        lane = r["combo"].split("+")[0]
        bounds, repl = deckbuilder._structure(set_code, lane)
        if not bounds:
            continue
        has[i] = True
        x = {k: X[k][i] for k in CATS}
        for k in CATS:
            lo, best, hi = bounds[k]
            out["below_min"][i] += max(0.0, lo - x[k])
            out["above_max"][i] += max(0.0, x[k] - hi)
            out["off_best"][i] += abs(x[k] - best)
            pos[k][i] = x[k] - best
            sh = best - x[k] - slack
            if sh > 0:
                cv = deckbuilder.covered(repl.get(k, ()), x, bounds)
                short_k[k][i], cov_k[k][i] = sh, min(cv, ratio * sh)
                unc_k[k][i] = max(0.0, ratio * sh - cv)
        out["short"][i] = sum(short_k[k][i] for k in CATS)
        out["covered"][i] = sum(cov_k[k][i] for k in CATS)
        out["uncovered"][i] = sum(unc_k[k][i] for k in CATS)
        out["dev"][i] = out["below_min"][i] + out["above_max"][i] + out["uncovered"][i]
    return has, out, pos, dict(short=short_k, covered=cov_k, uncovered=unc_k), (slack, ratio)


def coef_table(Xc, Xd, D, lam_d, names, scale=PTS):
    X = np.hstack([Xc, Xd])
    lam = lams(Xc.shape[1], Xd.shape[1], lam_d)
    b = fit(X, D["w"], D["n"], lam, np.ones(len(D["w"]), bool))
    V = sandwich(X, D["w"], D["n"], b, lam, D["draft"])
    se = np.sqrt(np.diag(V))
    k = Xc.shape[1]
    return [(names[i], scale * b[k + i], scale * se[k + i], b[k + i] / se[k + i]) for i in range(len(names))]


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    D0 = load_data(set_code, outcome)
    from companion.advisor_v2 import config, deckbuilder
    import companion.advisor_v2.engine  # noqa: F401

    config.activate(set_code)
    has0 = np.array([bool(deckbuilder._structure(set_code, r["combo"].split("+")[0])[0]) for r in D0["rows"]])
    D = subset(D0, has0)
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    has, cur, pos, ck, (slack, ratio) = current_features(D, set_code)
    name = lambda s: tables.path(set_code, f"10_current_{outcome}_{s}.csv")
    print(f"{outcome}: {len(D['w'])} decks with a structure section ({has0.sum()} of {len(has0)}), {int(D['n'].sum())} games; COMP_SLACK {slack}, COMP_RATIO {ratio}", flush=True)
    print("  decks off structure (dev > 0): {:.1f}%; mean dev {:.2f}; below min {:.2f}, above max {:.2f}, short {:.2f}, covered {:.2f}, uncovered {:.2f}".format(
        100 * (cur["dev"] > 0).mean(), cur["dev"].mean(), cur["below_min"].mean(), cur["above_max"].mean(), cur["short"].mean(), cur["covered"].mean(), cur["uncovered"].mean()), flush=True)

    Q = np.hstack([sets["quality"], sets["colour"]])
    lam_q = choose_lam(Xc, Q, D)
    z_base, _ = crossfit(Xc, Q, D, lam_q)
    gq, seq = gain(*(lambda z0: (z0, z_base))(crossfit(Xc, np.zeros((len(D["w"]), 0)), D, 0)[0]), D)
    print(f"  quality + colour over context: {gq:.5f} (SE {seq:.5f})", flush=True)
    S = sets["structure"]
    st_idx = {f: i for i, f in enumerate(structure)}
    rest = [i for i, f in enumerate(structure) if f not in LINEAR_OUT]
    cat_lin = standardise(np.column_stack([cur_k for cur_k in (np.array([r[COLS.get(k, k)] for r in D["rows"]], float) for k in CATS)]))
    der = standardise(np.column_stack([cols[f] for f in DERIVED if f not in LINEAR_OUT] + [cols["lands"] if "lands" in cols else F["lands"]]))
    one = lambda v: standardise(np.asarray(v)[:, None])
    posb = {}
    for k in CATS:
        d = np.clip(np.round(pos[k]), -4, 4)
        posb[k] = np.column_stack([(d == v).astype(float) for v in range(-4, 5) if v != 0])
    bins = np.hstack([posb[k] for k in CATS])
    raw = lambda k: np.array([r[COLS.get(k, k)] for r in D["rows"]], float)

    variants = [
        ("builder's structure_dev (cards off structure)", one(cur["dev"])),
        ("  its parts: below min, above max, uncovered shortfall", standardise(np.column_stack([cur["below_min"], cur["above_max"], cur["uncovered"]]))),
        ("  shortfall below best and the part covered, separately", standardise(np.column_stack([cur["short"], cur["covered"]]))),
        ("cards away from best (sum over categories)", one(cur["off_best"])),
        ("11 categories, linear", cat_lin),
        ("11 categories, in bins around best", bins),
        ("11 categories linear + derived splits (early/late creatures, cheap removal ...)", np.hstack([cat_lin, der])),
        ("11 categories in bins + derived splits", np.hstack([bins, der])),
        ("all 133 structure fields", S),
    ]
    res = []
    for label, Xd in variants:
        lam = choose_lam(Xc, np.hstack([Q, Xd]), D)
        z, _ = crossfit(Xc, np.hstack([Q, Xd]), D, lam)
        g, se = gain(z_base, z, D)
        gh = [gain(z_base, z, D, D["half"] == h) for h in (0, 1)]
        res.append(dict(model=label.strip(), columns=Xd.shape[1], lam=lam, gain_over_quality_colour=round(g, 5), se=round(se, 5), z=round(g / se, 1),
                        half0=round(gh[0][0], 5), half1=round(gh[1][0], 5)))
        print(f"  {label:78} cols {Xd.shape[1]:3} gain {g:+.5f} (SE {se:.5f}, z {g / se:+.1f}) halves {gh[0][0]:+.5f} / {gh[1][0]:+.5f}  [{time.time() - t0:.0f}s]", flush=True)
    write(name("models"), res)

    # Coefficients of the builder's parts (points of win rate per card, not standardised), controls = quality + colour + all other fields
    bg = np.hstack([Q, S[:, rest]])
    lam_bg = choose_lam(Xc, bg, D)
    comp_rows = []
    for label, cols_, nm in (("structure_dev", [cur["dev"]], ["cards off structure"]),
                             ("parts", [cur["below_min"], cur["above_max"], cur["uncovered"]], ["below min (per card)", "above max (per card)", "uncovered shortfall (per card)"]),
                             ("shortfall / covered", [cur["short"], cur["covered"]], ["shortfall below best less slack (per card)", "of it covered by replacements (per card)"])):
        Xd = np.column_stack(cols_)
        tab = coef_table(Xc, np.hstack([bg, Xd]), D, lam_bg, [f"bg{i}" for i in range(bg.shape[1])] + nm)[-len(nm):]
        for n_, c, s, z in tab:
            comp_rows.append(dict(model=label, term=n_, points_per_card=round(c, 2), se=round(s, 2), z=round(z, 1)))
            print(f"  {label:20} {n_:48} {c:+.2f} points (SE {s:.2f}, z {z:+.1f})", flush=True)
    for k in CATS:  # shortfall and covered per category (compensation of each)
        Xd = np.column_stack([ck["short"][k], ck["covered"][k]])
        if (Xd[:, 0] > 0).sum() < 300:
            continue
        tab = coef_table(Xc, np.hstack([bg, Xd]), D, lam_bg, [f"bg{i}" for i in range(bg.shape[1])] + ["short", "covered"])[-2:]
        comp_rows.append(dict(model=f"category {k}", term=f"shortfall ({int((Xd[:, 0] > 0).sum())} decks)", points_per_card=round(tab[0][1], 2), se=round(tab[0][2], 2), z=round(tab[0][3], 1)))
        comp_rows.append(dict(model=f"category {k}", term="covered by replacements", points_per_card=round(tab[1][1], 2), se=round(tab[1][2], 2), z=round(tab[1][3], 1)))
        print(f"  category {k:14} {int((Xd[:, 0] > 0).sum()):5} decks short: shortfall {tab[0][1]:+.2f} (z {tab[0][3]:+.1f}), covered {tab[1][1]:+.2f} (z {tab[1][3]:+.1f})", flush=True)
    write(name("compensation"), comp_rows)

    # Bins around best: effect of each category by distance from best, all categories jointly, other fields controlled
    names = [f"{k}:{v:+d}" for k in CATS for v in range(-4, 5) if v != 0]
    Xb = np.hstack([bg, bins])
    lam_vec = lams(Xc.shape[1], Xb.shape[1], lam_bg)
    lam_vec[Xc.shape[1] + bg.shape[1]:] = 10.0  # the bins: weak shrinkage
    X = np.hstack([Xc, Xb])
    b = fit(X, D["w"], D["n"], lam_vec, np.ones(len(D["w"]), bool))
    V = sandwich(X, D["w"], D["n"], b, lam_vec, D["draft"])
    se = np.sqrt(np.diag(V))
    k0 = Xc.shape[1] + bg.shape[1]
    binrows = []
    for j, nm in enumerate(names):
        cat, v = nm.split(":")
        cnt = int((np.clip(np.round(pos[cat]), -4, 4) == int(v)).sum())
        binrows.append(dict(category=cat, distance_from_best=int(v), decks=cnt, points=round(PTS * b[k0 + j], 2), se=round(PTS * se[k0 + j], 2), z=round(b[k0 + j] / se[k0 + j], 1)))
    write(name("bins"), binrows)
    print("  effect by distance from best (points of win rate vs at best; * = |z| >= 2.5):")
    for k in CATS:
        cells = [r for r in binrows if r["category"] == k]
        print(f"    {k:14}" + " ".join(f"{r['distance_from_best']:+d}:{r['points']:+5.1f}{'*' if abs(r['z']) >= 2.5 else ' '}({r['decks']})" for r in cells), flush=True)

    # Failure flags: bottom fifth within colour combination, beyond the linear trend of every field
    cq = {"few creatures": "creatures", "few late creatures (cmc 4+)": "cr_late", "few early creatures (cmc <=3)": "cr_early", "few removal": "removal",
          "few card flow": "flow", "few lands": "lands", "few cheap removal": "rm_cheap", "few finishers+top end": "cr_big"}
    flags = {}
    for label, f in cq.items():
        v = F[f] if f in F else cols[f]
        fl = np.zeros(len(v))
        for c in combos:
            m = base == c
            fl[m] = v[m] <= np.percentile(v[m], 20)
        flags[label] = fl
    fl_names = list(flags)
    Xf = np.column_stack([flags[k] for k in fl_names])
    pair_names = [f"{a} + {b}" for a, b in combinations(fl_names, 2)]
    Xp = np.column_stack([flags[a] * flags[b] for a, b in combinations(fl_names, 2)])
    flag_rows = []
    for title, Xd, nm in (("flag beyond linear trend", Xf, fl_names), ("both flags, beyond the two flags and the trend", np.hstack([Xf, Xp]), fl_names + pair_names)):
        Xall = np.hstack([bg, Xd])
        lam_v = lams(Xc.shape[1], Xall.shape[1], lam_bg)
        lam_v[Xc.shape[1] + bg.shape[1]:] = 10.0
        Xfull = np.hstack([Xc, Xall])
        b = fit(Xfull, D["w"], D["n"], lam_v, np.ones(len(D["w"]), bool))
        V = sandwich(Xfull, D["w"], D["n"], b, lam_v, D["draft"])
        se = np.sqrt(np.diag(V))
        for j, nm_ in enumerate(nm):
            jj = Xc.shape[1] + bg.shape[1] + j
            decks = int(Xd[:, j].sum())
            flag_rows.append(dict(analysis=title, term=nm_, decks=decks, points=round(PTS * b[jj], 2), se=round(PTS * se[jj], 2), z=round(b[jj] / se[jj], 1)))
            if title.startswith("flag") or abs(b[jj] / se[jj]) >= 2 or "creatures + few removal" in nm_:
                print(f"  {title[:22]:22} {nm_:62} {decks:6} decks  {PTS * b[jj]:+.2f} points (SE {PTS * se[jj]:.2f}, z {b[jj] / se[jj]:+.1f})", flush=True)
    zs = np.array([r["z"] for r in flag_rows if r["analysis"].startswith("both")])
    print(f"  pairs: {len(zs)} tested, |z| >= 2: {int((np.abs(zs) >= 2).sum())} (chance ~{0.0455 * len(zs):.1f}), |z| >= 3: {int((np.abs(zs) >= 3).sum())}", flush=True)
    write(name("flags"), flag_rows)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
