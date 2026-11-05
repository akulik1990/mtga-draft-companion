"""Builder comparison on top players' own pools: structure-first (A) against the gate off (R), scored by a held-out model.

    uv run python tools/builder_compare.py FRA --n 20        # pilot, about 1-2 min
    uv run python tools/builder_compare.py FRA --n 100

Pools: builder_check.sample (60%+ players, 2-3 base colours, hash buckets 0-300 of 1000), builds in each arm with the set's fitted colour rules.
A: deckbuilder.STRUCTURE on (the set's setting). R: STRUCTURE off, everything else the same (the earlier mode).
Scoring: the criteria_score model (context + quality + colour + 133 structure fields, ridge logistic) fitted on decks of 100+ games players OUTSIDE hash
buckets 0-300, so the check pools and the registered decks (T) are held out. The deck part in win-rate points (logit x 24.5), split into quality,
colour and structure. Control: a leave-out model with the removal group, finishers, lands, spells and dual-land fields removed; an arm that wins only
on the fields it optimises (removal, finishers, lands) does not win there. SEs: the pool-to-pool SE of the paired difference and the coefficient SE
(dx' V dx, cluster-robust by draft) are combined in quadrature. Builds are cached in <data>/Temp/compare_builds_SET_N.pkl.
"""

import argparse
import hashlib
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import GROUPS, build, write  # noqa: E402
from tools import builder_check as bc  # noqa: E402

LEAVE_OUT = set(GROUPS["removal"].split()) | {"finishers", "lands", "spells", "land_multi", "land_mono"}
_T = {}


def finisher_info(set_code):
    if not _T:
        t = card_table(set_code)
        gw = np.array([float(t[n].get("gihwr") or 0) for n in t if not (t[n].get("land") or 0) and not isinstance(t[n].get("gihwr"), str)])
        _T.update(table=t, thr=gw[gw > 0].mean() + gw[gw > 0].std())
    return _T["table"], _T["thr"]


def arms(item):
    from companion.advisor_v2 import deckbuilder
    from companion.advisor_v2.lanes import castable

    lane, splash, deck, pool = item
    A = bc.ARGS
    by_name = A["by_name"]
    table, thr = finisher_info(A["set"])
    cards = [by_name[n] for n, k in pool.items() if n in by_name for _ in range(k)]
    theirs = Counter({n: k for n, k in deck.items() if n in by_name and "Land" not in (by_name[n].get("types") or [])})
    out = {"lane": lane, "their": dict(deck)}
    for arm, flag in (("A", True), ("R", False)):
        deckbuilder.STRUCTURE = flag
        builds = deckbuilder.build(cards, A["model"], A["targets"], top=1, rule=A["rule"], three=A["three"])
        if not builds:
            out[arm] = None
            continue
        b = builds[0]
        counts = Counter(c["name"] for c in b["spells"])
        counts.update(b["basics"])
        counts.update(c["name"] for c in b["nonbasics"])
        ours = Counter(c["name"].split(" // ")[0] for c in b["spells"])
        fin_left = sum(1 for c in b["sideboard"] if "Land" not in (c.get("types") or []) and castable(c, b["lane"] + b["splash"])
                       and c["name"] in table and (table[c["name"]].get("cmc") or 0) >= 5 and float(table[c["name"]].get("gihwr") or 0) >= thr)
        out[arm] = dict(counts=dict(counts), lane=b["lane"], splash=b["splash"], predicted=float(b["predicted"]), names=sorted(ours.elements()),
                        overlap=sum((ours & theirs).values()) / max(1, sum(theirs.values())), fin_left=fin_left)
    deckbuilder.STRUCTURE = True
    return out


def fit_model(D, drop=frozenset(), lam_d=None):
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    structure = [f for f in structure if f not in drop]
    rawS = np.column_stack([cols[f] for f in structure])
    rawQ = np.column_stack([cols[f] for f in quality])
    muS, sdS = rawS.mean(0), np.where(rawS.std(0) == 0, 1, rawS.std(0))
    muQ, sdQ = rawQ.mean(0), np.where(rawQ.std(0) == 0, 1, rawQ.std(0))
    Xd = np.hstack([(rawQ - muQ) / sdQ, sets["colour"], (rawS - muS) / sdS])
    lam_d = lam_d or choose_lam(Xc, Xd, D)
    X = np.hstack([Xc, Xd])
    lam = lams(Xc.shape[1], Xd.shape[1], lam_d)
    b = fit(X, D["w"], D["n"], lam, np.ones(len(D["w"]), bool))
    V = sandwich(X, D["w"], D["n"], b, lam, D["draft"])
    k = Xc.shape[1]
    return dict(structure=structure, quality=quality, combos=combos, muS=muS, sdS=sdS, muQ=muQ, sdQ=sdQ, bd=b[k:], Vd=V[k:, k:], lam=lam_d,
                nq=len(quality), nc=len(combos) - 1, Xd=Xd)


def design(M, f):
    sq = (np.column_stack([f[q] for q in M["quality"]]) - M["muQ"]) / M["sdQ"]
    ss = (np.column_stack([f[s] for s in M["structure"]]) - M["muS"]) / M["sdS"]
    cl = np.array([[bb == c for c in M["combos"][1:]] for bb in f["base"]], float)
    return np.hstack([sq, cl, ss])


def parts(M, X):
    nq, nc, b = M["nq"], M["nc"], M["bd"]
    return dict(total=X @ b, quality=X[:, :nq] @ b[:nq], colour=X[:, nq:nq + nc] @ b[nq:nq + nc], structure=X[:, nq + nc:] @ b[nq + nc:])


def paired(M, XA, XB, label):
    """Mean of (B - A) in points, pool SE, coefficient SE, combined, share of pools that differ."""
    d = PTS * ((XB - XA) @ M["bd"])
    n = len(d)
    se_pool = d.std(ddof=1) / np.sqrt(n)
    dm = (XB - XA).mean(0)
    se_coef = PTS * np.sqrt(dm @ M["Vd"] @ dm)
    se = float(np.hypot(se_pool, se_coef))
    return dict(label=label, diff=round(float(d.mean()), 2), se_pool=round(float(se_pool), 2), se_coef=round(float(se_coef), 2), se=round(se, 2),
                lo=round(float(d.mean() - 1.96 * se), 2), hi=round(float(d.mean() + 1.96 * se), 2), differ=round(100 * float((np.abs(d) > 1e-9).mean())), sd=round(float(d.std(ddof=1)), 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("set")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--reuse", action="store_true", help="use the cached builds")
    a = p.parse_args()
    t0 = time.time()
    cache = DATA_DIR / "Temp" / f"compare_builds_{a.set}_{a.n}.pkl"
    if a.reuse and cache.exists():
        rows = pickle.loads(cache.read_bytes())
    else:
        bc.init(a.set, [], False)
        items = bc.sample(a.set, a.n, colours=(2, 3), buckets=bc.CHECK)
        with Pool(a.jobs, initializer=bc.init, initargs=(a.set, [], False)) as pool:
            rows = pool.map(arms, items, chunksize=1)
        cache.write_bytes(pickle.dumps(rows))
    n0 = len(rows)
    rows = [r for r in rows if r["A"] and r["R"]]
    print(f"{len(rows)} of {n0} pools built in both arms ({time.time() - t0:.0f}s)", flush=True)

    D0 = load_data(a.set, "all")
    out_of_check = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 1000 >= bc.CHECK[1] for d in D0["draft"]])
    D = subset(D0, out_of_check)
    print(f"scoring model fitted on {len(D['w'])} decks outside the check buckets", flush=True)
    full = fit_model(D)
    leave = fit_model(D, LEAVE_OUT, full["lam"])
    print(f"models fitted ({time.time() - t0:.0f}s); lam {full['lam']}", flush=True)

    sets_ = {"A": [r["A"]["counts"] for r in rows], "R": [r["R"]["counts"] for r in rows], "T": [r["their"] for r in rows]}
    F = {k: deck_fields(a.set, v) for k, v in sets_.items()}
    X = {m: {k: design(M, F[k]) for k in F} for m, M in (("full", full), ("leave-out", leave))}
    chk = [d["counts"] for d in D["decks"][:200]]
    print(f"design check on 200 real decks: max |difference| {np.abs(design(full, deck_fields(a.set, chk)) @ full['bd'] - full['Xd'][:200] @ full['bd']).max():.2e}")

    res = []
    for m, M in (("full", full), ("leave-out", leave)):
        print(f"\n== scoring model: {m} ({len(M['structure'])} structure fields)")
        print(f"{'arm':4} {'total':>7} {'quality':>8} {'colour':>7} {'structure':>9}   (mean deck part, points; T = their registered deck)")
        for k in ("A", "R", "T"):
            pt = {q: PTS * v.mean() for q, v in parts(M, X[m][k]).items()}
            print(f"{k:4} {pt['total']:7.2f} {pt['quality']:8.2f} {pt['colour']:7.2f} {pt['structure']:9.2f}")
        for lab, ka, kb in (("R - A (gate off minus structure-first)", "A", "R"), ("T - A (their deck minus A)", "A", "T"), ("T - R", "R", "T")):
            r = paired(M, X[m][ka], X[m][kb], lab)
            pa, pb = parts(M, X[m][ka]), parts(M, X[m][kb])
            sub = {q: round(float(PTS * (pb[q] - pa[q]).mean()), 2) for q in ("quality", "colour", "structure")}
            print(f"{lab:40} {r['diff']:+.2f}  95% CI {r['lo']:+.2f} to {r['hi']:+.2f}  (pool SE {r['se_pool']}, coef SE {r['se_coef']}); builds differ in {r['differ']}% of pools; SD of per-pool diff {r['sd']}; parts {sub}")
            res.append(dict(model=m, **r, **{f"part_{q}": v for q, v in sub.items()}))
    print("\n== what the builds look like (means over pools)")
    f_ = lambda k, field: float(np.mean(F[k][field]))
    print(f"{'arm':4} {'removal':>7} {'finish':>6} {'duals':>5} {'lands':>5} {'lcyc':>5} {'creat':>5} {'splash%':>7} {'three%':>6} {'same pair%':>10} {'overlap%':>8} {'pred WR':>7} {'fin left%':>9}")
    for k in ("A", "R", "T"):
        arm = [r.get(k) or {} for r in rows] if k != "T" else None
        if k != "T":
            lanes = [r[k]["lane"] for r in rows]
            sp = 100 * np.mean([bool(r[k]["splash"]) for r in rows]); th = 100 * np.mean([len(l) == 3 for l in lanes])
            same = 100 * np.mean([r[k]["lane"] == r["lane"] for r in rows]); ov = 100 * np.mean([r[k]["overlap"] for r in rows])
            pr = np.mean([r[k]["predicted"] for r in rows]); fl = 100 * np.mean([r[k]["fin_left"] > 0 for r in rows])
            extra = f"{sp:7.1f} {th:6.1f} {same:10.1f} {ov:8.1f} {pr:7.2f} {fl:9.1f}"
        else:
            extra = f"{'':7} {100 * np.mean([len(r['lane']) == 3 for r in rows]):6.1f}"
        print(f"{k:4} {f_(k, 'removal'):7.2f} {f_(k, 'finishers'):6.2f} {f_(k, 'land_multi'):5.2f} {f_(k, 'lands'):5.2f} {f_(k, 'landcycling'):5.2f} {f_(k, 'creatures'):5.2f} {extra}")
    differ = [i for i, r in enumerate(rows) if Counter(r["A"]["names"]) != Counter(r["R"]["names"])]
    print(f"\nspells differ between A and R in {len(differ)} of {len(rows)} pools; card differences in the first 5 of those:")
    for i in differ[:5]:
        ca, cr = Counter(rows[i]["A"]["names"]), Counter(rows[i]["R"]["names"])
        print(f"  pool {i} ({rows[i]['lane']}, A {rows[i]['A']['lane']}{'+' + rows[i]['A']['splash'] if rows[i]['A']['splash'] else ''}, R {rows[i]['R']['lane']}{'+' + rows[i]['R']['splash'] if rows[i]['R']['splash'] else ''}): "
              f"only A: {', '.join((ca - cr).elements()) or '-'} | only R: {', '.join((cr - ca).elements()) or '-'}")
    write(tables.path(a.set, f"14_builder_compare_{a.n}.csv"), res)
    print(f"\ndone ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
