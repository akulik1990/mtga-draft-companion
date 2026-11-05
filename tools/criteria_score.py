"""Score decks with a model fitted on the clean design, and show the 2 Nov UB pool under it.

    uv run python tools/criteria_score.py FRA [LOG]      # about 1 min

Model: context + quality (rank buckets, rarity) + colour combination + structure (133 fields), ridge logistic fitted on all
35,646 decks of players with 100+ games (all their games). The deck part is scored in win-rate points (logit x 24.5) against the
average UB deck of those players, with a cluster-robust SE for differences between decks. It is a descriptive score of
make-up, not the builder's objective: its card-quality fields are coarse (rank buckets), so it cannot rank individual cards.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build  # noqa: E402


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    log = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "Temp" / "live" / "FRA-quickdraft-2026-11-02.Player.log"
    D = load_data(set_code, "all")
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    rawS = np.column_stack([cols[f] for f in structure])
    rawQ = np.column_stack([cols[f] for f in quality])
    muS, sdS = rawS.mean(0), np.where(rawS.std(0) == 0, 1, rawS.std(0))
    muQ, sdQ = rawQ.mean(0), np.where(rawQ.std(0) == 0, 1, rawQ.std(0))
    Xd = np.hstack([(rawQ - muQ) / sdQ, sets["colour"], (rawS - muS) / sdS])
    lam_d = choose_lam(Xc, Xd, D)
    X = np.hstack([Xc, Xd])
    lam = lams(Xc.shape[1], Xd.shape[1], lam_d)
    b = fit(X, D["w"], D["n"], lam, np.ones(len(D["w"]), bool))
    V = sandwich(X, D["w"], D["n"], b, lam, D["draft"])
    k = Xc.shape[1]
    bd, Vd = b[k:], V[k:, k:]
    print(f"model fitted on {len(D['w'])} decks, lam {lam_d}", flush=True)

    def design(counts_list):
        f = deck_fields(set_code, counts_list)
        sq = (np.column_stack([f[q] for q in quality]) - muQ) / sdQ
        ss = (np.column_stack([f[s] for s in structure]) - muS) / sdS
        cl = np.array([[bb == c for c in combos[1:]] for bb in f["base"]], float)
        return np.hstack([sq, cl, ss]), f["base"]

    ub = base == "UB"
    s_all = Xd @ bd
    ref, sd_ub = s_all[ub].mean(), s_all[ub].std()
    print(f"UB decks of these players: {int(ub.sum())}; SD of the deck part {PTS * sd_ub:.1f} points", flush=True)

    # check: the designed rows reproduce table 3 for real decks
    chk = [d["counts"] for d in D["decks"][:200]]
    Xchk, _ = design(chk)
    print(f"design check on 200 real decks: max |difference| in the deck part {np.abs(Xchk @ bd - s_all[:200]).max():.2e}", flush=True)

    import companion.advisor_v2.engine  # noqa: F401
    from companion.advisor_v2 import composition, config, deckbuilder
    from tools.deck_model import DeckModel
    from tools.live_diff import picks
    from tools.replay.run import replay

    config.activate(set_code)
    app = replay.App(str(DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json"))
    ratings = {kk: dict(v, set=set_code) for kk, v in app.dataset.get_card_ratings().items()}
    last = picks(log)[-1]
    pool = [ratings[i] for i in list(last[3]) + list(last[2]) if i in ratings]
    model = DeckModel(set_code, deckbuilder.MODEL)
    targets = config.fitted(set_code, "decks")["pairs"]
    rule, three = config.fitted(set_code, "splash"), config.fitted(set_code, "three")
    cands = []
    for label, structure_on in (("current builder", True), ("builder without structure-first (the earlier mode)", False)):
        deckbuilder.STRUCTURE = structure_on
        for bld in deckbuilder.build(pool, model, targets, top=4, rule=rule, three=three):
            counts = Counter(c["name"] for c in bld["spells"])
            counts.update(bld["basics"])
            counts.update(c["name"] for c in bld["nonbasics"])
            cands.append((f"{label}: {bld['lane']} {bld.get('variant') or 'first'}{' +' + bld['splash'] if bld['splash'] else ''}", dict(counts), bld))
    deckbuilder.STRUCTURE = True
    uniq, seen = [], set()
    for lab, c, bld in cands:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append((lab, c, bld))
    Xn, bases = design([c for _, c, _ in uniq])
    sn = Xn @ bd
    ref_row = None
    print(f"\n{'build':74} {'WR pts vs avg UB':>16} {'pctile in UB':>12} {'cr':>3} {'rm':>3} {'lands':>5} {'dual':>4} {'finish':>6} {'pred':>6}")
    rl = lambda cs: sum("removal" in config.tags(c) for c in cs)
    rows = []
    for (lab, c, bld), s, x, bb in zip(uniq, sn, Xn, bases):
        fl = deck_fields(set_code, [c])
        pct = 100 * (s_all[ub] < s).mean()
        print(f"{lab:74} {PTS * (s - ref):+10.1f}       {pct:9.0f}% {int(fl['creatures'][0]):3} {int(fl['removal'][0]):3} {int(fl['lands'][0]):5} {int(fl['land_multi'][0]):4} {int(fl['finishers'][0]):6} {bld['predicted']:6.1f}")
        rows.append((lab, s, x))
    # differences against the first build with SE, and what moves them
    s0, x0 = rows[0][1], rows[0][2]
    print("\ndifference against the first build (points, cluster-robust SE) and its largest parts:")
    names = [f"q:{q}" for q in quality] + [f"colour:{c}" for c in combos[1:]] + structure
    for lab, s, x in rows[1:]:
        dx = x - x0
        se = np.sqrt(dx @ Vd @ dx)
        parts = np.argsort(-np.abs(dx * bd))[:4]
        print(f"  {lab[:70]:70} {PTS * (s - s0):+.2f} (SE {PTS * se:.2f}); " + ", ".join(f"{names[j]} {PTS * dx[j] * bd[j]:+.2f}" for j in parts))
    first = uniq[0][2]
    print("\nfirst build:", ", ".join(sorted(c["name"] for c in first["spells"])), "| basics", first["basics"], "| nonbasics", [c["name"] for c in first["nonbasics"]])
    print("its deck part broken down by group (points vs the average UB deck):")
    from criteria_search import GROUP_OF

    mean_x = Xd[ub].mean(0)
    dx = rows[0][2] - mean_x
    grp = {}
    for j, nme in enumerate(names):
        g = "quality" if nme.startswith("q:") else ("colour" if nme.startswith("colour:") else GROUP_OF.get(nme, "other"))
        grp[g] = grp.get(g, 0.0) + PTS * dx[j] * bd[j]
    for g, v in sorted(grp.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {g:10} {v:+.2f}")
    big = np.argsort(-np.abs(dx * bd))[:8]
    print("largest single fields:", ", ".join(f"{names[j]} {PTS * dx[j] * bd[j]:+.2f}" for j in big))


if __name__ == "__main__":
    main()
