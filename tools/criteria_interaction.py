"""Do strong cards need the rest of the deck? Quality x structure, on the clean design.

    uv run python tools/criteria_interaction.py FRA all|first      # about 1 min

Stage 1 (cross-fit by draft half): context + card quality + colour + structure, as tools/criteria_search.py. Each deck gets
out-of-fold scores: zq (the quality part, a logit), zs (the colour + structure part), zc (context).
Stage 2: the additive model [zc, zq, zs] against (a) + zq*zs, (b) the slope of zq inside each structure tercile, (c) the slope of
zs inside each quality tercile. Held-out gain over additive (log-loss per game, SE clustered by draft) and the slopes with
cluster-robust SEs. A slope of 1 = the score means what it says. "Strong cards can't win in a bad deck" predicts a smaller
zq slope in the bottom structure tercile. Cells: actual win rate minus the context expectation, in points, by tercile pair.
Stage-2 inputs come from stage-1 models fitted on the other half; stage 2 has 4-9 parameters, so the reuse is a small leak.
"""

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
    Xq, Xs = sets["quality"], np.hstack([sets["colour"], sets["structure"]])
    Xd = np.hstack([Xq, Xs])
    lam_d = choose_lam(Xc, Xd, D)
    X = np.hstack([Xc, Xd])
    lam = lams(Xc.shape[1], Xd.shape[1], lam_d)
    kc, kq = Xc.shape[1], Xq.shape[1]
    n = len(D["w"])
    zc, zq, zs = np.zeros(n), np.zeros(n), np.zeros(n)
    for h in (0, 1):
        b = fit(X, D["w"], D["n"], lam, D["half"] != h)
        te = D["half"] == h
        zc[te], zq[te], zs[te] = Xc[te] @ b[:kc], Xq[te] @ b[kc:kc + kq], Xs[te] @ b[kc + kq:]
    print(f"{outcome}: {n} decks, lam {lam_d}; SD of quality score {PTS * zq.std():.2f} pts, structure score {PTS * zs.std():.2f} pts, "
          f"correlation {np.corrcoef(zq, zs)[0, 1]:+.2f}", flush=True)
    cut = lambda z: np.digitize(z, np.percentile(z, [100 / 3, 200 / 3]))
    tq, ts = cut(zq), cut(zs)
    drafts = np.array(D["draft"])

    def cell(m):
        r = (D["w"] - D["n"] * sigmoid(zc))[m]
        nn = D["n"][m]
        mean = r.sum() / nn.sum()
        cl = {}
        for d, x in zip(drafts[m], r - mean * nn):
            cl[d] = cl.get(d, 0.0) + x
        return 100 * mean, 100 * np.sqrt(sum(v * v for v in cl.values())) / nn.sum()

    print("\nactual minus context expectation, points (SE); rows = structure tercile, columns = quality tercile")
    print(f"{'':16}{'quality low':>20}{'quality mid':>20}{'quality high':>20}{'high - low':>18}")
    for j, lab in enumerate(("structure low", "structure mid", "structure high")):
        cells = [cell((ts == j) & (tq == i)) for i in range(3)]
        d = cells[2][0] - cells[0][0]
        se = np.hypot(cells[2][1], cells[0][1])
        print(f"{lab:16}" + "".join(f"{c[0]:+12.1f} ({c[1]:.1f})  " for c in cells) + f"{d:+10.1f} ({se:.1f})")
    print(f"{'decks per cell':16}" + "".join(f"{int(((ts == 0) & (tq == i)).sum()):>6}/{int(((ts == 1) & (tq == i)).sum())}/{int(((ts == 2) & (tq == i)).sum()):<12}" for i in range(3)))

    one = np.ones(n)
    ind = lambda t, j: (t == j).astype(float)

    def design(kind):
        if kind == "additive":
            return [one, zc, zq, zs], ["1", "zc", "zq", "zs"]
        if kind == "product":
            return [one, zc, zq, zs, zq * zs], ["1", "zc", "zq", "zs", "zq*zs"]
        if kind == "quality slope by structure tercile":
            return [one, zc, zs, ind(ts, 1), ind(ts, 2)] + [zq * ind(ts, j) for j in range(3)], ["1", "zc", "zs", "s mid", "s high", "zq | s low", "zq | s mid", "zq | s high"]
        return [one, zc, zq, ind(tq, 1), ind(tq, 2)] + [zs * ind(tq, j) for j in range(3)], ["1", "zc", "zq", "q mid", "q high", "zs | q low", "zs | q mid", "zs | q high"]

    def crossfit2(Xm):
        z = np.zeros(n)
        lam2 = np.full(Xm.shape[1], 1e-3)
        lam2[0] = 0
        for h in (0, 1):
            b = fit(Xm, D["w"], D["n"], lam2, D["half"] != h)
            z[D["half"] == h] = Xm[D["half"] == h] @ b
        return z

    z_add = crossfit2(np.column_stack(design("additive")[0]))
    print()
    for kind in ("additive", "product", "quality slope by structure tercile", "structure slope by quality tercile"):
        cl, names = design(kind)
        Xm = np.column_stack(cl)
        g, se = gain(z_add, crossfit2(Xm), D)
        lam2 = np.full(Xm.shape[1], 1e-3)
        lam2[0] = 0
        b = fit(Xm, D["w"], D["n"], lam2, np.ones(n, bool))
        sd = np.sqrt(np.diag(sandwich(Xm, D["w"], D["n"], b, lam2, D["draft"])))
        show = [i for i, nm in enumerate(names) if nm.startswith("z") and nm != "zc"]
        print(f"  {kind:38} gain over additive {g:+.5f} (SE {se:.5f}, z {g / se if se else 0:+.1f}); " + ", ".join(f"{names[i]} {b[i]:.2f} (SE {sd[i]:.2f})" for i in show))


if __name__ == "__main__":
    main()
