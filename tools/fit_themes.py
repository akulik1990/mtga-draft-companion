"""Test synergy themes against real games: does "enablers x payoffs" in a deck predict wins beyond its cards and shape?

    uv run python tools/fit_themes.py FRA        # writes the synergy section of the set's fitted.json

Themes are hypotheses in the set's manual.json ("themes": patterns over oracle text, from guides and videos, with
sources). For each theme a deck gets x = enablers x payoffs / 23 (cards tagged by companion/advisor_v2/synergy.roles),
or for a "count" theme the number of enablers (or, with "shortfall": n, how far below n the deck is).
The deck model's design (context, shape, card counts; tools/deck_model.py) is refitted with the theme columns added, on
its own draft-id split, at its chosen penalty. Reported per theme, alone and all together: held-out log-loss gain over
the model without themes, the coefficient with an approximate SE (Wald, theme block only), and what a typical package
(the median x among decks with any) is worth in win-rate points at 50%. Nothing here changes the engine: the results
are recorded so guide claims can be judged against data.
"""

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.synergy import roles  # noqa: E402
from tools.deck_model import SHAPE, card_facts, fit_logistic, read  # noqa: E402

DATA = {}


def design(ctx, shp, C, test):
    mu, sd = shp[~test].mean(0), shp[~test].std(0)
    sd[sd == 0] = 1.0
    S = (shp - mu) / sd
    S2 = np.column_stack([S, S[:, SHAPE.index("creatures")] ** 2, S[:, SHAPE.index("lands")] ** 2, S[:, SHAPE.index("cmc2")] ** 2])
    return sparse.hstack([sparse.csr_matrix(ctx), sparse.csr_matrix(S2), C]).tocsr()


def offset_fit(z, x, y):
    """Logistic fit of y on a fixed linear predictor z plus beta * x: (intercept shift, beta)."""
    from scipy.optimize import minimize

    def f(w):
        q = 1 / (1 + np.exp(-(z + w[0] + w[1] * x)))
        loss = -(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12)).sum()
        r = q - y
        return loss, np.array([r.sum(), (r * x).sum()])

    return minimize(f, np.zeros(2), jac=True, method="L-BFGS-B").x


def run(cols):
    """The deck model fitted once without themes; each theme then fitted alone on top of its linear predictor (an
    offset), so the test asks what the theme adds given the cards and shape. (held-out gain, beta, cluster-bootstrap SE)."""
    z_tr, z_te, T, y, test, groups = DATA["z_tr"], DATA["z_te"], DATA["T"], DATA["y"], DATA["test"], DATA["groups"]
    i = cols[0]
    x_tr, x_te = T[~test][:, i], T[test][:, i]
    b0, b = offset_fit(z_tr, x_tr, y[~test])
    gain = logloss_z(z_te, y[test]) - logloss_z(z_te + b0 + b * x_te, y[test])
    rng = np.random.default_rng(i)
    g_tr = groups[~test]
    ids = np.unique(g_tr)
    boots = []
    for _ in range(20):
        pick = rng.choice(ids, size=len(ids))
        rows = np.concatenate([DATA["rows_by_group"][g] for g in pick])
        boots.append(offset_fit(z_tr[rows], x_tr[rows], y[~test][rows])[1])
    return gain, float(b), float(np.std(boots))


def logloss_z(z, y):
    q = 1 / (1 + np.exp(-z))
    return float(-(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12)).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args(argv)

    themes = config.layer(a.set, "manual").get("themes") or {}
    if not themes:
        raise SystemExit(f"no themes in {config.set_file(a.set, 'manual')}")
    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {}
    for c in data.values():
        if "Basic" in (c.get("types") or []):
            continue
        for n in (c["name"], c["name"].split(" // ")[0]):
            by.setdefault(n, c)
    names, C, ctx, shp, y, test, drafts = read(a.set, card_facts(a.set), ids=True)
    print(f"{len(y)} games, {int(test.sum())} held out")
    X = design(ctx, shp, C, test)
    T, info = [], {}
    for t, th in themes.items():
        tags = [roles(by[n], th) if n in by else (False, False) for n in names]
        e = np.array([float(x[0]) for x in tags], np.float32)
        pay = np.array([float(x[1]) for x in tags], np.float32)
        if th.get("count"):
            # A deck-level count ("blue decks want 3-4 card advantage spells"): the number of tagged cards, or how far
            # short of a target it falls ("shortfall": n).
            k = C @ e
            x = np.maximum(0.0, th["shortfall"] - k) if th.get("shortfall") else k
        else:
            x = (C @ e) * (C @ pay) / 23.0
        typical = float(np.median(x[x > 0])) if (x > 0).any() else 0.0
        sd = float(x[~test].std()) or 1.0
        T.append(x / sd)
        info[t] = {"enablers": sorted(n for n, (en, _) in zip(names, tags) if en), "payoffs": sorted(n for n, (_, pa) in zip(names, tags) if pa),
                   "decks_with": round(float((x > 0).mean()), 3), "typical_x": round(typical, 3), "sd_x": round(sd, 3)}
    T = np.column_stack(T).astype(np.float32)
    l2 = (config.fitted(a.set, "deckmodel") or {}).get("l2", 30.0)
    w = fit_logistic(X[~test], y[~test], l2)
    z = X @ w[1:] + w[0]
    groups = np.array(drafts)
    tr_groups = groups[~test]
    order = np.argsort(tr_groups, kind="stable")
    uniq, starts = np.unique(tr_groups[order], return_index=True)
    rows_by_group = dict(zip(uniq, np.split(order, starts[1:])))
    DATA.update(z_tr=z[~test], z_te=z[test], T=T, y=y, test=test, groups=groups, rows_by_group=rows_by_group)
    base = logloss_z(z[test], y[test])
    keys = list(themes)
    with get_context("fork").Pool(a.jobs) as pool:
        out = pool.map(run, [[i] for i in range(len(keys))])
    print(f"held-out log-loss of the deck model without themes {base:.5f}")
    print(f"{'theme':12s} {'decks':>6s} {'held-out gain x1e-5':>19s} {'beta/SD':>8s} {'SE':>6s} {'z':>5s} {'typical pkg, WR pts':>19s}")
    result = {"method": "offset: deck model fitted once, each theme fitted alone on its linear predictor; SE from 20 draft-cluster bootstraps",
              "heldout_logloss_base": round(base, 6), "l2": l2, "themes": {}}
    for i, t in enumerate(keys):
        gain, b, se = out[i]
        pts = 25.0 * b * info[t]["typical_x"] / info[t]["sd_x"]  # d(win %)/d(logit) = 25 at 50%
        print(f"{t:12s} {100 * info[t]['decks_with']:5.1f}% {1e5 * gain:+19.1f}  {b:+8.3f} {se:6.3f} {(b / se if se else float("nan")):+5.1f} {pts:+19.2f}")
        result["themes"][t] = {**info[t], "heldout_gain": round(gain, 8), "beta_per_sd": round(b, 4), "se": round(se, 4),
                               "typical_points": round(pts, 2), "claim": themes[t].get("claim", ""), "sources": themes[t].get("sources", [])}
    path = config.write_fitted(a.set, "synergy", result, "tools/fit_themes.py: manual.json themes tested as enablers x payoffs terms in the deck model, held-out games")
    print(f"wrote synergy to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
