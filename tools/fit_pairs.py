"""Discover card pairs that win together beyond their own ratings: a low-rank interaction term on top of the deck model.

    uv run python tools/fit_pairs.py FRA [--k 4] [--l2 30 100 300]

The deck model (tools/deck_model.py: context, shape, card counts) is fitted once on its training split; on top of its
linear predictor z each card gets a k-dimensional vector v, and a deck's logit becomes
z + sum over card pairs i<j of n_i n_j <v_i, v_j> (a factorisation machine: pairs share strength through the vectors,
so rare pairs are not fitted from their own few games). Kept only if held-out log-loss improves. Reports the strongest
positive and negative pairs among pairs registered together in at least --min-decks decks, in win-rate points at 50%.
Writes the "pairs" section of the set's fitted.json; nothing in the engine reads it yet.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion.advisor_v2 import config  # noqa: E402
from tools.deck_model import card_facts, fit_logistic, read  # noqa: E402
from tools.fit_themes import design, logloss_z  # noqa: E402


def fit_fm(C, C2, z, y, k, l2, seed=0):
    """Vectors V (cards x k) maximising the penalised likelihood of y given offset z."""
    n_cards = C.shape[1]
    rng = np.random.default_rng(seed)
    v0 = rng.normal(0, 0.01, n_cards * k)

    def f(flat):
        V = flat.reshape(n_cards, k)
        S = C @ V
        pair = 0.5 * ((S * S).sum(1) - C2 @ (V * V).sum(1))
        q = 1 / (1 + np.exp(-(z + pair)))
        loss = -(y * np.log(q + 1e-12) + (1 - y) * np.log(1 - q + 1e-12)).sum() + l2 * (flat @ flat)
        r = q - y
        g = C.T @ (r[:, None] * S) - (C2.T @ r)[:, None] * V + 2 * l2 * V
        return loss, g.ravel()

    res = minimize(f, v0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return res.x.reshape(n_cards, k)


def pair_term(C, C2, V):
    S = C @ V
    return 0.5 * ((S * S).sum(1) - C2 @ (V * V).sum(1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--l2", type=float, nargs="+", default=[1000.0, 3000.0, 10000.0])
    p.add_argument("--min-decks", type=int, default=300)
    a = p.parse_args(argv)

    names, C, ctx, shp, y, test = read(a.set, card_facts(a.set))
    X = design(ctx, shp, C, test)
    l2_base = (config.fitted(a.set, "deckmodel") or {}).get("l2", 30.0)
    w = fit_logistic(X[~test], y[~test], l2_base)
    z = X @ w[1:] + w[0]
    facts = card_facts(a.set)
    # Lands out of the pair term: card x basic land pairs only say which colours a card plays best in, which the
    # per-pair win rates already carry. Card-to-card pairs are the question.
    keep = np.array([0.0 if (facts.get(n) or {}).get("land") else 1.0 for n in names])
    C = (C.astype(np.float64) @ sparse.diags(keep)).tocsr()
    C2 = C.multiply(C).tocsr()
    tr, te = ~test, test
    base = logloss_z(z[te], y[te])
    print(f"{len(y)} games; held-out log-loss of the deck model {base:.5f}")
    best = None
    for l2 in a.l2:
        V = fit_fm(C[tr], C2[tr], z[tr], y[tr], a.k, l2)
        ll = logloss_z(z[te] + pair_term(C[te], C2[te], V), y[te])
        print(f"  k={a.k} l2={l2:g}: held-out {ll:.5f} (gain {1e5 * (base - ll):+.1f} x1e-5)", flush=True)
        if best is None or ll < best[0]:
            best = (ll, l2, V)
    ll, l2, V = best
    gain = base - ll
    # Pairs registered together often enough to read: co-occurrence over decks (rows) in the training split.
    B = (C[tr] > 0).astype(np.float32)
    co = (B.T @ B).tocoo()
    rows = []
    for i, j, n in zip(co.row, co.col, co.data):
        if i < j and n >= a.min_decks and keep[i] and keep[j]:
            rows.append((names[i], names[j], int(n), float(V[i] @ V[j]) * 25.0))
    rows.sort(key=lambda r: -r[3])
    print(f"best l2 {l2:g}: held-out gain {1e5 * gain:+.1f} x1e-5; {len(rows)} pairs with >= {a.min_decks} decks")
    print("strongest together (win-rate points per copy pair, at 50%):")
    for r in rows[:20]:
        print(f"  {r[3]:+.2f}  {r[0]} + {r[1]}  ({r[2]} decks)")
    print("weakest together:")
    for r in rows[-10:]:
        print(f"  {r[3]:+.2f}  {r[0]} + {r[1]}  ({r[2]} decks)")
    config.write_fitted(a.set, "pairs", {"k": a.k, "l2": l2, "heldout_gain": round(gain, 8), "min_decks": a.min_decks,
                                          "top": [list(r) for r in rows[:50]], "bottom": [list(r) for r in rows[-20:]]},
                        "tools/fit_pairs.py: low-rank card-pair interactions on top of the deck model, 17Lands game data")


if __name__ == "__main__":
    main()
