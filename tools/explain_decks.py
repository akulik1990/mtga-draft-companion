"""How much of deck win rate each kind of information explains, all categories at once (not one at a time).

    uv run python tools/explain_decks.py FRA      # ~12 min; results recorded in data/tables/FRA/9_explain.csv and docs/deck-findings-FRA.md

Deck-level binomial logistic (wins of games), ridge, controls = player skill bucket + format; fit on draft-hash half A,
judged on half B and the reverse. Feature sets: card quality (rank buckets, rarity), structure (every other table-3
field, duplicates merged), both. Spread = SD across decks of the predicted win rate from the deck part alone."""
import hashlib, sys
import numpy as np
from scipy.optimize import minimize
import companion
from tools import deck_counts
from tools.analyse import TEXT, META, merge_fields

rows = [r for r in deck_counts.read("FRA") if r["games"] > 0 and 2 <= r["n_base"] <= 3]
fields, _ = merge_fields(rows, [k for k in rows[0] if k not in TEXT and k not in META])
quality = [f for f in fields if f.startswith("rank_") or f in ("common", "uncommon", "rare", "mythic")]
structure = [f for f in fields if f not in quality]
g = np.array([r["games"] for r in rows]); w = np.array([r["wins"] for r in rows])
skills = sorted({r["skill"] for r in rows}); fmts = sorted({r["format"] for r in rows})
ctx = np.array([[r["skill"] == s for s in skills[1:]] + [r["format"] == f for f in fmts[1:]] for r in rows], float)
half = np.array([int(hashlib.md5(r["draft"].encode()).hexdigest(), 16) % 2 for r in rows])
model_p = np.clip(np.array([r["pred"] for r in rows]) / g, 0.01, 0.99)

def design(cols):
    X = np.array([[r[f] for f in cols] for r in rows], float) if cols else np.zeros((len(rows), 0))
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.column_stack([np.ones(len(rows)), ctx, (X - mu) / sd])

def fit(X, m, l2=30.0):
    def f(b):
        z = X[m] @ b; p = 1 / (1 + np.exp(-z))
        ll = (w[m] * np.log(p + 1e-12) + (g[m] - w[m]) * np.log(1 - p + 1e-12)).sum()
        grad = X[m].T @ (w[m] - g[m] * p)
        pen = l2 * (b[1 + ctx.shape[1]:] ** 2).sum()
        gp = np.zeros_like(b); gp[1 + ctx.shape[1]:] = 2 * l2 * b[1 + ctx.shape[1]:]
        return -ll + pen, -grad + gp
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x

def loss(p, m):
    return -(w[m] * np.log(p[m]) + (g[m] - w[m]) * np.log(1 - p[m])).sum() / g[m].sum()

def spread(X, b, m):
    # SD across decks of the predicted win rate from the deck part only (context held at its mean)
    k = 1 + ctx.shape[1]
    z = X[m, k:] @ b[k:]
    base = (X[m, :k] @ b[:k]).mean()
    return 100 * (1 / (1 + np.exp(-(base + z)))).std()

res = {}
for name, cols in (("context only", []), ("card quality", quality), ("structure (all categories)", structure), ("quality + structure", quality + structure)):
    X = design(cols); L = []; sp = []
    for h in (0, 1):
        b = fit(X, half == h)
        p = np.clip(1 / (1 + np.exp(-(X @ b))), 0.01, 0.99)
        L.append(loss(p, half != h)); sp.append(spread(X, b, half != h))
    res[name] = (np.mean(L), np.mean(sp))
base = res["context only"][0]
print(f"{len(rows)} decks, {g.sum()} games; {len(quality)} quality fields, {len(structure)} structure fields")
print(f"{'':30} {'held-out loss':>14} {'gain vs context':>16} {'spread of deck WR (SD)':>24}")
for k, (L, sp) in res.items():
    print(f"{k:30} {L:14.5f} {base - L:16.5f} {sp:22.2f} pts")
print(f"{'current deck model':30} {loss(model_p, np.ones(len(rows), bool)):14.5f} {base - loss(model_p, np.ones(len(rows), bool)):16.5f}   (includes player context; fitted on Premier games incl. these)")
