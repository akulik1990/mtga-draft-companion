"""Does deck content explain results beyond card quality, on a design without the two known biases?

    uv run python tools/explain_clean.py FRA      # about 1-3 min; needs the cache of tools/deck_context_controls.py

Design (docs/context-controls-FRA.md): one row per deck, the outcome is the deck's FIRST game only (record 0-0: no
record-based pairing yet), players with 100+ games only (their skill bucket barely contains this deck's results), decks
with 2-3 base colours. Controls: format, games bucket, rank of that game, play/draw, mulligans, and (variant "skill") the
player's skill bucket. Feature sets as tools/explain_decks.py: card quality (rank buckets, rarity), structure (every other
table-3 field, duplicates merged), both. Ridge logistic, 2-fold cross-fit by draft hash; the penalty is chosen on a
validation fifth of each training half (grid below), never on the half it is judged on.
Gain = out-of-fold log-loss of the context model minus the model's, per game, SE clustered by draft.
Quintiles: decks ordered by the deck part's score alone; actual win rate minus the context model's expectation.
Writes <data>/tables/SET/9_explain_clean.csv.
"""

import csv
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from tools import deck_counts, tables  # noqa: E402
from tools.analyse import META, TEXT, merge_fields  # noqa: E402

BIG = {"100", "500"}
GRID = (10, 30, 100, 300, 1000, 3000)
CTX_L2 = 1.0


def load(set_code):
    games = pickle.loads((DATA_DIR / "Temp" / f"context_games_{set_code}.pkl").read_bytes())
    first, order = {}, []
    for x in games:  # (fmt, draft, build, time, skill, games, rank, on_play, mull, omull, won)
        k = (x[0], x[1], x[2])
        if k not in first:
            order.append(k)
            first[k] = x
        elif x[3] < first[k][3]:
            first[k] = x
    rows = deck_counts.read(set_code)
    if len(rows) != len(order) or any((k[0], k[1]) != (r["format"], r["draft"]) for k, r in zip(order, rows)):
        sys.exit("table 3 does not line up with the game data: rebuild tables 1 and 3")
    keep = [i for i, (k, r) in enumerate(zip(order, rows)) if 2 <= r["n_base"] <= 3 and first[k][5] in BIG]
    return [rows[i] for i in keep], [first[order[i]] for i in keep]


def context(fg, skill):
    cols, data = {}, []
    for x in fg:
        fmt, _, _, _, sk, gm, rk, op, mu, om, _ = x
        ks = [("fmt", fmt), ("games", fmt, gm), ("rank", fmt, rk), ("play", op), ("mull", mu), ("omull", om), ("mp", mu, op)]
        if skill:
            ks.append(("skill", fmt, sk))
        data.append([cols.setdefault(k, len(cols)) for k in ks])
    X = np.zeros((len(fg), len(cols) + 1))
    X[:, 0] = 1
    for r, idx in enumerate(data):
        X[r, [i + 1 for i in idx]] = 1
    return X


def fit(Xc, Xd, y, m, l2):
    X = np.hstack([Xc, Xd])
    k = Xc.shape[1]
    pen = np.r_[np.full(k, CTX_L2), np.full(Xd.shape[1], l2)]
    pen[0] = 0

    def f(b):
        z = X[m] @ b
        ll = (y[m] * z - np.logaddexp(0, z)).sum()
        p = 1 / (1 + np.exp(-z))
        return -ll + (pen * b * b).sum(), -(X[m].T @ (y[m] - p)) + 2 * pen * b

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


def loss(z, y):
    return np.logaddexp(0, z) - y * z


def crossfit(Xc, Xd, y, half, val):
    """Out-of-fold (full logit, deck-part logit, chosen l2 per fold)."""
    z, zd, chosen = np.zeros(len(y)), np.zeros(len(y)), []
    k = Xc.shape[1]
    for h in (0, 1):
        tr, te = half != h, half == h
        l2 = GRID[0]
        if Xd.shape[1]:
            best = None
            for g in GRID:
                b = fit(Xc, Xd, y, tr & ~val, g)
                v = loss(np.hstack([Xc, Xd])[tr & val] @ b, y[tr & val]).mean()
                if best is None or v < best[0]:
                    best, l2 = (v, g), g
        chosen.append(l2)
        b = fit(Xc, Xd, y, tr, l2)
        X = np.hstack([Xc, Xd])
        z[te] = X[te] @ b
        zd[te] = Xd[te] @ b[k:] if Xd.shape[1] else 0
    return z, zd, chosen


def cluster_gain(d, groups):
    """mean(d) and its SE with drafts as the clusters."""
    n = len(d)
    mean = d.mean()
    s = {}
    for g, x in zip(groups, d - mean):
        s[g] = s.get(g, 0.0) + x
    return mean, float(np.sqrt(sum(v * v for v in s.values())) / n)


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    rows, fg = load(set_code)
    y = np.array([x[10] for x in fg], float)
    drafts = [x[1] for x in fg]
    h = lambda d, m: int(hashlib.md5(d.encode()).hexdigest(), 16) % m
    half = np.array([h(d, 2) for d in drafts])
    val = np.array([h(d, 5) == 0 for d in drafts])
    fields, _ = merge_fields(rows, [k for k in rows[0] if k not in TEXT and k not in META])
    quality = [f for f in fields if f.startswith("rank_") or f in ("common", "uncommon", "rare", "mythic")]
    structure = [f for f in fields if f not in quality]

    def deck(cols):
        if not cols:
            return np.zeros((len(rows), 0))
        X = np.array([[r[f] for f in cols] for r in rows], float)
        sd = X.std(0)
        sd[sd == 0] = 1
        return (X - X.mean(0)) / sd

    print(f"{len(rows)} decks (first game only, 100+ games, 2-3 base colours), win rate {100 * y.mean():.1f}%, "
          f"{len(set(drafts))} drafts; {len(quality)} quality, {len(structure)} structure fields", flush=True)
    out, lines = [], []
    for skill in (True, False):
        Xc = context(fg, skill)
        res = {}
        for name, cols in (("context only", []), ("card quality", quality), ("structure", structure), ("quality + structure", quality + structure)):
            z, zd, chosen = crossfit(Xc, deck(cols), y, half, val)
            res[name] = (z, zd, chosen)
            print(f"  skill={skill} {name}: l2 {chosen}", flush=True)
        z0 = res["context only"][0]
        base_mu = z0.mean()
        for name, (z, zd, chosen) in res.items():
            gain, se = cluster_gain(loss(z0, y) - loss(z, y), drafts)
            spread = 100 * (1 / (1 + np.exp(-(base_mu + zd)))).std() if name != "context only" else 0.0
            row = dict(controls="with skill bucket" if skill else "without skill bucket", model=name, decks=len(y), loss=round(float(loss(z, y).mean()), 5),
                       gain=round(float(gain), 5), gain_se=round(se, 5), spread_sd_points=round(float(spread), 2), l2=" / ".join(map(str, chosen)))
            if name != "context only":
                order = np.argsort(zd)
                q = np.array_split(order, 5)
                diff = [100 * (y[i] - 1 / (1 + np.exp(-z0[i]))).mean() for i in q]
                ses = [100 * (y[i] - 1 / (1 + np.exp(-z0[i]))).std() / np.sqrt(len(i)) for i in q]
                row.update(q1_minus_expected=round(diff[0], 2), q5_minus_expected=round(diff[4], 2), q5_minus_q1=round(diff[4] - diff[0], 2),
                           q5_minus_q1_se=round(float(np.hypot(ses[0], ses[4])), 2))
            out.append(row)
    keys = ["controls", "model", "decks", "loss", "gain", "gain_se", "spread_sd_points", "q1_minus_expected", "q5_minus_expected",
            "q5_minus_q1", "q5_minus_q1_se", "l2"]
    path = tables.path(set_code, "9_explain_clean.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, restval="")
        w.writeheader()
        w.writerows(out)
    print("\n" + " | ".join(keys))
    for r in out:
        print(" | ".join(str(r.get(k, "")) for k in keys))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
