"""Stage 1: every deck's results against a context-only expectation (no deck content at all).

    uv run python tools/deck_context.py FRA      # ~2 min; writes <data>/tables/FRA/9_context.csv.gz and docs/context-FRA.md

A logistic model of game wins on context only: per format the player's win-rate bucket, games bucket, 17Lands rank,
play/draw, own and opponent's mulligans (every bucket kept: the deck model's context stopped at 0.68 and read the
0.70+ buckets as the lowest one). Fitted on 80% of drafts (draft id hash), checked on the other 20%; the expectation
per deck is the sum over its own games. Per deck: games, wins, expected wins, binomial variance, residual.
`deck` is the row index in table 1.

Also the number the later stages depend on: how much of a deck's result is the deck at all. Each game's probability is
fixed before it is played, so for a deck E[r^2] = V + tau^2 V^2 (r = wins - expected, V = sum of p(1-p)); tau is the SD
of the deck effect (logit units) that context cannot see, by method of moments with a deck bootstrap SE.
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from tools import tables  # noqa: E402
from tools.deck_model import fit_logistic, logloss  # noqa: E402

FORMATS = ["PremierDraft", "TradDraft", "PickTwoDraft"]  # the order tools/deck_table.py reads them: table 1's row order
NAME = "9_context.csv.gz"


def band(sk):
    return "no bucket" if sk is None else ("0.70+" if sk > 0.69 else ("0.60-0.68" if sk >= 0.595 else "below 0.60"))


def read_games(set_code):
    games = []
    for fmt in FORMATS:
        src = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.{fmt}.csv.gz"
        with gzip.open(src, "rt", newline="") as f:
            r = csv.reader(f)
            ix = {k: i for i, k in enumerate(next(r))}
            need = [ix[k] for k in ("draft_id", "build_index", "user_game_win_rate_bucket", "user_n_games_bucket", "rank", "on_play",
                                    "num_mulligans", "opp_num_mulligans", "won")]
            for row in r:
                d, b, sk, gm, rk, op, mu, om, won = (row[i] for i in need)
                games.append((fmt, d, b, sk or "?", gm or "?", rk or "?", op == "True", min(2, int(float(mu or 0))), min(2, int(float(om or 0))),
                              won == "True", int(hashlib.md5(d.encode()).hexdigest()[:2], 16) < 52))
        print(f"  read {fmt}: {len(games)} games so far", flush=True)
    return games


def design(games):
    cols, rows, ci = {}, [], []
    for r, g in enumerate(games):
        fmt = g[0]
        for f in (("fmt", fmt), ("skill", fmt, g[3]), ("games", fmt, g[4]), ("rank", fmt, g[5]), ("play", g[6]), ("mull", g[7]), ("omull", g[8]),
                  ("mullplay", g[7], g[6])):
            rows.append(r)
            ci.append(cols.setdefault(f, len(cols)))
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, ci)), shape=(len(games), len(cols))), cols


def tau2(rs, vs, boot=200):
    rs, vs = np.asarray(rs), np.asarray(vs)
    est = lambda r, v: (r ** 2 - v).sum() / (v ** 2).sum()
    rng = np.random.default_rng(1)
    n = len(rs)
    bs = [est(rs[s], vs[s]) for s in (rng.integers(0, n, n) for _ in range(boot))]
    return float(est(rs, vs)), float(np.std(bs))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--l2", type=float, default=1.0)
    a = p.parse_args(argv)
    print("reading games ...", flush=True)
    games = read_games(a.set)
    y = np.array([g[9] for g in games], np.float32)
    test = np.array([g[10] for g in games])
    X, levels = design(games)
    tr, te = ~test, test
    print(f"{len(y)} games, {X.shape[1]} context columns, {int(te.sum())} held out", flush=True)
    fc = [levels[("fmt", f)] for f in FORMATS]
    w_f = fit_logistic(X[:, fc][tr], y[tr], a.l2)
    w = fit_logistic(X[tr], y[tr], a.l2)
    ll_f, ll_c = logloss(w_f, X[:, fc][te], y[te]), logloss(w, X[te], y[te])
    print(f"held-out log-loss: format only {ll_f:.5f}, context {ll_c:.5f}", flush=True)
    w = fit_logistic(X, y, a.l2)
    p_g = 1 / (1 + np.exp(-(X @ w[1:] + w[0])))

    acc, keys = {}, []
    for g, pg, won in zip(games, p_g, y):
        k = (g[0], g[1], g[2])
        if k not in acc:
            acc[k] = [0, 0, 0.0, 0.0]
            keys.append(k)
        s = acc[k]
        s[0] += 1; s[1] += int(won); s[2] += float(pg); s[3] += float(pg * (1 - pg))
    with gzip.open(tables.path(a.set, tables.DECKS), "rt") as f:
        decks = json.load(f)
    if len(keys) != len(decks):
        sys.exit(f"{len(keys)} decks in games, {len(decks)} in table 1: rebuild table 1 first")
    bad = sum(1 for k, d in zip(keys, decks) if (k[0], k[1]) != (d["format"], d["draft"]) or acc[k][0] != d["games"] or acc[k][1] != d["wins"])
    if bad:
        sys.exit(f"{bad} decks do not line up with table 1 (format, draft, games, wins)")
    out = tables.path(a.set, NAME)
    with gzip.open(out, "wt", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["deck", "format", "skill", "games", "wins", "expected", "variance", "resid", "pred_table1"])
        for i, (k, d) in enumerate(zip(keys, decks)):
            n, wins, e, v = acc[k]
            wr.writerow([i, d["format"], "" if d["skill"] is None else d["skill"], n, wins, round(e, 4), round(v, 4), round(wins - e, 4), d["pred"]])
    print(f"wrote {out}", flush=True)
    report(a.set, keys, decks, acc, ll_f, ll_c)


def report(set_code, keys, decks, acc, ll_f, ll_c):
    groups, by = defaultdict(list), defaultdict(lambda: [0, 0, 0.0, 0.0])
    for k, d in zip(keys, decks):
        n, w, e, v = acc[k]
        b = band(d["skill"])
        for g in (("all", "all"), (d["format"], "all"), ("all", b), (d["format"], b)):
            groups[g].append((n, w, e, v))
        x = by[(d["format"], b)]
        x[0] += n; x[1] += w; x[2] += d["pred"]; x[3] += e
    L = [f"# Stage 1: results against a context-only expectation ({set_code})", "",
         f"Table 9 (`tools/deck_context.py`). Held-out log-loss: format only {ll_f:.5f}, context {ll_c:.5f}.", "",
         "## Deck effect the context cannot see", "",
         "tau = SD of the deck effect in logit units (method of moments, deck-bootstrap SE); WR points = tau x 25.",
         "It includes player skill within a bucket, so it is an upper bound on what deck content can explain.", "",
         "| format | skill band | decks | games | wins - expected | tau^2 | SE | tau | WR points |", "|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(groups, key=lambda g: (g[0] != "all", g[0], g[1] != "all", g[1])):
        rows = groups[g]
        if len(rows) < 300:
            continue
        r = [x[1] - x[2] for x in rows]
        t2, se = tau2(r, [x[3] for x in rows])
        t = float(np.sqrt(max(t2, 0)))
        L.append(f"| {g[0]} | {g[1]} | {len(rows)} | {sum(x[0] for x in rows)} | {sum(r):+.0f} | {t2:.4f} | {se:.4f} | {t:.3f} | {25 * t:.1f} |")
    L += ["", "## Actual win rate against table 1's prediction and the context-only expectation", "",
          "| format | skill band | games | actual WR | table 1 pred | context-only |", "|---|---|---|---|---|---|"]
    for (fmt, b), x in sorted(by.items()):
        L.append(f"| {fmt} | {b} | {x[0]} | {100 * x[1] / x[0]:.1f} | {100 * x[2] / x[0]:.1f} | {100 * x[3] / x[0]:.1f} |")
    path = ROOT / "docs" / f"context-{set_code}.md"
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
