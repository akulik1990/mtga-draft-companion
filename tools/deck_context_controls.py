"""Stage 1 again, with controls that do not contain the deck's own games.

    uv run python tools/deck_context_controls.py FRA [--slice 100]

17Lands' skill bucket (user_game_win_rate_bucket) is the player's win rate in the set, this deck's games included. Variants:
  A  leaky      format, skill bucket, games bucket, rank per game, play/draw, mulligans   (what tools/deck_context.py did)
  B  main       format, games bucket, rank fixed per deck (rank of the deck's first game), play/draw, mulligans; no skill bucket
  C  100+ skill the same with the skill bucket, players with 100+ games only (one deck barely moves their rate)
  D  100+ none  C without the skill bucket: C against D is what the skill bucket adds when the leak is small
Per variant: held-out log-loss and tau^2 (method of moments on per-deck residuals, deck-bootstrap SE) on held-out decks
(draft id hash, as everywhere) and on all decks. tau^2 > 0 is needed for a deck effect to exist; B and D include player
skill that the controls cannot see, so they are upper bounds on what deck content can explain.
--slice N: the N drafts with the smallest hash, fitted and judged within themselves (a smoke test of the code, not a result).
Game rows are cached in <data>/Temp/context_games_SET.pkl.
"""

import argparse
import csv
import gzip
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from tools.deck_context import FORMATS, tau2  # noqa: E402
from tools.deck_model import fit_logistic, logloss  # noqa: E402

FIELDS = ("draft_id", "build_index", "game_time", "user_game_win_rate_bucket", "user_n_games_bucket", "rank", "on_play",
          "num_mulligans", "opp_num_mulligans", "won")
BIG = {"100", "500"}


def read_games(set_code):
    cache = DATA_DIR / "Temp" / f"context_games_{set_code}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    games = []
    for fmt in FORMATS:
        with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.{fmt}.csv.gz", "rt", newline="") as f:
            r = csv.reader(f)
            ix = {k: i for i, k in enumerate(next(r))}
            need = [ix[k] for k in FIELDS]
            for row in r:
                d, b, t, sk, gm, rk, op, mu, om, won = (row[i] for i in need)
                games.append((fmt, d, b, t, sk or "?", gm or "?", rk or "?", op == "True", min(2, int(float(mu or 0))),
                              min(2, int(float(om or 0))), won == "True"))
        print(f"  read {fmt}: {len(games)} games so far", flush=True)
    cache.write_bytes(pickle.dumps(games))
    return games


def keys(g, feats, first_rank):
    fmt, d, b, t, sk, gm, rk, op, mu, om, won = g
    out = [("fmt", fmt), ("play", op), ("mull", mu), ("omull", om), ("mullplay", mu, op)]
    if "skill" in feats:
        out.append(("skill", fmt, sk))
    if "games" in feats:
        out.append(("games", fmt, gm))
    if "rank_game" in feats:
        out.append(("rank", fmt, rk))
    if "rank_first" in feats:
        out.append(("rank", fmt, first_rank[(fmt, d, b)]))
    return out


def design(games, feats, first_rank):
    cols, rows, ci = {}, [], []
    for r, g in enumerate(games):
        for k in keys(g, feats, first_rank):
            rows.append(r)
            ci.append(cols.setdefault(k, len(cols)))
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, ci)), shape=(len(games), len(cols))), cols


def run_variant(label, games, first_rank, held, l2, feats):
    X, _ = design(games, feats, first_rank)
    y = np.array([g[10] for g in games], np.float32)
    tr, te = ~held, held
    w = fit_logistic(X[tr], y[tr], l2)
    ll = logloss(w, X[te], y[te])
    fmt0 = np.array([g[0] == FORMATS[0] for g in games], np.float32)
    p = 1 / (1 + np.exp(-(X @ w[1:] + w[0])))
    acc, order = {}, []
    for g, pg, won, h in zip(games, p, y, held):
        k = (g[0], g[1], g[2])
        if k not in acc:
            acc[k] = [0, 0, 0.0, 0.0, bool(h)]
            order.append(k)
        s = acc[k]
        s[0] += 1; s[1] += int(won); s[2] += float(pg); s[3] += float(pg * (1 - pg))
    res = {}
    for scope, sel in (("held-out decks", lambda s: s[4]), ("all decks", lambda s: True)):
        rows = [acc[k] for k in order if sel(acc[k])]
        r = [s[1] - s[2] for s in rows]
        t2, se = tau2(r, [s[3] for s in rows])
        res[scope] = (len(rows), sum(s[0] for s in rows), t2, se)
    return label, ll, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args(argv)
    t0 = time.time()
    games = read_games(a.set)
    print(f"{len(games)} games read in {time.time() - t0:.0f}s", flush=True)
    h = lambda d: int(hashlib.md5(d.encode()).hexdigest(), 16)
    if a.slice:
        drafts = sorted({(g[0], g[1]) for g in games}, key=lambda k: h(k[1]))[:a.slice]
        keep = set(drafts)
        games = [g for g in games if (g[0], g[1]) in keep]
        held = np.array([h(g[1]) % 4 == 0 for g in games])
        print(f"slice: {len(drafts)} drafts, {len(games)} games, {int(held.sum())} held-out", flush=True)
    else:
        held = np.array([int(hashlib.md5(g[1].encode()).hexdigest()[:2], 16) < 52 for g in games])
    first = {}
    for g in games:
        k = (g[0], g[1], g[2])
        if k not in first or g[3] < first[k][0]:
            first[k] = (g[3], g[6])
    first_rank = {k: v[1] for k, v in first.items()}
    big = [i for i, g in enumerate(games) if g[5] in BIG]
    sub = [games[i] for i in big]
    sub_held = held[big]
    out = []
    for label, gs, hd, feats in (
            ("A leaky: skill + games + rank per game", games, held, {"skill", "games", "rank_game"}),
            ("B games + rank of first game", games, held, {"games", "rank_first"}),
            ("E rank of first game only", games, held, {"rank_first"}),
            ("F format, play, mulligans only", games, held, set()),
            ("C 100+ games: skill + rank of first game", sub, sub_held, {"skill", "games", "rank_first"}),
            ("D 100+ games: rank of first game", sub, sub_held, {"games", "rank_first"}),
            ("H 100+ games: format, play, mulligans only", sub, sub_held, set())):
        t1 = time.time()
        out.append(run_variant(label, gs, first_rank, hd, a.l2, feats))
        print(f"  {label[0]} done in {time.time() - t1:.0f}s", flush=True)
    L = [f"| variant | held-out log-loss | decks (held-out) | tau^2 held-out | SE | tau^2 all decks | SE |", "|---|---|---|---|---|---|---|"]
    for label, ll, res in out:
        ho, al = res["held-out decks"], res["all decks"]
        L.append(f"| {label} | {ll:.5f} | {ho[0]} | {ho[2]:+.4f} | {ho[3]:.4f} | {al[2]:+.4f} | {al[3]:.4f} |")
    print("\n".join(L))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
