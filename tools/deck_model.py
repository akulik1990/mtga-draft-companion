"""A deck's chance to win a game, learnt from 17Lands' public game data: the deck-quality measure.

    uv run python tools/deck_model.py fit FRA      # writes the deckmodel section of companion/advisor_v2/sets/FRA/fitted.json

Each game row in game_data_public.<SET>.PremierDraft.csv.gz has the deck played (card counts), the
result, and context: the player's win-rate and games buckets, play/draw, both sides' mulligans.
A logistic regression on card counts plus deck shape (creatures, curve 1..6+, removal, lands,
colours, splash) with the context as controls separates "good deck" from "good player" and
"lucky draw". Scoring a deck holds the context at typical values (average player, 50% on the play,
no mulligans), so the score is the deck's win rate for an average player. Fitted on 80% of drafts
(by draft id hash), checked on the other 20% against a context-only model.
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import composition, config  # noqa: E402

SHAPE = ["creatures", "cmc1", "cmc2", "cmc3", "cmc4", "cmc5", "cmc6p", "removal", "lands", "nonbasic_lands", "colours", "splash"]
SKILLS = [0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]


def card_facts(set_code):
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    facts = {}
    for c in data.values():
        types, tags = c.get("types") or [], config.tags(c, set_code)
        f = {"creature": composition.is_creature(c), "land": "Land" in types, "basic": "Basic" in types,
             "cmc": int(c.get("cmc") or 0), "removal": "removal" in tags}
        facts.setdefault(c["name"], f)
        facts.setdefault(c["name"].split(" // ")[0], f)
    return facts


def shape(counts, facts, main_colours="", splash=""):
    """Deck-shape features from {card name: count}."""
    s = dict.fromkeys(SHAPE, 0.0)
    for name, n in counts.items():
        f = facts.get(name)
        if not f:
            continue
        if f["land"]:
            s["lands"] += n
            s["nonbasic_lands"] += 0 if f["basic"] else n
            continue
        s["creatures"] += n * f["creature"]
        s["removal"] += n * f["removal"]
        s[f"cmc{min(6, max(1, f['cmc']))}" if f["cmc"] < 6 else "cmc6p"] += n
    s["colours"] = len(main_colours)
    s["splash"] = 1.0 if splash else 0.0
    return s


def context_row(skill, games, on_play, mull, opp_mull):
    """Controls: skill bucket dummies, log games, on the play, mulligans."""
    out = [float(abs((skill or 0) - b) < 0.02) for b in SKILLS[1:]]
    out += [float(np.log1p(games or 0)), float(on_play), float(mull or 0), float(opp_mull or 0)]
    return out


TYPICAL = {"skill": 0.56, "games": 100, "on_play": 0.5, "mull": 0, "opp_mull": 0}


def read(set_code, facts, ids=False):
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        names = [n for _, n in deck_cols]
        rows, cols, vals, X_ctx, X_shape, y, test, drafts = [], [], [], [], [], [], [], []
        r = 0
        for row in reader:
            counts = {}
            for j, (i, name) in enumerate(deck_cols):
                v = row[i]
                if v and v != "0":
                    n = int(float(v))
                    counts[name] = n
                    rows.append(r); cols.append(j); vals.append(n)
            sk = row[col["user_game_win_rate_bucket"]]
            gm = row[col["user_n_games_bucket"]]
            X_ctx.append(context_row(float(sk) if sk not in ("", "None") else None, float(gm) if gm not in ("", "None") else None,
                                     row[col["on_play"]] == "True", float(row[col["num_mulligans"]] or 0), float(row[col["opp_num_mulligans"]] or 0)))
            s = shape(counts, facts, row[col["main_colors"]], row[col["splash_colors"]])
            X_shape.append([s[k] for k in SHAPE])
            y.append(row[col["won"]] == "True")
            test.append(int(hashlib.md5(row[col["draft_id"]].encode()).hexdigest()[:2], 16) < 52)  # ~20%
            drafts.append(row[col["draft_id"]])
            r += 1
    C = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(r, len(names)))
    out = (names, C, np.array(X_ctx, np.float32), np.array(X_shape, np.float32), np.array(y, np.float32), np.array(test))
    return out + (drafts,) if ids else out


def fit_logistic(X, y, l2, sw=None):
    """Ridge logistic regression (intercept unpenalised) on a sparse or dense design; sw: per-row weights."""
    n, k = X.shape
    sw = np.ones(n) if sw is None else sw

    def f(w):
        z = X @ w[1:] + w[0]
        p = 1 / (1 + np.exp(-z))
        eps = 1e-9
        loss = -(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() + l2 * (w[1:] @ w[1:])
        g = np.empty_like(w)
        r = sw * (p - y)
        g[0] = r.sum()
        g[1:] = X.T @ r + 2 * l2 * w[1:]
        return loss, g

    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return res.x


def logloss(w, X, y):
    p = 1 / (1 + np.exp(-(X @ w[1:] + w[0])))
    return float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())


def standardise(M):
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def cmd_fit(a):
    facts = card_facts(a.set)
    names, C, ctx, shp, y, test = read(a.set, facts)
    print(f"{len(y)} games, {C.shape[1]} card columns, {int(test.sum())} held out")
    mu_s, sd_s = standardise(shp[~test])
    S = (shp - mu_s) / sd_s
    S2 = np.column_stack([S, S[:, SHAPE.index("creatures")] ** 2, S[:, SHAPE.index("lands")] ** 2, S[:, SHAPE.index("cmc2")] ** 2])
    full = sparse.hstack([sparse.csr_matrix(ctx), sparse.csr_matrix(S2), C]).tocsr()
    base = sparse.csr_matrix(ctx)
    tr, te = ~test, test
    # Games by 60%+ players count top_weight times (their card choices are the ones the builder copies).
    top = ctx[:, [i for i, b in enumerate(SKILLS[1:]) if b >= 0.60 - 1e-9]].sum(axis=1) > 0
    sw = np.where(top, a.top_weight, 1.0)
    w0 = fit_logistic(base[tr], y[tr], 1.0)
    print(f"held-out log-loss, context only: {logloss(w0, base[te], y[te]):.5f}")
    best = None
    for l2 in a.l2:
        w = fit_logistic(full[tr], y[tr], l2, sw[tr])
        ll = logloss(w, full[te], y[te])
        print(f"held-out log-loss, deck + context, l2={l2}: {ll:.5f}")
        if best is None or ll < best[0]:
            best = (ll, l2, w)
    ll, l2, _ = best
    w = fit_logistic(full, y, l2, sw)  # refit on everything with the chosen penalty
    k_ctx, k_shape = ctx.shape[1], S2.shape[1]
    model = {"set": a.set, "l2": l2, "heldout_logloss": ll, "context_only_logloss": logloss(w0, base[te], y[te]),
             "intercept": float(w[0]), "context": [float(x) for x in w[1:1 + k_ctx]],
             "shape_names": SHAPE + ["creatures^2", "lands^2", "cmc2^2"], "shape": [float(x) for x in w[1 + k_ctx:1 + k_ctx + k_shape]],
             "shape_mu": [float(x) for x in mu_s], "shape_sd": [float(x) for x in sd_s],
             "cards": {n: float(x) for n, x in zip(names, w[1 + k_ctx + k_shape:])}, "typical": TYPICAL}
    model["top_weight"] = a.top_weight
    out = config.write_fitted(a.set, a.section, model, "tools/deck_model.py fit: regularised logistic of game wins on 17Lands public game data"
                              + (f", 60%+ players' games weighted x{a.top_weight}" if a.top_weight != 1.0 else ""))
    top = sorted(model["cards"].items(), key=lambda kv: -kv[1])[:6]
    print("strongest per copy:", ", ".join(f"{n} {v:+.3f}" for n, v in top))
    print("shape:", ", ".join(f"{n} {v:+.3f}" for n, v in zip(model["shape_names"], model["shape"])))
    print(f"wrote {out}")


class DeckModel:
    """Score a deck ({card name: count}, basics included) as a win rate for an average player."""

    def __init__(self, set_code, section="deckmodel"):
        self.m = config.fitted(set_code, section) or (config.fitted(set_code, "deckmodel") if section != "deckmodel" else None)
        if not self.m:
            raise FileNotFoundError(f"no deckmodel section in {config.set_file(set_code, 'fitted')}")
        self.facts = card_facts(set_code)
        t = self.m["typical"]
        self.ctx = float(np.dot(self.m["context"], context_row(t["skill"], t["games"], t["on_play"], t["mull"], t["opp_mull"])))

    def win_rate(self, counts, main_colours="", splash=""):
        s = shape(counts, self.facts, main_colours, splash)
        z = (np.array([s[k] for k in SHAPE]) - self.m["shape_mu"]) / self.m["shape_sd"]
        z2 = np.concatenate([z, [z[SHAPE.index("creatures")] ** 2, z[SHAPE.index("lands")] ** 2, z[SHAPE.index("cmc2")] ** 2]])
        cards = sum(self.m["cards"].get(n, self.m["cards"].get(n.split(" // ")[0], 0.0)) * c for n, c in counts.items())
        logit = self.m["intercept"] + self.ctx + float(np.dot(self.m["shape"], z2)) + cards
        return 100.0 / (1.0 + np.exp(-logit))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("set"); f.add_argument("--l2", type=float, nargs="+", default=[10.0, 30.0, 100.0, 300.0])
    f.add_argument("--top-weight", type=float, default=1.0, help="weight of 60%%+ players' games")
    f.add_argument("--section", default="deckmodel", help="fitted.json section to write (deckmodel is the mock yardstick; the builder can use another)")
    f.set_defaults(fn=cmd_fit)
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401  (registers the settings activate checks)
    config.activate(a.set)  # the set's settings (e.g. composition.TOKENS_AS_CREATURES) shape what is fitted
    a.fn(a)


if __name__ == "__main__":
    main()
