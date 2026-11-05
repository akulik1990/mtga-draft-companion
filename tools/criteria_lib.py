"""Shared pieces for the criteria search (tools/criteria_search.py, tools/criteria_current.py).

Data: one row per deck of players with 100+ games and 2-3 base colours (35,646 decks; rows line up with table 3).
Outcome "first": the deck's first game only (record 0-0, before record-based pairing); "all": all its games (more power,
but pairing compresses win rates toward 50%, so effects read smaller, not larger). Controls: format, games bucket, rank of
the deck's first game, skill bucket (the 100+ games players' bucket contains this deck's games only marginally), and for
"first" play/draw and mulligans. Models are binomial ridge logistic, fitted by Newton, 2-fold cross-fit by draft hash
(fit on one half of the drafts, scored on the other). Gains are log-loss per game with SEs clustered by draft.
"""

import hashlib
import json
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from tools import deck_counts, tables  # noqa: E402
from tools.analyse import META, TEXT, merge_fields  # noqa: E402
from tools.deck_table import cards as card_table  # noqa: E402

BIG = {"100", "500"}
PTS = 24.5  # win-rate points per logit at a 57% win rate


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_data(set_code="FRA", outcome="all"):
    games = pickle.loads((DATA_DIR / "Temp" / f"context_games_{set_code}.pkl").read_bytes())
    first, order, W, G = {}, [], {}, {}
    for x in games:  # (fmt, draft, build, time, skill, games bucket, rank, on_play, mull, omull, won)
        k = (x[0], x[1], x[2])
        if k not in first:
            order.append(k)
            first[k], W[k], G[k] = x, 0, 0
        elif x[3] < first[k][3]:
            first[k] = x
        W[k] += int(x[10])
        G[k] += 1
    rows = deck_counts.read(set_code)
    if len(rows) != len(order) or any((k[0], k[1]) != (r["format"], r["draft"]) for k, r in zip(order, rows)):
        sys.exit("table 3 does not line up with the game data")
    keep = [i for i, (k, r) in enumerate(zip(order, rows)) if 2 <= r["n_base"] <= 3 and first[k][5] in BIG]
    with gzip.open(tables.path(set_code, tables.DECKS), "rt") as f:
        decks = json.load(f)
    D = dict(idx=np.array(keep), rows=[rows[i] for i in keep], decks=[decks[i] for i in keep], fg=[first[order[i]] for i in keep])
    if outcome == "first":
        D["w"] = np.array([float(x[10]) for x in D["fg"]]); D["n"] = np.ones(len(keep))
    else:
        D["w"] = np.array([float(W[order[i]]) for i in keep]); D["n"] = np.array([float(G[order[i]]) for i in keep])
    D["draft"] = [x[1] for x in D["fg"]]
    h = lambda d, m: int(hashlib.md5(d.encode()).hexdigest(), 16) % m
    D["half"] = np.array([h(d, 2) for d in D["draft"]])
    D["val"] = np.array([h(d, 5) == 0 for d in D["draft"]])
    D["outcome"] = outcome
    return D


def context(D, skill=True):
    cols, data = {}, []
    for x in D["fg"]:
        fmt, _, _, _, sk, gm, rk, op, mu, om, _ = x
        ks = [("fmt", fmt), ("games", fmt, gm), ("rank", fmt, rk)]
        if D["outcome"] == "first":
            ks += [("play", op), ("mull", mu), ("omull", om), ("mp", mu, op)]
        if skill:
            ks.append(("skill", fmt, sk))
        data.append([cols.setdefault(k, len(cols)) for k in ks])
    X = np.zeros((len(D["fg"]), len(cols) + 1))
    X[:, 0] = 1
    for r, idx in enumerate(data):
        X[r, [i + 1 for i in idx]] = 1
    return X


def card_features(D, set_code="FRA"):
    """Per-deck sums over spells of card-level splits the table-3 fields do not have (early/late creatures, cheap removal ...)."""
    table = card_table(set_code)
    names = list(table)
    col = {n: i for i, n in enumerate(names)}
    num = lambda n, f: float(table[n].get(f) or 0) if not isinstance(table[n].get(f), str) else 0.0
    A = {f: np.array([num(n, f) for n in names]) for f in
         ("cmc", "creature", "power", "toughness", "land", "role_removal", "role_card_advantage", "role_evasion", "draw", "scry", "surveil", "loot",
          "cycling", "flying", "menace", "trample", "haste", "lifelink", "defender", "reach", "deathtouch", "first_strike", "gain_life", "instant",
          "sorcery", "tokens", "creature_tokens", "gihwr", "makes_mana", "role_mana", "tag_fixing_ramp", "landcycling", "land_search", "treasure",
          "tag_combat_trick", "pump", "flash", "etb", "dies", "recursion", "role_creatures")}
    sp = 1 - A["land"]
    rows, cols, vals = [], [], []
    for i, d in enumerate(D["decks"]):
        for n, c in d["counts"].items():
            if n in col:
                rows.append(i); cols.append(col[n]); vals.append(c)
    C = sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(D["decks"]), len(names)))
    gw = A["gihwr"][sp > 0]
    gm, gs = gw[gw > 0].mean(), gw[gw > 0].std()
    cr = A["role_creatures"] > 0
    flow = ((A["draw"] + A["scry"] + A["surveil"] + A["loot"] + A["cycling"] + A["role_card_advantage"]) > 0).astype(float)
    interact = (A["role_removal"] > 0).astype(float)
    m = lambda x: (C @ (x * sp))
    out = {
        "creatures": m(cr.astype(float)),
        "cr_early": m((cr & (A["cmc"] <= 3)).astype(float)),
        "cr_late": m((cr & (A["cmc"] >= 4)).astype(float)),
        "cr_big": m((cr & (A["cmc"] >= 5)).astype(float)),
        "cr_blockers": m((cr & (A["cmc"] <= 3) & (A["toughness"] >= 3)).astype(float)),
        "cr_power": m(A["power"] * cr),
        "cr_toughness": m(A["toughness"] * cr),
        "removal": m(interact),
        "rm_cheap": m(interact * (A["cmc"] <= 2)),
        "rm_late": m(interact * (A["cmc"] >= 4)),
        "flow": m(flow),
        "flow_cheap": m(flow * (A["cmc"] <= 2)),
        "evasion": m((A["role_evasion"] > 0).astype(float)),
        "finishers": m(((A["cmc"] >= 5) & (A["gihwr"] >= gm + gs)).astype(float)),
        "noncreature": m((~cr).astype(float)),
        "lands": C @ A["land"],
        "mana": m((A["role_mana"] > 0).astype(float)),
        "gih_sum": m(np.where(A["gihwr"] > 0, A["gihwr"], gm)),
    }
    n_sp = m(np.ones(len(names)))
    out["spells"] = n_sp
    out["gih_mean"] = out["gih_sum"] / np.maximum(n_sp, 1)
    out["cr_late_share"] = out["cr_late"] / np.maximum(out["creatures"], 1)
    # weakest cards: spells with GIH WR more than 1 SD under the set mean, and the best (top SD)
    out["weak"] = m(((A["gihwr"] > 0) & (A["gihwr"] < gm - gs)).astype(float))
    out["strong"] = m((A["gihwr"] >= gm + gs).astype(float))
    return out


def standardise(X):
    X = np.asarray(X, float)
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def fit(X, w, n, lam, mask, iters=30):
    """Binomial ridge by Newton. lam: per-coefficient penalty (0.5*lam*b^2), lam[0] = 0 for the intercept."""
    Xm, wm, nm = X[mask], w[mask], n[mask]
    b = np.zeros(X.shape[1])
    b[0] = np.log((wm.sum() + 1) / (nm.sum() - wm.sum() + 1))
    L = np.diag(lam)
    for _ in range(iters):
        p = sigmoid(Xm @ b)
        g = Xm.T @ (wm - nm * p) - lam * b
        H = (Xm * (nm * p * (1 - p))[:, None]).T @ Xm + L
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def lams(k_ctx, k_deck, lam_deck, lam_ctx=2.0):
    lam = np.r_[np.full(k_ctx, lam_ctx), np.full(k_deck, lam_deck)]
    lam[0] = 0
    return lam


def dev(z, w, n):
    """Per-deck log-loss summed over its games."""
    return n * np.logaddexp(0, z) - w * z


def crossfit(Xc, Xd, D, lam_deck, lam_ctx=2.0):
    """Out-of-fold (full logit z, deck-part logit zd)."""
    X = np.hstack([Xc, Xd]) if Xd.shape[1] else Xc
    lam = lams(Xc.shape[1], Xd.shape[1], lam_deck, lam_ctx)
    z, zd = np.zeros(len(D["w"])), np.zeros(len(D["w"]))
    for h in (0, 1):
        b = fit(X, D["w"], D["n"], lam, D["half"] != h)
        te = D["half"] == h
        z[te] = X[te] @ b
        if Xd.shape[1]:
            zd[te] = Xd[te] @ b[Xc.shape[1]:]
    return z, zd


def choose_lam(Xc, Xd, D, grid=(30, 100, 300, 1000, 3000, 10000)):
    """Penalty by loss on the validation fifth of each training half (never the half being scored)."""
    X = np.hstack([Xc, Xd])
    best = None
    for g in grid:
        lam = lams(Xc.shape[1], Xd.shape[1], g)
        tot = 0.0
        for h in (0, 1):
            tr = D["half"] != h
            b = fit(X, D["w"], D["n"], lam, tr & ~D["val"])
            m = tr & D["val"]
            tot += dev(X[m] @ b, D["w"][m], D["n"][m]).sum() / D["n"][m].sum()
        if best is None or tot < best[0]:
            best = (tot, g)
    return best[1]


def gain(zbase, zmodel, D, mask=None):
    """(gain per game, SE clustered by draft): log-loss of zbase minus zmodel (positive = zmodel better)."""
    m = np.ones(len(zbase), bool) if mask is None else mask
    d = (dev(zbase, D["w"], D["n"]) - dev(zmodel, D["w"], D["n"]))[m]
    n = D["n"][m]
    g = d.sum() / n.sum()
    res = d - g * n
    cl = {}
    for dr, r in zip(np.array(D["draft"])[m], res):
        cl[dr] = cl.get(dr, 0.0) + r
    return float(g), float(np.sqrt(sum(v * v for v in cl.values())) / n.sum())


def sandwich(X, w, n, b, lam, drafts):
    """Cluster-robust (by draft) covariance of the ridge-logistic coefficients."""
    p = sigmoid(X @ b)
    H = (X * (n * p * (1 - p))[:, None]).T @ X + np.diag(lam)
    S = X * (w - n * p)[:, None]
    _, inv = np.unique(np.asarray(drafts), return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(inv.max() + 1, len(inv))) @ S
    M = A.T @ A
    Hi = np.linalg.inv(H)
    return Hi @ M @ Hi


def subset(D, mask):
    """D restricted to the decks where mask is True."""
    ix = np.where(mask)[0]
    out = dict(D)
    for k in ("idx", "w", "n", "half", "val"):
        out[k] = D[k][ix]
    for k in ("rows", "decks", "fg", "draft"):
        out[k] = [D[k][i] for i in ix]
    return out


def deck_fields(set_code, counts_list):
    """Table-3 style fields for arbitrary decks ([{card name: copies}], basics and nonbasics included): {field: array}, plus
    "base": the base colours by the user's rule and the derived splits of card_features. Mirrors tools/deck_counts.py."""
    import json as _json

    from tools.deck_counts import CURVE, SKIP, colours_of

    table = card_table(set_code)
    names = list(table)
    col = {n: i for i, n in enumerate(names)}
    data = {}
    for c in _json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
        data.setdefault(c["name"], c)
    feats = [k for k, v in next(iter(table.values())).items() if isinstance(v, float) and k not in SKIP]
    slot = lambda v: v if isinstance(v, str) else f"{int(v)}"
    num = lambda v: v if isinstance(v, float) else 0.0
    spell = np.array([0.0 if table[n].get("land") else 1.0 for n in names])
    Fm = np.column_stack([np.array([num(table[n].get(f)) for n in names]) for f in feats]
                         + [np.array([1.0 if slot(table[n].get("slot")) == s else 0.0 for n in names]) for s in CURVE]
                         + [np.array([num(table[n].get(k)) for n in names]) for k in ("cmc", "power", "toughness")])
    Fm = Fm * np.where(np.array([f.startswith("land_") for f in feats] + [False] * (len(CURVE) + 3)), 1 - spell[:, None], spell[:, None])
    rows, cols, vals = [], [], []
    for i, d in enumerate(counts_list):
        for n, k in d.items():
            if n in col:
                rows.append(i); cols.append(col[n]); vals.append(k)
    C = sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(counts_list), len(names)))
    S = C @ Fm
    k = len(feats)
    n_sp, n_l = C @ spell, C @ (1 - spell)
    out = {f: S[:, i] for i, f in enumerate(feats)}
    out.update({c: S[:, k + i] for i, c in enumerate(CURVE)})
    out.update(spells=n_sp, lands=n_l, avg_mv=S[:, k + 6] / np.maximum(n_sp, 1), power=S[:, k + 7], toughness=S[:, k + 8])
    out["base"] = np.array([colours_of([data[n] for n, c in d.items() if n in data and "Land" not in (data[n].get("types") or []) for _ in range(c)])[0]
                            for d in counts_list])
    der = card_features({"decks": [{"counts": d} for d in counts_list]}, set_code)
    out.update({f: v for f, v in der.items()})
    return out
