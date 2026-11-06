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
from companion.advisor_v2.lanes import pool_fixing  # noqa: E402

SHAPE = ["creatures", "cmc1", "cmc2", "cmc3", "cmc4", "cmc5", "cmc6p", "removal", "lands", "nonbasic_lands", "colours", "splash"]
# Version 2 (2026-11-06): true nonbasic lands (the card table has no "Basic" type, so version 1's nonbasic_lands equals lands),
# a second splash-colour flag (2+ splash colours), sources per splash colour (basics, duals, any-colour fixers) and duals for the main pair.
SHAPE_V2 = SHAPE + ["splash2p", "splash_src", "main_duals"]
BASICS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G", "Wastes": ""}
SKILLS = [0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]


def card_facts(set_code):
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    facts = {}
    for c in data.values():
        types, tags = c.get("types") or [], config.tags(c, set_code)
        f = {"creature": composition.is_creature(c), "land": "Land" in types, "basic": "Basic" in types,
             "cmc": int(c.get("cmc") or 0), "removal": "removal" in tags}
        base = c["name"].removeprefix("Snow-Covered ")
        f["basic2"] = base in BASICS and "Land" in types
        f["bcol"] = BASICS.get(base, "") if f["basic2"] else ""
        f["makes"] = "" if f["basic2"] else "".join(k for k, v in pool_fixing([c]).items() if v)  # colours a nonbasic land or fixer makes
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


def shape_v2(counts, facts, main_colours="", splash=""):
    """Version 2 deck-shape features: shape() with true nonbasic lands plus splash2p, splash_src and main_duals."""
    s = shape(counts, facts, main_colours, splash)
    nb, duals, src = 0.0, 0.0, dict.fromkeys("WUBRG", 0.0)
    for name, n in counts.items():
        f = facts.get(name)
        if not f:
            continue
        if f["basic2"]:
            if f["bcol"]:
                src[f["bcol"]] += n
            continue
        nb += n * f["land"]
        for col in f["makes"]:
            src[col] += n
        duals += n * (sum(col in main_colours for col in f["makes"]) >= 2)
    s["nonbasic_lands"] = nb
    s["splash2p"] = 1.0 if len(splash) >= 2 else 0.0
    s["splash_src"] = sum(src[c] for c in splash) / len(splash) if splash else 0.0
    s["main_duals"] = duals
    return s


# 17Lands' skill bucket is 0.02 wide, but SKILLS has a dummy per 0.04 and matches within < 0.02, so every odd bucket (0.42, 0.46, 0.50, 0.54, 0.58,
# 0.62, 0.66 ...: about 55% of games) got no dummy and was coded as the 0.40 baseline. Fine skill (version 2 "s" fits): a dummy per 0.02 bucket, ends pooled.
SKILLS_FINE = [round(0.36 + 0.02 * i, 2) for i in range(19)]  # 0.36 ... 0.72


def context_row(skill, games, on_play, mull, opp_mull, fine=False):
    """Controls: skill bucket dummies, log games, on the play, mulligans."""
    if fine:
        b = min(max(round((skill or 0) / 0.02) * 0.02, SKILLS_FINE[0]), SKILLS_FINE[-1])
        out = [float(abs(b - k) < 0.01) for k in SKILLS_FINE[1:]]
    else:
        out = [float(abs((skill or 0) - b) < 0.02) for b in SKILLS[1:]]
    out += [float(np.log1p(games or 0)), float(on_play), float(mull or 0), float(opp_mull or 0)]
    return out


TYPICAL = {"skill": 0.56, "games": 100, "on_play": 0.5, "mull": 0, "opp_mull": 0}


def read(set_code, facts, ids=False, skills=False, v2=False):
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        names = [n for _, n in deck_cols]
        rows, cols, vals, X_ctx, X_shape, y, test, drafts, skill = [], [], [], [], [], [], [], [], []
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
            skill.append(float(sk) if sk not in ("", "None") else 0.0)
            gm = row[col["user_n_games_bucket"]]
            X_ctx.append(context_row(float(sk) if sk not in ("", "None") else None, float(gm) if gm not in ("", "None") else None,
                                     row[col["on_play"]] == "True", float(row[col["num_mulligans"]] or 0), float(row[col["opp_num_mulligans"]] or 0)))
            s = (shape_v2 if v2 else shape)(counts, facts, row[col["main_colors"]], row[col["splash_colors"]])
            X_shape.append([s[k] for k in (SHAPE_V2 if v2 else SHAPE)])
            y.append(row[col["won"]] == "True")
            test.append(int(hashlib.md5(row[col["draft_id"]].encode()).hexdigest()[:2], 16) < 52)  # ~20%
            drafts.append(row[col["draft_id"]])
            r += 1
    C = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(r, len(names)))
    out = (names, C, np.array(X_ctx, np.float32), np.array(X_shape, np.float32), np.array(y, np.float32), np.array(test))
    out = out + (drafts,) if ids else out
    return out + (np.array(skill),) if skills else out


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
    names, C, ctx, shp, y, test, skill = read(a.set, facts, skills=True)
    print(f"{len(y)} games, {C.shape[1]} card columns, {int(test.sum())} held out")
    mu_s, sd_s = standardise(shp[~test])
    S = (shp - mu_s) / sd_s
    S2 = np.column_stack([S, S[:, SHAPE.index("creatures")] ** 2, S[:, SHAPE.index("lands")] ** 2, S[:, SHAPE.index("cmc2")] ** 2])
    full = sparse.hstack([sparse.csr_matrix(ctx), sparse.csr_matrix(S2), C]).tocsr()
    base = sparse.csr_matrix(ctx)
    tr, te = ~test, test
    # Games by 60%+ players count top_weight times (their card choices are the ones the builder copies).
    # From the raw bucket (2026-11-06): the skill dummies match within 0.02 of 0.60/0.64/0.68 only, so buckets 0.62, 0.66 and
    # 0.72+ (about 42% of 60%+ players' games) were missing the weight. Not refit since.
    top = skill >= 0.60 - 1e-9
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

    def __init__(self, set_code, section=None, model=None):
        # section None: the mock yardstick, deckmodel_v2 where the set has it (shape_v2 + skill dummies), else deckmodel
        section = section or ("deckmodel_v2" if config.fitted(set_code, "deckmodel_v2") else "deckmodel")
        self.m = model or config.fitted(set_code, section) or (config.fitted(set_code, "deckmodel") if section != "deckmodel" else None)
        if not self.m:
            raise FileNotFoundError(f"no deckmodel section in {config.set_file(set_code, 'fitted')}")
        self.facts = card_facts(set_code)
        t = self.m["typical"]
        self.ctx = float(np.dot(self.m["context"], context_row(t["skill"], t["games"], t["on_play"], t["mull"], t["opp_mull"], fine=bool(self.m.get("context_fine")))))

    def win_rate(self, counts, main_colours="", splash=""):
        v2 = self.m.get("version") == 2
        names = SHAPE_V2 if v2 else SHAPE
        s = (shape_v2 if v2 else shape)(counts, self.facts, main_colours, splash)
        z = (np.array([s[k] for k in names]) - self.m["shape_mu"]) / self.m["shape_sd"]
        z2 = np.concatenate([z, [z[SHAPE.index("creatures")] ** 2, z[SHAPE.index("lands")] ** 2, z[SHAPE.index("cmc2")] ** 2]])
        cards = sum(self.m["cards"].get(n, self.m["cards"].get(n.split(" // ")[0], 0.0)) * c for n, c in counts.items())
        logit = self.m["intercept"] + self.ctx + float(np.dot(self.m["shape"], z2)) + cards
        return 100.0 / (1.0 + np.exp(-logit))


_G = {}  # shared with the forked workers of fit2


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _fit_job(job):
    variant, tw, l2, which = job
    X, y, tr, skill = _G["X"][variant], _G["y"], _G["tr"], _G["skill"]
    sw = np.where(skill >= 0.60 - 1e-9, tw, 1.0)
    rows = tr if which == "train" else np.ones(len(y), bool) if which == "all" else _G["fold"] != which  # an int: leave that fold out
    return job, fit_logistic(X[rows], y[rows], l2, sw[rows])


def cluster_se(X, y, w, l2, sw, groups):
    """Draft-clustered sandwich SEs of a ridge logistic fit (penalty treated as fixed; coefficients are shrunk, so SEs are conditional on l2)."""
    Xi = sparse.hstack([sparse.csr_matrix(np.ones((X.shape[0], 1))), X]).tocsr()
    p = _sigmoid(Xi @ w)
    H = (Xi.T @ Xi.multiply((sw * p * (1 - p))[:, None]).tocsr()).toarray()
    pen = np.full(len(w), 2 * l2)
    pen[0] = 0
    H += np.diag(pen)
    g = np.unique(groups, return_inverse=True)[1]
    ind = sparse.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))))
    Sg = (ind @ Xi.multiply((sw * (p - y))[:, None]).tocsr()).toarray()
    Hi = np.linalg.inv(H)
    return np.sqrt(np.diag(Hi @ (Sg.T @ Sg) @ Hi))


def _groups(skill):
    return {"all": np.ones(len(skill), bool), "good": skill >= 0.60 - 1e-9, "avg": (skill >= 0.52 - 1e-9) & (skill <= 0.58 + 1e-9), "weak": skill < 0.52 - 1e-9}


def cmd_fit2(a):
    """Version 2 deck model (SHAPE_V2): fit with and without 60%+ weighting, compare to version 1 on the same split, write staged JSON."""
    import concurrent.futures as cf
    import multiprocessing as mp

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts = card_facts(a.set)
    names, C, ctx, shp, y, test, drafts, skill = read(a.set, facts, ids=True, skills=True, v2=True)
    tr = ~test
    print(f"{len(y)} games, {len(set(drafts))} drafts, {int(test.sum())} games held out", flush=True)
    shp1 = shp[:, :len(SHAPE)].copy()
    shp1[:, SHAPE.index("nonbasic_lands")] = shp1[:, SHAPE.index("lands")]  # version 1 as fitted: nonbasic_lands == lands
    cols = {"v1": (shp1, SHAPE), "v2": (shp, SHAPE_V2), "v2s": (shp, SHAPE_V2)}
    ctx_fine = np.column_stack([np.array([context_row(k, 100, 0, 0, 0, fine=True)[:len(SKILLS_FINE) - 1] for k in skill], np.float32), ctx[:, -4:]])
    ctxs = {"v1": ctx, "v2": ctx, "v2s": ctx_fine}
    X, stand = {}, {}
    for v, (M, nm) in cols.items():
        mu, sd = standardise(M[tr])
        S = (M - mu) / sd
        S2 = np.column_stack([S, S[:, nm.index("creatures")] ** 2, S[:, nm.index("lands")] ** 2, S[:, nm.index("cmc2")] ** 2])
        X[v] = sparse.hstack([sparse.csr_matrix(ctxs[v]), sparse.csr_matrix(S2), C]).tocsr()
        stand[v] = (mu, sd, S2.shape[1])
    _G.update(X=X, y=y, tr=tr, skill=skill)
    jobs = [(v, tw, l2, "train") for v in cols for tw in a.top_weights for l2 in a.l2]
    with cf.ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        W = dict(ex.map(_fit_job, jobs))
    best, ev = {}, {}
    for (v, tw, l2, _), w in W.items():
        ll = logloss(w, X[v][test], y[test])
        print(f"held-out log-loss {v} top_weight={tw} l2={l2}: {ll:.5f}", flush=True)
        if (v, tw) not in best or ll < best[(v, tw)][0]:
            best[(v, tw)] = (ll, l2)
    final_jobs = [(v, tw, best[(v, tw)][1], "all") for (v, tw) in best]
    with cf.ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        WF = dict(ex.map(_fit_job, final_jobs))
    base = sparse.csr_matrix(ctx)
    w0 = fit_logistic(base[tr], y[tr], 1.0)
    base_f = sparse.csr_matrix(ctx_fine)
    w0f = fit_logistic(base_f[tr], y[tr], 1.0)
    grp = _groups(skill)
    te_ll = lambda w, XX: -(y * np.log(_sigmoid(XX @ w[1:] + w[0]) + 1e-9) + (1 - y) * np.log(1 - _sigmoid(XX @ w[1:] + w[0]) + 1e-9))
    rowloss = {"ctx": te_ll(w0, base), "ctx_fine": te_ll(w0f, base_f)}
    for (v, tw), (ll, l2) in best.items():
        rowloss[(v, tw)] = te_ll(W[(v, tw, l2, "train")], X[v])
    dg = np.unique(np.array(drafts), return_inverse=True)[1]
    def clustered(d, m):
        """mean of d over rows m with draft-clustered SE."""
        mm = m & test
        s = np.bincount(dg[mm], weights=d[mm], minlength=dg.max() + 1)
        n = np.bincount(dg[mm], minlength=dg.max() + 1)
        N = n.sum(); mean = s.sum() / N
        r = s - mean * n
        return float(mean), float(np.sqrt((r ** 2).sum()) / N)
    report = {"games": int(len(y)), "drafts": int(len(set(drafts))), "heldout_games": int(test.sum()), "best": {f"{v}|{tw}": {"logloss": ll, "l2": l2} for (v, tw), (ll, l2) in best.items()}, "groups": {}}
    for g, m in grp.items():
        row = {"n_heldout": int((m & test).sum()), "ctx": clustered(rowloss["ctx"], m)[0], "ctx_fine": clustered(rowloss["ctx_fine"], m)[0]}
        for k in best:
            row[f"{k[0]}|{k[1]}"] = clustered(rowloss[k], m)
        for tw in a.top_weights:  # positive = the second is better
            row[f"v1-v2|{tw}"] = clustered(rowloss[("v1", tw)] - rowloss[("v2", tw)], m)
            row[f"v2-v2s|{tw}"] = clustered(rowloss[("v2", tw)] - rowloss[("v2s", tw)], m)
            row[f"v1-v2s|{tw}"] = clustered(rowloss[("v1", tw)] - rowloss[("v2s", tw)], m)
        report["groups"][g] = row
        print(g, json.dumps(row), flush=True)
    for (v, tw, l2, _), w in WF.items():
        sw = np.where(skill >= 0.60 - 1e-9, tw, 1.0)
        se = cluster_se(X[v], y, w, l2, sw, np.array(drafts))
        mu, sd, k_shape = stand[v]
        nm = (SHAPE if v == "v1" else SHAPE_V2) + ["creatures^2", "lands^2", "cmc2^2"]
        k_ctx = ctxs[v].shape[1]
        fine = v == "v2s"
        model = {"set": a.set, "version": 1 if v == "v1" else 2, "context_fine": fine, "l2": l2, "heldout_logloss": best[(v, tw)][0], "context_only_logloss": logloss(w0f, base_f[test], y[test]) if fine else logloss(w0, base[test], y[test]),
                 "intercept": float(w[0]), "context": [float(x) for x in w[1:1 + k_ctx]], "shape_names": nm,
                 "shape": [float(x) for x in w[1 + k_ctx:1 + k_ctx + k_shape]], "shape_se": [float(x) for x in se[1 + k_ctx:1 + k_ctx + k_shape]],
                 "shape_mu": [float(x) for x in mu], "shape_sd": [float(x) for x in sd],
                 "cards": {n: float(x) for n, x in zip(names, w[1 + k_ctx + k_shape:])}, "typical": TYPICAL, "top_weight": tw}
        wt = W[(v, tw, l2, "train")]
        train = dict(model, intercept=float(wt[0]), context=[float(x) for x in wt[1:1 + k_ctx]], shape=[float(x) for x in wt[1 + k_ctx:1 + k_ctx + k_shape]],
                     cards={n: float(x) for n, x in zip(names, wt[1 + k_ctx + k_shape:])}, trained_on="train drafts only (hash < 52 held out)")
        tag = f"{v}_tw{int(tw)}"
        (out / f"{tag}.json").write_text(json.dumps(model))
        (out / f"{tag}.train.json").write_text(json.dumps(train))
        print(tag, "shape terms (per SD, SE):", ", ".join(f"{n} {b:+.3f} ({e:.3f})" for n, b, e in zip(nm, model["shape"], model["shape_se"])), flush=True)
    (out / "fit2_report.json").write_text(json.dumps(report, indent=1))
    print("staged in", out)


def fold_of(draft_id, k=5):
    """Cross-fitting fold of a draft (md5 of its id mod k), independent of the 80/20 split above."""
    return int(hashlib.md5(draft_id.encode()).hexdigest(), 16) % k


def cmd_cvfit(a):
    """k-fold cross-fit of deck models (specs variant:top_weight:l2, variant v1 | v2 | v2s as in fit2): fold j leaves the drafts with fold_of == j out.
    Writes DIR/cv_<variant>_tw<w>_fold<j>.json; every deck of fold j is then predicted by a model that never saw its draft."""
    import concurrent.futures as cf
    import multiprocessing as mp

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts = card_facts(a.set)
    names, C, ctx, shp, y, test, drafts, skill = read(a.set, facts, ids=True, skills=True, v2=True)
    fold = np.array([fold_of(d) for d in drafts])
    shp1 = shp[:, :len(SHAPE)].copy()
    shp1[:, SHAPE.index("nonbasic_lands")] = shp1[:, SHAPE.index("lands")]
    ctx_fine = np.column_stack([np.array([context_row(k, 100, 0, 0, 0, fine=True)[:len(SKILLS_FINE) - 1] for k in skill], np.float32), ctx[:, -4:]])
    cols = {"v1": (shp1, SHAPE, ctx), "v2": (shp, SHAPE_V2, ctx), "v2s": (shp, SHAPE_V2, ctx_fine)}
    specs = [(sp.split(":")[0], float(sp.split(":")[1]), float(sp.split(":")[2])) for sp in a.specs]
    X, stand = {}, {}
    for v in {sp[0] for sp in specs}:
        M, nm, cx = cols[v]
        mu, sd = standardise(M)
        S = (M - mu) / sd
        S2 = np.column_stack([S, S[:, nm.index("creatures")] ** 2, S[:, nm.index("lands")] ** 2, S[:, nm.index("cmc2")] ** 2])
        X[v] = sparse.hstack([sparse.csr_matrix(cx), sparse.csr_matrix(S2), C]).tocsr()
        stand[v] = (mu, sd, S2.shape[1], cx.shape[1], nm)
    _G.update(X=X, y=y, tr=~test, skill=skill, fold=fold)
    jobs = [(v, tw, l2, j) for (v, tw, l2) in specs for j in range(5)]
    print(f"{len(y)} games, {len(set(drafts))} drafts, folds {np.bincount(fold).tolist()}, {len(jobs)} fits", flush=True)
    with cf.ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        for (v, tw, l2, j), w in ex.map(_fit_job, jobs):
            mu, sd, k_shape, k_ctx, nm = stand[v]
            model = {"set": a.set, "version": 1 if v == "v1" else 2, "context_fine": v == "v2s", "l2": l2, "intercept": float(w[0]),
                     "context": [float(x) for x in w[1:1 + k_ctx]], "shape_names": nm + ["creatures^2", "lands^2", "cmc2^2"],
                     "shape": [float(x) for x in w[1 + k_ctx:1 + k_ctx + k_shape]], "shape_mu": [float(x) for x in mu], "shape_sd": [float(x) for x in sd],
                     "cards": {n: float(x) for n, x in zip(names, w[1 + k_ctx + k_shape:])}, "typical": TYPICAL, "top_weight": tw, "left_out_fold": j}
            (out / f"cv_{v}_tw{int(tw)}_fold{j}.json").write_text(json.dumps(model))
            print("done", v, tw, l2, j, flush=True)


def cmd_install(a):
    """Write the staged version 2 models into fitted.json as new sections (the existing sections are untouched)."""
    out = Path(a.out_dir)
    for tw, section in ((1, "deckmodel_v2"), (3, "deckmodel_top3_v2")):
        m = json.loads((out / f"{a.variant}_tw{tw}.json").read_text())
        path = config.write_fitted(a.set, section, m, "tools/deck_model.py fit2 (version 2 shape: true nonbasic lands, 2+ splash colours, splash sources, main duals" + (", skill dummy per 0.02 bucket)" if a.variant == "v2s" else ")")
                                   + (f", 60%+ players' games weighted x{tw}" if tw != 1 else ""))
        print("wrote", section, "to", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("set"); f.add_argument("--l2", type=float, nargs="+", default=[10.0, 30.0, 100.0, 300.0])
    f.add_argument("--top-weight", type=float, default=1.0, help="weight of 60%%+ players' games")
    f.add_argument("--section", default="deckmodel", help="fitted.json section to write (deckmodel is the mock yardstick; the builder can use another)")
    f.set_defaults(fn=cmd_fit)
    f2 = sub.add_parser("fit2"); f2.add_argument("set"); f2.add_argument("--out-dir", required=True)
    f2.add_argument("--l2", type=float, nargs="+", default=[10.0, 30.0, 100.0, 300.0]); f2.add_argument("--top-weights", type=float, nargs="+", default=[1.0, 3.0])
    f2.set_defaults(fn=cmd_fit2)
    cv = sub.add_parser("cvfit"); cv.add_argument("set"); cv.add_argument("--out-dir", required=True); cv.add_argument("--specs", nargs="+", required=True)
    cv.set_defaults(fn=cmd_cvfit)
    ins = sub.add_parser("install"); ins.add_argument("set"); ins.add_argument("--out-dir", required=True); ins.add_argument("--variant", default="v2s", choices=["v2", "v2s"])
    ins.set_defaults(fn=cmd_install)
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401  (registers the settings activate checks)
    config.activate(a.set)  # the set's settings (e.g. composition.TOKENS_AS_CREATURES) shape what is fitted
    a.fn(a)


if __name__ == "__main__":
    main()
