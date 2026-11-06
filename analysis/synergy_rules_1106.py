"""Within-deck synergy of the 2026-11-06 per-card rules (Ghalta, the Ways, Danitha), by player skill and population, against placebo sets.
Pre-registered before any outcome was read (docs/synergy-rules-2026-11-06.md section 1): statistic = synergy (effect with the payoff in hand
minus effect with the payoff in the deck but not drawn; WR points per enabler, deck-demeaned, cards and lands in hand held fixed, keyed by
draft id as tools/within_deck.py) minus the mean synergy of 30 random ordinary-spell sets of the same size held as often; z = that difference over
sqrt(bootstrap SE^2 + placebo SD^2). Groups by the player's skill bucket: good >= 0.60, average 0.52-0.58, weak < 0.52; populations ALL and P100
(user_n_games_bucket >= 100). Also the card_needs.py logistic (product term). Cache: analysis/skill_groups_cache.py output (syn_games.npz).
    systemd-run --user --scope -p MemoryMax=4G uv run python analysis/synergy_rules_1106.py NPZ OUT.json
"""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MTGA_COMPANION_DATA", f"{R}/data")
sys.path.insert(0, R)
NPZ, OUT = sys.argv[1], sys.argv[2]
B, NPL = 400, 30
GROUPS = ["all", "good", "average", "weak"]

Z = np.load(NPZ, allow_pickle=True)
H, D, names = Z["H"], Z["D"], [str(x) for x in Z["names"]]
won = Z["won"].astype(float); skill = Z["skill"]; ngames = Z["ngames"]; play = Z["play"].astype(float)
dnames, dcode = np.unique(np.asarray(Z["draft"], dtype=str), return_inverse=True); ND = len(dnames)
nidx = {n: i for i, n in enumerate(names)}
ratings = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
lands = {c["name"] for c in ratings.values() if "Land" in (c.get("types") or [])}
lcols = [nidx[n] for n in lands if n in nidx]
ncard = H.sum(1, dtype=np.int32).astype(float); Lh = H[:, lcols].sum(1, dtype=np.int32).astype(float)
s = np.round(skill.astype(float), 2)
GM = {"all": np.ones(len(s), bool), "good": s >= .60, "average": (s >= .52) & (s < .60), "weak": s < .52}
POPS = {"ALL": np.ones(len(won), bool), "P100": ngames >= 100}
rng = np.random.default_rng(20261106)
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=np.float32)


def demean(rows, X):
    ku, k = np.unique(dcode[rows], return_inverse=True); nk = len(ku)
    cnt = np.bincount(k).astype(float)
    A = np.column_stack([X, won[rows]]).astype(float)
    for j in range(A.shape[1]):
        A[:, j] -= (np.bincount(k, A[:, j], minlength=nk) / cnt)[k]
    keep = cnt[k] >= 2
    return A[keep], dcode[rows][keep]


def point(rows, X):
    A, _ = demean(rows, X)
    if len(np.unique(dcode[rows])) < 50 or A[:, 0].std() == 0: return None
    p = A.shape[1] - 1
    return 100 * np.linalg.lstsq(A[:, :p], A[:, p], rcond=None)[0][0]


def boot(rows, X):
    """(estimate, bootstrap draws) of the first coefficient, bootstrap over drafts with the shared multiplicities."""
    A, dd = demean(rows, X)
    if len(np.unique(dd)) < 50 or A[:, 0].std() == 0: return None
    p = A.shape[1] - 1
    uc, ci = np.unique(dd, return_inverse=True)
    S = np.zeros((len(uc), p + 1, p + 1))
    for i in range(p + 1):
        for j in range(i, p + 1): S[:, i, j] = S[:, j, i] = np.bincount(ci, A[:, i] * A[:, j], minlength=len(uc))
    solve = lambda T: 100 * np.linalg.pinv(T[..., :p, :p]) @ T[..., :p, p:p + 1]
    est = solve(S.sum(0))[0, 0]
    bs = solve(np.einsum("bc,cij->bij", W[:, uc].astype(np.float64), S))[:, 0, 0]
    return est, bs, len(uc)


def logit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(X @ w)))
        Hh = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(Hh, X.T @ (y - p)); w += step
        if np.abs(step).max() < 1e-8: break
    return w, np.sqrt(np.diag(np.linalg.inv(Hh)))


def one(task):
    label, payoffs, enablers = task
    Pc = [nidx[x] for x in payoffs if x in nidx]
    Ec = [nidx[e] for e in enablers if e in nidx and e not in lands and e not in payoffs]
    if not Pc or not Ec: return label, None
    inh = H[:, Pc].sum(1) > 0; ind = D[:, Pc].sum(1) > 0
    pool_games = np.where(inh)[0]
    held = D[pool_games].sum(0)
    en_rate = np.mean(held[Ec]); isp = set(Pc) | set(Ec) | set(lcols)
    pool = [i for i in range(len(names)) if i not in isp and 0.5 * en_rate <= held[i] <= 2 * en_rate]
    pick = random.Random(0); fakes = [pick.sample(pool, len(Ec)) for _ in range(NPL)] if len(pool) >= len(Ec) else []
    out = {"enablers": len(Ec), "pool": len(pool), "res": {}, "needs": {}}
    sk = np.where(np.isnan(skill), .55, skill).astype(float)
    for pn, pm in POPS.items():
        for g in GROUPS:
            m = pm & GM[g]
            r_with, r_alone = np.where(inh & m)[0], np.where(ind & ~inh & m)[0]
            def X(rows, cols): return np.column_stack([H[rows][:, cols].sum(1, dtype=np.int32), ncard[rows], Lh[rows]]).astype(float)
            w, a = boot(r_with, X(r_with, Ec)) if len(r_with) else None, boot(r_alone, X(r_alone, Ec)) if len(r_alone) else None
            if w is None or a is None:
                out["res"][f"{pn}|{g}"] = None; continue
            syn = w[0] - a[0]; sb = w[1] - a[1]
            pl = []
            for f in fakes:
                xw, xa = point(r_with, X(r_with, f)), point(r_alone, X(r_alone, f))
                if xw is not None and xa is not None: pl.append(xw - xa)
            out["res"][f"{pn}|{g}"] = dict(decks_with=w[2], decks_alone=a[2], with_=w[0], alone=a[0], syn=syn, syn_se=float(np.std(sb, ddof=1)),
                                           pl_mean=float(np.mean(pl)) if len(pl) >= 5 else None, pl_sd=float(np.std(pl)) if len(pl) >= 5 else None,
                                           npl=len(pl), sb=sb.tolist() if g in ("good", "weak") else None)
            # card_needs.py logistic: games whose deck held the payoff; product of (deck enabler count) and (payoff in hand)
            rows = np.where(ind & m)[0]
            if len(rows) >= 200 and inh[rows].sum() >= 30:
                nen = D[rows][:, Ec].sum(1, dtype=np.int32).astype(float); hand = inh[rows].astype(float)
                Xm = np.column_stack([np.ones(len(rows)), nen, hand, nen * hand, sk[rows] - sk[rows].mean(), play[rows]])
                if Xm[:, 3].std() > 0:
                    wv, se = logit(Xm, won[rows]); out["needs"][f"{pn}|{g}"] = dict(est=wv[3] * 24.75, se=se[3] * 24.75, n=len(rows))
    return label, out


if __name__ == "__main__":
    from companion.advisor_v2 import config, needs
    data = {}
    for c in ratings.values(): data.setdefault(c["name"], c)
    tasks = []
    for name, spec in (config.layer("FRA", "manual")["card_wants"]).items():
        if name.startswith("_") or not (name.startswith(("Way of", "Ghalta the U", "Danitha, Spear", "Kiora of Salt"))): continue
        for j, sp in enumerate([spec] + (spec.get("extra") or [])):
            en = sorted(n for n, c in data.items() if n != name and "Basic" not in (c.get("types") or []) and needs.is_enabler(c, sp))
            tasks.append((f"{name} | {'main' if j == 0 else sp['name']}", [name], en))
    ways = [n for n in data if n.startswith("Way of the")]
    emp = sorted(n for n, c in data.items() if needs.matches(c, {"text": "empower jace", "not": {"text": "when way of the"}}) and "Basic" not in c["types"])
    tasks.append(("POOLED any Way | empower Jace cards (descriptive)", ways, emp))
    print(len(tasks), "lists", flush=True)
    with Pool(6) as pool:
        res = {}
        for label, out in pool.imap_unordered(one, tasks):
            res[label] = out; print("done", label, flush=True)
    json.dump(res, open(OUT, "w"))
