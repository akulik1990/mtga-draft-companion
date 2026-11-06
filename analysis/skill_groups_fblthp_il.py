"""Fblthp, Impossibly Lost: within-deck synergy (tools/within_deck.py design, as analysis/skill_groups_synergy.py S1) with enablers of three kinds,
pre-registered 2026-11-06: mill, evasion (tag), draw (tag card_advantage or draw text), plus the combined list (manual.json match, mode any).
Deck-demeaned linear probability of winning on enablers in hand (cards and lands in hand held fixed), games with Fblthp in hand ("with") vs in the deck
but not in hand ("alone"); synergy = with - alone, WR points per enabler in hand. Groups good >= 0.60 / average 0.52-0.58 / weak < 0.52 skill.
Primary: P100 (user_n_games_bucket >= 100), group "all". Four tests (combined + three sub-tests), Bonferroni alpha 0.05/4 = 0.0125 two-sided (|z| >= 2.50).
Needs scratchpad syn_games.npz (analysis/skill_groups_cache.py) and il_lists.json (enabler lists from needs.matches). Bootstrap over drafts, 400."""
import json, os
import numpy as np, pandas as pd

SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NAME = "Fblthp, Impossibly Lost"
B = 400
rng = np.random.default_rng(20261106)
Z = np.load(f"{SCR}/syn_games.npz", allow_pickle=True)  # our own cache
H, D, names = Z["H"], Z["D"], [str(x) for x in Z["names"]]
won = Z["won"].astype(float); skill = np.round(Z["skill"].astype(float), 2); ngames = Z["ngames"]
draft = np.asarray(Z["draft"], dtype=str); nidx = {n: i for i, n in enumerate(names)}
dcode, _ = pd.factorize(draft); ND = dcode.max() + 1
ratings = json.load(open("/home/morris/Projects/mtga-draft-companion/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
lands = {c["name"] for c in ratings.values() if "Land" in (c.get("types") or [])}
lcols = [nidx[n] for n in lands if n in nidx]
ncard = H.sum(1, dtype=np.int32).astype(float); Lh = H[:, lcols].sum(1, dtype=np.int32).astype(float)
fi = nidx[NAME]; inh = H[:, fi] > 0; ind = D[:, fi] > 0
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=np.float64)
GR = {"all": np.ones(len(won), bool), "good": skill >= .60, "average": (skill >= .52) & (skill < .60), "weak": skill < .52}
POP = {"P100": ngames >= 100, "ALL": np.ones(len(won), bool)}
LISTS = json.load(open(f"{SCR}/il_lists.json"))


def fit(rows, X):
    """deck-keyed (draft id, as tools/within_deck.py) demeaned OLS; coefficient of the first column in WR points, bootstrap draws."""
    k = pd.factorize(dcode[rows])[0]
    ok = np.bincount(k)[k] >= 2
    rows = rows[ok]; k = pd.factorize(dcode[rows])[0]; nk = k.max() + 1 if len(rows) else 0
    A = np.column_stack([X[rows], won[rows]]); c = np.bincount(k).astype(float)
    for j in range(4):
        A[:, j] -= (np.bincount(k, A[:, j], minlength=nk) / c)[k]
    if nk < 50 or A[:, 0].std() == 0:
        return None
    S = np.zeros((nk, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            S[:, i, j] = S[:, j, i] = np.bincount(k, A[:, i] * A[:, j], minlength=nk)
    uk = np.unique(k)
    solve = lambda T: 100 * (np.linalg.pinv(T[..., :3, :3]) @ T[..., :3, 3:4])[..., 0, 0]
    # draft id is the key, so decks and drafts coincide: bootstrap the per-draft blocks
    d_of = np.zeros(nk, int); d_of[k] = dcode[rows]
    return solve(S.sum(0)), solve(np.einsum("bc,cij->bij", W[:, d_of], S)), nk


def z(e, bs):
    return f"{e:+6.2f} (SE {bs.std():.2f}, z {e / bs.std():+.1f})"


print(f"games {len(won)}; Fblthp in deck {ind.sum()}, in hand {inh.sum()}")
res = {}
for lab, names_ in LISTS.items():
    cols = [nidx[n] for n in names_ if n in nidx]
    X = np.column_stack([H[:, cols].sum(1, dtype=np.int32).astype(float), ncard, Lh])
    print(f"\n=== {lab}: {len(cols)} enablers ===")
    for pop, pm in POP.items():
        for g, gm in GR.items():
            m = pm & gm
            w = fit(np.where(inh & m)[0], X); a = fit(np.where(ind & ~inh & m)[0], X)
            res[(lab, pop, g)] = (w, a)
            if w is None or a is None:
                print(f"{pop:5s} {g:8s} too few"); continue
            d = w[0] - a[0]; bs = w[1] - a[1]
            print(f"{pop:5s} {g:8s} with {z(w[0], w[1])} | alone {z(a[0], a[1])} | synergy {z(d, bs)}  decks {w[2]}/{a[2]}")
        r = {g: res[(lab, pop, g)] for g in ("good", "weak")}
        if all(v is not None for t in r.values() for v in t):
            d = (r["good"][0][0] - r["good"][1][0]) - (r["weak"][0][0] - r["weak"][1][0])
            bs = (r["good"][0][1] - r["good"][1][1]) - (r["weak"][0][1] - r["weak"][1][1])
            print(f"{pop:5s} good minus weak synergy {z(d, bs)}")
