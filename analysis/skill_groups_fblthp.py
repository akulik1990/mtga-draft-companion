"""Fblthp, Knows the Way: does having it in hand help more in decks with 3+ basic land types than in 2? Within deck (draft x build key),
games whose deck holds it: deck-demeaned linear probability of winning on copies in hand, cards and lands in hand held fixed (as
tools/within_deck.py; skill cancels inside a deck; no game-count term). Basic types in the deck = distinct of Plains/Island/Swamp/Mountain/Forest
among the 40 (Theorist's Sanctum counts as an Island; duals and Heartwood tokens do not). Groups: good skill >= 0.60, average 0.52-0.58, weak < 0.52,
population P100 (user_n_games_bucket >= 100) and ALL. Bootstrap over drafts (400), shared multiplicities. Needs the cache from analysis/skill_groups_cache.py.
Run: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 uv run python analysis/skill_groups_fblthp.py"""
import os, sys
import numpy as np, pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NAME = "Fblthp, Knows the Way"
B = 400
rng = np.random.default_rng(20261106)

Z = np.load(f"{SCR}/syn_games.npz", allow_pickle=True)
H, D, names = Z["H"], Z["D"], [str(x) for x in Z["names"]]
won = Z["won"].astype(float); skill = np.round(Z["skill"].astype(float), 2); ngames = Z["ngames"]
draft = np.asarray(Z["draft"], dtype=str); build = Z["build"]
nidx = {n: i for i, n in enumerate(names)}
dcode, _ = pd.factorize(draft); ND = dcode.max() + 1
key = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))[0]
fi = nidx[NAME]
ratings = __import__("json").load(open(f"/home/morris/Projects/mtga-draft-companion/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
lands = {c["name"] for c in ratings.values() if "Land" in (c.get("types") or [])}
lcols = [nidx[n] for n in lands if n in nidx]
ncard = H.sum(1, dtype=np.int32).astype(float); Lh = H[:, lcols].sum(1, dtype=np.int32).astype(float)
types = np.zeros(len(won), int)
for t in ("Plains", "Island", "Swamp", "Mountain", "Forest"):
    has = D[:, nidx[t]] > 0
    if t == "Island" and "Theorist's Sanctum" in nidx:
        has = has | (D[:, nidx["Theorist's Sanctum"]] > 0)
    types += has
indeck = D[:, fi] > 0
hand = H[:, fi].astype(float)
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=np.float64)
GR = {"all": np.ones(len(won), bool), "good": skill >= .60, "average": (skill >= .52) & (skill < .60), "weak": skill < .52}
POP = {"P100": ngames >= 100, "ALL": np.ones(len(won), bool)}
print(f"games {len(won)}; Fblthp in deck {indeck.sum()}; basic types among those: {dict(zip(*np.unique(types[indeck], return_counts=True)))}")
print(f"in hand (copies) among deck games: {(hand[indeck] > 0).mean():.3f}")


def fit(rows):
    """deck-demeaned OLS of won on [hand, ncard, lands in hand]; coefficient on hand in WR points, bootstrap B draws, number of decks."""
    k, ku = pd.factorize(key[rows])
    cnt = np.bincount(k)
    ok = cnt[k] >= 2
    rows, k = rows[ok], pd.factorize(key[rows[ok]])[0]
    nk = k.max() + 1 if len(rows) else 0
    A = np.column_stack([hand[rows], ncard[rows], Lh[rows], won[rows]])
    c = np.bincount(k).astype(float)
    for j in range(4):
        A[:, j] -= (np.bincount(k, A[:, j], minlength=nk) / c)[k]
    if nk < 50 or A[:, 0].std() == 0:
        return None
    uc, ci = np.unique(dcode[rows], return_inverse=True)
    S = np.zeros((len(uc), 4, 4))
    for i in range(4):
        for j in range(i, 4):
            S[:, i, j] = S[:, j, i] = np.bincount(ci, A[:, i] * A[:, j], minlength=len(uc))
    solve = lambda T: 100 * (np.linalg.pinv(T[..., :3, :3]) @ T[..., :3, 3:4])[..., 0, 0]
    return solve(S.sum(0)), solve(np.einsum("bc,cij->bij", W[:, uc], S)), nk, len(rows)


def line(label, e, bs, nk, n):
    se = bs.std()
    return f"{label:34s} {e:+6.2f} (SE {se:.2f}, z {e / se:+.1f})  decks {nk}, games {n}"


for pop, pm in POP.items():
    print(f"\n== {pop} ==")
    for g, gm in GR.items():
        base = indeck & pm & gm
        r2 = fit(np.where(base & (types == 2))[0]); r3 = fit(np.where(base & (types >= 3))[0])
        out = []
        for lab, r in (("2 basic types", r2), ("3+ basic types", r3)):
            if r is None:
                out.append(f"{lab:34s} too few")
            else:
                out.append(line(lab, r[0], r[1], r[2], r[3]))
        print(f"-- {g}")
        print("\n".join(out))
        if r2 and r3:
            d, bs = r3[0] - r2[0], r3[1] - r2[1]
            print(f"{'3+ minus 2':34s} {d:+6.2f} (SE {bs.std():.2f}, z {d / bs.std():+.1f})")
    print("-- good minus weak (3+ types / 2 types / 3+ minus 2)")
    res = {}
    for g in ("good", "weak"):
        base = indeck & pm & GR[g]
        res[g] = (fit(np.where(base & (types == 2))[0]), fit(np.where(base & (types >= 3))[0]))
    if all(res[g][i] for g in res for i in (0, 1)):
        for lab, f in (("3+ types", lambda g: res[g][1][0]), ("2 types", lambda g: res[g][0][0]),
                       ("3+ minus 2", lambda g: res[g][1][0] - res[g][0][0])):
            pass
        for lab, i in (("3+ types", 1), ("2 types", 0)):
            d, bs = res["good"][i][0] - res["weak"][i][0], res["good"][i][1] - res["weak"][i][1]
            print(f"{lab:34s} {d:+6.2f} (SE {bs.std():.2f}, z {d / bs.std():+.1f})")
        d = (res["good"][1][0] - res["good"][0][0]) - (res["weak"][1][0] - res["weak"][0][0])
        bs = (res["good"][1][1] - res["good"][0][1]) - (res["weak"][1][1] - res["weak"][0][1])
        print(f"{'3+ minus 2, good minus weak':34s} {d:+6.2f} (SE {bs.std():.2f}, z {d / bs.std():+.1f})")
