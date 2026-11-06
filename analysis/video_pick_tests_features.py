"""Stage 2 for analysis/video_pick_tests_h1a.py, _h1b.py, _h5.py: build the H1a pick table and the H5 draft x pair table from the memmaps of video_pick_tests_cache.py.
Designs are fixed in analysis/video_pick_tests_h1a.py, _h1b.py, _h5.py (header) and docs/video-test-picks-FRA.md; this file only computes features.
Output data/Temp/vid1106p/h1.npz, h5.npz. Run like video_pick_tests_cache.py."""
import os, sys, json, zlib, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, f"{R}/analysis"); os.environ.setdefault("MTGA_COMPANION_DATA", f"{R}/data")
import video_counters as vc
from companion.advisor_v2.lanes import LANES, pool_fixing, pips, castable, off_pips
D = f"{R}/data/Temp/vid1106p"
names = json.load(open(f"{D}/cols.json")); by = vc.cards(); C = len(names); L = len(LANES); COL = "WUBRG"
cards = [by[n.split(" // ")[0]] for n in names]
island = np.array(["Land" in c["types"] for c in cards]); spell = ~island
gA = np.array([((c["deck_colors"].get("All Decks") or {}).get("gihwr") or 0.0) for c in cards])
W = np.array([[((c["deck_colors"].get(l) or {}).get("gihwr") or (c["deck_colors"].get("All Decks") or {}).get("gihwr") or 0.0) for l in LANES] for c in cards])
cast = np.array([[castable(c, l) for l in LANES] for c in cards]) & spell[:, None]
noff = np.array([[len(off_pips(c, l)) for l in LANES] for c in cards])
offmask = np.zeros((C, L, 5), bool)
for i, c in enumerate(cards):
    if island[i]: continue
    for j, l in enumerate(LANES):
        o = off_pips(c, l)
        if len(o) == 1:
            for x in o[0]: offmask[i, j, COL.index(x)] = True
fixland = np.array([island[i] and "Basic" not in cards[i]["types"] and sum(pool_fixing([cards[i]]).values()) >= 2 for i in range(C)])
makes = np.array([[pool_fixing([c])[x] > 0 if island[i] else False for x in COL] for i, c in enumerate(cards)])  # colours a land makes
laneset = np.array([[x in l for x in COL] for l in LANES])  # L x 5
surplus = np.where(spell & (gA > 0), np.maximum(0, gA - 50), 0.0)
LS = cast * surplus[:, None]  # card x lane: a card's surplus in each lane (choice.top_lane)
nlanes = cast.sum(1); ncol = np.array([len(c["colors"]) for c in cards])
gold = spell & (ncol == 2) & (nlanes == 1); mono = spell & (ncol == 1)
band = (gA >= 55.0) & (gA < 60.0)
G = np.zeros((C, L), np.float32); Mo = np.zeros((C, 5), np.float32)
for i in range(C):
    if gold[i] and band[i]: G[i, int(np.argmax(cast[i]))] = 1
    if mono[i] and band[i]: Mo[i, COL.index(cards[i]["colors"][0])] = 1
PC = np.zeros((5, L), np.float32)  # pair p = colours X,Y
for j, l in enumerate(LANES):
    for x in l: PC[COL.index(x), j] = 1
alsa = np.array([((c["deck_colors"].get("All Decks") or {}).get("alsa") or 0.0) for c in cards])
latew = ((np.arange(14)[:, None] + 1 > alsa[None, :]) & (alsa[None, :] > 0)).astype(np.float32)   # card seen later than its ALSA at that pick (post hoc variant)
col_of = {n: i for i, n in enumerate(names)}
col_of.update({n.split(" // ")[0]: i for i, n in enumerate(names)})
print("cards", C, "fixer lands", [names[i] for i in np.flatnonzero(fixland)], "gold band", int(G.sum()), "mono band", int(Mo.sum()), flush=True)

A = np.load(f"{D}/pack.npy", mmap_mode="r"); B = np.load(f"{D}/pool.npy", mmap_mode="r"); M = pd.read_pickle(f"{D}/meta.pkl")
N = len(M); did = M.draft_id.values
starts = np.r_[0, np.flatnonzero(did[1:] != did[:-1]) + 1]; cnt = np.diff(np.r_[starts, N])
ok = (cnt == 42) & np.array([(M.pack_number.values[s:s + 42] * 14 + M.pick_number.values[s:s + 42] == np.arange(42)).all() for s in starts])
print("drafts", len(starts), "with 42 ordered picks", int(ok.sum()), "skill NaN rows", int(M.user_game_win_rate_bucket.isna().sum()), flush=True)
starts = starts[ok]; ND = len(starts)
skill = M.user_game_win_rate_bucket.values[starts].astype(float); ids = did[starts]
half = np.array([zlib.crc32(str(x).encode()) % 2 for x in ids])
takencol = np.array([col_of.get(p, -1) for p in M.pick.values])

h1 = {k: [] for k in ("gid", "draft", "chosen", "gih", "isfix", "match", "splash", "fixheld", "pack", "pick", "skill", "half")}
cov = dict(rows=0, both=0, taken_in=0, taken_fix=0); gid0 = 0
h5 = {k: [] for k in ("g", "m", "g7", "m7", "c4", "av3", "avg3", "avm3", "fr3", "av2", "avg2", "avm2", "fr2", "av1l", "fr1l", "end", "skill", "half", "idx", "gl", "ml", "c9")}
for b0 in range(0, ND, 3000):
    S = starts[b0:b0 + 3000]; n = len(S); rows = (S[:, None] + np.arange(42)).ravel()
    # drafts are not necessarily adjacent in the memmap if some were dropped: gather by fancy index (sorted, so memmap-friendly)
    Pk = np.asarray(A[rows]); Po = np.asarray(B[rows])
    pick = np.tile(np.arange(14), 3 * n); pk = np.tile(np.repeat(np.arange(3), 14), n)
    # ---------------- H1a
    sel = pick >= 5
    Pos = Po.astype(np.float32); lane = np.argmax(Pos @ LS, 1)
    splash = np.zeros(len(rows)); need = np.zeros((len(rows), 5), np.float32)
    for j in range(L):
        spj = spell & (noff[:, j] == 1) & (W[:, j] >= 56.5); r = lane == j
        splash[r] = Pos[r] @ spj.astype(np.float32)
        for x in range(5): need[r, x] = Pos[r] @ (spj & offmask[:, j, x]).astype(np.float32)
    fixheld = Pos @ fixland.astype(np.float32)
    pres = Pk > 0
    useful = fixland[None, :] & np.array([((makes & laneset[j]).any(1) & (makes & ~laneset[j]).any(1)) for j in range(L)])[lane]  # taps the lane, makes an outside colour
    best = np.where(useful & pres, gA[None, :], -1).max(1)
    play = pres & cast[:, lane].T & (gA[None, :] > 0) & (np.abs(gA[None, :] - best[:, None]) <= 2.0)
    cs = (pres & useful) | play
    tk = takencol[rows]; has = (useful & pres).any(1) & play.any(1) & sel
    inn = has & (tk >= 0) & cs[np.arange(len(rows)), np.maximum(tk, 0)]
    cov["rows"] += int(sel.sum()); cov["both"] += int(has.sum()); cov["taken_in"] += int(inn.sum()); cov["taken_fix"] += int((inn & useful[np.arange(len(rows)), np.maximum(tk, 0)]).sum())
    r_i, c_i = np.nonzero(cs & inn[:, None])
    gids = gid0 + np.unique(r_i, return_inverse=True)[1]; gid0 = gids.max() + 1 if len(gids) else gid0
    h1["gid"].append(gids); h1["draft"].append(b0 + r_i // 42); h1["chosen"].append(c_i == tk[r_i]); h1["gih"].append(gA[c_i]); h1["isfix"].append(useful[r_i, c_i])
    mt = (makes[c_i] & ~laneset[lane[r_i]] & (need[r_i] > 0)).any(1) & useful[r_i, c_i]  # fixer makes a colour a held splash card needs
    h1["match"].append(mt); h1["splash"].append(splash[r_i]); h1["fixheld"].append(fixheld[r_i]); h1["pack"].append(pk[r_i]); h1["pick"].append(pick[r_i])
    h1["skill"].append(skill[b0 + r_i // 42]); h1["half"].append(half[b0 + r_i // 42])
    # ---------------- H5
    Pr = Pk.reshape(n, 42, C).astype(np.float32); Pl = Po.reshape(n, 42, C).astype(np.float32)
    def pair_counts(rng, lo, hi):
        X = Pr[:, lo:hi, :].sum(1); g = X @ G; mc = X @ Mo; return g, mc @ PC, g + mc @ PC
    g, m, _ = pair_counts(0, 4, 9); g7, m7, _ = pair_counts(0, 4, 8)
    XL = (Pr[:, 4:9, :] * latew[4:9][None]).sum(1); gl_, ml_ = XL @ G, (XL @ Mo) @ PC
    c9 = np.stack([Pl[:, 9, :] @ LS[:, j] for j in range(L)], 1)
    c4 = np.stack([Pl[:, 4, :] @ LS[:, j] for j in range(L)], 1)
    def outcome(lo, hi, fresh):
        ga, ma, av = pair_counts(0, lo, hi); f = Pr[:, fresh, :]; return av, ga, ma, f @ G + (f @ Mo) @ PC
    av3, avg3, avm3, fr3 = outcome(29, 36, 28); av2, avg2, avm2, fr2 = outcome(15, 22, 14); av1l, _, _, fr1l = outcome(9, 14, 0)
    fin = Pl[:, 41, :].copy(); fin[np.arange(n), takencol[S + 41]] += (takencol[S + 41] >= 0)
    end = np.eye(L)[np.argmax(fin @ LS, 1)]
    for k, v in dict(g=g, m=m, g7=g7, m7=m7, c4=c4, av3=av3, avg3=avg3, avm3=avm3, fr3=fr3, av2=av2, avg2=avg2, avm2=avm2, fr2=fr2, av1l=av1l, fr1l=fr1l, end=end, skill=skill[b0:b0 + n], half=half[b0:b0 + n], idx=np.arange(b0, b0 + n), gl=gl_, ml=ml_, c9=c9).items(): h5[k].append(v)
    print(b0, ND, flush=True)
np.savez_compressed(f"{D}/h1.npz", **{k: np.concatenate(v) for k, v in h1.items()}, cov=np.array(list(cov.values())))
np.savez_compressed(f"{D}/h5.npz", **{k: np.concatenate(v) for k, v in h5.items()}, ids=ids)
print("coverage", cov, flush=True)
