"""Video splash test H4 (mqZ-31): do cheap splash cards lose more value than expensive ones? Within deck, B5 design (docs/b5-splash-cast-FRA.md), FRA Premier game data.
Output analysis/out/video_splash_within.txt. Reuses B5's cache (b5_games.npz, built by analysis/b5_splash_cast.py cache; copied to data/Temp/vid1106/). Design fixed before outcomes:
  Population, splashed-card rule, treatment, controls (cards in hand, lands in hand, deck fixed effects), sources strata, placebo (same-deck on-colour card, same cmc, 40 draws),
  300 draft bootstraps, halves by md5(draft) parity: all as B5. Only change: single-pip splash cards are split by mana value into 1-2 / 3-4 / 5+ (placebo cards follow their splash card's group).
  PRIMARY: single-pip net value per card in hand, pooled over sources (B5 M1 form), adjusted = estimate minus placebo mean, 1-2 minus 5+.
  Secondary (fixed now): same contrast within source strata 0 / 1 / 2+ (B5 M2 form); 1-2 minus 3-4; 3-4 minus 5+; double-pip stays one pooled term.
  Skill groups by the player's skill bucket (user_game_win_rate_bucket of the game; a deck has one player): good >=0.60, avg 0.52-0.58, weak <0.52; good minus weak on independent drafts.
Run: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_splash_within.py"""
import hashlib, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
NPZ = f"{R}/data/Temp/vid1106/b5_games.npz"
by = {}
for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
    n = v["name"].split(" // ")[0]
    if n in by and not v.get("isprimarycard", 1): continue
    by[n] = v
rng = np.random.default_rng(20261109)
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
Z = np.load(NPZ, allow_pickle=True)
spells, landn = list(Z["spells"]), list(Z["landn"])
splash_all = np.asarray(Z["splash"], dtype=str)
sel = np.where(np.char.str_len(splash_all) == 1)[0]
draft = np.asarray(Z["draft"], dtype=str)[sel]; build = Z["build"][sel]; won = Z["won"][sel].astype(float); skill = Z["skill"][sel].astype(float)
H = Z["H"][sel]; HL = Z["HL"][sel]; Dk = Z["Dk"][sel]; main = np.asarray(Z["main"], dtype=str)[sel]; spl = splash_all[sel]
del Z
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))
o = np.argsort(codes, kind="stable")
draft, build, won, skill, H, HL, Dk, main, spl, codes = draft[o], build[o], won[o], skill[o], H[o], HL[o], Dk[o], main[o], spl[o], codes[o]
codes, _ = pd.factorize(codes)
dcode, dnames = pd.factorize(draft); ND = dcode.max() + 1
half = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 2 for d in dnames])[dcode]
G = len(won); NDK = codes.max() + 1
starts = np.r_[0, np.where(np.diff(codes) != 0)[0] + 1, G]
grp = np.full(G, -1); grp[skill >= .6 - 1e-9] = 0; grp[(skill >= .52 - 1e-9) & (skill <= .58 + 1e-9)] = 1; grp[skill < .52 - 1e-9] = 2
P(f"single-colour splash games {G}, decks {NDK}, drafts {ND}; skill groups good/avg/weak games {[(grp == i).sum() for i in range(3)]}, unassigned {(grp < 0).sum()}")
COL = "WUBRG"
cmc = np.array([by[n]["cmc"] or 0 for n in spells])
pips = np.zeros((len(spells), 5), int); hyb = np.zeros((len(spells), 5), bool); cyc = np.zeros(len(spells), bool); gih = np.zeros(len(spells))
for j, n in enumerate(spells):
    v = by[n]; front = (v.get("mana_cost") or "").split("//")[0]
    for k, c in enumerate(COL): pips[j, k] = len(re.findall(r"\{%s\}" % c, front))
    for a, b in re.findall(r"\{([WUBRG])/([WUBRG])\}", front): hyb[j, COL.index(a)] = hyb[j, COL.index(b)] = True
    cyc[j] = bool(re.search(r"landcycling", v.get("oracle_text") or "", re.I)); gih[j] = (v.get("deck_colors", {}).get("All Decks", {}) or {}).get("gihwr", 0) or 0
BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
lmake = np.zeros((len(landn), 5), bool)
for i, n in enumerate(landn):
    base = n.replace("Snow-Covered ", "")
    cs = set(COL) if n == "Room of Refuge" else {BASIC[base]} if base in BASIC else {c for c in (by[n].get("colors") or []) if c in COL}
    for c in cs: lmake[i, COL.index(c)] = True
grpc = np.where(cmc <= 2, 0, np.where(cmc <= 4, 1, 2))   # cmc group of every spell
info = []
for d in range(NDK):
    a, b = starts[d], starts[d + 1]; s = COL.index(spl[a]); mc = {COL.index(c) for c in main[a]}; dk = Dk[a]; cols = np.where(dk > 0)[0]
    isspl = [j for j in cols if pips[j, s] >= 1 and not cyc[j]]
    ok = [j for j in cols if j not in isspl and not cyc[j] and not hyb[j, s] and pips[j, s] == 0 and all((pips[j, k] == 0) or (k in mc) for k in range(5))]
    info.append((a, b, s, isspl, ok))
sg = np.repeat(np.array([x[2] for x in info]), np.diff(starts))
ncard = H.sum(1) + HL.sum(1); Lh = HL.sum(1).astype(float)
src = (HL.astype(np.int16) @ lmake.astype(np.int16))[np.arange(G), sg]; st = np.minimum(src, 2)
def counts(draw_fn):
    """columns: single g0, g1, g2, double (counts in hand) and the same weighted by GIH."""
    n = np.zeros((G, 4)); r = np.zeros((G, 4))
    for item in draw_fn():
        a, b, j, k, dbl = item   # splash card j (group via cmc), placed card k
        c = 3 if dbl else grpc[j]
        n[a:b, c] += H[a:b, k]; r[a:b, c] += H[a:b, k] * gih[k]
    return n, r
def real():
    for (a, b, s, isspl, ok) in info:
        for j in isspl: yield a, b, j, j, pips[j, s] >= 2
def placebo():
    for (a, b, s, isspl, ok) in info:
        used = set(); dk = Dk[a]
        for j in rng.permutation(isspl) if isspl else []:
            bc = min(cmc[j], 6); c = [k for k in ok if k not in used and min(cmc[k], 6) == bc]; c2 = [k for k in c if dk[k] == dk[j]]
            if c2: c = c2
            if not c:
                c = [k for k in ok if k not in used]
                if c: bm = min(abs(cmc[k] - cmc[j]) for k in c); c = [k for k in c if abs(cmc[k] - cmc[j]) == bm]
            if not c: continue
            k = int(c[rng.integers(len(c))]); used.add(k); yield a, b, j, k, pips[j, s] >= 2
NS, RS = counts(real)
NPL = 40; PL = []; PR = []
for i in range(NPL):
    n, r = counts(placebo); PL.append(n.astype(np.float32)); PR.append(r)
    if i % 10 == 0: print("placebo", i, flush=True)
GN = ["1-2", "3-4", "5+", "double"]
P(f"games with a splash card of group in hand: " + ", ".join(f"{GN[i]} {np.mean(NS[:, i] > 0):.3f} (cards per game {NS[:, i].mean():.3f}, placebo {np.mean([p[:, i].mean() for p in PL]):.3f})" for i in range(4)))
P(f"splash cards (distinct, over decks): " + ", ".join(f"{GN[i]} {sum(1 for x in info for j in x[3] if (3 if pips[j, x[2]] >= 2 else grpc[j]) == i)}" for i in range(4)))
P(f"source strata 0/1/2+ of games: {np.mean(st == 0):.3f}/{np.mean(st == 1):.3f}/{np.mean(st == 2):.3f}")

def demean(X, g):
    cnt = np.bincount(g).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=len(cnt)) / np.maximum(cnt, 1))[g]
    return o
B = 300
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=float)
def fit(treat, ctrl, mask, boot=False):
    m = np.where(mask)[0]; gg = pd.factorize(codes[m])[0]; ok = np.bincount(gg)[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    X = np.column_stack([t[m] for t in treat] + [c[m] for c in ctrl]); k = len(treat)
    A = demean(np.column_stack([X, won[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    keep = Xd.std(0) > 1e-9; keep[k:] = True; kt = keep[:k]; Xk = Xd[:, keep]
    cf = np.full(k, np.nan); cf[kt] = 100 * np.linalg.lstsq(Xk, yd, rcond=None)[0][:kt.sum()]
    if not boot: return cf, None
    cl = dcode[m]; od = np.argsort(cl, kind="stable"); cl = cl[od]; Xs, ys = Xk[od], yd[od]
    u, stt = np.unique(cl, return_index=True)
    XtX = np.add.reduceat(Xs[:, :, None] * Xs[:, None, :], stt, axis=0); Xty = np.add.reduceat(Xs * ys[:, None], stt, axis=0)
    A_ = np.einsum("bd,dij->bij", W[:, u], XtX); b_ = W[:, u] @ Xty
    bs = np.full((B, k), np.nan)
    for r in range(B):
        try: bs[r, kt] = 100 * np.linalg.solve(A_[r], b_[r])[:kt.sum()]
        except np.linalg.LinAlgError: pass
    return cf, bs
ctrl1 = [ncard.astype(float), Lh]
ctrl2 = [(st == 1).astype(float), (st == 2).astype(float), ncard.astype(float), Lh]
def d1(N): return [N[:, i].astype(float) for i in range(4)]
def d2(N): return [N[:, i].astype(float) * (st == s) for i in range(4) for s in range(3)]   # index 3*i + s
CELLS = {"all": np.ones(G, bool), "half 0": half == 0, "half 1": half == 1, "good": grp == 0, "average": grp == 1, "weak": grp == 2}
RES = {}
for cn, mk in CELLS.items():
    e1, b1 = fit(d1(NS), ctrl1, mk, True); p1 = np.array([fit(d1(p), ctrl1, mk)[0] for p in PL])
    e2, b2 = fit(d2(NS), ctrl2, mk, True); p2 = np.array([fit(d2(p), ctrl2, mk)[0] for p in PL])
    RES[cn] = dict(M1=(e1, b1, p1), M2=(e2, b2, p2)); print(cn, np.round(e1, 2), np.round(np.nanmean(p1, 0), 2), flush=True)
def vec(n, pos): v = np.zeros(n); [v.__setitem__(i, c) for i, c in pos.items()]; return v
def lin(cn, M, v):
    e, bs, pl = RES[cn][M]; return float(v @ e - v @ np.nanmean(pl, 0)), bs @ v, pl @ v
def show(a, b=None, sign=1):
    """adjusted combination: a = (cell, model, vec) ; b optional second (subtracted). Returns text."""
    va, ba, pa = lin(*a)
    if b: vb, bb, pb = lin(*b); v, bs, pl = va - vb, ba - bb, pa - pb
    else: v, bs, pl = va, ba, pa
    se = np.nanstd(bs, ddof=1); ps = np.nanstd(pl, ddof=1); ase = np.hypot(se, ps)
    return f"{v:+.2f} ({ase:.2f}) [{v-1.96*ase:+.2f}, {v+1.96*ase:+.2f}] z {v/ase:+.2f}", v, ase
CN = list(CELLS)
P("\n# H4 results: adjusted (estimate minus placebo mean), WR points per splash card in hand; (SE incl. placebo noise) [95% CI] z\n")
P("## M1 (pooled over sources), single-pip by mana value\n")
P("| cell | 1-2 | 3-4 | 5+ | double-pip | **1-2 minus 5+ (PRIMARY)** | 1-2 minus 3-4 | 3-4 minus 5+ |\n|---|---|---|---|---|---|---|---|")
for cn in CN:
    r = [show((cn, "M1", vec(4, {i: 1})))[0] for i in range(4)]
    r += [show((cn, "M1", vec(4, {0: 1, 2: -1})))[0], show((cn, "M1", vec(4, {0: 1, 1: -1})))[0], show((cn, "M1", vec(4, {1: 1, 2: -1})))[0]]
    P(f"| {cn} | " + " | ".join(r) + " |")
def gw(M, v, label):
    t, _, _ = show(("good", M, v), ("weak", M, v)); P(f"good minus weak, {label}: {t}")
gw("M1", vec(4, {0: 1, 2: -1}), "M1 1-2 minus 5+ (difference of contrasts)")
for i in range(3): gw("M1", vec(4, {i: 1}), f"M1 {GN[i]} level")
P("\n## M2 (by sources of the splash colour in hand: lands only), single-pip by mana value\n")
P("| cell | group | 0 sources | 1 source | 2+ sources |\n|---|---|---|---|---|")
for cn in CN:
    for i in range(3): P(f"| {cn} | {GN[i]} | " + " | ".join(show((cn, "M2", vec(12, {3 * i + s: 1})))[0] for s in range(3)) + " |")
P("\n| cell | contrast (1-2 minus 5+) | 0 sources | 1 source | 2+ sources | 1+ sources pooled (mean of strata 1, 2+; post hoc) |\n|---|---|---|---|---|---|")
for cn in CN:
    P(f"| {cn} | | " + " | ".join(show((cn, "M2", vec(12, {s: 1, 6 + s: -1})))[0] for s in range(3)) + " | " + show((cn, "M2", vec(12, {1: .5, 2: .5, 7: -.5, 8: -.5})))[0] + " |")
for s in range(3): gw("M2", vec(12, {s: 1, 6 + s: -1}), f"M2 1-2 minus 5+ at {s if s < 2 else '2+'} sources")
# smallest-effect summaries and rating benchmark
P("\n## Rating benchmark (indicative): splash cards' All-Decks GIH WR minus matched placebo cards', exposure-weighted, per card in hand (WR points)\n\n| group | gap | cards in hand |\n|---|---|---|")
for i in range(4):
    r = (RS[:, i].sum() - np.mean([p[:, i].sum() for p in PR])) / max(NS[:, i].sum(), 1); P(f"| {GN[i]} | {r:+.2f} | {int(NS[:, i].sum())} |")
P("\nThe raw level terms above are adjusted by the same-cmc placebo, so 'value' is relative to an on-colour card of the same cost. A cost-dependent shortfall shows as 1-2 minus 5+ < 0.")
open(f"{OUT}/video_splash_within.txt", "w").write("\n".join(lines) + "\n")
