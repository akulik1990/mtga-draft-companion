"""B5 (docs/test-plan-2026-11-07.md): splash castability within deck, FRA Premier game data. Design: docs/b5-splash-cast-FRA.md (written before outcomes).
  python3 analysis/b5_splash_cast.py cache   # stream the game CSV in 20k-row chunks into an int8 npz cache in SCR
  python3 analysis/b5_splash_cast.py run     # analysis -> analysis/out/b5_splash_cast.txt
Run as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ..."""
import hashlib, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad/b5"
NPZ = f"{SCR}/b5_games.npz"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
by = {}
for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
    n = v["name"].split(" // ")[0]
    if n in by and not v.get("isprimarycard", 1): continue
    by[n] = v
island = {n for n, v in by.items() if "Land" in v["types"]}

if sys.argv[1] == "cache":
    os.makedirs(SCR, exist_ok=True)
    hdr = pd.read_csv(GD, nrows=0).columns.tolist()
    spells = sorted(n for n in by if n not in island and f"opening_hand_{n}" in hdr)
    landn = sorted(n for n in set(island) | {"Plains", "Island", "Swamp", "Mountain", "Forest"} if f"opening_hand_{n}" in hdr)
    meta = ["draft_id", "build_index", "won", "on_play", "num_mulligans", "main_colors", "splash_colors", "user_game_win_rate_bucket"]
    use = meta + [f"{p}{n}" for n in spells + landn for p in ("opening_hand_", "drawn_")] + [f"deck_{n}" for n in spells]
    dt = {c: np.float32 for c in use if c not in meta}
    parts = []
    for i, ch in enumerate(pd.read_csv(GD, usecols=use, chunksize=20000, dtype=dt)):
        f = lambda p, ns: ch[[f"{p}{n}" for n in ns]].fillna(0).values
        parts.append((ch[meta].copy(), (f("opening_hand_", spells) + f("drawn_", spells)).astype(np.int8),
                      (f("opening_hand_", landn) + f("drawn_", landn)).astype(np.int8), f("deck_", spells).astype(np.int8))); print("chunk", i, flush=True)
    M = pd.concat([p[0] for p in parts], ignore_index=True)
    np.savez_compressed(NPZ, H=np.vstack([p[1] for p in parts]), HL=np.vstack([p[2] for p in parts]), Dk=np.vstack([p[3] for p in parts]),
                        spells=np.array(spells), landn=np.array(landn), draft=M.draft_id.values.astype(str), build=M.build_index.values.astype(np.int16),
                        won=M.won.values.astype(np.int8), play=M.on_play.values.astype(np.int8), mull=M.num_mulligans.fillna(0).values.astype(np.int8),
                        main=M.main_colors.fillna("").values.astype(str), splash=M.splash_colors.fillna("").values.astype(str),
                        skill=M.user_game_win_rate_bucket.values.astype(np.float32))
    print("cached"); sys.exit()

# ---------------------------------------------------------------- run
rng = np.random.default_rng(20261108)
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
Z = np.load(NPZ, allow_pickle=True)
spells, landn = list(Z["spells"]), list(Z["landn"])
splash_all = np.asarray(Z["splash"], dtype=str)
sel = np.where(np.char.str_len(splash_all) == 1)[0]
P(f"games total {len(splash_all)}; with a splash flag {int((np.char.str_len(splash_all) > 0).sum())}; single-colour splash {len(sel)}; multi-colour splash {int((np.char.str_len(splash_all) > 1).sum())}")
draft = np.asarray(Z["draft"], dtype=str)[sel]; build = Z["build"][sel]; won = Z["won"][sel].astype(float)
H = Z["H"][sel]; HL = Z["HL"][sel]; Dk = Z["Dk"][sel]; main = np.asarray(Z["main"], dtype=str)[sel]; spl = splash_all[sel]
del Z
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))
o = np.argsort(codes, kind="stable")
draft, build, won, H, HL, Dk, main, spl, codes = draft[o], build[o], won[o], H[o], HL[o], Dk[o], main[o], spl[o], codes[o]
codes, _ = pd.factorize(codes)
dcode, dnames = pd.factorize(draft); ND = dcode.max() + 1; ND_ = len(dnames)
half = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 2 for d in dnames])[dcode]
G = len(won); NDK = codes.max() + 1
starts = np.r_[0, np.where(np.diff(codes) != 0)[0] + 1, G]
P(f"selected games {G}, decks {NDK}, drafts {ND}; skill not used (within deck)")

# ---- card attributes
COL = "WUBRG"
cmc = np.array([by[n]["cmc"] or 0 for n in spells])
pips = np.zeros((len(spells), 5), int); hyb = np.zeros((len(spells), 5), bool); cyc = np.zeros(len(spells), bool); gih = np.zeros(len(spells)); heart = np.zeros(len(spells), bool)
for j, n in enumerate(spells):
    v = by[n]; front = (v.get("mana_cost") or "").split("//")[0]
    for k, c in enumerate(COL): pips[j, k] = len(re.findall(r"\{%s\}" % c, front))
    for a, b in re.findall(r"\{([WUBRG])/([WUBRG])\}", front): hyb[j, COL.index(a)] = hyb[j, COL.index(b)] = True
    t = (v.get("oracle_text") or ""); cyc[j] = bool(re.search(r"landcycling", t, re.I)); heart[j] = ("Heartwood token" in t and "reate" in t)
    gih[j] = (v.get("deck_colors", {}).get("All Decks", {}) or {}).get("gihwr", 0) or 0
BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
lmake = np.zeros((len(landn), 5), bool)
for i, n in enumerate(landn):
    base = n.replace("Snow-Covered ", "")
    cs = set("WUBRG") if n == "Room of Refuge" else {BASIC[base]} if base in BASIC else {c for c in (by[n].get("colors") or []) if c in COL}
    for c in cs: lmake[i, COL.index(c)] = True
mv = spells.index("Murmuring Volume") if "Murmuring Volume" in spells else None
P(f"landcyclers {[spells[j] for j in np.where(cyc)[0]]}; heartwood makers {[spells[j] for j in np.where(heart)[0]]}; spells without a GIH rating {int((gih == 0).sum())}")

# ---- per deck: splash cards, placebo candidates, speed
sidx = np.array([COL.index(c) for c in spl[starts[:-1]]]); dkrows = starts[:-1]
info = []
for d in range(NDK):
    a, b = starts[d], starts[d + 1]; s = COL.index(spl[a]); mc = {COL.index(c) for c in main[a]}; dk = Dk[a]
    cols = np.where(dk > 0)[0]
    isspl = [j for j in cols if pips[j, s] >= 1 and not cyc[j]]
    ok = [j for j in cols if j not in isspl and not cyc[j] and not hyb[j, s] and pips[j, s] == 0 and all((pips[j, k] == 0) or (k in mc) for k in range(5))]
    base = [j for j in cols if j not in isspl]
    sp = float(np.average(cmc[base], weights=dk[base])) if base else np.nan
    info.append((a, b, s, isspl, ok, sp))
speed = np.array([x[5] for x in info]); good = ~np.isnan(speed)
cut = np.nanpercentile(speed, [100 / 3, 200 / 3]); P(f"speed (mean cmc of the base non-land spells) tercile cuts over splash decks: {cut[0]:.3f}, {cut[1]:.3f}; decks with a splash card {sum(1 for x in info if x[3])} of {NDK}")
terc_deck = np.where(speed <= cut[0], 0, np.where(speed <= cut[1], 1, 2)); terc = np.repeat(terc_deck, np.diff(starts))

# ---- per-game variables
def dgame(arr): return np.repeat(arr, np.diff(starts))
sg = dgame(sidx)
nSP = np.zeros(G); nDP = np.zeros(G); rSP = np.zeros(G); rDP = np.zeros(G)
for d, (a, b, s, isspl, ok, sp) in enumerate(info):
    for j in isspl:
        (nDP if pips[j, s] >= 2 else nSP)[a:b] += H[a:b, j]; (rDP if pips[j, s] >= 2 else rSP)[a:b] += H[a:b, j] * gih[j]
ncard = H.sum(1) + HL.sum(1); Lh = HL.sum(1).astype(float)
srcall = (HL.astype(np.int16) @ lmake.astype(np.int16))  # G x 5
src = srcall[np.arange(G), sg]
cycH = (H[:, cyc].sum(1)); heartH = H[:, heart].sum(1) * np.isin(sg, [COL.index("R"), COL.index("G")]); mvH = H[:, mv] if mv is not None else 0
src2 = src + cycH + heartH + mvH
st = np.minimum(src, 2); st2 = np.minimum(src2, 2)
P(f"games with a single-pip splash card in hand {np.mean(nSP>0):.3f}, double-pip {np.mean(nDP>0):.3f}; source strata 0/1/2+ of games: {np.mean(st==0):.3f}/{np.mean(st==1):.3f}/{np.mean(st==2):.3f}")
P(f"decks with a single-pip splash card {sum(1 for x in info if any(pips[j, x[2]] == 1 for j in x[3]))}, with a double-pip {sum(1 for x in info if any(pips[j, x[2]] >= 2 for j in x[3]))}")

# ---- placebo draws
def placebo():
    pS = np.zeros(G); pD = np.zeros(G); qS = np.zeros(G); qD = np.zeros(G); miss = 0; tot = 0
    for d, (a, b, s, isspl, ok, sp) in enumerate(info):
        used = set(); dk = Dk[a]
        for j in rng.permutation(isspl) if isspl else []:
            tot += 1; bc = min(cmc[j], 6)
            c = [k for k in ok if k not in used and min(cmc[k], 6) == bc]
            c2 = [k for k in c if dk[k] == dk[j]]
            if c2: c = c2
            if not c:
                c = [k for k in ok if k not in used]
                if c: bm = min(abs(cmc[k] - cmc[j]) for k in c); c = [k for k in c if abs(cmc[k] - cmc[j]) == bm]
            if not c: miss += 1; continue
            k = int(c[rng.integers(len(c))]); used.add(k)
            if pips[j, s] >= 2: pD[a:b] += H[a:b, k]; qD[a:b] += H[a:b, k] * gih[k]
            else: pS[a:b] += H[a:b, k]; qS[a:b] += H[a:b, k] * gih[k]
    return pS, pD, qS, qD, miss / max(tot, 1)

# ---- estimation
def demean(X, g):
    cnt = np.bincount(g).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=len(cnt)) / np.maximum(cnt, 1))[g]
    return o
B, NPL = 300, 40
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=float)
def fit(treat, ctrl, mask, boot=False):
    """treat: list of (G,) arrays; ctrl: list of (G,) arrays. Returns coef (k treat, pts) and bootstrap (B x k) or None."""
    m = np.where(mask)[0]; gg = pd.factorize(codes[m])[0]; ok = np.bincount(gg)[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    X = np.column_stack([t[m] for t in treat] + [c[m] for c in ctrl]); k = len(treat)
    A = demean(np.column_stack([X, won[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    keep = Xd.std(0) > 1e-9; keep[k:] = True
    kt = keep[:k]; Xk = Xd[:, keep]
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

def one(d, i, n):  # indicator-weighted counts: d (G,) times stratum==i
    return d * (n == i)
def design(nS, nD, stratum):
    tr = [one(nS, i, stratum) for i in range(3)] + [one(nD, i, stratum) for i in range(3)]
    return tr, [(stratum == 1).astype(float), (stratum == 2).astype(float), ncard.astype(float), Lh]
NAMES = ["single, 0 sources", "single, 1 source", "single, 2+ sources", "double, 0 sources", "double, 1 source", "double, 2+ sources"]

# placebo draws once, stored (G x NPL arrays would be 40*4*G floats = small enough for G ~ 100k: 130 MB at float32)
P(f"\ndrawing {NPL} placebo sets ...")
PL = []; missr = []
for _ in range(NPL):
    pS, pD, qS, qD, mr = placebo(); PL.append((pS.astype(np.float32), pD.astype(np.float32), qS.astype(np.float32), qD.astype(np.float32))); missr.append(mr)
P(f"placebo cards unmatched (share of splash cards): {np.mean(missr):.4f}")
qSm = np.mean([p[2] for p in PL[:10]], 0); qDm = np.mean([p[3] for p in PL[:10]], 0); pSm = np.mean([p[0] for p in PL[:10]], 0); pDm = np.mean([p[1] for p in PL[:10]], 0)

CELLS = {"all": np.ones(G, bool), "half 0": half == 0, "half 1": half == 1, "slow (high cmc)": terc == 2, "middle": terc == 1, "fast (low cmc)": terc == 0,
         "slow, half 0": (terc == 2) & (half == 0), "slow, half 1": (terc == 2) & (half == 1), "fast, half 0": (terc == 0) & (half == 0), "fast, half 1": (terc == 0) & (half == 1)}
RES = {}
for cn, mk in CELLS.items():
    mk = mk & good[np.repeat(np.arange(NDK), np.diff(starts))] if False else mk
    tr, ct = design(nSP, nDP, st); est, bs = fit(tr, ct, mk, boot=True)
    pl = np.array([fit(*design(p[0].astype(float), p[1].astype(float), st), mk)[0] for p in PL] if False else [fit(design(p[0].astype(float), p[1].astype(float), st)[0], ct, mk)[0] for p in PL])
    RES[cn] = (est, bs, pl); print(cn, np.round(est, 2), np.round(np.nanmean(pl, 0), 2), flush=True)
# pooled single/double (M1) and sensitivity (secondary source count)
M1 = {}
for cn in ["all", "half 0", "half 1", "slow (high cmc)", "middle", "fast (low cmc)"]:
    mk = CELLS[cn]; ct = [ncard.astype(float), Lh]
    est, bs = fit([nSP, nDP], ct, mk, boot=True)
    pl = np.array([fit([p[0].astype(float), p[1].astype(float)], ct, mk)[0] for p in PL]); M1[cn] = (est, bs, pl)
tr, ct = design(nSP, nDP, st2); SENS = (*fit(tr, ct, CELLS["all"], boot=True), np.array([fit(design(p[0].astype(float), p[1].astype(float), st2)[0], ct, CELLS["all"])[0] for p in PL]))

def line(label, est, bs, pl):
    se = np.nanstd(bs, ddof=1); pm = np.nanmean(pl); ps = np.nanstd(pl, ddof=1); adj = est - pm; ase = np.hypot(se, ps); h = 1.96 * ase
    return f"| {label} | {est:+.2f} ({se:.2f}) | {pm:+.2f} ({ps:.2f}) | {adj:+.2f} ({ase:.2f}) | {adj/ase:+.1f} | [{adj-h:+.2f}, {adj+h:+.2f}] |"
HEAD = "| term | pts per card in hand (boot SE) | placebo mean (SD) | adjusted (SE) | z | adjusted 95% interval (null rules out beyond) |\n|---|---|---|---|---|---|"
P("\n# Results (win-rate points per splashed card in hand; adjusted = estimate minus placebo mean; 300 draft bootstraps, 40 placebo sets)")
P("\n## M1: pooled by pip type (no source strata)\n")
for cn, (e, bs, pl) in M1.items():
    P(f"\n### {cn}\n\n{HEAD}"); P(line("single-pip", e[0], bs[:, 0], pl[:, 0])); P(line("double-pip", e[1], bs[:, 1], pl[:, 1]))
P("\n## M2: by sources of the splash colour in hand (lands only)")
for cn, (e, bs, pl) in RES.items():
    P(f"\n### {cn}\n\n{HEAD}")
    for i, nme in enumerate(NAMES): P(line(nme, e[i], bs[:, i], pl[:, i]))
P("\n### sensitivity: sources = lands + landcyclers + Murmuring Volume + Heartwood makers (R/G), all\n\n" + HEAD)
for i, nme in enumerate(NAMES): P(line(nme, SENS[0][i], SENS[1][:, i], SENS[2][:, i]))

def diff_line(label, A, Bb, ia, ib, sign=1):
    ea, ba, pa = A; eb, bb, pb = Bb
    d = (ea[ia] - np.nanmean(pa[:, ia])) - (eb[ib] - np.nanmean(pb[:, ib])); se = np.nanstd(ba[:, ia] - bb[:, ib], ddof=1)
    ps = np.nanstd(pa[:, ia] - pb[:, ib], ddof=1); ase = np.hypot(se, ps); h = 1.96 * ase
    return f"| {label} | {d:+.2f} | {ase:.2f} | {d/ase:+.1f} | [{d-h:+.2f}, {d+h:+.2f}] |"
P("\n# Contrasts (adjusted points, joint bootstrap + placebo noise)\n\n| contrast | adjusted diff | SE | z | 95% interval |\n|---|---|---|---|---|")
a = RES["all"]
for t, o in (("single", 0), ("double", 3)):
    P(diff_line(f"{t}: 2+ sources minus 0 sources, all", a, a, o + 2, o))
    P(diff_line(f"{t}: 1 source minus 0 sources, all", a, a, o + 1, o))
    for cn in ("half 0", "half 1"): P(diff_line(f"{t}: 2+ minus 0 sources, {cn}", RES[cn], RES[cn], o + 2, o))
    for lvl in (0, 1, 2): P(diff_line(f"{t}: slow minus fast tercile, {NAMES[o+lvl].split(', ')[1]}", RES["slow (high cmc)"], RES["fast (low cmc)"], o + lvl, o + lvl))
    P(diff_line(f"{t}: (2+ minus 0 sources) slow minus fast: interaction", RES["slow (high cmc)"], RES["fast (low cmc)"], o + 2, o + 2))  # placeholder replaced below

# interaction slow-fast of (2+ minus 0)
def inter(A, Bb, o):
    ea, ba, pa = A; eb, bb, pb = Bb
    f = lambda e, p: (e[o + 2] - e[o]) - np.nanmean(p[:, o + 2] - p[:, o])
    d = f(ea, pa) - f(eb, pb); se = np.nanstd((ba[:, o + 2] - ba[:, o]) - (bb[:, o + 2] - bb[:, o]), ddof=1)
    ps = np.nanstd((pa[:, o + 2] - pa[:, o]) - (pb[:, o + 2] - pb[:, o]), ddof=1); ase = np.hypot(se, ps)
    return d, ase
lines[:] = [l for l in lines if "interaction" not in l]
for t, o in (("single", 0), ("double", 3)):
    d, ase = inter(RES["slow (high cmc)"], RES["fast (low cmc)"], o); h = 1.96 * ase
    P(f"| {t}: (2+ minus 0 sources) slow minus fast: interaction | {d:+.2f} | {ase:.2f} | {d/ase:+.1f} | [{d-h:+.2f}, {d+h:+.2f}] |")

P("\n# Replication of the speed contrast by draft-hash half (single-pip, 0 sources, and the 2+ minus 0 interaction)\n\n| contrast | adjusted diff | SE | z | 95% interval |\n|---|---|---|---|---|")
for h in (0, 1):
    P(diff_line(f"single, 0 sources: slow minus fast, half {h}", RES[f"slow, half {h}"], RES[f"fast, half {h}"], 0, 0))
    d, ase = inter(RES[f"slow, half {h}"], RES[f"fast, half {h}"], 0); P(f"| single interaction, half {h} | {d:+.2f} | {ase:.2f} | {d/ase:+.1f} | [{d-1.96*ase:+.2f}, {d+1.96*ase:+.2f}] |")

# rating benchmark: per-card rating gap (splash cards in hand minus matched placebo in hand), by stratum, all games
P("\n# Rating benchmark: GIH WR (All Decks) of splash cards in hand minus matched placebo cards in hand, per card (points; indicative)\n\n| stratum | single-pip gap | double-pip gap | cards in hand per game, single / double |\n|---|---|---|---|")
for i, nme in enumerate(["0 sources", "1 source", "2+ sources"]):
    m = st == i; r = []
    for nn, rr, qq in ((nSP, rSP, 2), (nDP, rDP, 3)):
        r.append((rr[m].sum() - np.mean([p[qq][m].sum() for p in PL[:10]])) / max(nn[m].sum(), 1))
    P(f"| {nme} | {r[0]:+.2f} | {r[1]:+.2f} | {nSP[m].mean():.3f} / {nDP[m].mean():.3f} |")
P("\nPlacebo draw count check: mean single / double placebo cards in hand per game "
  f"{pSm.mean():.3f} / {pDm.mean():.3f} vs splash {nSP.mean():.3f} / {nDP.mean():.3f}")
open(f"{OUT}/b5_splash_cast.txt", "w").write("\n".join(lines)); P("done")
