"""B1 (docs/test-plan-2026-11-07.md): final within-deck estimate of removal by class and counterspells, FRA Premier game data.
Design is in docs/b1-within-removal-FRA.md (written before outcomes). Two stages, memory-capped:
  python3 analysis/b1_within_removal.py cache   # stream the game CSV in 20k-row chunks into an npz cache (int8) in SCR
  python3 analysis/b1_within_removal.py run     # analysis -> analysis/out/b1_within_removal.txt
Run both as: systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ..."""
import hashlib, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
def cards():  # as video_counters.cards (that module runs its analysis on import, so it is not imported)
    by = {}
    for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
        n = v["name"].split(" // ")[0]
        if n in by and not v.get("isprimarycard", 1): continue
        by[n] = v
    return by
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NPZ = f"{SCR}/b1_games.npz"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
by = cards()
island = {n for n, v in by.items() if "Land" in v["types"]}

if sys.argv[1] == "cache":
    hdr = pd.read_csv(GD, nrows=0).columns.tolist()
    spells = sorted(n for n in by if n not in island and f"opening_hand_{n}" in hdr)
    landn = [n for n in island if f"opening_hand_{n}" in hdr]
    meta = ["draft_id", "build_index", "won", "num_turns", "on_play", "opp_colors", "num_mulligans", "user_game_win_rate_bucket"]
    use = meta + [f"{p}{n}" for n in spells + landn for p in ("opening_hand_", "drawn_")] + [f"deck_{n}" for n in spells + landn] + [f"tutored_{n}" for n in spells if f"tutored_{n}" in hdr]
    dt = {c: np.float32 for c in use if c not in meta}
    parts = []
    for i, ch in enumerate(pd.read_csv(GD, usecols=use, chunksize=20000, dtype=dt)):
        f = lambda p, ns: ch[[f"{p}{n}" for n in ns]].fillna(0).values
        H = (f("opening_hand_", spells) + f("drawn_", spells)).astype(np.int8)
        Lh = (f("opening_hand_", landn) + f("drawn_", landn)).sum(1).astype(np.int8)
        Tu = f("tutored_", spells).astype(np.int8)
        Dk = f("deck_", spells).astype(np.int8)
        parts.append((ch[meta].copy(), H, Lh, Tu, Dk)); print("chunk", i, flush=True)
    M = pd.concat([p[0] for p in parts], ignore_index=True)
    np.savez_compressed(NPZ, H=np.vstack([p[1] for p in parts]), Lh=np.concatenate([p[2] for p in parts]), Tu=np.vstack([p[3] for p in parts]),
                        Dk=np.vstack([p[4] for p in parts]), spells=np.array(spells), draft=M.draft_id.values.astype(str), build=M.build_index.values.astype(np.int16),
                        won=M.won.values.astype(np.int8), turns=M.num_turns.fillna(0).values.astype(np.int16), play=M.on_play.values.astype(np.int8),
                        opp=M.opp_colors.fillna("").values.astype(str), mull=M.num_mulligans.fillna(0).values.astype(np.int8),
                        skill=M.user_game_win_rate_bucket.values.astype(np.float32))
    print("cached"); sys.exit()

# ---------------------------------------------------------------- run
rng = np.random.default_rng(20261107)
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
Z = np.load(NPZ, allow_pickle=True)
H, Lh, Tu, Dk, spells = Z["H"], Z["Lh"], Z["Tu"], Z["Dk"], list(Z["spells"])
draft = np.asarray(Z["draft"], dtype=str); build = Z["build"]; won = Z["won"].astype(float); turns = Z["turns"]; play = Z["play"]; mull = Z["mull"]; skill = Z["skill"]
sidx = {n: i for i, n in enumerate(spells)}
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))
dcode, dnames = pd.factorize(draft)
ND = dcode.max() + 1
half = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 2 for d in dnames])[dcode]
ncard = H.sum(1) + Lh
cmc = np.array([by[n]["cmc"] or 0 for n in spells])
P(f"games {len(won)}, decks {codes.max()+1}, drafts {ND}, spells {len(spells)}; on play {play.mean():.3f}; skill missing {np.isnan(skill).mean():.3f}; games with any tutored spell {(Tu.sum(1)>0).mean():.4f}")

RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
cls = {}
for n, x in RC.items():
    if n in sidx: cls.setdefault(x["cls"].rstrip("*"), []).append(sidx[n])
cls["counter"] = [sidx[n] for n in ("Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany") if n in sidx]
CLASSES = ["unconditional", "damage", "conditional", "tempo", "counter"]
excluded = set(j for v in cls.values() for j in v) | {sidx[n] for n in ("Fatehold Charm", "Unsummon", "Consign to Memory") if n in sidx}
P("class sizes: " + ", ".join(f"{c} {len(cls[c])}" for c in cls))
pool_all = [j for j in range(len(spells)) if j not in excluded and cmc[j] > 0 and Dk[:, j].sum() > 0]

def demean(X, g):
    cnt = np.bincount(g).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=len(cnt)) / np.maximum(cnt, 1))[g]
    return o
def fit(idx, mask, W=None):
    """deck-demeaned OLS of won on [count of idx in hand, cards in hand, lands in hand]; games with a tutored copy of idx dropped.
    Returns (est pts, bootstrap array of est, n decks). Bootstrap: per-draft sufficient statistics, shared draft multiplicities W (B x ND)."""
    m = mask & (Tu[:, idx].sum(1) == 0)
    m = np.where(m)[0]; gg = pd.factorize(codes[m])[0]; ok = np.bincount(gg)[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    if len(m) < 500: return None
    X = np.column_stack([H[m][:, idx].sum(1), ncard[m], Lh[m]]); A = demean(np.column_stack([X, won[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    if Xd[:, 0].std() == 0: return None
    est = 100 * np.linalg.lstsq(Xd, yd, rcond=None)[0][0]
    if W is None: return est, None, gg.max() + 1
    cl = dcode[m]; o = np.argsort(cl, kind="stable"); cl = cl[o]; Xs, ys = Xd[o], yd[o]
    u, st = np.unique(cl, return_index=True)
    XtX = np.add.reduceat(Xs[:, :, None] * Xs[:, None, :], st, axis=0); Xty = np.add.reduceat(Xs * ys[:, None], st, axis=0)
    Wu = W[:, u]; A_ = np.einsum("bd,dij->bij", Wu, XtX); b_ = Wu @ Xty
    bs = np.array([100 * np.linalg.solve(A_[k], b_[k])[0] if abs(np.linalg.det(A_[k])) > 0 else np.nan for k in range(len(Wu))])
    return est, bs, gg.max() + 1

def placebo_set(idx, mask):
    """one random set: for each class card a non-removal, non-counter spell of the same cmc (cmc 6+ pooled) with deck presence in the mask
    within 0.5-2x of the class card's; fallback to cmc match only. Sampling without replacement."""
    rate = Dk[mask].mean(0); out = []
    for j in idx:
        bnd = lambda c: min(c, 6)
        c = [k for k in pool_all if k not in out and bnd(cmc[k]) == bnd(cmc[j]) and 0.5 * rate[j] <= rate[k] <= 2 * rate[j]]
        if len(c) < 3: c = [k for k in pool_all if k not in out and bnd(cmc[k]) == bnd(cmc[j])]
        out.append(int(rng.choice(c)))
    return out

B, NPL = 300, 40
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=float)
SPLITS = [("all decks", np.ones(len(won), bool)), ("skill >= 0.60", skill >= 0.60), ("on the play", play == 1), ("on the draw", play == 0),
          ("half 0 (draft hash)", half == 0), ("half 1 (draft hash)", half == 1), ("no mulligan", mull == 0)]
RES = {}
def run(split, mk, c):
    idx = cls[c]; mask = mk & (Dk[:, idx].sum(1) > 0)
    r = fit(idx, mask, W)
    if r is None: return None
    est, bs, nd = r
    pl = [x[0] for x in (fit(placebo_set(idx, mask), mask) for _ in range(NPL)) if x is not None]
    pm, ps = float(np.mean(pl)), float(np.std(pl, ddof=1))
    se = float(np.nanstd(bs, ddof=1)); adj = est - pm; ase = float(np.hypot(se, ps))
    return dict(split=split, cls=c, decks=nd, est=est, se=se, ci=(float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))), pm=pm, ps=ps,
                adj=adj, ase=ase, z=adj / ase, bs=bs, pl=pl)
for sl, mk in SPLITS:
    for c in CLASSES:
        r = run(sl, mk, c)
        if r: RES[(sl, c)] = r; print(sl, c, f"{r['est']:+.2f} pl {r['pm']:+.2f} z {r['z']:+.1f}", flush=True)

def table(sl):
    P(f"\n### {sl}\n\n| class | decks | pts per card in hand (SE) [95% boot CI] | placebo mean (SD), 40 sets | adjusted (SE) | z | rules out beyond (95%) |\n|---|---|---|---|---|---|---|")
    for c in CLASSES:
        r = RES.get((sl, c))
        if not r: P(f"| {c} | too few | | | | | |"); continue
        h = 1.96 * r["ase"]
        P(f"| {c} | {r['decks']} | {r['est']:+.2f} ({r['se']:.2f}) [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['pm']:+.2f} ({r['ps']:.2f}) | {r['adj']:+.2f} ({r['ase']:.2f}) | {r['z']:+.1f} | adjusted outside [{r['adj']-h:+.2f}, {r['adj']+h:+.2f}] |")
P("\n# Results (win-rate points per card of the class in hand; adjusted = estimate minus placebo mean)")
for sl, _ in SPLITS: table(sl)

# ratios to unconditional: joint bootstrap (shared W); placebo noise added by drawing the placebo means from their empirical sets
P("\n# Weights: class effect as a ratio to unconditional removal (adjusted = placebo mean subtracted from numerator and denominator)")
P("\n| split | class | raw ratio [95% CI] | adjusted ratio [95% CI] |\n|---|---|---|---|")
for sl, _ in SPLITS:
    u = RES.get((sl, "unconditional"))
    if not u: continue
    for c in CLASSES[1:]:
        r = RES.get((sl, c))
        if not r: continue
        n = len(r["bs"]); pu = rng.choice(u["pl"], n); pc = rng.choice(r["pl"], n)
        raw = r["bs"] / u["bs"]; adj = (r["bs"] - pc) / (u["bs"] - pu)
        ok = lambda a: a[np.isfinite(a)]
        q = lambda a: np.percentile(ok(a), [2.5, 97.5])
        P(f"| {sl} | {c} | {r['est']/u['est']:+.2f} [{q(raw)[0]:+.2f}, {q(raw)[1]:+.2f}] | {r['adj']/u['adj']:+.2f} [{q(adj)[0]:+.2f}, {q(adj)[1]:+.2f}] |")
P("\nRatio CIs are percentile intervals; when the denominator interval spans 0 they are unbounded and uninformative (read the absolute table).")
# differences vs unconditional in pts (stable)
P("\n# Differences from unconditional removal (pts per card, adjusted, joint bootstrap + placebo noise), all decks")
u = RES[("all decks", "unconditional")]
for c in CLASSES[1:]:
    r = RES[("all decks", c)]; d = r["adj"] - u["adj"]; s = np.hypot(np.nanstd(r["bs"] - u["bs"], ddof=1), np.hypot(r["ps"], u["ps"]))
    P(f"{c} minus unconditional: {d:+.2f} +/- {s:.2f}  (z {d/s:+.1f})")
# halves replication
P("\n# Halves: adjusted estimates")
for c in CLASSES:
    a, b = RES.get(("half 0 (draft hash)", c)), RES.get(("half 1 (draft hash)", c))
    if a and b: P(f"{c}: half 0 {a['adj']:+.2f} (se {a['ase']:.2f}), half 1 {b['adj']:+.2f} (se {b['ase']:.2f})")
open(f"{OUT}/b1_within_removal.txt", "w").write("\n".join(lines)); P("done")
