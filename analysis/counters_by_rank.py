"""Task B: counterspells (and unconditional removal as the comparison) within deck, split by the opponent's rank and the player's skill group. FRA Premier game data.
Doc: docs/heldout-leads-FRA.md section B. Design = analysis/b1_within_removal.py (deck-demeaned OLS of won on [class cards in hand, cards in hand, lands in hand], games with a tutored copy dropped,
bootstrap by draft, placebo-adjusted with same-cmc non-removal spells), re-used by importing nothing: the b1 cache (SCR/b1_games.npz, same row order as the game CSV) is joined to a column cache of opp_rank.
  python3 analysis/counters_by_rank.py cache   # streams opp_rank (and won, as an alignment check) from the game CSV
  python3 analysis/counters_by_rank.py run [own]  # -> analysis/out/counters_by_rank.txt; 'own' splits by the player's own rank (opp_rank is empty in the public file) -> counters_by_rank_ownrank.txt
PRE-REGISTRATION (before any output): opponent rank groups Bronze-Silver / Gold-Platinum / Diamond-Mythic from opp_rank; skill groups good >= 0.60, average [0.52, 0.60), weak < 0.52 (+ all);
 classes counter (Countersculpt, Icy Reception, Precise Redaction, Sublime Epiphany) and unconditional removal; estimate = adjusted (minus mean of 40 placebo sets) pts per card in hand.
 Hypothesis (Akshin): counters pay against weaker opponents and not against strong ones. Test: adjusted counter effect in Bronze-Silver minus Diamond-Mythic (all skill groups pooled, primary), SE from the bootstrap of the
 difference plus placebo noise; the same contrast for unconditional removal as the comparison. Confirmed if the counter contrast is positive with a 95% CI excluding 0. Skill-group cells are descriptive (12 cells, no correction claimed).
 Rank groups are by the OPPONENT's rank; a game is in a cell by its opponent, so the deck-demeaning compares the same deck's games across opponents of that rank only.
Run as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ..."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NPZ = f"{SCR}/b1_games.npz"; RK = f"{SCR}/opp_rank.npz"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"

if sys.argv[1] == "cache":
    d = pd.concat(ch for ch in pd.read_csv(GD, usecols=["won", "rank", "opp_rank"], chunksize=100000, dtype=str))
    np.savez_compressed(RK, opp=d.opp_rank.fillna("").values.astype(str), rank=d["rank"].fillna("").values.astype(str), won=(d.won == "True").values.astype(np.int8))
    print("cached", len(d), d.opp_rank.value_counts(dropna=False).to_dict()); sys.exit()

def cards():
    by = {}
    for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
        n = v["name"].split(" // ")[0]
        if n in by and not v.get("isprimarycard", 1): continue
        by[n] = v
    return by
by = cards(); island = {n for n, v in by.items() if "Land" in v["types"]}
rng = np.random.default_rng(20261108)
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
Z = np.load(NPZ, allow_pickle=True); RZ = np.load(RK, allow_pickle=True)
H, Lh, Tu, Dk, spells = Z["H"], Z["Lh"], Z["Tu"], Z["Dk"], list(Z["spells"])
draft = np.asarray(Z["draft"], dtype=str); build = Z["build"]; won = Z["won"].astype(float); play = Z["play"]; mull = Z["mull"]; skill = Z["skill"]
assert len(RZ["won"]) == len(won) and (RZ["won"] == Z["won"]).all(), "row alignment with the b1 cache failed"
SRC = sys.argv[2] if len(sys.argv) > 2 else "opp"   # "opp" = the pre-registered opponent rank; "own" = the PLAYER's rank (proxy, see header)
orank = np.asarray(RZ["opp" if SRC == "opp" else "rank"], dtype=str)
OUTF = f"{OUT}/counters_by_rank.txt" if SRC == "opp" else f"{OUT}/counters_by_rank_ownrank.txt"
if SRC == "opp" and not np.isin(np.char.lower(orank), ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]).any():
    open(OUTF, "w").write("opp_rank is 'None' in all %d games of game_data_public.FRA.PremierDraft.csv.gz (also Trad and Pick-Two): the pre-registered split by opponent rank cannot be run.\nThe player's own `rank` column is populated; run `run own` for that proxy (a different question, see docs/heldout-leads-FRA.md).\n" % len(orank))
    print(open(OUTF).read()); sys.exit()
RG = np.array([{"bronze": "BS", "silver": "BS", "gold": "GP", "platinum": "GP", "diamond": "DM", "mythic": "DM"}.get(r.lower(), "NA") for r in orank])
sidx = {n: i for i, n in enumerate(spells)}
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str))); dcode, dnames = pd.factorize(draft); ND = dcode.max() + 1
ncard = H.sum(1) + Lh; cmc = np.array([by[n]["cmc"] or 0 for n in spells])
P(f"rank source: {SRC}; games {len(won)}, decks {codes.max()+1}, drafts {ND}; opponent rank groups: " + ", ".join(f"{g} {int((RG==g).sum())}" for g in ("BS", "GP", "DM", "NA")) + "; raw opp_rank: " + str(pd.Series(orank).value_counts().to_dict()))
RC = json.load(open(f"{OUT}/removal_classes_FRA.json")); cls = {}
for n, x in RC.items():
    if n in sidx: cls.setdefault(x["cls"].rstrip("*"), []).append(sidx[n])
cls["counter"] = [sidx[n] for n in ("Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany") if n in sidx]
CLASSES = ["counter", "unconditional"]
excluded = set(j for v in cls.values() for j in v) | {sidx[n] for n in ("Fatehold Charm", "Unsummon", "Consign to Memory") if n in sidx}
pool_all = [j for j in range(len(spells)) if j not in excluded and cmc[j] > 0 and Dk[:, j].sum() > 0]
P("class sizes: " + ", ".join(f"{c} {len(cls[c])}" for c in CLASSES))

def demean(X, g):
    cnt = np.bincount(g).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=len(cnt)) / np.maximum(cnt, 1))[g]
    return o
def fit(idx, mask, W):
    m = mask & (Tu[:, idx].sum(1) == 0); m = np.where(m)[0]; gg = pd.factorize(codes[m])[0]; ok = np.bincount(gg)[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    if len(m) < 500: return None
    X = np.column_stack([H[m][:, idx].sum(1), ncard[m], Lh[m]]); A = demean(np.column_stack([X, won[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    if Xd[:, 0].std() == 0: return None
    est = 100 * np.linalg.lstsq(Xd, yd, rcond=None)[0][0]
    if W is None: return est, None, gg.max() + 1, len(m)
    cl = dcode[m]; o = np.argsort(cl, kind="stable"); cl = cl[o]; Xs, ys = Xd[o], yd[o]; u, st = np.unique(cl, return_index=True)
    XtX = np.add.reduceat(Xs[:, :, None] * Xs[:, None, :], st, axis=0); Xty = np.add.reduceat(Xs * ys[:, None], st, axis=0)
    A_ = np.einsum("bd,dij->bij", W[:, u], XtX); b_ = W[:, u] @ Xty
    bs = np.array([100 * np.linalg.solve(A_[k], b_[k])[0] if abs(np.linalg.det(A_[k])) > 0 else np.nan for k in range(len(W))])
    return est, bs, gg.max() + 1, len(m)
def placebo_set(idx, mask):
    rate = Dk[mask].mean(0); out = []
    for j in idx:
        b = lambda c: min(c, 6)
        c = [k for k in pool_all if k not in out and b(cmc[k]) == b(cmc[j]) and 0.5 * rate[j] <= rate[k] <= 2 * rate[j]]
        if len(c) < 3: c = [k for k in pool_all if k not in out and b(cmc[k]) == b(cmc[j])]
        out.append(int(rng.choice(c)))
    return out
B, NPL = 300, 40
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=float)
SK = {"all": np.ones(len(won), bool), "good": skill >= .60, "avg": (skill >= .52) & (skill < .60), "weak": skill < .52}
RK_ = {"all ranks": np.ones(len(won), bool), "BS": RG == "BS", "GP": RG == "GP", "DM": RG == "DM"}
RES = {}
for sk, ms in SK.items():
    for rk, mr in RK_.items():
        for c in CLASSES:
            idx = cls[c]; mask = ms & mr & (Dk[:, idx].sum(1) > 0); r = fit(idx, mask, W)
            if r is None: continue
            est, bs, nd, ng = r
            pl = [x[0] for x in (fit(placebo_set(idx, mask), mask, None) for _ in range(NPL)) if x is not None]
            RES[(sk, rk, c)] = dict(est=est, bs=bs, nd=nd, ng=ng, pl=pl, pm=float(np.mean(pl)), ps=float(np.std(pl, ddof=1)), se=float(np.nanstd(bs, ddof=1)))
            print(sk, rk, c, f"{est:+.2f}", flush=True)
def cell(r):
    if r is None: return "n/a"
    a = r["est"] - r["pm"]; s = np.hypot(r["se"], r["ps"]); return f"{a:+.2f} ({s:.2f}) [{a-1.96*s:+.2f}, {a+1.96*s:+.2f}]"
def dcell(a, b):
    if a is None or b is None: return "n/a"
    d = (a["est"] - a["pm"]) - (b["est"] - b["pm"]); s = np.hypot(np.nanstd(a["bs"] - b["bs"], ddof=1) if False else np.hypot(a["se"], b["se"]), np.hypot(a["ps"], b["ps"]))
    # opponent-rank groups share decks, so the two bootstraps are paired through W: use the joint bootstrap difference
    s = np.hypot(np.nanstd(a["bs"] - b["bs"], ddof=1), np.hypot(a["ps"], b["ps"])); return f"{d:+.2f} ({s:.2f}) [{d-1.96*s:+.2f}, {d+1.96*s:+.2f}]"
for c in CLASSES:
    P(f"\n## {c}: adjusted pts per card in hand (SE) [95% CI]; games n / decks n in brackets for the raw cell")
    P("| skill | all ranks | Bronze-Silver | Gold-Platinum | Diamond-Mythic | BS minus DM | GP minus DM |\n|---|---|---|---|---|---|---|")
    for sk in SK:
        g = lambda rk: RES.get((sk, rk, c))
        P(f"| {sk} | " + " | ".join(cell(g(rk)) + (f" [{g(rk)['ng']}g/{g(rk)['nd']}d]" if g(rk) else "") for rk in RK_) + f" | {dcell(g('BS'), g('DM'))} | {dcell(g('GP'), g('DM'))} |")
P("\nRaw (unadjusted) estimates and placebo means: (skill, rank, class): est / placebo mean")
for k, r in RES.items(): P(f"  {k}: {r['est']:+.2f} / {r['pm']:+.2f} (placebo SD {r['ps']:.2f})")
P("\nNote: the BS and DM cells are disjoint game sets sharing decks (a deck must have 2+ games in the cell); the difference CI uses the joint bootstrap (shared draft multiplicities) plus placebo noise.")
open(OUTF, "w").write("\n".join(lines)); P("done")
