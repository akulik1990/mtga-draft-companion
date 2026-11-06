"""Review check: the flood / screw curve over the whole game instead of the opening hand only, within deck, conditioned on game length
and play/draw (so lands seen is draw luck at a fixed number of cards seen). Prices 18 vs 17 lands by the hypergeometric shift of lands
seen (the opening-hand method of hand_land_curve.py extended to all cards seen), and tests the sinks x flood claim (video item 5 / B6)
as "pts per land seen above expectation" by sink bucket. Uses the npz cache written by review_within.py.
Caveats printed with the numbers: game length is itself an outcome (a flooded deck may lose fast or slow), so this is descriptive;
num_turns counts both players' turns? (17Lands: turns of the game); cards seen = opening hand + draws incl. mulligan bottoms.
Output analysis/out/review_flood.txt."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import hypergeom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, cards
import video_sinks
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/9dfdf4f9-4b3f-4b70-a9fc-640913d44225/scratchpad"
Z = np.load(f"{SCR}/review_games.npz", allow_pickle=True)  # own cache written by analysis/review_within.py this session (string arrays), not downloaded data
H, Lh, Dk, Dl, spells = Z["H"], Z["Lh"].astype(int), Z["Dk"], Z["Dl"].astype(int), list(Z["spells"])
draft, build, won, turns, play, mull = np.asarray(Z["draft"], dtype=str), Z["build"], Z["won"].astype(float), Z["turns"].astype(int), Z["play"].astype(int), Z["mull"].astype(int)
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str))); dcode, _ = pd.factorize(draft)
ncard = H.sum(1).astype(int) + Lh
rng = np.random.default_rng(2); lines = []
def P(s=""): print(s, flush=True); lines.append(s)

def demean(X, g):
    n = g.max() + 1; cnt = np.bincount(g, minlength=n).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=n) / np.maximum(cnt, 1))[g]
    return o
def wfit(X, y, mask, boot=100):
    m = np.where(mask)[0]; gg = pd.factorize(codes[m])[0]; cnt = np.bincount(gg); ok = cnt[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    if len(m) < 500: return None
    A = demean(np.column_stack([X[m], y[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    keep = Xd.std(0) > 0
    if not keep.any(): return None
    est = np.full(X.shape[1], np.nan); est[keep] = 100 * np.linalg.lstsq(Xd[:, keep], yd, rcond=None)[0]
    cl = pd.factorize(dcode[m])[0]; order = np.argsort(cl, kind="stable"); Xs, ys = Xd[order][:, keep], yd[order]
    starts = np.r_[0, np.cumsum(np.bincount(cl))]; nc = cl.max() + 1
    XtX = np.array([Xs[a:b].T @ Xs[a:b] for a, b in zip(starts[:-1], starts[1:])]); Xty = np.array([Xs[a:b].T @ ys[a:b] for a, b in zip(starts[:-1], starts[1:])])
    bs = []
    for _ in range(boot):
        i = rng.integers(0, nc, nc)
        try: bs.append(100 * np.linalg.solve(XtX[i].sum(0), Xty[i].sum(0)))
        except np.linalg.LinAlgError: pass
    se = np.full(X.shape[1], np.nan); se[keep] = np.std(bs, axis=0)
    return est, se, gg.max() + 1

P("# Lands seen and winning, within deck, by game length and play/draw (kept 7-card hands only, num_mulligans = 0)")
P(f"games {len(won)}; mean cards seen {ncard.mean():.1f}; deck lands mode {np.bincount(Dl).argmax()}")
BANDS = [("<=6", 0, 6), ("7-8", 7, 8), ("9-10", 9, 10), ("11-12", 11, 12), ("13-14", 13, 14), ("15+", 15, 99)]
KS = list(range(2, 13))
curves = {}
P("\nPts vs the band's modal lands-seen, dummies per lands seen (k), deck fixed effects, bootstrap over drafts (100).\n")
for pl, pn in ((1, "play"), (0, "draw")):
    for lab, lo, hi in BANDS:
        mask = (turns >= lo) & (turns <= hi) & (play == pl) & (mull == 0)
        if mask.sum() < 3000: continue
        n_seen = int(np.median(ncard[mask])); ref = int(np.bincount(Lh[mask]).argmax())
        ks = [k for k in KS if k != ref and (Lh[mask] == k).sum() >= 200]
        X = np.column_stack([(Lh == k).astype(float) for k in ks])
        r = wfit(X, won, mask)
        if r is None: continue
        est, se, nd = r; curves[(pl, lab)] = (n_seen, ref, dict(zip(ks, est)), int(mask.sum()))
        P(f"{pn}, turns {lab}: games {int(mask.sum())}, decks {nd}, cards seen {n_seen}, ref {ref} lands: " + "; ".join(f"{k}: {est[i]:+.1f}±{se[i]:.1f}" for i, k in enumerate(ks)))

P("\n## 18 vs 17 lands priced by the hypergeometric shift of lands seen (40 cards, n = median cards seen in the band), per band and pooled by game share")
tot, wsum = 0.0, 0
for (pl, lab), (n_seen, ref, w, ng) in curves.items():
    d = 0.0
    for k in range(0, 20):
        p17, p18 = hypergeom.pmf(k, 40, 17, n_seen), hypergeom.pmf(k, 40, 18, n_seen)
        wk = w.get(k, 0.0 if k == ref else np.nan)
        if np.isnan(wk): wk = w[min(w, key=lambda j: abs(j - k))]  # outside the fitted range: nearest fitted bin
        d += (p18 - p17) * wk
    P(f"{'play' if pl else 'draw'}, turns {lab}: D18 = {d:+.2f} pts (n seen {n_seen})"); tot += d * ng; wsum += ng
P(f"Pooled D18 (games-share weighted): {tot / wsum:+.2f} pts per game. The hand-only method gave +0.00 [-0.04, +0.04]. Not a causal estimate: holds game length fixed, and\n"
  "the 18th land replaces a spell whose draw value is not priced here; it is the flood/screw side of the trade only.")

P("\n## Sinks x flood: pts per land seen above expectation (Lh - deck_lands/40 x cards seen), within deck, by sink bucket, inside turn bands")
by = cards(); T = video_sinks.tag(by)
sidx = {n: i for i, n in enumerate(spells)}
sink_i = [sidx[n] for n in T if n in sidx]
sinks = Dk[:, sink_i].sum(1).astype(int)
bk = np.where(sinks == 0, 0, np.where(sinks <= 2, 1, 2))
P(f"sink cards in deck (copies, spells only; land sinks not in the spell matrix): 0: {(bk==0).sum()} games, 1-2: {(bk==1).sum()}, 3+: {(bk==2).sum()}")
excess = Lh - Dl / 40.0 * ncard
P("\n| turns | sinks | games | decks | pts per excess land |\n|---|---|---|---|---|")
for lab, lo, hi in (("all", 0, 99), ("<=8", 0, 8), ("9-12", 9, 12), ("13+", 13, 99)):
    for b, bn in ((0, "0"), (1, "1-2"), (2, "3+")):
        mask = (turns >= lo) & (turns <= hi) & (bk == b) & (mull == 0)
        r = wfit(np.column_stack([excess, play]).astype(float), won, mask, boot=60)
        if r: P(f"| {lab} | {bn} | {int(mask.sum())} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} |")
# interaction form: one fit, excess x sinks (count, capped at 6), all games
mask = mull == 0
sc = np.minimum(sinks, 6).astype(float)
r = wfit(np.column_stack([excess, excess * sc, play]).astype(float), won, mask, boot=60)
if r: P(f"\nall games, excess land x sink count (0..6): per excess land at 0 sinks {r[0][0]:+.2f} ± {r[1][0]:.2f}; change per sink {r[0][1]:+.3f} ± {r[1][1]:.3f} (decks {r[2]})")
for lab, lo, hi in (("<=8", 0, 8), ("9-12", 9, 12), ("13+", 13, 99)):
    mask = (turns >= lo) & (turns <= hi) & (mull == 0)
    r = wfit(np.column_stack([excess, excess * sc, play]).astype(float), won, mask, boot=60)
    if r: P(f"turns {lab}: per excess land at 0 sinks {r[0][0]:+.2f} ± {r[1][0]:.2f}; change per sink {r[0][1]:+.3f} ± {r[1][1]:.3f}")
P("\nThe sink count here is a deck property, so 'change per sink' compares decks (not within deck); only the excess-land slope is within deck. A placebo\n(random spell sets of the same size) would be needed before reading the per-sink term as a sink effect rather than a deck-type effect.")
open(f"{OUT}/review_flood.txt", "w").write("\n".join(lines))
P("done")
