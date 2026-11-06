"""Review checks on the 17Lands game data (FRA Premier, 297,964 games), one load cached as npz in the scratchpad.
(a) Up-time / game-length bias in GIH WR: per card, GIH WR minus the win rate of the decks that run it, by cmc band, overall and in
    short (<= 8 turns) vs long (>= 11 turns) games and by deck speed tercile (mean cmc of the deck's spells). Shows how much of
    "mean GIH WR" is a game-length artefact that the deck GLMs inherit.
(b) Within-deck removal by class (REM48 classes): removal card of class c in hand vs other games of the same deck, cards and lands in
    hand fixed, against 20 placebo sets of ordinary spells; the design the lead's B5 should have used.
(c) Counters (CORE) in hand within deck, by opponent colour and by game length, next to unconditional removal in hand under the same
    split: the outcome the "counters answer the format's threats" claim needs.
(d) Ways in hand within deck by whether the opponent is in B or R (removal-heavy colours): fragility to removal.
Output analysis/out/review_within.txt. Run (about 3 min, one core for the load):
nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/review_within.py"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, cards
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/9dfdf4f9-4b3f-4b70-a9fc-640913d44225/scratchpad"
NPZ = f"{SCR}/review_games.npz"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
rng = np.random.default_rng(1)
lines = []
def P(s=""): print(s, flush=True); lines.append(s)

by = cards()
island = {n for n, v in by.items() if "Land" in v["types"]}
if not os.path.exists(NPZ):
    hdr = pd.read_csv(GD, nrows=0).columns.tolist()
    spells = sorted(n for n in by if n not in island and f"opening_hand_{n}" in hdr)
    landn = [n for n in island if f"opening_hand_{n}" in hdr]
    meta = ["draft_id", "build_index", "won", "num_turns", "on_play", "opp_colors", "main_colors", "num_mulligans"]
    use = meta + [f"{p}{n}" for n in spells + landn for p in ("opening_hand_", "drawn_")] + [f"deck_{n}" for n in spells + landn]
    parts = []
    for i, ch in enumerate(pd.read_csv(GD, usecols=use, chunksize=40000)):
        H = (ch[[f"opening_hand_{n}" for n in spells]].fillna(0).values + ch[[f"drawn_{n}" for n in spells]].fillna(0).values).astype(np.int8)
        Lh = (ch[[f"opening_hand_{n}" for n in landn]].fillna(0).values + ch[[f"drawn_{n}" for n in landn]].fillna(0).values).sum(1).astype(np.int8)
        Dk = ch[[f"deck_{n}" for n in spells]].fillna(0).values.astype(np.int8)
        Dl = ch[[f"deck_{n}" for n in landn]].fillna(0).values.sum(1).astype(np.int8)
        parts.append((ch[meta].copy(), H, Lh, Dk, Dl)); P(f"chunk {i}")
    M = pd.concat([p[0] for p in parts], ignore_index=True)
    np.savez_compressed(NPZ, H=np.vstack([p[1] for p in parts]), Lh=np.concatenate([p[2] for p in parts]), Dk=np.vstack([p[3] for p in parts]),
                        Dl=np.concatenate([p[4] for p in parts]), spells=np.array(spells), draft=M.draft_id.values.astype(str), build=M.build_index.values.astype(np.int16),
                        won=M.won.values.astype(np.int8), turns=M.num_turns.fillna(0).values.astype(np.int16), play=M.on_play.values.astype(np.int8),
                        opp=M.opp_colors.fillna("").values.astype(str), main=M.main_colors.fillna("").values.astype(str), mull=M.num_mulligans.fillna(0).values.astype(np.int8))
Z = np.load(NPZ, allow_pickle=True)  # own cache in the scratchpad
H, Lh, Dk, Dl, spells = Z["H"], Z["Lh"], Z["Dk"], Z["Dl"], list(Z["spells"])
draft, build, won, turns, play, opp, main, mull = np.asarray(Z["draft"], dtype=str), Z["build"], Z["won"].astype(float), Z["turns"], Z["play"], np.asarray(Z["opp"], dtype=str), np.asarray(Z["main"], dtype=str), Z["mull"]
sidx = {n: i for i, n in enumerate(spells)}
codes, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))
dcode, _ = pd.factorize(draft)
ncard = H.sum(1) + Lh
cmc = np.array([by[n]["cmc"] or 0 for n in spells])
P(f"games {len(won)}, decks {codes.max()+1}, spells {len(spells)}; mean turns {turns.mean():.1f}; on play {play.mean():.3f}")

# ---------- (a) up-time bias
P("\n# (a) GIH WR minus deck WR by cmc band (pts). Deck WR = win rate of games where the card is in the deck.")
deck_cmc = (Dk * cmc).sum(1) / np.maximum(Dk.sum(1), 1)
ter = np.digitize(deck_cmc, np.quantile(deck_cmc, [1/3, 2/3]))
def gap(mask):
    inh = (H > 0) & mask[:, None]; ind = (Dk > 0) & mask[:, None]
    w = won[:, None]
    gih = (inh * w).sum(0) / np.maximum(inh.sum(0), 1); dwr = (ind * w).sum(0) / np.maximum(ind.sum(0), 1)
    ok = inh.sum(0) >= 500
    out = {}
    for lo, hi, lab in ((0, 2, "<=2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 99, "6+")):
        sel = ok & (cmc >= lo) & (cmc <= hi)
        out[lab] = (100 * (gih - dwr)[sel].mean(), int(sel.sum()))
    return out
rows = [("all games", np.ones(len(won), bool)), ("short games (<= 8 turns)", turns <= 8), ("long games (>= 11 turns)", turns >= 11),
        ("fast decks (cmc tercile 1)", ter == 0), ("slow decks (tercile 3)", ter == 2)]
P("\n| games | " + " | ".join(f"cmc {k}" for k in ("<=2", "3", "4", "5", "6+")) + " |\n|---|---|---|---|---|---|")
for lab, m in rows:
    g = gap(m); P(f"| {lab} | " + " | ".join(f"{g[k][0]:+.2f} (n {g[k][1]})" for k in g) + " |")
P("Reading: a positive gap = a card wins more when drawn than its decks do on average (the usual 'drawn' premium); the spread across cmc\n"
  "bands inside one row is the up-time bias (expensive cards sit in hand longer and in short games never get cast). Rows by game length show\n"
  "the game-length bias: in short games all drawn cards sit below deck WR for aggressive decks and above for slow ones.")

# ---------- within-deck machinery (as tools/within_deck.py: deck-demeaned OLS, cards and lands in hand fixed)
def demean(X, g):
    n = g.max() + 1; cnt = np.bincount(g, minlength=n).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=n) / np.maximum(cnt, 1))[g]
    return o
def wfit(X, y, mask, boot=0, keep=1):
    m = np.where(mask)[0]; gg = pd.factorize(codes[m])[0]; cnt = np.bincount(gg); ok = cnt[gg] >= 2; m = m[ok]; gg = pd.factorize(gg[ok])[0]
    if len(m) < 200: return None
    A = demean(np.column_stack([X[m], y[m]]), gg); Xd, yd = A[:, :-1], A[:, -1]
    if Xd[:, 0].std() == 0: return None
    est = 100 * np.linalg.lstsq(Xd, yd, rcond=None)[0][:keep]
    if not boot: return est, None, gg.max() + 1
    cl = pd.factorize(dcode[m])[0]; order = np.argsort(cl, kind="stable"); Xs, ys = Xd[order], yd[order]
    starts = np.r_[0, np.cumsum(np.bincount(cl))]; nc = cl.max() + 1
    XtX = np.array([Xs[a:b].T @ Xs[a:b] for a, b in zip(starts[:-1], starts[1:])]); Xty = np.array([Xs[a:b].T @ ys[a:b] for a, b in zip(starts[:-1], starts[1:])])
    bs = []
    for _ in range(boot):
        i = rng.integers(0, nc, nc)
        try: bs.append(100 * np.linalg.solve(XtX[i].sum(0), Xty[i].sum(0))[:keep])
        except np.linalg.LinAlgError: pass
    return est, np.std(bs, axis=0), gg.max() + 1
def placebo(idx, mask, n=20):
    """random sets of ordinary spells, same size, deck-presence rate within 0.5-2x of the test cards' (as video_empower.py)."""
    rate = Dk[mask].mean(0); er = rate[idx].mean()
    pool = [j for j in range(len(spells)) if j not in idx and 0.5 * er <= rate[j] <= 2 * er]
    nul = []
    for _ in range(n):
        fk = rng.choice(pool, len(idx), replace=False)
        x = wfit(np.column_stack([H[:, fk].sum(1), ncard, Lh]), won, mask)
        if x is not None: nul.append(x[0][0])
    return float(np.mean(nul)), float(np.std(nul))
def test(lab, idx, mask, boot=200, plac=20):
    r = wfit(np.column_stack([H[:, idx].sum(1), ncard, Lh]), won, mask, boot=boot)
    if r is None: P(f"{lab}: too few"); return
    pm, ps = placebo(idx, mask, plac) if plac else (float("nan"), float("nan"))
    z = (r[0][0] - pm) / np.hypot(r[1][0], ps) if plac else float("nan")
    P(f"| {lab} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} | {pm:+.2f} ± {ps:.2f} | {z:+.1f} |")

RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
MERGE = {"tempo": "other", "sweeper": "other", "bite/fight": "bite"}
cls = {}
for n, x in RC.items():
    if n in sidx: cls.setdefault(MERGE.get(x["cls"].rstrip("*"), x["cls"].rstrip("*")), []).append(sidx[n])
allrem = sorted(j for v in cls.values() for j in v)
P("\n# (b) Within deck: removal card in hand, pts per card, cards and lands in hand fixed (decks that run >= 1 card of the class)")
P("\n| class | decks | pts per card in hand | placebo (20 sets) | z vs placebo |\n|---|---|---|---|---|")
test("all removal (REM48)", allrem, Dk[:, allrem].sum(1) > 0)
for c in ("unconditional", "damage", "conditional", "bite", "other"):
    test(c, cls[c], Dk[:, cls[c]].sum(1) > 0)
P("\nSame, by game length (short <= 8 turns / long >= 11), no placebo:")
P("\n| class | games | decks | pts per card |\n|---|---|---|---|")
for c in ("unconditional", "damage", "conditional"):
    for gl, gm in (("short", turns <= 8), ("long", turns >= 11)):
        r = wfit(np.column_stack([H[:, cls[c]].sum(1), ncard, Lh]), won, (Dk[:, cls[c]].sum(1) > 0) & gm, boot=100)
        if r: P(f"| {c} | {gl} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} |")

# ---------- (c) counters
CORE = [sidx[n] for n in ("Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany") if n in sidx]
P("\n# (c) Within deck: CORE counter in hand vs unconditional removal in hand, by opponent colour and game length")
P("\n| cards | split | decks | pts per card in hand |\n|---|---|---|---|")
for lab, idx in (("counter (CORE)", CORE), ("unconditional removal", cls["unconditional"])):
    base = Dk[:, idx].sum(1) > 0
    for sl, sm in (("all", np.ones(len(won), bool)), ("on play", play == 1), ("on draw", play == 0), ("short <= 8 turns", turns <= 8), ("long >= 11 turns", turns >= 11)):
        r = wfit(np.column_stack([H[:, idx].sum(1), ncard, Lh]), won, base & sm, boot=100)
        if r: P(f"| {lab} | {sl} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} |")
    for col in "WUBRG":
        sm = np.array([col in o for o in opp])
        r = wfit(np.column_stack([H[:, idx].sum(1), ncard, Lh]), won, base & sm, boot=100)
        if r: P(f"| {lab} | opp plays {col} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} |")

# ---------- (d) Ways by opponent removal colours
WAYS = [sidx[n] for n in spells if n.startswith("Way of the")]
P("\n# (d) Within deck: any Way in hand, by whether the opponent is in B or R (removal-heavy)")
P("\n| split | decks | pts per Way in hand |\n|---|---|---|")
base = Dk[:, WAYS].sum(1) > 0
for sl, sm in (("all", np.ones(len(won), bool)), ("opp in B or R", np.array([("B" in o) or ("R" in o) for o in opp])), ("opp not in B or R", np.array([("B" not in o) and ("R" not in o) for o in opp]))):
    r = wfit(np.column_stack([H[:, WAYS].sum(1), ncard, Lh]), won, base & sm, boot=100)
    if r: P(f"| {sl} | {r[2]} | {r[0][0]:+.2f} ± {r[1][0]:.2f} |")
open(f"{OUT}/review_within.txt", "w").write("\n".join(lines))
P("done")
