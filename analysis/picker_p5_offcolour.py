"""P5 pack direction and Power/Fit off-colour disagreements (2026-11-06, FRA, 600 train top-player drafts).

    python3 analysis/picker_p5_offcolour.py > analysis/out/picker_p5_offcolour.txt

Measures are fixed in the docstrings below before any outcome was read. Clustered by draft throughout (bootstrap over drafts,
2000 resamples, seed 20261106). Halves for Task 2 are by md5 of the draft id.
"""
import gzip, hashlib, json, csv, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

R = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261106)
COL = "WUBRG"
front = lambda n: n.split(" // ")[0]
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)
info = lambda n: cards.get(front(n), {})
gih = lambda n: ((info(n).get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
cols = lambda n: [c for c in COL if c in (info(n).get("colors") or [])]
is_land = lambda n: "Land" in (info(n).get("types") or [])
drafts = json.load(gzip.open(R / "data/Temp/replay/FRA.top600.train.drafts.json.gz", "rt"))
log = json.loads((R / "data/Temp/replay/logs/tokenfix.json").read_text())
allg = [gih(n) for n in cards if gih(n) > 0 and not is_land(n)]
BASE = float(np.mean(allg))
print(f"drafts {len(drafts)}, log picks {len(log)}, baseline GIH (mean over {len(allg)} non-land cards) {BASE:.2f}")
excess = lambda n: max(gih(n) - BASE, 0.0)


def colvec(n, w=None):
    """card's excess quality split equally over its colours (colourless and lands: zeros)."""
    v = np.zeros(5); cs = cols(n)
    if cs and not is_land(n):
        for c in cs: v[COL.index(c)] = (excess(n) if w is None else w) / len(cs)
    return v


def boot_ci(stat, n, B=2000):
    s = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    return np.percentile(s, [2.5, 97.5])


# ---------------------------------------------------------------- Task 1a
def avail(d, pack, lo, hi):
    v = np.zeros(5); k = 0
    for p in d["picks"]:
        if p[0] == pack and lo <= p[1] <= hi:
            for n in p[3]: v += colvec(n)
            k += 1
    return v / max(k, 1)

def build(lo, hi):
    A = np.array([[avail(d, p, lo, hi) for p in (1, 2, 3)] for d in drafts])  # draft, pack, colour
    return A
def residual(A):
    A = A - A.mean(axis=0, keepdims=True)         # remove set-level colour means per pack (over drafts)
    return A - A.mean(axis=2, keepdims=True)      # centre across colours inside draft-pack

def ols(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]; return b

def task1a(lo, hi, label):
    A = residual(build(lo, hi)); n = len(A)
    z = lambda x: x / x.std()
    q1, q2, q3 = (z(A[:, i, :]) for i in range(3))
    sd = [A[:, i, :].std() for i in range(3)]
    def stat(idx):
        y, a, b = q3[idx].ravel(), q1[idx].ravel(), q2[idx].ravel()
        r1, r2 = np.corrcoef(y, a)[0, 1], np.corrcoef(y, b)[0, 1]
        bb = ols(y, np.column_stack([a, b]))
        return np.array([r1, r2, r1 - r2, bb[1], bb[2], bb[1] - bb[2]])
    full = stat(np.arange(n)); ci = np.array([boot_ci(lambda i, k=k: stat(i)[k], n, 1000) for k in range(6)])
    names = ["r(P1,P3)", "r(P2,P3)", "r diff", "beta P1 (joint)", "beta P2 (joint)", "beta diff"]
    print(f"\n1a. pack 3 availability at picks {lo}-{hi} on pack 1 and pack 2 availability at the same picks ({label}); {n} drafts x 5 colours, standardised")
    for k in range(6): print(f"   {names[k]:18s} {full[k]:+.3f}  [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")
    # reference: reliability ceiling, P1 vs P2 correlation (same neighbour pairs differ)
    print(f"   r(P1,P2) {np.corrcoef(q1.ravel(), q2.ravel())[0,1]:+.3f}")
    return A

A38 = task1a(3, 8, "primary")
task1a(5, 10, "robustness: later window")
# placebo-ish: split-half reliability of P3 itself is unavailable; report n of observations only

# ---------------------------------------------------------------- Task 1b1 (mechanism: what I cut flows back in pack 2)
def taken(d, pack):
    v = np.zeros(5)
    for p in d["picks"]:
        if p[0] == pack: v += colvec(p[2])
    return v
T1 = np.array([taken(d, 1) for d in drafts]); T1 = T1 - T1.mean(axis=0, keepdims=True); T1 = T1 - T1.mean(axis=1, keepdims=True)
T12 = np.array([taken(d, 1) + taken(d, 2) for d in drafts]); T12 = T12 - T12.mean(axis=0, keepdims=True); T12 = T12 - T12.mean(axis=1, keepdims=True)
n = len(drafts); z = lambda x: x / x.std()
q = [z(A38[:, i, :]) for i in range(3)]
def b1(idx, tgt):
    X = np.column_stack([z(T1)[idx].ravel(), q[0][idx].ravel()]); y = q[tgt][idx].ravel()
    return ols(y, X)[1]
print("\n1b-i. Mechanism check: does a colour I took in pack 1 (excess GIH taken, centred) come back more in pack 2 (opposite neighbour, starved of it) than in pack 3?")
print("   coefficient on pack-1 take share, controlling for pack-1 availability; standardised, picks 3-8")
res = {}
for tgt, lab in ((1, "pack 2 availability"), (2, "pack 3 availability")):
    f = b1(np.arange(n), tgt); ci = boot_ci(lambda i: b1(i, tgt), n, 1000); res[tgt] = (f, ci)
    print(f"   {lab}: {f:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
dd = lambda i: b1(i, 1) - b1(i, 2)
print(f"   difference (pack 2 - pack 3): {dd(np.arange(n)):+.3f} {np.round(boot_ci(dd, n, 1000), 3)}")

# ---------------------------------------------------------------- Task 1b2 (conditional logit on picks)
def cond_logit(X, ch, mask, cl, ridge=1e-6):
    """X: G x K x F, ch: G chosen index, mask: G x K, cl: draft index per group. Returns beta, cluster-robust cov."""
    G, K, F = X.shape
    def nll(b):
        u = np.where(mask, X @ b, -1e9); m = u.max(1, keepdims=True); e = np.exp(u - m) * mask; s = e.sum(1, keepdims=True)
        p = e / s; ll = (u[np.arange(G), ch] - m[:, 0]) - np.log(s[:, 0])
        g = -(X[np.arange(G), ch] - (p[:, :, None] * X).sum(1)).sum(0)
        return -ll.sum() + ridge * b @ b, g + 2 * ridge * b
    b = minimize(lambda b: nll(b)[0], np.zeros(F), jac=lambda b: nll(b)[1], method="BFGS").x
    u = np.where(mask, X @ b, -1e9); m = u.max(1, keepdims=True); e = np.exp(u - m) * mask; p = e / e.sum(1, keepdims=True)
    mean = (p[:, :, None] * X).sum(1); sc = X[np.arange(G), ch] - mean            # G x F scores
    H = np.zeros((F, F))
    for k in range(F):
        for l in range(F):
            H[k, l] = (p * (X[:, :, k] - mean[:, k:k+1]) * (X[:, :, l] - mean[:, l:l+1])).sum()
    nd = cl.max() + 1; S = np.zeros((nd, F)); np.add.at(S, cl, sc)
    Hi = np.linalg.inv(H); cov = Hi @ (S.T @ S) @ Hi * nd / (nd - 1)
    return b, cov

dix = {d["id"]: i for i, d in enumerate(drafts)}
human = {(d["id"], p[0], p[1]): p[2] for d in drafts for p in d["picks"]}
def poolshare(names, w=None):
    v = np.zeros(5)
    for nm in names: v += colvec(nm, 1.0) if w is None else colvec(nm, w)
    return v / v.sum() if v.sum() else np.full(5, 0.2)
# per draft: share of P1 picks by colour (count weight, coloured nonland cards)
cut = {}
for d in drafts:
    cut[d["id"]] = poolshare([p[2] for p in d["picks"] if p[0] == 1])
def feats(e, chooser, packn):
    d, pk, pn, cand, pool = e[0], e[1], e[2], e[3], e[4]
    names = list(dict.fromkeys(n for n, _ in cand)); ps = poolshare(pool); cs = cut[d]
    rows = []
    for n in names:
        c = cols(n); 
        rows.append([(gih(n) - BASE) / 3.0, float(np.mean([ps[COL.index(x)] for x in c])) if c else 0.0, 0.0 if c else 1.0,
                     float(np.mean([cs[COL.index(x)] for x in c])) if c else 0.0])
    return names, np.array(rows)

def stage_logit(packs, lo, hi, who, use_pool=True, inter=False):
    """packs: list; inter=True stacks packs and adds every feature x (pack == packs[0]) so the difference is estimated jointly (cluster-robust)."""
    Xs, chs, cls = [], [], []
    for e in log:
        if e[1] not in packs or not (lo <= e[2] <= hi) or not e[3]: continue
        names, F = feats(e, who, e[1])
        pick = human.get(tuple(e[:3])) if who == "human" else e[3][0][0]
        if pick not in names: continue
        F = F[:, [0, 2] + ([1] if use_pool else []) + [3]]
        if inter: F = np.hstack([F, F * float(e[1] == packs[0])])
        Xs.append(F); chs.append(names.index(pick)); cls.append(dix[e[0]])
    K = 14; G = len(Xs); Fn = Xs[0].shape[1]
    X = np.zeros((G, K, Fn)); mask = np.zeros((G, K), bool)
    for i, x in enumerate(Xs): X[i, :len(x)] = x; mask[i, :len(x)] = True
    b, cov = cond_logit(X, np.array(chs), mask, np.array(cls))
    return b, np.sqrt(np.diag(cov)), G, cov

print("\n1b-ii. Conditional logit of the chosen card on the pack (candidates = the pack as logged): GIH (per 3 points above baseline), colourless dummy,")
print("   pool colour share of the card's colours, P1 take share of the card's colours (count share of pack-1 picks; pack 1 itself excluded, it would leak).")
print("   Picks 2-9, cluster-robust SE by draft. Human = top drafters; v2 = the engine's top pick (replay on the human pool). Coefficients in logits.")
for pack in (2, 3):
    for who in ("human", "v2"):
        out = {}
        for use_pool in (True, False):
            b, se, G, cov = stage_logit([pack], 2, 9, who, use_pool)
            names = (["GIH", "colourless"] + (["pool share"] if use_pool else []) + ["P1 take share"])
            out[use_pool] = " ".join(f"{nm} {bb:+.2f}({s:.2f})" for nm, bb, s in zip(names, b, se))
        print(f"   pack {pack} {who:5s} picks {G:5d} | with pool: {out[True]}\n{'':27s}| without pool: {out[False]}")
print("   Pack 2 minus pack 3 for the P1-take coefficient (joint model with pack-2 interactions, cluster-robust):")
for who in ("human", "v2"):
    for use_pool in (True, False):
        b, se, G, cov = stage_logit([2, 3], 2, 9, who, use_pool, inter=True)
        F = len(b) // 2; k = F - 1
        print(f"   {who:5s} {'with pool   ' if use_pool else 'without pool'} P1-take pack 3 {b[k]:+.2f} ({se[k]:.2f}); pack 2 minus pack 3 {b[F+k]:+.2f} ({se[F+k]:.2f})"
              + (f"; pool-share diff {b[F+2]:+.2f} ({se[F+2]:.2f})" if use_pool else ""))

# ================================================================ Task 2
# Pre-registered before outcomes were read:
#  disagreement = engine top-1 != human pick, human pick among the logged candidates; stages P1 picks 2-10, P2 picks 1-14.
#  pool pair = the two colours held by most non-land pool cards (ties by excess-GIH sum); commitment = their share of all colour-card counts.
#  OFF = card has a colour outside the pair; ON = everything else (inside the pair, colourless, lands).
#  classes: (i) engine OFF & human ON; (ii) engine ON & human OFF; (iii) other (both ON, both OFF).
#  rule: if the engine's top card is OFF and the best-scoring ON candidate b has score gap sa - sb < T, take b. T from a grid, per stage
#  (P1 2-4, P1 5-10, P2), either constant (F1) or t x commitment (F2); chosen on the fit half (md5 of draft id, even), top-1 on the odd half.
print("\n\n==================== Task 2: Power/Fit off-colour disagreements ====================")
md = {}
ids = set(dix)
with gzip.open(R / "data/Temp/replay/draft_data_public.FRA.PremierDraft.top.csv.gz", "rt", newline="") as f:
    rd = csv.reader(f); hd = next(rd); ix = {h: i for i, h in enumerate(hd)}
    for row in rd:
        if row[2] in ids:
            md[(row[2], int(row[7]) + 1, int(row[8]) + 1)] = (row[9], float(row[11] or 0))
chk = [(human[k], v[0]) for k, v in md.items() if k in human]
print(f"maindeck join: {len(md)} CSV picks for these drafts, {sum(1 for a, b in chk if front(a) == front(b))}/{len(chk)} agree with the drafts file's pick name")
half = lambda did: int(hashlib.md5(did.encode()).hexdigest(), 16) % 2

def pair_of(pool):
    cnt = Counter(); ex = Counter()
    for n in pool:
        if is_land(n): continue
        for c in cols(n): cnt[c] += 1; ex[c] += excess(n)
    top = sorted(cnt, key=lambda c: (-cnt[c], -ex[c]))[:2]
    return set(top), sum(cnt[c] for c in top) / max(sum(cnt.values()), 1)
off = lambda n, pr: bool(set(cols(n)) - pr)

rows = []
for e in log:
    _, pk, pn = e[:3]
    if not ((pk == 1 and 2 <= pn <= 10) or pk == 2) or not e[3]: continue
    h = human.get(tuple(e[:3])); sc = dict(e[3])
    if h is None or h not in sc: continue
    pr, share = pair_of(e[4]); A = e[3][0][0]
    onb = [(s, n) for n, s in e[3] if not off(n, pr)]
    rows.append(dict(d=e[0], pk=pk, pn=pn, stage="P1 2-4" if pk == 1 and pn <= 4 else "P1 5-10" if pk == 1 else "P2", A=A, h=h, sa=e[3][0][1], sh=sc[h],
                     share=share, poolsz=len(e[4]), Aoff=off(A, pr), hoff=off(h, pr), agree=(A == h), best_on=onb[0] if onb else None,
                     md=md.get(tuple(e[:3]), (None, None))[1], half=half(e[0])))
print(f"picks in scope {len(rows)} ({len({r['d'] for r in rows})} drafts); top-1 {100*np.mean([r['agree'] for r in rows]):.2f}%")
dis = [r for r in rows if not r["agree"]]
print(f"disagreements {len(dis)}")
for r in dis: r["cls"] = "(i) eng OFF / human ON" if r["Aoff"] and not r["hoff"] else "(ii) eng ON / human OFF" if not r["Aoff"] and r["hoff"] else "(iii) both ON" if not r["Aoff"] else "(iii) both OFF"
mdbase = {st: np.mean([r["md"] for r in rows if r["stage"] == st and r["md"] is not None]) for st in ("P1 2-4", "P1 5-10", "P2")}
print("human pick's maindeck rate, all picks in stage:", {k: round(float(v), 3) for k, v in mdbase.items()})
def table(sub, label):
    print(f"\n{label}: {len(sub)} disagreements")
    print(f"{'class':24s}{'n':>6s}{'share':>7s}{'GIH eng-hum':>12s}{'hum higher':>11s}{'commit':>8s}{'pick':>6s}{'score gap':>10s}{'hum maindeck':>13s}{'(base)':>8s}")
    for c in ["(i) eng OFF / human ON", "(ii) eng ON / human OFF", "(iii) both ON", "(iii) both OFF"]:
        s = [r for r in sub if r["cls"] == c]
        if not s: continue
        g = np.array([gih(r["A"]) - gih(r["h"]) for r in s]); m = [r["md"] for r in s if r["md"] is not None]
        base = np.mean([mdbase[r["stage"]] for r in s if r["md"] is not None])
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0
        print(f"{c:24s}{len(s):6d}{len(s)/len(sub):7.1%}{g.mean():+8.2f}±{1.96*se:.2f}{np.mean(g < 0):11.1%}{np.mean([r['share'] for r in s]):8.2f}{np.mean([r['pn'] for r in s]):6.1f}"
              f"{np.mean([r['sa']-r['sh'] for r in s]):10.2f}{np.mean(m):13.3f}{base:8.3f}")
table(dis, "all stages")
for st in ("P1 2-4", "P1 5-10", "P2"): table([r for r in dis if r["stage"] == st], st)
table([r for r in dis if r["sa"] - r["sh"] > 4], "score gap > 4 (the earlier 'big miss' cut)")
print("\ncommitment bands (disagreements, share of class (i) and (ii)):")
for lo, hi in ((0, .75), (.75, .9), (.9, 1.01)):
    s = [r for r in dis if lo <= r["share"] < hi]
    print(f"  commit {lo:.2f}-{hi:.2f}: n {len(s)}, (i) {np.mean([r['cls'].startswith('(i) ') for r in s]):.1%}, (ii) {np.mean([r['cls'].startswith('(ii)') for r in s]):.1%}")
# cluster bootstrap of class shares
dd_ = sorted({r["d"] for r in dis}); byd = defaultdict(list)
for r in dis: byd[r["d"]].append(r)
def share_stat(cls):
    def f(idx):
        s = [r for i in idx for r in byd[dd_[i]]]; return np.mean([r["cls"].startswith(cls) for r in s])
    return f
for cls in ("(i) ", "(ii)"):
    f = share_stat(cls); print(f"  share of disagreements in class {cls}: {f(np.arange(len(dd_))):.1%} {np.round(boot_ci(f, len(dd_), 1000), 3)} (draft bootstrap)")

# ---- rule
GRID = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15]
def apply(r, fam, t):
    """returns the pick under the rule; unchanged unless engine card OFF and an ON candidate within T."""
    if not r["Aoff"] or r["best_on"] is None: return r["A"]
    gap = r["sa"] - r["best_on"][0]; T = t if fam == "F1" else t * r["share"]
    return r["best_on"][1] if gap < T else r["A"]
def top1(sub, fam, t): return np.mean([apply(r, fam, t) == r["h"] for r in sub])
print("\n---- rule: take the best ON-colour candidate when the engine's OFF-colour top card leads it by less than T ----")
print("fit on even-hash drafts (grid chosen by top-1; ties to the smaller T), tested on odd-hash drafts, then the halves swapped")
for fit_h in (0, 1):
    print(f"\nfit half {fit_h} -> test half {1-fit_h}")
    tot = {"F1": [], "F2": []}
    for st in ("P1 2-4", "P1 5-10", "P2"):
        fit = [r for r in rows if r["stage"] == st and r["half"] == fit_h]; te = [r for r in rows if r["stage"] == st and r["half"] != fit_h]
        for fam in ("F1", "F2"):
            curve = [(top1(fit, fam, t), -t) for t in GRID]; best = GRID[max(range(len(GRID)), key=lambda i: curve[i])]
            b0 = top1(te, fam, 0); b1 = top1(te, fam, best)
            gains = sum(apply(r, fam, best) == r["h"] and r["A"] != r["h"] for r in te); losses = sum(r["A"] == r["h"] and apply(r, fam, best) != r["h"] for r in te)
            changed = sum(apply(r, fam, best) != r["A"] for r in te)
            ted = sorted({r["d"] for r in te}); bd = defaultdict(list)
            for r in te: bd[r["d"]].append(r)
            def stat(idx, fam=fam, best=best, ted=ted, bd=bd):
                s = [r for i in idx for r in bd[ted[i]]]; return top1(s, fam, best) - top1(s, fam, 0)
            ci = boot_ci(stat, len(ted), 1000)
            print(f"  {st:8s} {fam}: T {best:>2d} | fit {100*curve[GRID.index(best)][0]:.2f} (T=0: {100*curve[0][0]:.2f}) | held-out {100*b0:.2f} -> {100*b1:.2f} "
                  f"({100*(b1-b0):+.2f} pts, 95% CI {100*ci[0]:+.2f} to {100*ci[1]:+.2f}); changed {changed}, gained {gains}, lost {losses}, n {len(te)}")
    # full-curve on the fit half for reference
for st in ("P1 2-4", "P1 5-10", "P2"):
    print(f"\ntop-1 by T, all drafts, {st} (F1 constant | F2 x commitment):")
    sub = [r for r in rows if r["stage"] == st]
    print("  T      " + " ".join(f"{t:6d}" for t in GRID)); 
    print("  F1     " + " ".join(f"{100*top1(sub,'F1',t):6.2f}" for t in GRID)); print("  F2     " + " ".join(f"{100*top1(sub,'F2',t):6.2f}" for t in GRID))
# GIH check on changed picks
for st in ("P1 2-4", "P1 5-10", "P2"):
    sub = [r for r in rows if r["stage"] == st and r["Aoff"] and r["best_on"] is not None and r["sa"] - r["best_on"][0] < 4]
    if sub: print(f"{st}: engine OFF top with an ON candidate within 4 points: {len(sub)} picks; the ON card's GIH is {np.mean([gih(r['best_on'][1]) - gih(r['A']) for r in sub]):+.2f} vs the engine's; human took the ON candidate {np.mean([r['h']==r['best_on'][1] for r in sub]):.1%}, the engine's card {np.mean([r['h']==r['A'] for r in sub]):.1%}")
