"""H1a (mqZ-16, mqZ-48), picks. Question: do drafters take a fixing land over a comparable playable more when the pool already holds splash-worthy
off-colour cards, and less once they hold six or more fixers?  Fixed before any outcome was looked at (design in docs/video-test-picks-FRA.md):
  data      17Lands FRA PremierDraft picks (56,475 complete drafts), pick_number >= 5 (pool of 5+ cards; zero-indexed pick_number 5 = 6th pick).
  lane      the pool's top pair at the pick (choice.top_lane rule: summed GIH surplus over 50 of castable spells).
  fixer     a dual / any-colour LAND (pool_fixing counts >= 2 colours: 15 duals + Room of Refuge) that taps a lane colour and makes a colour outside the lane.
  playable  spell castable in the lane (no off pip) with All-Decks GIH WR within +-2.0 of the best fixer in the pack.
  choice set  the useful fixers and the comparable playables in the pack; picks with both present and the pick inside the set (others are dropped).
  splash held   cards in the pool with exactly one off-lane pip and lane GIH WR >= 56.5 (the builder's floor): cells 0 / 1 / 2+.
  fixers held   dual / any-colour lands in the pool: cells 0-2 / 3-5 / 6+.
  model     conditional logit, utility = b_gih*GIH + fixer*(a + s1 + s2+ + f3-5 + f6+ + pack2 + pack3 + picks 10-14); SEs clustered by draft.
  primary   (1) splash 2+ minus 0 (coefficient s2+), good group;  (2) fixers 6+ minus 3-5 (f6+ - f3-5), good group.   Bonferroni over 6 primaries in the three H-scripts.
Post hoc (labelled): fixer makes a colour a held splash card needs ("match"); cell table.
Run: systemd-run --user --scope -q -p MemoryMax=4G nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_pick_tests_h1a.py"""
import os, sys, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f"{R}/analysis")
from video_pick_tests_common import *
d = np.load(f"{R}/data/Temp/vid1106p/h1.npz"); OUT = f"{R}/analysis/out/video_pick_h1a.txt"; lines = []
def P(s=""): print(s, flush=True); lines.append(s)
keep = ~np.isnan(d["skill"]); d = {k: d[k][keep] for k in d.files if k != "cov"}
f = d["isfix"].astype(float); s1 = (d["splash"] == 1) * 1.0; s2 = (d["splash"] >= 2) * 1.0; f35 = ((d["fixheld"] >= 3) & (d["fixheld"] <= 5)) * 1.0; f6 = (d["fixheld"] >= 6) * 1.0
pk2, pk3, late = (d["pack"] == 1) * 1.0, (d["pack"] == 2) * 1.0, (d["pick"] >= 9) * 1.0
base = np.column_stack([d["gih"], f, f * s1, f * s2, f * f35, f * f6, f * pk2, f * pk3, f * late]); NAMES = ["gih", "fixer", "s1", "s2+", "f3-5", "f6+", "pack2", "pack3", "pick10+"]
post = np.column_stack([base, f * d["match"]]); PN = NAMES + ["match"]
def fit(mask, X):
    b, V = clogit(X[mask], d["gid"][mask], d["chosen"][mask], d["draft"][mask]); return b, V
def contrasts(b, V):
    k = len(b); w1 = np.zeros(k); w1[3] = 1; w2 = np.zeros(k); w2[5] = 1; w2[4] = -1; return lin(b, V, w1), lin(b, V, w2)
P("# H1a: fixer vs comparable playable (FRA PremierDraft picks), log-odds of the fixer's utility; * p<.05, ** beyond Bonferroni (p<.05/6) (z>%.3f, 6 primaries)" % ZB)
P("Cells (choice sets / picks; fixer taken share) per group, splash held x fixers held:")
first = np.r_[True, d["gid"][1:] != d["gid"][:-1]]
for k in GR:
    m = gmask(d["skill"], k)
    sets = first & m; fx = (d["isfix"] & d["chosen"]) & m
    P(f"  {k}: sets {int(sets.sum())}, drafts {len(np.unique(d['draft'][m]))}, fixer taken {fx.sum()/sets.sum():.3f}")
    if k == "good" or k == "all":
        for sl, sm in (("0", d["splash"] == 0), ("1", d["splash"] == 1), ("2+", d["splash"] >= 2)):
            row = []
            for fl, fmk in (("0-2", d["fixheld"] <= 2), ("3-5", (d["fixheld"] >= 3) & (d["fixheld"] <= 5)), ("6+", d["fixheld"] >= 6)):
                c = sets & sm & fmk; row.append(f"f{fl}: n={int(c.sum())}, taken {(fx & sm & fmk).sum()/max(1,c.sum()):.3f}")
            P(f"     splash {sl}: " + "; ".join(row))
res = {}
P("\n## Coefficients by group (all terms; fixer terms are log-odds shifts for a fixer vs a playable)")
for k in GR:
    b, V = fit(gmask(d["skill"], k), base); res[k] = (b, V); se = np.sqrt(np.diag(V))
    P(f"[{k}] " + "  ".join(f"{n}={b[i]:+.3f}({se[i]:.3f})" for i, n in enumerate(NAMES)))
P("\n## Primary contrasts (estimate [95% CI] z; OR = exp(est))")
for lab, i, hyp in (("C1 splash 2+ minus 0 (hyp +)", 0, "+"), ("C2 fixers 6+ minus 3-5 (hyp -)", 1, "-")):
    P(f"\n{lab}")
    cs = {k: contrasts(*res[k])[i] for k in GR}
    for k in GR: P(f"  {k:5s} {fm(*cs[k])}  OR {np.exp(cs[k][0]):.2f}")
    dg = diff(cs["good"], cs["weak"]); P(f"  good minus weak {fm(*dg)}")
    for h in (0, 1):
        r = {}
        for k in ("all", "good"):
            b, V = fit(gmask(d["skill"], k) & (np.load(f"{R}/data/Temp/vid1106p/h1.npz")["half"][keep] == h), base); r[k] = contrasts(b, V)[i]
        P(f"  half {h}: all {fm(*r['all'])} | good {fm(*r['good'])}")
    e, se = cs["good"]; lo, hi = e - 1.96 * se, e + 1.96 * se
    P(f"  smallest effect ruled out (good, 95%): {'effects above %+.3f (OR %.2f)' % (hi, np.exp(hi)) if hyp == '+' else 'effects below %+.3f (OR %.2f)' % (lo, np.exp(lo))} in the hypothesised direction")
P("\n## Post hoc: add 'fixer makes a colour a held splash card needs' (match)")
for k in GR:
    b, V = fit(gmask(d["skill"], k), post); se = np.sqrt(np.diag(V)); P(f"[{k}] " + "  ".join(f"{n}={b[i]:+.3f}({se[i]:.3f})" for i, n in enumerate(PN) if n in ("s1", "s2+", "f3-5", "f6+", "match")))
open(OUT, "w").write("\n".join(lines) + "\n")
