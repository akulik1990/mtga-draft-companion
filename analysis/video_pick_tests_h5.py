"""H5 (mqZ-23): do late gold cards of a pair signal that the pair is open less reliably than late mono cards?  Fixed before any outcome was looked at:
  unit      one row per draft x pair XY (10 pairs), 56,475 complete drafts; skill NaN dropped.
  exposure  pack 1, picks 5-9 (zero-indexed rows 4-8, the packs the drafter saw): g = gold-XY cards seen, m = mono cards of X or Y seen, both counted only
            in the GIH band [55, 60) (All Decks), so gold and mono are of similar quality (band mean GIH printed). Pick 9 is the wheel of the drafter's first pack
            (sensitivity: picks 5-8 only, g7/m7).
  outcome i   later availability: band cards castable in the pair (gold XY + mono X + mono Y) in pack 3, picks 2-8 (zero-indexed 1-7; pick 9 is a wheel of
            the drafter's own first pack and is left out); same direction of passing as pack 1. Control: the pair's band cards in the drafter's own fresh pack 3.
  outcome ii  the drafter's final pool (45 cards) has this pair as its top pair (choice.top_lane rule), linear probability, in percentage points.
  model     y ~ pair fixed effects + g + m + c4 (summed GIH surplus of the pair's castable cards in the pool after 4 picks) [+ fresh-pack control for i]; SEs clustered by draft.
  primary   per-card gold effect minus per-card mono effect, (i) and (ii), all drafters. Hypothesis: gold - mono < 0.
  placebo   pack 2 (passed from the other neighbour) availability: pack-1 exposure should not predict it.
  secondary pack 1 picks 10-14 availability; gold-only / mono-only availability in pack 3 (mechanism); decks_FRA lane not used (a subset of drafts).
Run like video_pick_tests_h1a.py."""
import os, sys, numpy as np, json
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f"{R}/analysis")
from video_pick_tests_common import *
d = np.load(f"{R}/data/Temp/vid1106p/h5.npz"); OUT = f"{R}/analysis/out/video_pick_h5.txt"; lines = []
def P(s=""): print(s, flush=True); lines.append(s)
ok = ~np.isnan(d["skill"]); D = {k: d[k][ok] for k in d.files if k != "ids"}; nd = ok.sum(); L = 10
D["did"] = D["av3"] - D["av2"]; D["frd"] = D["fr3"] - D["fr2"]
skill, half = D["skill"], D["half"]; cl = np.repeat(np.arange(nd), L)
FE = np.tile(np.eye(L), (nd, 1))
flat = lambda a: a.reshape(-1)
def run(outcome, g, m, ctrl, sel, scale=1.0):
    X = np.column_stack([FE[np.repeat(sel, L)], flat(D[g][sel]), flat(D[m][sel]), flat(D["c4"][sel])] + ([flat(D[ctrl][sel])] if ctrl else []))
    y = flat(D[outcome][sel]); b, V = ols(X, y, cl[np.repeat(sel, L)])
    return b, V, L
def contrast(b, V):
    k = len(b); wg = np.zeros(k); wg[L] = 1; wm = np.zeros(k); wm[L + 1] = 1; return lin(b, V, wg), lin(b, V, wm), lin(b, V, wg - wm)
os.environ.setdefault("MTGA_COMPANION_DATA", f"{R}/data"); sys.path.insert(0, R)
import video_counters as vc
_by = vc.cards(); _g = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
_sp = [v for v in _by.values() if "Land" not in v["types"] and 55 <= _g(v) < 60]
_gold = [_g(v) for v in _sp if len(v["colors"]) == 2]; _mono = [_g(v) for v in _sp if len(v["colors"]) == 1]
BAND_INFO = f"band [55,60): gold cards {len(_gold)} (mean GIH {np.mean(_gold):.2f}), mono cards {len(_mono)} (mean GIH {np.mean(_mono):.2f})"
P(f"# H5: gold vs mono late signals (FRA PremierDraft), drafters {nd}; units: cards seen per extra card in exposure (i), percentage points (ii); * p<.05, ** beyond Bonferroni (p<.05/6) z>{ZB:.3f} (6 primaries)")
P(BAND_INFO)
P(f"mean exposure per draft x pair: gold {D['g'].mean():.3f}, mono {D['m'].mean():.3f}; mean pair availability pack 3: {D['av3'].mean():.3f}; P(end in pair) {D['end'].mean():.3f}")
spec = [("i  availability pack 3 picks 2-8 (PRIMARY)", "av3", "g", "m", "fr3", 1.0, True),
        ("ii ends in the pair, pp (PRIMARY)", "end", "g", "m", None, 100.0, True),
        ("placebo: availability pack 2 picks 2-8 (expect 0)", "av2", "g", "m", "fr2", 1.0, False),
        ("secondary: availability pack 1 picks 10-14", "av1l", "g", "m", "fr1l", 1.0, False),
        ("secondary: gold-only availability pack 3", "avg3", "g", "m", "fr3", 1.0, False),
        ("secondary: mono-only availability pack 3", "avm3", "g", "m", "fr3", 1.0, False),
        ("sensitivity (i), exposure picks 5-8", "av3", "g7", "m7", "fr3", 1.0, False),
        ("sensitivity (ii), exposure picks 5-8", "end", "g7", "m7", None, 100.0, False),
        ("POST HOC (i) exposure counted only for cards seen later than their ALSA", "av3", "gl", "ml", "fr3", 1.0, False),
        ("POST HOC (ii) exposure counted only for cards seen later than their ALSA", "end", "gl", "ml", None, 100.0, False),
        ("POST HOC (ii) net of the pair's surplus the drafter took in picks 5-9 (post-treatment control c9)", "end", "g", "m", "c9", 100.0, False),
        ("POST HOC (i) difference-in-differences: pack 3 minus pack 2 availability (removes the table / time environment)", "did", "g", "m", "frd", 1.0, False)]
for lab, out, g, m, ctrl, sc, prim in spec:
    P(f"\n## {lab}")
    res = {}
    for k in GR:
        sel = gmask(skill, k); b, V, _ = run(out, g, m, ctrl, sel); res[k] = contrast(b, V)
        P(f"  {k:5s} n={int(sel.sum())}  gold {fm(*res[k][0], 4, sc)} | mono {fm(*res[k][1], 4, sc)} | gold-mono {fm(*res[k][2], 4, sc)}")
    dg = diff(res["good"][2], res["weak"][2]); P(f"  good minus weak (gold-mono): {fm(*dg, 4, sc)}")
    hv = []
    for h in (0, 1):
        r = {}
        for k in ("all", "good"):
            sel = gmask(skill, k) & (half == h); b, V, _ = run(out, g, m, ctrl, sel); r[k] = contrast(b, V)[2]
        hv.append(r); P(f"  half {h}: all gold-mono {fm(*r['all'], 4, sc)} | good {fm(*r['good'], 4, sc)}")
    e, se = res["all"][2]
    P(f"  halves difference (all) z = {(hv[0]['all'][0]-hv[1]['all'][0])/np.hypot(hv[0]['all'][1], hv[1]['all'][1]):+.2f}")
    if prim: P(f"  smallest effect ruled out (all, 95%, hypothesis gold-mono < 0): gold-mono below {(e-1.96*se)*sc:+.4f} ({'null' if abs(e/se)<1.96 else 'NOT null'})")
open(OUT, "w").write("\n".join(lines) + "\n")
