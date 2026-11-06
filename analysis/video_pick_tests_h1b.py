"""H1b (mqZ-16, mqZ-48), deck level. Question: does a deck's win rate rise per extra fixing land, less so beyond five fixers, more so with more splash cards?
Fixed before any outcome was looked at:
  decks     decks_FRA.json.gz, two-colour core lane, >= 1 game, >= 20 spells with a GIH WR (as foundations_deck POP1); all formats (C(format) control); all skills with skill as control, NO game-count term.
  fixers    dual / any-colour lands in the main deck (the 16 FRA lands pool_fixing counts for >= 2 colours), copies counted.
  splash cards  spells in the deck with >= 1 pip outside the lane's two colours, copies counted: 0 / 1 / 2+.
  model     binomial GLM on deck win rate, weights = deck games, SEs clustered by draft; controls skill, lane, format, quality (mean lane GIH WR of the spells), total lands (15-18 cells).
            Effects in WR points at 55% (24.75 x coefficient).
  M1        piecewise slope per fixer: 0-2 (fa = min(f,2)), 3-5 (fb = clip(f-2,0,3)), 6+ (fc = max(f-5,0)).  PRIMARY (3): slope 6+ minus slope 3-5 (hypothesis < 0).
  M2        slope per fixer (min(f,5)) separately for splash cards 0 / 1 / 2+, splash bucket main effects.  PRIMARY (4): slope at 2+ splash cards minus slope at 0 (hypothesis > 0).
Post hoc (labelled): in-lane duals vs duals that touch an off-lane colour counted separately.
Run like video_pick_tests_h1a.py."""
import os, sys, gzip, json, zlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm, statsmodels.formula.api as smf
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R); sys.path.insert(0, f"{R}/analysis"); os.environ.setdefault("MTGA_COMPANION_DATA", f"{R}/data")
import video_counters as vc
from companion.advisor_v2.lanes import pool_fixing, off_pips
from video_pick_tests_common import *
OUT = f"{R}/analysis/out/video_pick_h1b.txt"; lines = []
def P(s=""): print(s, flush=True); lines.append(s)
K = 24.75; by = vc.cards()
gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
gL = lambda v, l: (v["deck_colors"].get(l) or {}).get("gihwr") or gA(v)
isl = lambda v: "Land" in v["types"]
fixl = {n for n, v in by.items() if isl(v) and "Basic" not in v["types"] and sum(pool_fixing([v]).values()) >= 2}
rows = []
for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
    lane = d["lane"]
    if len(lane) != 2 or d["deck_games"] < 1: continue
    q, f, fin, fout, sp, nl = [], 0, 0, 0, 0, 0
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if not v: continue
        if isl(v):
            nl += k
            if n in fixl:
                f += k; cols = pool_fixing([v]); touch = any(c not in lane and x > 0 for c, x in cols.items()); fout += k * touch; fin += k * (not touch)
            continue
        g = gL(v, lane)
        if g: q += [g] * k
        sp += k * (len(off_pips(v, lane)) > 0)
    if len(q) < 20: continue
    rows.append(dict(draft=d["draft"], fmt=d["format"], lane=lane, skill=d["skill"], y=d["deck_wins"] / d["deck_games"], g=d["deck_games"], f=f, fin=fin, fout=fout, sp=sp, q=np.mean(q), nl=int(np.clip(nl, 15, 18))))
D = pd.DataFrame(rows); D["half"] = D.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
D["fa"], D["fb"], D["fc"] = D.f.clip(upper=2), (D.f - 2).clip(0, 3), (D.f - 5).clip(lower=0)
D["f5"] = D.f.clip(upper=5); D["sb"] = D.sp.clip(upper=2)
for j in range(3): D[f"f5s{j}"] = D.f5 * (D.sb == j)
CTL = "skill + q + C(lane) + C(fmt) + C(nl)"
def fit(df, f):
    return smf.glm("y ~ " + f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
def est(m, w):
    nm = list(m.params.index); v = np.zeros(len(nm))
    for t, c in w.items(): v[nm.index(t)] = c
    return lin(m.params.values, m.cov_params().values, v)
def sub(k, h=None):
    s = D.skill; m = gmask(s.values, k); 
    if h is not None: m = m & (D.half.values == h)
    return D[m]
P(f"# H1b: deck win rate by fixers and splash cards (FRA, {len(D)} two-colour decks); WR points per extra fixer; * p<.05, ** beyond Bonferroni (p<.05/6) z>{ZB:.3f} (6 primaries)")
P("Decks by fixers x splash cards (all): " + "; ".join(f"splash {s}: " + "/".join(str(int(((D.sb == s) & fb).sum())) for fb in (D.f <= 2, (D.f >= 3) & (D.f <= 5), D.f >= 6)) + " (f0-2/3-5/6+)" for s in (0, 1, 2)))
P("Groups (decks): " + ", ".join(f"{k} {len(sub(k))}" for k in GR) + f"; fixers mean {D.f.mean():.2f}, 6+: {(D.f>=6).sum()}")
def scaled(e): return (e[0] * K, e[1] * K)
def block(title, formula, prim_w, hyp):
    P(f"\n## {title}")
    r = {}
    for k in GR:
        m = fit(sub(k), formula); r[k] = {lab: scaled(est(m, w)) for lab, w in prim_w.items()}
        P(f"  {k:5s} n={len(sub(k))}  " + " | ".join(f"{lab}: {fm(*r[k][lab], 2)}" for lab in prim_w))
    last = list(prim_w)[-1]; dg = diff(r["good"][last], r["weak"][last]); P(f"  good minus weak ({last}): {fm(*dg, 2)}")
    hv = []
    for h in (0, 1):
        rr = {k: scaled(est(fit(sub(k, h), formula), prim_w[last])) for k in ("all", "good")}; hv.append(rr)
        P(f"  half {h}: all {fm(*rr['all'], 2)} | good {fm(*rr['good'], 2)}")
    P(f"  halves difference (all) z = {(hv[0]['all'][0]-hv[1]['all'][0])/np.hypot(hv[0]['all'][1], hv[1]['all'][1]):+.2f}")
    e, se = r["all"][last]; lo, hi = e - 1.96 * se, e + 1.96 * se
    P(f"  smallest effect ruled out (all, 95%): {'above %+.2f' % hi if hyp == '+' else 'below %+.2f' % lo} WR points per fixer in the hypothesised direction ({'null' if abs(e/se) < 1.96 else 'NOT null'})")
block("M1 piecewise slope per fixer; PRIMARY contrast = slope 6+ minus slope 3-5 (hyp -)", f"fa + fb + fc + {CTL}", {"0-2": {"fa": 1}, "3-5": {"fb": 1}, "6+": {"fc": 1}, "6+ minus 3-5": {"fc": 1, "fb": -1}}, "-")
block("M2 slope per fixer (up to 5) by splash cards; PRIMARY contrast = 2+ minus 0 (hyp +)", f"f5s0 + f5s1 + f5s2 + C(sb) + {CTL}", {"0 splash": {"f5s0": 1}, "1": {"f5s1": 1}, "2+": {"f5s2": 1}, "2+ minus 0": {"f5s2": 1, "f5s0": -1}}, "+")
P("\n## Post hoc: in-lane duals vs duals touching an off-lane colour (one slope each, same controls, up to 5)")
D["fin5"], D["fout5"] = D.fin.clip(upper=5), D.fout.clip(upper=5)
for k in GR:
    m = fit(sub(k), f"fin5 + fout5 + {CTL}"); P(f"  {k:5s} in-lane {fm(*scaled(est(m, {'fin5': 1})), 2)} | off-touching {fm(*scaled(est(m, {'fout5': 1})), 2)}")
open(OUT, "w").write("\n".join(lines) + "\n")
