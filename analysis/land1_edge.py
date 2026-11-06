"""EDGE from engine replays of the 200 train drafts (arms: off, e4, e8, e12). Beta from the off arm (score-only logit),
held fixed; per arm the model's expected take rate of main-lane duals (main-dual mask from the off arm) and the share
where the dual is v2's #1. EDGE = interpolated value where the expected rate equals the observed rate; bootstrap over drafts.
    uv run python analysis/land1_edge.py"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "data/Temp/land1"
sys.argv = [sys.argv[0], str(L / "train200.off.json"), "/dev/null"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(ROOT / "analysis/land1_fit.py").read().split("# EDGE: floor")[0])   # reuse rows/info/MD/Y/th (beta, gamma)
beta = th[0]
EDG = [0, 4, 8, 12, 24, 36]
arms = {0: {(r["d"], r["pk"], r["pi"]): r for r in rows}}
for e in EDG[1:]:
    lg = {(r[0], r[1], r[2]): r for r in json.load(open(L / f"train200.e{e}.json"))}
    arms[e] = lg
keys = [(r["d"], r["pk"], r["pi"]) for r in rows]
def arm_stats(e):
    exp = np.zeros(len(rows)); top = np.zeros(len(rows)); hit = np.zeros(len(rows))
    for k, (key, r) in enumerate(zip(keys, rows)):
        if not r["md"].any(): continue
        if e == 0: nm, sc = r["names"], r["sc"]
        else: nm, sc = [n for n, _ in arms[e][key][3]], np.array([s for _, s in arms[e][key][3]])
        u = beta * sc / 10; p = np.exp(u - np.logaddexp.reduce(u)); idx = [nm.index(r["names"][i]) for i in np.where(r["md"])[0]]
        exp[k] = p[idx].sum(); top[k] = (np.argmax(sc) in idx); hit[k] = r["md"][r["y"]]
    return exp, top, hit
st = {e: arm_stats(e) for e in EDG}
mdn = MD.sum(1)
def rate(e, w, j): return (w * st[e][j]).sum() / (w * mdn).sum()
def solve(w):
    obs = (w * st[0][2]).sum() / (w * mdn).sum()
    ex = np.array([rate(e, w, 0) for e in EDG]); return np.interp(obs, ex, EDG) if obs <= ex.max() else np.nan, obs, ex
rng = np.random.default_rng(1); ids = sorted(set(DID)); idx_ = {i: k for k, i in enumerate(ids)}; di = np.array([idx_[d] for d in DID])
e0, obs, ex = solve(np.ones(len(rows)))
bs = [solve(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[di].astype(float))[0] for _ in range(300)]
lo, hi = np.nanpercentile(bs, [2.5, 97.5])
out = [f"observed main-dual take rate {100*obs:.1f}%  (opportunities {int(mdn.sum())})",
       "arm EDGE: expected take rate (score-only logit, beta fixed) / share where v2's #1 is a main dual: " + ", ".join(f"{e}: {100*rate(e,np.ones(len(rows)),0):.1f}% / {100*rate(e,np.ones(len(rows)),1):.1f}%" for e in EDG),
       f"EDGE by interpolation: {e0:.2f} (95% CI {lo:.2f} to {hi:.2f}; {int(np.isnan(bs).sum())} of 300 draws above the largest arm)"]
print("\n".join(out)); open(ROOT / "analysis/out/land_fit.txt", "a").write("\n".join(out) + "\n")
