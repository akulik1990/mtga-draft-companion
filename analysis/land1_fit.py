"""Fit LAND_MAIN_DUAL_EDGE on train drafts (log from land1_replay.py, switch off) and the planeswalker split.
    uv run python analysis/land1_fit.py <train log> [out txt]
Route: own conditional logit (tools/choice_model.py has no main-dual feature): utility = b*score + g*maindual;
shift = g/b in score points. Then EDGE = the value of the floor (replacement+EDGE, applied after the engine's own
score scale) at which the model-free simulation reproduces the observed take rate: see edge_for_rate()."""
import gzip, json, re, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import video_draft as V  # noqa: E402

log = json.load(open(sys.argv[1]))
drafts = json.load(gzip.open(ROOT / "data/Temp/land1/train200.drafts.json.gz"))
took = {(d["id"], p[0], p[1]): p[2] for d in drafts for p in d["picks"]}
_, cards = V.load_set("FRA_PremierDraft_All_Data.json")
names = sorted({n for r in log for n, _ in r[3]} | {n for r in log for n in r[4]})
info = {c["name"].split(" // ")[0]: c for c in cards(names)}
info = {n: info.get(n.split(" // ")[0]) for n in names}

def is_land(n): return "Land" in (info[n].get("types") or [])
def nonbasic(n): return is_land(n) and "Basic" not in (info[n].get("types") or [])
def lcolours(n): return info[n].get("colors") or []
def pw_dual(n): return nonbasic(n) and len(lcolours(n)) >= 2 and "planeswalker" in (info[n].get("oracle_text") or "").lower()
def has_pw(pool): return any("Planeswalker" in (info[n].get("types") or []) for n in pool if n in info and info[n])

rows = []  # one per pick: dict
for r in log:
    did, pk, pi, ranked, pool, lanes, repl, top = r[:8]
    t = took.get((did, pk, pi))
    if not ranked or t is None: continue
    names_ = [n for n, _ in ranked]
    if t not in names_: continue
    lane = max(lanes, key=lanes.get) if lanes else ""
    sc = np.array([s for _, s in ranked], float)
    md = np.array([nonbasic(n) and len(lcolours(n)) >= 2 and all(c in lane for c in lcolours(n)) for n in names_])
    rows.append(dict(d=did, pk=pk, pi=pi, names=names_, sc=sc, md=md, y=names_.index(t), repl=repl, lane=lane,
                     pwd=np.array([pw_dual(n) for n in names_]), pw=has_pw(pool)))
print(len(rows), "picks;", sum(r["md"].any() for r in rows), "with a main-lane dual in the pack")

W = max(len(r["names"]) for r in rows)
def pad(f):
    return np.array([np.pad(f(r), (0, W - len(r["names"]))) for r in rows], float)
SC, MD = pad(lambda r: r["sc"] / 10), pad(lambda r: r["md"].astype(float))
MASK = pad(lambda r: np.ones(len(r["names"])))
Y = np.array([r["y"] for r in rows]); DID = np.array([r["d"] for r in rows])
def fit(w=None):
    w = np.ones(len(rows)) if w is None else w
    def f(th):
        u = np.where(MASK > 0, th[0] * SC + th[1] * MD, -1e9)
        m = u.max(1, keepdims=True); lse = m[:, 0] + np.log(np.exp(u - m).sum(1))
        p = np.exp(u - lse[:, None])
        n = rows_n = np.arange(len(rows))
        ll = u[n, Y] - lse
        g0 = SC[n, Y] - (p * SC).sum(1); g1 = MD[n, Y] - (p * MD).sum(1)
        return -(w * ll).sum(), -np.array([(w * g0).sum(), (w * g1).sum()])
    return minimize(f, np.array([0.5, 0.0]), jac=True, method="BFGS").x

th = fit()
sub = [r for r in rows if r["md"].any()]
# score-only beta from all picks, gamma from all picks jointly; shift in score points = 10*g/b
shift = 10 * th[1] / th[0]
rng = np.random.default_rng(0)
ids = sorted({r["d"] for r in rows}); by = {i: [r for r in rows if r["d"] == i] for i in ids}
boots = []
idx = {i: k for k, i in enumerate(ids)}; didx = np.array([idx[d] for d in DID])
for _ in range(300):
    cnt = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    b = fit(cnt[didx].astype(float)); boots.append(10 * b[1] / b[0])
lo, hi = np.percentile(boots, [2.5, 97.5])
out = [f"picks {len(rows)}, with main-lane dual {len(sub)}; beta (per 10 score pts) {th[0]:.3f}, gamma {th[1]:.3f}",
       f"shift in final-score points = {shift:.2f} (95% CI {lo:.2f} to {hi:.2f}, 300 bootstrap draws over drafts)"]
# observed vs model take rate of main duals (opportunities), no shift
opp = tk = 0; exp0 = 0.0
for r in rows:
    u = th[0] * r["sc"] / 10; p = np.exp(u - np.logaddexp.reduce(u))
    for i in np.where(r["md"])[0]:
        opp += 1; tk += (i == r["y"]); exp0 += p[i]
out.append(f"main-lane dual opportunities {opp}: taken {tk} ({100*tk/opp:.1f}%), score-only model expects {exp0:.1f} ({100*exp0/opp:.1f}%)")
# rank-based view: how far below the pack's best the duals score
gaps = [r["sc"][0] - r["sc"][i] for r in rows for i in np.where(r["md"])[0]]
out.append(f"main dual score gap to pack #1: median {np.median(gaps):.1f}, mean {np.mean(gaps):.1f}; replacement median {np.median([r['repl'] for r in rows]):.1f}")
# planeswalker split (descriptive)
for lab, flt in (("planeswalker dual, any lane", lambda r, i: r["pwd"][i]),):
    for pw in (False, True):
        o = t_ = 0; e = 0.0; cl = {}
        for r in rows:
            if r["pw"] != pw: continue
            u = th[0] * r["sc"] / 10; p = np.exp(u - np.logaddexp.reduce(u))
            for i in np.where(r["pwd"])[0]:
                o += 1; t_ += (i == r["y"]); e += p[i]; cl.setdefault(r["d"], []).append((1, int(i == r["y"]), p[i]))
        ks = list(cl)
        bs = []
        for _ in range(500):
            s = rng.choice(len(ks), len(ks)); a = np.array([x for j in s for x in cl[ks[j]]]); bs.append(a[:, 1].sum() / max(1, a[:, 0].sum()))
        l, h = np.percentile(bs, [2.5, 97.5])
        out.append(f"{lab}, planeswalker in pool={pw}: opportunities {o}, taken {t_} ({100*t_/max(o,1):.1f}%, CI {100*l:.1f}-{100*h:.1f}), score-model expects {100*e/max(o,1):.1f}%")
print("\n".join(out))
Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "analysis/out/land_fit.txt").write_text("\n".join(out) + "\n")
