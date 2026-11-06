"""GIH_SLOPE held-out. Design in docs/gih-slope-heldout-FRA.md. Output analysis/out/gih_slope_heldout.txt.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/gih_slope_heldout.py"""
import json, os, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{R}/analysis"); import video_counters as vc
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
rng = np.random.default_rng(0)

by = vc.cards()
SPELL = {n for n, v in by.items() if "Land" not in v["types"]}
import csv, gzip
SRC = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
# Streamed, never a whole DataFrame (memory cap). Counters per half h, quarter q, lane (row 0 = All Decks): games in hand and wins in hand.
f = gzip.open(SRC, "rt", newline=""); rd = csv.reader(f); hdr = next(rd); col = {c: i for i, c in enumerate(hdr)}
names = [c[5:] for c in hdr if c.startswith("deck_") and c[5:].split(" // ")[0] in SPELL and f"opening_hand_{c[5:]}" in col and f"drawn_{c[5:]}" in col]
NC = len(names); ci = {k: col[k] for k in ("draft_id", "build_index", "main_colors", "splash_colors", "won", "user_game_win_rate_bucket", "user_n_games_bucket")}
iOH = [col[f"opening_hand_{c}"] for c in names]; iDR = [col[f"drawn_{c}"] for c in names]; iDK = [col[f"deck_{c}"] for c in names]
del hdr, col
LANES = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]; LI = {l: i + 1 for i, l in enumerate(LANES)}
CN = np.zeros((2, 2, len(LANES) + 1, NC), np.int64); CW = np.zeros_like(CN)
rec = {}   # deck key -> [draft, lane, splash, skill, bucket, wins, games, cards idx, copies, half, q]
ngames = 0; drafts = set(); hcache = {}
for row in rd:
    ngames += 1
    dr = row[ci["draft_id"]]
    if dr not in hcache: hh = zlib.crc32(dr.encode()); hcache[dr] = (hh % 2, (hh // 2) % 2)
    h, q = hcache[dr]; w = 1 if row[ci["won"]] in ("True", "1", "true") else 0
    ln = row[ci["main_colors"]]
    ih = np.fromiter((int(row[a] or 0) + int(row[b] or 0) > 0 for a, b in zip(iOH, iDR)), bool, NC)
    CN[h, q, 0] += ih; CW[h, q, 0] += ih * w
    if ln in LI: CN[h, q, LI[ln]] += ih; CW[h, q, LI[ln]] += ih * w
    k = (dr, row[ci["build_index"]])
    r = rec.get(k)
    if r is None:
        dk = np.fromiter((int(row[a] or 0) for a in iDK), np.int16, NC); nz = np.nonzero(dk)[0]
        sk = row[ci["user_game_win_rate_bucket"]]
        r = rec[k] = [dr, ln, int(bool(row[ci["splash_colors"]])), float(sk) if sk else np.nan, int(float(row[ci["user_n_games_bucket"]] or 0)), 0, 0,
                      nz.astype(np.int16), dk[nz], h, q]
    r[5] += w; r[6] += 1
f.close()
P(f"games {ngames}, decks {len(rec)}, spell columns {NC}")
D = pd.DataFrame([x[:7] + x[9:] for x in rec.values()], columns=["draft_id", "lane", "splash", "skill", "user_n_games_bucket", "w", "n", "half", "q"])
D["y"] = D.w / D.n; Dk = [(x[7], x[8]) for x in rec.values()]; D["_row"] = np.arange(len(D)); del rec

def ratings(hs, qs=(0, 1), minn=100):
    """lane -> percent GIH WR over cards (nan below minn), 'All' fallback; from rating halves hs / quarters qs."""
    n = CN[list(hs)][:, list(qs)].sum((0, 1)); w = CW[list(hs)][:, list(qs)].sum((0, 1))
    pr = np.where(n >= minn, 100 * w / np.maximum(n, 1), np.nan)
    out = {"All": pr[0]}
    for l in LANES: out[l] = pr[LI[l]]
    return out

def quality(rt, rows, lanes):
    out = np.full(len(rows), np.nan)
    for i, (r, l) in enumerate(zip(rows, lanes)):
        v = rt.get(l); a = np.where(np.isnan(v), rt["All"], v) if v is not None else rt["All"]
        ix, k = Dk[r]; x = a[ix]; ok = ~np.isnan(x)
        out[i] = (x[ok] * k[ok]).sum() / k[ok].sum() if ok.any() else np.nan
    return out

def ame(res, d, var="m"):
    p = res.predict(d); b = res.params[var]
    f = np.average(p * (1 - p), weights=d.n)
    return 100 * b * f, 100 * (b - 1.96 * res.bse[var]) * f, 100 * (b + 1.96 * res.bse[var]) * f

def fit(d):
    d = d.dropna(subset=["m", "skill"]).copy(); d["dg"] = pd.factorize(d.draft_id)[0]
    r = smf.glm("y ~ m + skill + C(lane) + splash", d, family=sm.families.Binomial(), freq_weights=d.n).fit(cov_type="cluster", cov_kwds={"groups": d.dg})
    s, lo, hi = ame(r, d)
    d1, d0 = d.copy(), d.copy(); d1["splash"] = 1; d0["splash"] = 0
    f = lambda dd: np.average(r.predict(dd), weights=d.n)
    sp = 100 * (f(d1) - f(d0))
    # CI for splash by scaling its coefficient CI
    sc = sp / r.params["splash"]
    return dict(slope=s, lo=lo, hi=hi, splash=sp, splo=sc * (r.params["splash"] - 1.96 * r.bse["splash"]), sphi=sc * (r.params["splash"] + 1.96 * r.bse["splash"]),
                n=len(d), drafts=d.dg.nunique(), beta=r.params["m"], se=r.bse["m"]), d

def resid(d, col):
    X = pd.get_dummies(d[["skill", "lane", "splash"]], columns=["lane"], drop_first=True).astype(float); X["c"] = 1.0
    b = np.linalg.lstsq(X.values, d[col].values, rcond=None)[0]; return d[col].values - X.values @ b

def reliability(rate_half, deck_half, pop, minn):
    """conditional split-half reliability of m, ratings from the two quarters of rate_half, decks in deck_half; Spearman-Brown to the half."""
    d = pop[pop.half == deck_half].copy()
    r1 = ratings([rate_half], (0,), minn); r2 = ratings([rate_half], (1,), minn)
    d["m1"] = quality(r1, d._row.values, d.lane.values); d["m2"] = quality(r2, d._row.values, d.lane.values)
    d = d.dropna(subset=["m1", "m2", "skill"]).reset_index(drop=True)
    e1, e2 = resid(d, "m1"), resid(d, "m2")
    c = lambda idx: np.corrcoef(e1[idx], e2[idx])[0, 1]
    rq = c(np.arange(len(d))); rh = 2 * rq / (1 + rq)
    # bootstrap over drafts
    dr = d.draft_id.values; u = pd.Series(np.arange(len(d))).groupby(dr).apply(np.array).tolist(); bs = []
    for _ in range(200):
        idx = np.concatenate([u[i] for i in rng.integers(0, len(u), len(u))]); q = c(idx); bs.append(2 * q / (1 + q))
    raw_c = np.corrcoef(d.m1, d.m2)[0, 1]
    return rh, np.percentile(bs, [2.5, 97.5]), rq, 2 * raw_c / (1 + raw_c), len(d)

def run(label, popfilter, minn=100, rel=True):
    P(f"\n=== {label} (threshold {minn}) ===")
    pop = D[popfilter].copy(); P(f"decks {len(pop)}, games {int(pop.n.sum())}, mean y {np.average(pop.y, weights=pop.n):.4f}")
    res = {}
    for rate_h in (0, 1):
        rt = ratings([rate_h], (0, 1), minn)
        for deck_h, tag in ((1 - rate_h, "held-out"), (rate_h, "in-sample")):
            d = pop[pop.half == deck_h].copy(); d["m"] = quality(rt, d._row.values, d.lane.values)
            f, dd = fit(d); res[(rate_h, tag)] = f
            P(f"ratings {'AB'[rate_h]} -> decks {'AB'[deck_h]} {tag:9s}: slope {f['slope']:.3f} [{f['lo']:.3f}, {f['hi']:.3f}] WR pts per GIH pt; splash {f['splash']:+.2f} [{f['splo']:+.2f}, {f['sphi']:+.2f}]; "
              f"decks {f['n']}, drafts {f['drafts']}, sd(m) {dd.m.std():.2f}")
    out = {}
    for tag in ("held-out", "in-sample"):
        a, b = res[(0, tag)], res[(1, tag)]
        # inverse-variance mean of the two directions (independent draft sets)
        wa, wb = 1 / a["se"] ** 2, 1 / b["se"] ** 2
        sa, sb = (a["hi"] - a["lo"]) / 3.92, (b["hi"] - b["lo"]) / 3.92
        wa, wb = 1 / sa ** 2, 1 / sb ** 2; s = (a["slope"] * wa + b["slope"] * wb) / (wa + wb); se = (wa + wb) ** -.5
        spa, spb = (a["sphi"] - a["splo"]) / 3.92, (b["sphi"] - b["splo"]) / 3.92
        w1, w2 = 1 / spa ** 2, 1 / spb ** 2; sp = (a["splash"] * w1 + b["splash"] * w2) / (w1 + w2); spse = (w1 + w2) ** -.5
        out[tag] = dict(slope=s, lo=s - 1.96 * se, hi=s + 1.96 * se, splash=sp, splo=sp - 1.96 * spse, sphi=sp + 1.96 * spse)
        P(f"POOLED {tag:9s}: slope {s:.3f} [{s - 1.96 * se:.3f}, {s + 1.96 * se:.3f}]; splash {sp:+.2f} [{sp - 1.96 * spse:+.2f}, {sp + 1.96 * spse:+.2f}]; "
          f"directions {a['slope']:.2f} / {b['slope']:.2f} ({'overlap' if max(a['lo'], b['lo']) <= min(a['hi'], b['hi']) else 'DISAGREE'})")
    if rel:
        rs = []
        for rate_h in (0, 1):
            r = reliability(rate_h, 1 - rate_h, pop, minn); rs.append(r)
            P(f"reliability ratings {'AB'[rate_h]} -> decks {'AB'[1 - rate_h]}: conditional {r[0]:.3f} [{r[1][0]:.3f}, {r[1][1]:.3f}] (quarter corr {r[2]:.3f}; unconditional SB {r[3]:.3f}; decks {r[4]})")
            ho = res[(rate_h, "held-out")]
            P(f"   disattenuated slope {ho['slope'] / r[0]:.3f} [{ho['lo'] / r[0]:.3f}, {ho['hi'] / r[0]:.3f}]; with reliability CI ends {ho['slope'] / r[1][1]:.3f} .. {ho['slope'] / r[1][0]:.3f} (point)")
        lam = np.mean([r[0] for r in rs]); o = out["held-out"]
        P(f"POOLED reliability {lam:.3f}; disattenuated slope {o['slope'] / lam:.3f} [{o['lo'] / lam:.3f}, {o['hi'] / lam:.3f}]")
        P(f"SPLASH_TERM 4.07 in GIH points: at 1.72 {4.07 / 1.72:.2f}; at held-out slope {4.07 / o['slope']:.2f}; at disattenuated {4.07 / (o['slope'] / lam):.2f}; at in-sample {4.07 / out['in-sample']['slope']:.2f}")
        P(f"model splash cost {o['splash']:+.2f} WR pts = {-o['splash'] / o['slope']:.2f} GIH pts at held-out slope, {-o['splash'] / (o['slope'] / lam):.2f} at disattenuated")
        out["lam"] = lam
    return out

two = (D.lane.str.len() == 2)
if __name__ == "__main__":
    P(f"decks total {len(D)}; two-colour {int(two.sum())}; 100+ players {int((D.user_n_games_bucket >= 100).sum())}")
    HEAD = two & (D.user_n_games_bucket >= 100)
    run("HEADLINE: two-colour, 100+ games players, lane rating", HEAD, 100)
    run("sens: threshold 30", HEAD, 30)
    run("sens: threshold 300", HEAD, 300)
    run("sens: all skill-bucket players (skill bucket contains own games, see doc)", two, 100, rel=False)
    run("sens: two- and three-colour mains, 100+ players (lane[:2] for 3-colour is not the builder's pair, so not used for lookup beyond All Decks)", (D.lane.str.len() >= 2) & (D.user_n_games_bucket >= 100), 100, rel=False)
    open(f"{R}/analysis/out/gih_slope_heldout.txt", "w").write("\n".join(lines) + "\n")
