"""Held-out check: 196 drafts (adv1 sp3 + two), switch off vs on. Logs from land1_replay.py:
data/Temp/land1/{sp3,two}.{off,on}.json. Match = v2's #1 is the player's card; clustered bootstrap over drafts.
    uv run python analysis/land1_heldout.py"""
import gzip, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import video_draft as V  # noqa: E402
L = ROOT / "data/Temp/land1"
ARM = sys.argv[1]  # e.g. e20
drafts = [d for n in ("sp3", "two") for d in json.load(gzip.open(ROOT / f"data/Temp/adv1/{n}.drafts.json.gz"))]
took = {(d["id"], p[0], p[1]): p[2] for d in drafts for p in d["picks"]}
logs = {a: {(r[0], r[1], r[2]): r for n in ("sp3", "two") for r in json.load(open(L / f"{n}.{a}.json"))} for a in ("off", ARM)}
names = sorted({n for r in logs["off"].values() for n, _ in r[3]})
_, cards = V.load_set("FRA_PremierDraft_All_Data.json")
info = {c["name"].split(" // ")[0]: c for c in cards(names)}
def kind(n):
    c = info[n.split(" // ")[0]]; t = c.get("types") or []; x = (c.get("oracle_text") or "").lower()
    if "Land" not in t: return "spell"
    if "Basic" in t: return "basic"
    return "dual" if re.search(r"add \{[wubrg]\} or \{[wubrg]\}", x) else "otherland"
rows = []
for key, t in took.items():
    a, b = logs["off"].get(key), logs[ARM].get(key)
    if not a or not b or not a[3] or t not in dict(a[3]): continue
    ra, rb = [n for n, _ in a[3]], [n for n, _ in b[3]]
    rows.append((key[0], kind(t), ra[0] == t, rb[0] == t, t in ra[:3], t in rb[:3]))
ids = sorted({r[0] for r in rows}); ix = {i: k for k, i in enumerate(ids)}
D = np.array([ix[r[0]] for r in rows]); K = np.array([r[1] for r in rows]); M = np.array([r[2:] for r in rows], float)
rng = np.random.default_rng(2); W = [np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(1000)]
def est(mask, col):
    m = mask.astype(float); out = []
    for w in W:
        ww = w[D] * m; out.append((ww * M[:, col]).sum() / max(ww.sum(), 1))
    return np.percentile(out, [2.5, 97.5])
def line(label, mask):
    n = int(mask.sum()); s = []
    for nm, ca, cb in (("top-1", 0, 1), ("top-3", 2, 3)):
        a, b = M[mask, ca].mean(), M[mask, cb].mean()
        d = [((w[D] * mask) * (M[:, cb] - M[:, ca])).sum() / max((w[D] * mask).sum(), 1) for w in W]
        l, h = np.percentile(d, [2.5, 97.5])
        s.append(f"{nm} {100*a:.1f} -> {100*b:.1f} (diff {100*(b-a):+.2f}, CI {100*l:+.2f} to {100*h:+.2f})")
    return f"{label} (n={n}): " + "; ".join(s)
allm = np.ones(len(rows), bool)
out = [f"{len(ids)} drafts, {len(rows)} picks", line("all picks", allm), line("player took a dual land", K == "dual"),
       line("player took any nonbasic land", (K == "dual") | (K == "otherland")), line("player took a non-land", K == "spell")]
print("\n".join(out)); open(ROOT / "analysis/out/land_fit.txt", "a").write(f"\nHeld-out, 196 drafts, switch off vs {ARM}\n" + "\n".join(out) + "\n")
