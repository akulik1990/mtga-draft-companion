"""Train check by kind of the player's pick (spell / dual / other nonbasic / basic): top-1 and top-3, base vs new replay log, clustered bootstrap over drafts.
Also: what v2 picked instead where it lost (new #1 kind) and non-land losses by the base pick's kind.
    uv run python analysis/land2_picks.py <replay_base.jsonl> <replay_new.jsonl> <drafts.json.gz>"""
import gzip, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import video_draft as V  # noqa: E402
took = {(d["id"], p[0], p[1]): p[2] for d in json.load(gzip.open(sys.argv[3], "rt")) for p in d["picks"]}
logs = [{(r[0], r[1], r[2]): r for r in json.load(open(p))} for p in sys.argv[1:3]]
names = sorted({n for r in logs[0].values() for n, _ in r[3]})
_, cards = V.load_set("FRA_PremierDraft_All_Data.json")
info = {c["name"].split(" // ")[0]: c for c in cards(names)}
def kind(n):
    c = info[n.split(" // ")[0]]; t = c.get("types") or []; x = (c.get("oracle_text") or "").lower()
    if "Land" not in t: return "spell"
    if "Basic" in t: return "basic"
    return "dual" if re.search(r"add \{[wubrg]\} or \{[wubrg]\}", x) else "otherland"
rows = []; lost = Counter(); newk = Counter()
for key, t in took.items():
    a, b = logs[0].get(key), logs[1].get(key)
    if not a or not b or not a[3] or t not in dict(a[3]): continue
    ra, rb = [n for n, _ in a[3]], [n for n, _ in b[3]]
    kt = kind(t)
    rows.append((key[0], kt, ra[0] == t, rb[0] == t, t in ra[:3], t in rb[:3]))
    if ra[0] == t and rb[0] != t: lost[(kt, kind(rb[0]))] += 1
    if ra[0] != rb[0]: newk[(kind(ra[0]), kind(rb[0]))] += 1
ids = sorted({r[0] for r in rows}); ix = {i: k for k, i in enumerate(ids)}
D = np.array([ix[r[0]] for r in rows]); K = np.array([r[1] for r in rows]); M = np.array([r[2:] for r in rows], float)
rng = np.random.default_rng(2); W = [np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(1000)]
def line(label, mask):
    s = []
    for nm, ca, cb in (("top-1", 0, 1), ("top-3", 2, 3)):
        a, b = M[mask, ca].mean(), M[mask, cb].mean()
        d = [((w[D] * mask) * (M[:, cb] - M[:, ca])).sum() / max((w[D] * mask).sum(), 1) for w in W]
        l, h = np.percentile(d, [2.5, 97.5])
        s.append(f"{nm} {100*a:.1f} -> {100*b:.1f} (diff {100*(b-a):+.2f}, CI {100*l:+.2f} to {100*h:+.2f})")
    print(f"{label} (n={int(mask.sum())}): " + "; ".join(s))
line("all picks", np.ones(len(rows), bool))
for k in ("spell", "dual", "otherland", "basic"): line(f"player took {k}", K == k)
print("base #1 right, new #1 wrong: (player kind, new #1 kind) ->", dict(lost))
print("all #1 changes (base kind, new kind) ->", dict(newk))
