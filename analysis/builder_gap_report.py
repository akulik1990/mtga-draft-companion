"""Report for builder_gap_pools.py (design in its header). Bootstrap over drafts, 2000 resamples, seed 1.
Output: analysis/out/builder_gap_pools.txt"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in open(REPO / "data" / "Temp" / "gap1106" / "pool_gaps.jsonl")]
lines = []


def P(s=""):
    print(s)
    lines.append(s)


rng = np.random.default_rng(1)


def boot_diff(a, b, B=2000):
    """mean(a) - mean(b), a/b lists of (draft, value); resample drafts within each side."""
    def means(x):
        by = defaultdict(list)
        for d, v in x:
            by[d].append(v)
        keys = list(by)
        s = np.array([sum(by[k]) for k in keys])
        n = np.array([len(by[k]) for k in keys])
        idx = rng.integers(0, len(keys), (B, len(keys)))
        return s[idx].sum(1) / n[idx].sum(1), s.sum() / n.sum()
    ma, pa = means(a)
    mb, pb = means(b)
    d = ma - mb
    return pa - pb, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.std())


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def gaps(sel, tag):
    return [(r["draft"], r["their"] - r[f"ours{tag}"]) for r in sel if r.get(f"ours{tag}") is not None]


half = lambda r: int(hashlib.md5(r["draft"].encode()).hexdigest(), 16) % 2
P(f"pools built {len(rows)}; errors A {sum('errA' in r for r in rows)}, B {sum('errB' in r for r in rows)}; no build A {sum(r.get('oursA', 0) is None for r in rows)}")
P("\ncell sizes (class x group) and mean deck-model rating: theirs / ours A (as configured) / ours B (model objective); gap = theirs - ours")
for c in ("clean2", "1spl", "2+spl", "tri", "c45"):
    for g in ("good", "avg", "weak", "all"):
        s = [r for r in rows if r["cls"] == c and (g == "all" or r["group"] == g)]
        if not s:
            continue
        gA, gB = gaps(s, "A"), gaps(s, "B")
        P(f"  {c:6s} {g:5s} n {len(s):4d}  theirs {np.mean([r['their'] for r in s]):6.2f}  ours A {np.mean([r['oursA'] for r in s if r.get('oursA') is not None]):6.2f}  ours B {np.mean([r['oursB'] for r in s if r.get('oursB') is not None]):6.2f}"
          f"  gap A {np.mean([v for _, v in gA]):+.2f}  gap B {np.mean([v for _, v in gB]):+.2f}  "
          f"their splash cols {np.mean([len(r['splash']) for r in s]):.2f}, our A splash {np.mean([len(r.get('spA') or r.get('splashA') or '') for r in s]):.2f}")

P("\n=== PRIMARY: gap(2+spl) - gap(1spl), variant A, WR points [95% CI by draft bootstrap]")
for g in ("all", "good", "avg", "weak"):
    sel = lambda c: [r for r in rows if r["cls"] == c and (g == "all" or r["group"] == g)]
    P(f"  {g:5s} {fmt(boot_diff(gaps(sel('2+spl'), 'A'), gaps(sel('1spl'), 'A')))}")
d = {g: boot_diff(gaps([r for r in rows if r["cls"] == "2+spl" and r["group"] == g], "A"), gaps([r for r in rows if r["cls"] == "1spl" and r["group"] == g], "A")) for g in ("good", "weak")}
se = (d["good"][3] ** 2 + d["weak"][3] ** 2) ** .5
P(f"  good minus weak {d['good'][0] - d['weak'][0]:+.2f} (SE {se:.2f}) [{d['good'][0] - d['weak'][0] - 1.96 * se:+.2f}, {d['good'][0] - d['weak'][0] + 1.96 * se:+.2f}]")
for h in (0, 1):
    sel = lambda c: [r for r in rows if r["cls"] == c and half(r) == h]
    P(f"  half {h}: {fmt(boot_diff(gaps(sel('2+spl'), 'A'), gaps(sel('1spl'), 'A')))}")

P("\n=== secondary (not adjusted)")
for tag in ("A", "B"):
    for a, b in (("2+spl", "1spl"), ("2+spl", "clean2"), ("1spl", "clean2"), ("c45", "tri")):
        sa, sb = [r for r in rows if r["cls"] == a], [r for r in rows if r["cls"] == b]
        if not sa or not sb:
            continue
        P(f"  variant {tag}: gap({a}) - gap({b}) {fmt(boot_diff(gaps(sa, tag), gaps(sb, tag)))}")
P("\nhow much of the gap variant B (builder maximising the yardstick, limits unchanged) closes: mean gap A -> B by class")
for c in ("clean2", "1spl", "2+spl", "tri", "c45"):
    s = [r for r in rows if r["cls"] == c and r.get("oursA") is not None and r.get("oursB") is not None]
    if not s:
        continue
    P(f"  {c:6s} n {len(s)}  gap A {np.mean([r['their'] - r['oursA'] for r in s]):+.2f}  gap B {np.mean([r['their'] - r['oursB'] for r in s]):+.2f}  "
      f"we splash/three A {np.mean([len(r['laneA']) + len(r['splashA']) > 2 for r in s]):.2f} B {np.mean([len(r['laneB']) + len(r['splashB']) > 2 for r in s]):.2f}; "
      f"our colours A {np.mean([len(r['laneA']) + len(r['splashA']) for r in s]):.2f} B {np.mean([len(r['laneB']) + len(r['splashB']) for r in s]):.2f} vs theirs {np.mean([len(r['lane']) + len(r['splash']) for r in s]):.2f}")
Path(REPO / "analysis" / "out" / "builder_gap_pools.txt").write_text("\n".join(lines) + "\n")
