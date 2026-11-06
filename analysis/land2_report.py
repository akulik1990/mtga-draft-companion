"""Paired report for LAND_MAIN_DUAL: mock deck WR per seed, non-land top-1 losses, 3 most different mock decks.
    python3 analysis/land2_report.py <mock_base.json> <mock_new.json> <replay_base.jsonl> <replay_new.jsonl> <drafts.json.gz>
"""
import gzip, json, math, sys
from collections import Counter
from statistics import mean, stdev

mb, mn = (json.load(open(p)) for p in sys.argv[1:3])
rb, rn = (json.load(open(p)) for p in sys.argv[3:5])
human = {(d["id"], p[0], p[1]): p[2] for d in json.load(gzip.open(sys.argv[5], "rt")) for p in d["picks"]}

B = {t["seed"]: t for t in mb}
N = {t["seed"]: t for t in mn}
seeds = sorted(set(B) & set(N))
d = [N[s]["v2"]["deck_wr"] - B[s]["v2"]["deck_wr"] for s in seeds if N[s]["v2"].get("deck_wr") is not None and B[s]["v2"].get("deck_wr") is not None]
m, se = mean(d), stdev(d) / math.sqrt(len(d))
t = m / se
p = math.erfc(abs(t) / math.sqrt(2))
print(f"mock v2.deck_wr: base {mean(B[s]['v2']['deck_wr'] for s in seeds):.2f} new {mean(N[s]['v2']['deck_wr'] for s in seeds):.2f} "
      f"diff {m:+.3f} (SE {se:.3f}, 95% CI {m-1.96*se:+.3f}..{m+1.96*se:+.3f}, t {t:.2f}, p {p:.3f}, n {len(d)})")

# per-pick comparison
keyb = {tuple(e[:3]): e for e in rb}
land_names = set()
lost, gained, both = Counter(), Counter(), 0
nl_b = nl_n = nl_tot = 0
for e in rn:
    k = tuple(e[:3]); b = keyb.get(k); h = human.get(k)
    if b is None or h is None or not e[3] or not b[3]:
        continue
    tb, tn = b[3][0][0], e[3][0][0]
    if tb != tn:
        if tb == h: lost[(tb, tn)] += 1
        elif tn == h: gained[(tb, tn)] += 1
    if h == tb or h == tn: pass
print("top-1 flips where base was right (base -> new):", sum(lost.values()), " new right:", sum(gained.values()))
print("top lost:", lost.most_common(12))
print("top gained:", gained.most_common(8))
# arms' pick changes: what are the new picks (are they lands?)
newpicks = Counter(n for (b, n) in lost)
print("new picks in lost cases:", newpicks.most_common(12))
oldpicks = Counter(b for (b, n) in gained)
print("new picks in gained cases:", Counter(n for (b, n) in gained).most_common(8))

# 3 mock decks with largest |diff|
rank = sorted(seeds, key=lambda s: -abs(N[s]["v2"]["deck_wr"] - B[s]["v2"]["deck_wr"]))[:3]
for s in rank:
    a, b = B[s]["v2"], N[s]["v2"]
    print(f"\nseed {s}: deck_wr {a['deck_wr']:.2f} -> {b['deck_wr']:.2f}  colours {a.get('deck_colours')} -> {b.get('deck_colours')}")
    da, db = Counter(a["deck"]), Counter(b["deck"])
    print("  only base:", dict(da - db)); print("  only new :", dict(db - da))
