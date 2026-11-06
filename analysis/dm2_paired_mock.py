"""Paired mock comparison of two `tools/mock_draft.py compare` result files (v2 side), per seed.
    python3 analysis/dm2_paired_mock.py <base.json> <new.json>
"""
import json, math, sys
from statistics import mean, stdev

a, b = (json.load(open(p)) for p in sys.argv[1:3])
B = {r["seed"]: r for r in b}
rows = [(r, B[r["seed"]]) for r in a if r["seed"] in B]
print("tables", len(rows))
for key in ("deck_wr", "deck_gih", "lane_score"):
    pairs = [(x["v2"].get(key), y["v2"].get(key)) for x, y in rows]
    ok = [(p, q) for p, q in pairs if p is not None and q is not None]
    if not ok:
        print(f"{key}: n/a (base {sum(p is not None for p, _ in pairs)}, new {sum(q is not None for _, q in pairs)} rows)"); continue
    d = [q - p for p, q in ok]
    m, se = mean(d), stdev(d) / math.sqrt(len(d))
    if not se:
        print(f"{key}: identical in all {len(d)} tables"); continue
    t = m / se
    p_ = math.erfc(abs(t) / math.sqrt(2))
    print(f"{key}: base {mean(p for p,_ in ok):.3f} new {mean(q for _,q in ok):.3f} diff {m:+.3f} SE {se:.3f} 95%CI [{m-1.96*se:+.3f},{m+1.96*se:+.3f}] t {t:.2f} p~{p_:.4f} n={len(d)}")
same_picks = sum(x["v2"]["picks"] == y["v2"]["picks"] for x, y in rows)
same_deck = sum(x["v2"]["deck"] == y["v2"]["deck"] for x, y in rows)
print(f"identical pick sequences {same_picks}/{len(rows)}; identical decks {same_deck}/{len(rows)}")
