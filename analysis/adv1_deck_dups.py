"""(1) paired primary with the builder's own yardstick (model_wr, counts name collisions correctly); (2) wheel duplicates in v2 pools."""
import gzip, json, random, statistics as S, collections as C
R = "/home/morris/Projects/mtga-draft-companion/data/Temp/adv1/"
import sys
res = json.load(open(R + (sys.argv[1] if len(sys.argv) > 1 else "deck_full.json"))); dr = {d["id"]: d for d in json.load(gzip.open(R + "sp3.drafts.json.gz", "rt"))}
rng = random.Random(1); B = 4000
def rep(lab, v):
    ms = sorted(S.mean(rng.choices(v, k=len(v))) for _ in range(B)); print(f"  {lab}: n={len(v)} mean {S.mean(v):+.2f} [{ms[int(.025*B)]:+.2f}, {ms[int(.975*B)]:+.2f}], v2>=player in {sum(x>=0 for x in v)/len(v):.0%}")
print("(1) builder's own yardstick (model_wr):")
rep("v2 build - player-pool build", [r["v2"]["model_wr"] - r["pl"]["model_wr"] for r in res])
rep("v2 build - registered deck (registered via deck_win_rate)", [r["v2"]["model_wr"] - r["reg_wr"] for r in res])
print("(2) wheel duplicates; pack sizes at picks 1, 9:", C.Counter((len(p[3]) for d in dr.values() for p in d["picks"] if p[1] in (1, 9) and p[0] == 1)).most_common(4))
bad_pools = 0; extra = 0; n_dup_pool = 0; ex = []
for r in res:
    d = dr[r["id"]]; packs = {(p[0], p[1]): C.Counter(p[3]) for p in d["picks"]}
    took = {(p[0], p[1]): t for p, t in zip(d["picks"], r["v2_picks"])}
    bad = 0
    for pk in (1, 2, 3):
        for k in range(1, 7):
            a, b = (pk, k), (pk, k + 8)
            if a not in took or b not in took: continue
            for name in {took[a], took[b]}:
                n_took = (took[a] == name) + (took[b] == name)
                if n_took > packs[a][name]: bad += n_took - packs[a][name]; ex.append((r["id"][:8], pk, k, name))
    if bad: bad_pools += 1; extra += bad
    c = C.Counter(r["v2_picks"]); n_dup_pool += any(v > 1 for v in c.values())
print(f"  v2 pools with a wheel-impossible extra copy: {bad_pools}/100, extra copies total {extra} (of {sum(len(r['v2_picks']) for r in res)} picks); examples {ex[:3]}")
print(f"  pools with any repeated card name at all: {n_dup_pool}/100")
