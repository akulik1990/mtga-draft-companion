"""adv1: does v2 keep pulling to two colours when a top player ends splash / three-colour? Reads data/Temp/adv1/{sp3,two}.log.json
(analysis/adv1_replay.py), meta.json (analysis/adv1_sample.py), prints everything; saved to analysis/out/adv1_results.txt.
Cuts fixed before outcomes: classes by FINAL deck, lane-weight threshold 0.10, bootstrap = resample drafts (B=4000, seed 1)."""
import gzip, json, random, statistics as S, collections as C, sys
sys.path.insert(0, "/home/morris/Projects/mtga-draft-companion")
from companion.advisor_v2.lanes import pool_fixing
R = "/home/morris/Projects/mtga-draft-companion/"
A = R + "data/Temp/adv1/"
front = lambda n: n.split(" // ")[0]
cards = {}
for c in json.load(open(R + "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values(): cards.setdefault(front(c["name"]), c)
meta = json.load(open(A + "meta.json"))
human = {}
for f in ("sp3", "two"):
    for d in json.load(gzip.open(A + f + ".drafts.json.gz", "rt")): human[d["id"]] = d["picks"]
logs = {f: json.load(open(A + f + ".log.json")) for f in ("sp3", "two")}
THR, B = 0.10, 4000
rng = random.Random(1)
is_land = lambda c: "Land" in (c.get("types") or [])
cols = lambda n: set((cards.get(front(n)) or {}).get("colors") or [])

def config(m):
    """(base pair/triple, third-or-splash colours S, pair) from the final deck."""
    cnt = C.Counter()
    for n, k in m["counts"].items():
        c = cards.get(front(n))
        if c and not is_land(c):
            for col in c.get("colors") or []: cnt[col] += k
    base, spl = set(m["lane"]), set(m["splash"])
    if len(base) == 3:
        third = min(base, key=lambda c: cnt[c]); pair = base - {third}
        return base | spl, ({third} | spl), pair
    return base | spl, spl, base

def klass(name, m):
    allc, S_, pair = config(m)
    c = cards.get(front(name))
    if c is None: return None
    if any(pool_fixing([c])[x] for x in S_) : return "fixer"
    cc = set(c.get("colors") or [])
    if cc & S_: return "splash/third colour card"
    if cc <= pair: return "on-pair card"
    return "other"

def boot(vals_by_draft, stat, B=B):
    ids = list(vals_by_draft); out = []
    for _ in range(B):
        out.append(stat([vals_by_draft[i] for i in rng.choices(ids, k=len(ids))]))
    out.sort(); return out[int(.025 * B)], out[int(.975 * B)]

def per_pick(log):
    rows = []
    for e in log:
        did, pk, pi, sc, pool, lanes = e; h = None
        for p in human[did]:
            if p[0] == pk and p[1] == pi: h = p[2]
        names = [n for n, _ in sc]
        if h is None or h not in names: continue
        rank = names.index(h) + 1
        rows.append(dict(d=did, pk=pk, pi=pi, h=h, v2=names[0], rank=rank, gap=sc[0][1] - dict(sc)[h], lanes=lanes or {}, n=len(pool), pool=pool))
    return rows
P = {f: per_pick(l) for f, l in logs.items()}
for r in P["sp3"]:
    r["kind"] = meta[r["d"]]["kind"]; r["cls"] = klass(r["h"], meta[r["d"]]); r["in_deck"] = front(r["h"]) in {front(x) for x in meta[r["d"]]["counts"]}
for r in P["two"]: r["kind"] = "two"; r["cls"] = klass(r["h"], meta[r["d"]])
def by_draft(rows, f):
    d = C.defaultdict(list)
    for r in rows: d[r["d"]].append(f(r))
    return d
def rate(groups): return sum(sum(g) for g in groups) / sum(len(g) for g in groups)  # pick-weighted
def mean_of_means(groups): return sum(sum(g) / len(g) for g in groups) / len(groups)
print("sample: picks scored", {f: len(r) for f, r in P.items()}, " drafts", {f: len({x['d'] for x in r}) for f, r in P.items()},
      " kinds", dict(C.Counter(m["kind"] for m in meta.values())))

print("\n== 1. top-1 / top-3, pick-weighted, CI = bootstrap over drafts ==")
res = {}
for label, rows in [("splash", [r for r in P["sp3"] if r["kind"] == "splash"]), ("three", [r for r in P["sp3"] if r["kind"] == "three"]),
                    ("splash+three", P["sp3"]), ("two-colour control", P["two"])]:
    for t, k in (("top-1", 1), ("top-3", 3)):
        g = by_draft(rows, lambda r: int(r["rank"] <= k)); lo, hi = boot(g, rate); res[(label, t)] = g
        print(f"  {label:20s} {t}  {rate(g.values()):.3f}  [{lo:.3f}, {hi:.3f}]  drafts={len(g)}")
for t, k in (("top-1", 1), ("top-3", 3)):
  for lab in ("splash", "three", "splash+three"):
      a, b = res[(lab, t)], res[("two-colour control", t)]
      ia, ib = list(a), list(b); ds = []
      for _ in range(B):
          ds.append(rate([a[i] for i in rng.choices(ia, k=len(ia))]) - rate([b[i] for i in rng.choices(ib, k=len(ib))]))
      ds.sort(); d0 = rate(a.values()) - rate(b.values())
      print(f"  diff {lab} minus control {t}: {d0:+.3f}  [{ds[int(.025*B)]:+.3f}, {ds[int(.975*B)]:+.3f}]  (two independent draft sets, so unpaired)")

print("\n== 2. top-1 by class of the taken card (class from final deck); CI over drafts ==")
def cls_table(rows, label):
    for c in ("fixer", "splash/third colour card", "on-pair card", "other"):
        sub = [r for r in rows if r["cls"] == c]
        if not sub: continue
        g = by_draft(sub, lambda r: int(r["rank"] == 1)); g3 = by_draft(sub, lambda r: int(r["rank"] <= 3)); lo, hi = boot(g, rate)
        print(f"  {label:8s} {c:26s} n={len(sub):5d} drafts={len(g):3d} top-1 {rate(g.values()):.3f} [{lo:.3f},{hi:.3f}]  top-3 {rate(g3.values()):.3f}  mean rank {S.mean(r['rank'] for r in sub):.2f}")
cls_table(P["sp3"], "sp3"); cls_table([r for r in P["sp3"] if r["kind"] == "splash"], "splash"); cls_table([r for r in P["sp3"] if r["kind"] == "three"], "three"); cls_table(P["two"], "control")
print("  -- only taken cards that made the final deck (sp3):")
cls_table([r for r in P["sp3"] if r["in_deck"]], "sp3-deck")
# on-pair minus splash/fixer contrast inside the same drafts (cluster bootstrap)
ids = sorted({r["d"] for r in P["sp3"]}); byd = C.defaultdict(list)
for r in P["sp3"]: byd[r["d"]].append(r)
def contrast(sample):
    def rt(cs):
        rows = [r for i in sample for r in byd[i] if r["cls"] in cs]; return sum(r["rank"] == 1 for r in rows) / max(len(rows), 1)
    return rt({"on-pair card"}) - rt({"fixer", "splash/third colour card"})
d0 = contrast(ids); bs = sorted(contrast(rng.choices(ids, k=len(ids))) for _ in range(B))
print(f"  top-1 on-pair minus (fixer+splash/third) in the same drafts: {d0:+.3f} [{bs[int(.025*B)]:+.3f}, {bs[int(.975*B)]:+.3f}]")

print("\n== 2b. misses on fixer / splash-third picks: what v2 took instead ==")
def v2class(r):
    m = meta[r["d"]]; allc, S_, pair = config(m); c = cards.get(front(r["v2"]))
    if c is None: return "unknown"
    top = max(r["lanes"], key=r["lanes"].get) if r["lanes"] else ""
    if any(pool_fixing([c])[x] for x in S_): return "fixer (final S)"
    cc = set(c.get("colors") or [])
    if cc & S_: return "card of final splash/third colour"
    if cc <= pair: return "on final pair" + (", in v2 top lane" if cc <= set(top) else ", outside v2 top lane")
    return "other colour" + (", in v2 top lane" if cc <= set(top) else ", outside v2 top lane")
miss = [r for r in P["sp3"] if r["cls"] in ("fixer", "splash/third colour card") and r["rank"] > 1]
tot = C.Counter(v2class(r) for r in miss)
print(f"  misses n={len(miss)}")
for k, v in tot.most_common(): print(f"    {k:42s} {v:4d}  {v/len(miss):.0%}")
for c in ("fixer", "splash/third colour card"):
    sub = [r for r in miss if r["cls"] == c]; print(f"   {c}: v2 took on-pair or its own top lane in {sum(v2class(r).startswith('on final pair') or 'in v2 top lane' in v2class(r) for r in sub)}/{len(sub)}; median gap to v2's top score {S.median(r['gap'] for r in sub):.1f} pts, median human rank {S.median(r['rank'] for r in sub)}")

print("\n== 3. timing: v2's lane weight on the player's final configuration ==")
def cfgw(r):
    m = meta[r["d"]]; allc, S_, pair = config(m)
    if not S_: return 0.0
    # mass on lanes holding the whole pair plus at least one of S (a triple), i.e. the player's configuration
    return sum(p for l, p in r["lanes"].items() if pair <= set(l) and set(l) & S_)
def pair_only(r):
    m = meta[r["d"]]; allc, S_, pair = config(m); return sum(p for l, p in r["lanes"].items() if set(l) == pair)
def any_three(r): return sum(p for l, p in r["lanes"].items() if len(l) == 3)
checkpoints = (5, 10, 15, 20, 25, 30, 35, 40)
print("  mean weight at the pick (v2 lanes before the pick is made): config = pair + a final third/splash colour; pair_only = exactly the final pair; any3 = any three-colour lane")
print("  pick  | sp3 config  pair_only  any3 | control any3")
for k in checkpoints:
    s = [r for r in P["sp3"] if r["n"] + 1 == k]; c = [r for r in P["two"] if r["n"] + 1 == k]
    print(f"  {k:4d}  | {S.mean(map(cfgw,s)):.3f}      {S.mean(map(pair_only,s)):.3f}      {S.mean(map(any_three,s)):.3f} | {S.mean(map(any_three,c)):.3f}   (n={len(s)}/{len(c)})")
first_v2, first_pl, first_fix, both = {}, {}, {}, []
for did, rows in byd.items():
    rows = sorted(rows, key=lambda r: r["n"]); m = meta[did]; allc, S_, pair = config(m)
    v = next((r["n"] + 1 for r in rows if cfgw(r) >= THR), None)
    pl = next((r["n"] + 1 for r in rows if r["cls"] == "splash/third colour card"), None)
    fx = next((r["n"] + 1 for r in rows if r["cls"] == "fixer"), None)
    first_v2[did], first_pl[did], first_fix[did] = v, pl, fx
for lab, kind in (("all sp3", None), ("splash", "splash"), ("three", "three")):
    ids_ = [i for i in byd if kind is None or meta[i]["kind"] == kind]
    v = [first_v2[i] for i in ids_ if first_v2[i]]; p = [first_pl[i] for i in ids_ if first_pl[i]]; f = [first_fix[i] for i in ids_ if first_fix[i]]
    diffs = [first_v2[i] - first_pl[i] for i in ids_ if first_v2[i] and first_pl[i]]
    print(f"  {lab:8s} n={len(ids_)} v2 reaches config>={THR}: {len(v)} drafts, median pick {S.median(v) if v else None}; player first takes S-colour card: {len(p)}, median {S.median(p) if p else None}; first fixer: {len(f)}, median {S.median(f) if f else None}; v2 minus player (both): median {S.median(diffs) if diffs else None}, n={len(diffs)}, v2 never reaches {len(ids_)-len(v)}")
    if diffs:
        ds = sorted(S.median(rng.choices(diffs, k=len(diffs))) for _ in range(B)); print(f"           median lag CI [{ds[int(.025*B)]}, {ds[int(.975*B)]}]; v2 later than player in {sum(x>0 for x in diffs)}/{len(diffs)}, same pick {sum(x==0 for x in diffs)}")
# peak weight over the draft
pk = [max(cfgw(r) for r in rows) for rows in byd.values()]
print(f"  peak config weight over the draft: median {S.median(pk):.2f}, share of drafts never above {THR}: {sum(x < THR for x in pk)/len(pk):.0%}, above 0.5: {sum(x>=.5 for x in pk)/len(pk):.0%}")
fin = [cfgw(sorted(rows, key=lambda r: r['n'])[-1]) for rows in byd.values()]
print(f"  weight at last pick (45th): mean {S.mean(fin):.3f}, median {S.median(fin):.3f}")

print("\n== 4. five examples (worst summed score gap on fixer/splash/third picks that the player put in the deck; illustrative, picked on the outcome) ==")
cand = []
for did, rows in byd.items():
    bad = [r for r in rows if r["cls"] in ("fixer", "splash/third colour card") and r["in_deck"] and r["rank"] > 1]
    cand.append((sum(r["gap"] for r in bad), did, bad))
cand.sort(key=lambda x: -x[0]); chosen, kinds_seen = [], C.Counter()
for tot_, did, bad in cand:
    if kinds_seen[meta[did]["kind"]] < 3 and len(chosen) < 5: chosen.append((tot_, did, bad)); kinds_seen[meta[did]["kind"]] += 1
for tot_, did, bad in chosen:
    m = meta[did]; allc, S_, pair = config(m); rows = sorted(byd[did], key=lambda r: r["n"])
    print(f"\n  draft {did[:8]}  skill {m['skill']}  final {m['lane']}+{m['splash'] or '-'}  ({m['kind']}); pair {''.join(sorted(pair))}, third/splash {''.join(sorted(S_))}; summed gap {tot_:.0f}")
    print("   final deck non-basic cards of S / fixers: " + ", ".join(n for n in m["counts"] if cards.get(front(n)) and (klass(n, m) in ("fixer", "splash/third colour card"))))
    for r in sorted(bad, key=lambda r: -r["gap"])[:4]:
        top = max(r["lanes"], key=r["lanes"].get) if r["lanes"] else "?"
        print(f"   P{r['pk']}p{r['pi']:<2d} took {r['h']} [{r['cls']}] (v2 rank {r['rank']}, {r['gap']:.1f} pts under); v2 took {r['v2']} [{v2class(r)}]; v2 top lane {top} {r['lanes'].get(top,0):.2f}, config weight {cfgw(r):.2f}")
