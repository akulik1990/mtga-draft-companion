"""Held-out replay report for LANE_OPEN: before = data/Temp/land1/{tag}.off.json, after = data/Temp/lane1/{tag}.open.json.
    env -C <repo> python3 analysis/lane1_report.py   -> analysis/out/lane_open.txt"""
import gzip, json, random, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
A, B, L = ROOT / "data/Temp/adv1", ROOT / "data/Temp/land1", ROOT / "data/Temp/lane1"
meta = json.load(open(A / "meta.json"))
ratings = json.load(open(ROOT / "data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
col = {}
for c in ratings.values():
    col.setdefault(c["name"], set() if "Land" in (c.get("types") or []) else set(c.get("colors") or []))
out = []
def P(s=""): out.append(s); print(s)

drafts, base, new = {}, {}, {}
for t in ("sp3", "two"):
    for d in json.load(gzip.open(A / f"{t}.drafts.json.gz")):
        drafts[d["id"]] = d
    for name, store, f in (("b", base, B / f"{t}.off.json"), ("n", new, L / f"{t}.open.json")):
        for r in json.load(open(f)):
            store[(r[0], r[1], r[2])] = r
    # baseline check against the adv1 (older) log
    old = {(r[0], r[1], r[2]): r[3][0][0] for r in json.load(open(A / f"{t}.log.json")) if r[3]}
    same = sum(1 for k, v in old.items() if base[k][3][0][0] == v)
    P(f"baseline check {t}: land1 off top card equals adv1 log top card on {same}/{len(old)} picks")

taken = {}
for d in drafts.values():
    for pack, pick, name, *_ in d["picks"]:
        taken[(d["id"], pack, pick)] = name
def rank(r, name):
    names = [n for n, _ in r[3]]
    return names.index(name) if name in names else 99
rows = []  # (draft, n, rb, rn, taken, rowb, rown)
for k, name in taken.items():
    if k in base and k in new:
        rows.append((k[0], (k[1] - 1) * 14 + k[2], rank(base[k], name), rank(new[k], name), name, base[k], new[k]))
P(f"picks compared: {len(rows)} in {len(drafts)} drafts")

random.seed(1)
ids = sorted(drafts)
def boot(fn, B_=4000):
    per = defaultdict(list)
    for r in rows: per[r[0]].append(r)
    diffs = []
    for _ in range(B_):
        s = [x for i in random.choices(ids, k=len(ids)) for x in per[i]]
        diffs.append(fn(s))
    diffs.sort()
    return fn(rows), diffs[int(.025 * B_)], diffs[int(.975 * B_)]
def top(k, lo=1, hi=42):
    def f(s):
        s = [r for r in s if lo <= r[1] <= hi]
        return 100 * (sum(r[3] < k for r in s) - sum(r[2] < k for r in s)) / len(s)
    return f
def lvl(rs, which, k): return 100 * sum(r[which] < k for r in rs) / len(rs)
P("\n== top-1/top-3 before -> after, difference (pp) with clustered bootstrap 95% CI over 196 drafts ==")
for lab, lo, hi in (("all", 1, 42), ("picks 1-10", 1, 10), ("picks 11-42", 11, 42)):
    rs = [r for r in rows if lo <= r[1] <= hi]
    for k in (1, 3):
        d, a, b = boot(top(k, lo, hi))
        P(f"{lab:12s} top-{k}: {lvl(rs,2,k):.2f} -> {lvl(rs,3,k):.2f}  diff {d:+.2f} [{a:+.2f}, {b:+.2f}]  n={len(rs)}")

P("\n== third/splash colour cards (final colours from meta; third = splash, or for 3-colour decks the lane colour with fewest basics) ==")
third = {}
for i, m in meta.items():
    if m["kind"] == "splash": third[i] = (m["splash"], set(m["lane"]))
    elif m["kind"] == "three":
        basics = {c: m["counts"].get({"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}[c], 0) for c in m["lane"]}
        t = min(basics, key=basics.get); third[i] = (t, set(m["lane"]) - {t})
tc = [r for r in rows if r[0] in third and third[r[0]][0] in col.get(r[4], ()) ]
tc_only = [r for r in tc if col[r[4]] <= {third[r[0]][0]}]
for lab, rs in (("card has third colour", tc), ("mono third-colour card", tc_only)):
    if not rs: continue
    P(f"{lab}: n={len(rs)}  rank#1 {lvl(rs,2,1):.1f}->{lvl(rs,3,1):.1f}  top3 {lvl(rs,2,3):.1f}->{lvl(rs,3,3):.1f}  mean rank(capped 15) {st.mean(min(r[2],15) for r in rs):.2f}->{st.mean(min(r[3],15) for r in rs):.2f}")
    for lo, hi in ((1, 10), (11, 28), (29, 42)):
        s = [r for r in rs if lo <= r[1] <= hi]
        if s: P(f"   picks {lo}-{hi}: n={len(s)} top1 {lvl(s,2,1):.1f}->{lvl(s,3,1):.1f} top3 {lvl(s,2,3):.1f}->{lvl(s,3,3):.1f}")
    ids_t = sorted({r[0] for r in rs}); per = defaultdict(list)
    for r in rs: per[r[0]].append(r)
    ds = []
    for _ in range(4000):
        s = [x for i in random.choices(ids_t, k=len(ids_t)) for x in per[i]]
        ds.append(lvl(s,3,1) - lvl(s,2,1))
    ds.sort(); P(f"   top-1 diff {lvl(rs,3,1)-lvl(rs,2,1):+.2f} [{ds[100]:+.2f},{ds[3900]:+.2f}] over {len(ids_t)} drafts")

P("\n== non-lane effects: non-land picks whose colours lie in the player's final top-2 colours ==")
def final_pair(i):
    m = meta[i]; return set(m["lane"][:2])
inc = [r for r in rows if col.get(r[4]) and col[r[4]] <= final_pair(r[0]) and r[1] >= 1]
rs = inc
P(f"in-colour non-land picks: n={len(rs)} top1 {lvl(rs,2,1):.2f}->{lvl(rs,3,1):.2f} top3 {lvl(rs,2,3):.2f}->{lvl(rs,3,3):.2f}")
rs = [r for r in inc if r[1] > 16]
P(f"   same, pick > 16 (ramp over): n={len(rs)} top1 {lvl(rs,2,1):.2f}->{lvl(rs,3,1):.2f}")
chg = sum(r[5][3][0][0] != r[6][3][0][0] for r in rows)
P(f"engine top card changes on {100*chg/len(rows):.1f}% of picks overall; for picks > 16: {100*sum(r[5][3][0][0]!=r[6][3][0][0] for r in rows if r[1]>16)/sum(1 for r in rows if r[1]>16):.1f}%")
for lo, hi in ((1, 5), (6, 10), (11, 16), (17, 28), (29, 42)):
    s = [r for r in rows if lo <= r[1] <= hi]
    P(f"   picks {lo}-{hi}: top card changes {100*sum(r[5][3][0][0]!=r[6][3][0][0] for r in s)/len(s):.1f}%")

P("\n== lane weight on top two-colour pair, after N cards in pool (new = flattened, as scored) ==")
def toppair(lanes): return max((v for l, v in (lanes or {}).items() if len(l) == 2), default=0)
for n in (0, 1, 3, 6, 10, 16):
    b = [toppair(r[5][5]) for r in rows if len(r[5][4]) == n]
    a = [toppair(r[6][5]) if False else toppair(r[6][5]) for r in rows if len(r[6][4]) == n]
    if b: P(f"pool={n}: before mean {st.mean(b):.3f} (median {st.median(b):.3f})  after mean {st.mean(a):.3f} (median {st.median(a):.3f})  n={len(b)}")

P("\n== first pool size at which lane mass >= 0.10 (splash/three drafts; uniform flatten makes this trivial early) ==")
def first(rowsd, fn):
    res = []
    for i in third:
        seq = sorted((len(r[4]), r) for r in rowsd if r[0] == i)
        hit = next((n for n, r in seq if fn(i, r) >= .10), None); res.append(hit)
    got = [x for x in res if x is not None]
    return (st.median(got) if got else None, len(got), len(res))
def pair_mass(i, r):
    pair = "".join(sorted(third[i][1], key="WUBRG".index)); return (r[5] or {}).get(pair, 0)
def third_mass(i, r): return sum(v for l, v in (r[5] or {}).items() if third[i][0] in l)
for lab, fn in (("final main pair", pair_mass), ("any lane containing the third colour", third_mass)):
    b = first([(k[0], *[]) and None for k in []] or [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in base.values()], fn)
    a = first([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in new.values()], fn)
    P(f"{lab}: median pool size at first >=0.10: before {b}, after {a} (median, n reached, n drafts)")

P("\nmean lane mass on the final main pair / on lanes containing the third colour, by pool size (splash+three drafts), before | after")
for n in (1, 3, 6, 10, 14, 20, 30):
    for lab, fn in (("pair", pair_mass), ("third", third_mass)):
        b = [fn(r[0], r[5:]) if False else fn(r[0], r) for r in base.values() if r[0] in third and len(r[4]) == n]
        a = [fn(r[0], r) for r in new.values() if r[0] in third and len(r[4]) == n]
        P(f"pool={n:2d} {lab:5s}: {st.mean(b):.3f} | {st.mean(a):.3f}  (n={len(b)})")
P("\n== per-pick rank table (rank of player's pick: #1/#2/#3/other %, before | after) and off-colour cells ==")
colr = col
def cell(r):
    return r
ranks = defaultdict(lambda: [Counter(), Counter()])
offc = defaultdict(lambda: [Counter(), Counter()])
for r in rows:
    for w, rk in ((0, r[2]), (1, r[3])):
        ranks[r[1]][w][min(rk, 3) if rk < 3 else "o"] += 1
    pool = r[5][4]
    if len(pool) >= 3:
        cnt = Counter(c for nme in pool for c in colr.get(nme, ()))
        main = {c for c, _ in cnt.most_common(2)}
        off_p = bool(colr.get(r[4], set()) - main)
        for w, rr in ((0, r[5]), (1, r[6])):
            off_e = bool(colr.get(rr[3][0][0], set()) - main)
            offc[r[1]][w][("both" if off_p and off_e else "player" if off_p else "engine" if off_e else "none")] += 1
def pc(c, k): n = sum(c.values()); return f"{100*c[k]/n:.0f}" if n else "-"
P("| Pick | n | #1 | #2 | #3 | other | both-off | player-only | engine-only |")
P("|---|---|---|---|---|---|---|---|---|")
tot = [Counter(), Counter()]; toto = [Counter(), Counter()]
for n in sorted(ranks):
    for w in (0, 1): tot[w].update(ranks[n][w]); toto[w].update(offc[n][w])
    c = ranks[n]; o = offc[n]
    lab = f"P{(n-1)//14+1}p{(n-1)%14+1}"
    P(f"| {lab} | {sum(c[0].values())} | " + " | ".join(f"{pc(c[0],k)}>{pc(c[1],k)}" for k in (0, 1, 2, "o")) + " | " + " | ".join((f"{pc(o[0],k)}>{pc(o[1],k)}" if sum(o[0].values()) else "-") for k in ("both", "player", "engine")) + " |")
P(f"| All | {sum(tot[0].values())} | " + " | ".join(f"{pc(tot[0],k)}>{pc(tot[1],k)}" for k in (0, 1, 2, "o")) + " | " + " | ".join(f"{pc(toto[0],k)}>{pc(toto[1],k)}" for k in ("both", "player", "engine")) + " |")
(ROOT / "analysis/out/lane_open.txt").write_text("\n".join(out) + "\n")
