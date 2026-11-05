"""Review of the 2 Nov Quick Draft (FRA), pick by pick and colour by colour.

    MTGA_COMPANION_DATA=$PWD/data uv run --with pandas python analysis/draft_review.py > analysis/out/draft_review.txt     # about 3 min

Ratings are 17Lands Premier GIH WR (Quick Draft has no public data), so 'best card' means best by that number. Sections:
1 final pool and the builder's builds in every pair it can build, scored by the criteria model against the average deck of the same colours
2 where the pool sits among 54k drafted pools (top-23 mean GIH WR in the best pair, removal, finishers)
3 pick table: your pick vs the best GIH WR card in the pack vs v2's top pick, with v2's lane belief
4 lane scores of the pool after pick 7 / 14 / 21 / 28 / 35 / 42
5 what the other seats took (cards of each colour that survived the wheel)
6 hindsight: commit to a pair from pick k and take the best card in that pair from every pack seen (packs taken as seen; wheels not re-simulated)
"""
import csv, gzip, itertools, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/morris/Projects/mtga-draft-companion")
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import composition, config, deckbuilder  # noqa: E402
from companion.advisor_v2.engine import DraftAdvisorV2  # noqa: E402
from companion.advisor_v2.lanes import LANES, all_decks, castable  # noqa: E402
from tools.live_diff import picks  # noqa: E402
from tools.mock_draft import apply_overrides  # noqa: E402
from tools.replay.run import replay  # noqa: E402

LOG = ROOT / "data/Temp/live/FRA-quickdraft-2026-11-02.Player.log"
app = replay.App(str(DATA_DIR / "Sets" / "FRA_PremierDraft_All_Data.json"))
ratings = {k: dict(v, set="FRA") for k, v in app.dataset.get_card_ratings().items()}
steps = picks(LOG)
assert len(steps) == 42, len(steps)
card = lambda i: ratings.get(i) or {"name": f"?{i}", "types": [], "deck_colors": {}, "colors": [], "cmc": 0}
cards_of = lambda ids: [ratings[i] for i in ids if i in ratings]
gih = lambda c: float(all_decks(c).get("gihwr") or 0.0)
island = lambda c: "Land" in (c.get("types") or [])
wr_in = lambda c, lane: float(((c.get("deck_colors") or {}).get(lane) or {}).get("gihwr") or 0.0) or gih(c)
vals = [gih(c) for c in ratings.values() if not island(c) and gih(c) > 0]
MU, SD = float(np.mean(vals)), float(np.std(vals))
took = [next(iter(Counter(nxt[3]) - Counter(cur[3]))) for cur, nxt in zip(steps, steps[1:])] + [steps[-1][2][0]]
final_ids = list(steps[-1][3]) + list(steps[-1][2])
pool = cards_of(final_ids)
print(f"pool {len(pool)} cards; set GIH WR mean {MU:.1f} SD {SD:.1f}\n")

def lane_score(cs, lane, n=23):
    r = sorted([wr_in(c, lane) for c in cs if not island(c) and castable(c, lane) and wr_in(c, lane) > 0], reverse=True)[:n]
    return (sum(r) + 45.0 * (n - len(r))) / n

# advisor replay first: it resets config
def advisor(overrides):
    config.RUNTIME.clear(); config.ACTIVE["set"] = None
    apply_overrides(overrides)
    seen, out = [], []
    for pack_no, pick_no, pack_ids, picked_ids in steps:
        pack = cards_of(pack_ids)
        seen.append((pack_no, pick_no, pack))
        adv = DraftAdvisorV2(app.metrics, cards_of(picked_ids), seen_packs=list(seen))
        recs = adv.evaluate_pack(pack, pick_no, current_pack=pack_no)
        out.append((recs, sorted(adv.lanes.items(), key=lambda kv: -kv[1])[:3]))
    return out
A = advisor([])
B = advisor(["FINISHER_POINTS=4"])
config.activate("FRA")

# 1 builds
model = __import__("tools.deck_model", fromlist=["DeckModel"]).DeckModel("FRA", deckbuilder.MODEL)
targets = config.fitted("FRA", "decks")["pairs"]
rule, three = config.fitted("FRA", "splash"), config.fitted("FRA", "three")
builds = {}
for flag in (True, False):
    deckbuilder.STRUCTURE = flag
    seen_k = set()
    for b in deckbuilder.build(pool, model, targets, top=12, rule=rule, three=three):
        key = (b["lane"], b["splash"])
        if key not in seen_k:
            seen_k.add(key); builds.setdefault(key, {})[flag] = b
deckbuilder.STRUCTURE = True
final_lane = next(iter(builds))[0]
print("== 1. what the pool can build (builder, every pair it can make 23 castable spells in)")
print(f"pairs with >= 23 castable spells in the pool: {[l for l in LANES if sum((not island(c)) and castable(c, l) for c in pool) >= 23]}")
print("castable spells per pair:", {l: sum((not island(c)) and castable(c, l) for c in pool) for l in LANES})
print("lane score (best 23 by GIH WR in the pair, filler 45) per pair:", {l: round(lane_score(pool, l), 2) for l in LANES})

from tools import builder_compare as bcmp  # noqa: E402
from tools.criteria_lib import PTS, deck_fields, load_data  # noqa: E402
D = load_data("FRA", "all")
M = bcmp.fit_model(D)
base_tr = np.array([r["combo"].split("+")[0] for r in D["rows"]])
s_tr = M["Xd"] @ M["bd"]
print(f"\ncriteria model fitted on {len(D['w'])} decks. Builds: predicted WR (deck model), deck part vs the average deck of the same colours, percentile among those decks. A = structure-first, R = gate off")
print(f"{'build':10} {'mode':4} {'pred':>5} {'pts':>6} {'pctile':>6} {'rm':>3} {'fin':>3} {'creat':>5} {'duals':>5} {'lands':>5} {'devs':>4} {'pen':>4}")
def describe(lane, splash, mode, b):
    counts = Counter(c["name"] for c in b["spells"]); counts.update(b["basics"]); counts.update(c["name"] for c in b["nonbasics"])
    f = deck_fields("FRA", [dict(counts)])
    s = float((bcmp.design(M, f) @ M["bd"])[0]); base = f["base"][0]
    m = base_tr == base
    ref = s_tr[m].mean() if m.any() else float("nan"); pct = 100 * (s_tr[m] < s).mean() if m.any() else float("nan")
    print(f"{lane + ('+' + splash if splash else ''):10} {mode:4} {b['predicted']:5.1f} {PTS * (s - ref):+6.1f} {pct:5.0f}% {int(f['removal'][0]):3d} {int(f['finishers'][0]):3d} {int(f['creatures'][0]):5d} {int(f['land_multi'][0]):5d} {int(f['lands'][0]):5d} {b['structure_dev']:4} {b['penalty']:4.1f}")
for (lane, splash), by in builds.items():
    for flag, b in sorted(by.items(), reverse=True):
        describe(lane, splash, "A" if flag else "R", b)
print("-- every pair and three-colour combination built directly (the builder's own cut on shape penalty is skipped here)")
for flag in (True, False):
    deckbuilder.STRUCTURE = flag
    for l in [x for x in LANES] + [x for x in targets if len(x) == 3]:
        if l not in targets: continue
        b = deckbuilder.build_pair(pool, l, model, targets[l], splash_cost=0.0 if len(l) == 3 else None)
        if b: describe(l, "", "A" if flag else "R", b)
deckbuilder.STRUCTURE = True
print("\ncurrent build (A) of the first pair:", ", ".join(sorted(c["name"] for c in builds[next(iter(builds))][True]["spells"])))

# 2 distribution
tab = {r["name"]: r for r in csv.DictReader(open(ROOT / "data/tables/FRA/2_cards.csv"))}
num = lambda r, k: float(r.get(k) or 0) if r.get(k) not in ("", None) and not str(r.get(k)).isalpha() else 0.0
gv = np.array([num(r, "gihwr") for r in tab.values() if not num(r, "land") and num(r, "gihwr") > 0])
mu2, sd2 = gv.mean(), gv.std()
def pool_stats(cs):
    best = None
    for pr in itertools.combinations(range(5), 2):
        out = [c for c in "WUBRG" if "WUBRG".index(c) not in pr]
        sel = []
        for c in cs:
            r = tab.get(c["name"]) or tab.get(c["name"].split(" // ")[0])
            if not r or num(r, "land"): continue
            if any(num(r, "col_" + x) for x in out): continue
            g = num(r, "gihwr") or mu2 - sd2
            sel.append((g, num(r, "cmc"), num(r, "role_removal") > 0))
        sel = sorted(sel, reverse=True)[:23]
        if not sel: continue
        q = np.mean([s[0] for s in sel])
        if best is None or q > best[0]:
            best = (q, sum(s[2] for s in sel), sum(s[1] >= 5 and s[0] >= mu2 + sd2 for s in sel), len(sel), "".join("WUBRG"[i] for i in pr))
    return best
q, rm, fin, n, pr = pool_stats(pool)
rows = list(csv.DictReader(gzip.open(ROOT / "analysis/out/draft_features_FRA.csv.gz", "rt")))
full = [r for r in rows if int(r["n"]) >= 23]
big = [r for r in full if float(r["user_n_games_bucket"] or 0) >= 100]
pct = lambda rs, k, v: 100 * np.mean([float(r[k]) < v for r in rs])
print(f"\n== 2. where the pool sits among {len(full)} drafted pools with a full 23 (same method: best pair by top-23 mean GIH WR)")
print(f"yours: pair {pr}, quality {q:.2f} (n {n}), removal {rm}, finishers {fin}")
print(f"quality percentile: all drafters {pct(full, 'quality', q):.0f}%, 100+ games players {pct(big, 'quality', q):.0f}% (median {np.median([float(r['quality']) for r in full]):.2f}, 90th {np.percentile([float(r['quality']) for r in full], 90):.2f})")
print(f"removal percentile {pct(full, 'removal', rm):.0f}% (median {np.median([float(r['removal']) for r in full]):.0f}), finishers percentile {pct(full, 'finishers', fin):.0f}% (share with 2+: {100 * np.mean([float(r['finishers']) >= 2 for r in full]):.0f}%)")

# 3 pick table
lane = final_lane
print(f"\n== 3. picks. final lane {lane}. 'best' = highest GIH WR card in the pack, 'v2' = top pick now, 'lane' = best card in the final lane (WR there)")
print(f"{'pick':6} {'took':30} {'gih':>5} {'rk':>3}/{'n':<2} {'best in pack':30} {'gih':>5}  {'v2 top':30} {'lane best':30} {'v2 lanes'}")
tbl = []
for i, ((pn, pk, pids, _), t, (recs, lanes), (recsB, _)) in enumerate(zip(steps, took, A, B)):
    pack = cards_of(pids); tc = card(t)
    order = sorted(pack, key=lambda c: -gih(c)) or [tc]
    rank = 1 + sum(gih(c) > gih(tc) for c in pack)
    lane_best = max((c for c in pack if castable(c, lane) and not island(c)), key=lambda c: wr_in(c, lane), default=None)
    v2 = recs[0].card_name if recs else "?"
    tbl.append((i + 1, f"P{pn}P{pk}", tc["name"], gih(tc), rank, len(pack), order[0]["name"], gih(order[0]), v2, lane_best["name"] if lane_best else "-", wr_in(lane_best, lane) if lane_best else 0, recsB[0].card_name if recsB else "?", lanes))
    print(f"P{pn}P{pk:<3} {tc['name'][:30]:30} {gih(tc):5.1f} {rank:3d}/{len(pack):<2} {order[0]['name'][:30]:30} {gih(order[0]):5.1f}  {v2[:30]:30} {(lane_best['name'] if lane_best else '-')[:22]:22}{wr_in(lane_best, lane) if lane_best else 0:5.1f}  " + " ".join(f"{l}{p:.2f}" for l, p in lanes))
with open(ROOT / "analysis/out/draft_review_picks.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["n", "pick", "took", "took_gih", "rank", "pack", "best", "best_gih", "v2", "lane_best", "lane_best_wr", "v2_finisher4", "v2_lanes"]); w.writerows([r[:-1] + (str(r[-1]),) for r in tbl])
agree = sum(r[2] == r[8] for r in tbl); agreeB = sum(r[2] == r[11] for r in tbl)
miss = [r for r in tbl if r[3] < r[7] - 1.5 and r[2] != r[8]]
print(f"\nyou matched v2's top pick {agree}/42 (with FINISHER_POINTS=4: {agreeB}); took the pack's top GIH card {sum(r[4] == 1 for r in tbl)}/42; v2's pick differed from yours by >1.5 GIH points of the card you took: {len(miss)}")
print("v2's top pick differs from yours:")
for r in tbl:
    if r[2] != r[8]:
        c8 = next((c for c in cards_of(steps[r[0] - 1][2]) if c["name"] == r[8]), None)
        print(f"  {r[1]:6} you {r[2]} ({r[3]:.1f}) | v2 {r[8]} ({gih(c8) if c8 else 0:.1f}) | lane best {r[9]} ({r[10]:.1f})")

# 4 lane trail
print("\n== 4. lane score of the pool after each stage (top 4 pairs; filler 45 until 23 playables)")
for k in (7, 14, 21, 28, 35, 42):
    pl = cards_of(list(steps[k - 1][3]) + [took[k - 1]])
    sc = sorted(((lane_score(pl, l), l) for l in LANES), reverse=True)[:4]
    print(f"  after pick {k:2}: " + "   ".join(f"{l} {s:.1f}" for s, l in sc) + f"   | {len(pl)} cards, castable in {final_lane}: {sum(castable(c, final_lane) and not island(c) for c in pl)}")

# 5 wheel
print("\n== 5. what the other seats took: cards of a colour above the set mean that were in a pack when you saw it (not your pick) and gone when it came back 8 picks later")
for rd in (1, 2, 3):
    base = (rd - 1) * 14
    tot, gone = Counter(), Counter()
    for k in range(6):
        a, b = steps[base + k], steps[base + k + 8]
        mine = took[base + k]
        for i in a[2]:
            if i == mine or i not in ratings: continue
            c = ratings[i]
            if island(c) or gih(c) < MU: continue
            for col in (c.get("colors") or ["C"]):
                tot[col] += 1; gone[col] += i not in b[2]
    print(f"  pack {rd}: " + "  ".join(f"{col} {gone[col]}/{tot[col]}" for col in "WUBRGC" if tot[col]) + f"   (sizes {len(steps[base][2])} -> {len(steps[base + 8][2])})")

# 6 hindsight
print("\n== 6. hindsight: from pick k take the best GIH WR card (in the pair) in every pack as seen; final lane score of that pair (yours: actual picks)")
real = {l: lane_score(pool, l) for l in LANES}
ks = [1, 4, 7, 10, 15, 22]
print(f"{'pair':5} {'actual':>6} " + " ".join(f"{'from ' + str(k):>7}" for k in ks))
for l in LANES:
    row = []
    for k in ks:
        have = [t for t in took[:k - 1]]
        for j in range(k - 1, 42):
            avail = [i for i in steps[j][2] if i in ratings and i not in have and not island(ratings[i]) and castable(ratings[i], l)]
            if not avail: avail = [i for i in steps[j][2] if i in ratings and i not in have]
            if not avail: continue
            have.append(max(avail, key=lambda i: wr_in(ratings[i], l) if castable(ratings[i], l) else gih(ratings[i]) - 5))
        row.append(lane_score(cards_of(have), l))
    print(f"{l:5} {real[l]:6.1f} " + " ".join(f"{x:7.1f}" for x in row))

# 7 removal and finishers seen
is_rm = lambda c: (not island(c)) and "removal" in config.tags(c)
is_fin = lambda c: (not island(c)) and (c.get("cmc") or 0) >= 5 and gih(c) >= MU + SD
for label, test in (("removal", is_rm), ("finishers (cmc 5+, GIH WR >= %.1f)" % (MU + SD), is_fin)):
    print(f"\n== 7. {label} you saw, castable in {final_lane}: first sighting, what you took then, whether it was still there 8 picks later")
    seen_names = {}
    for j, (pn, pk, pids, _) in enumerate(steps):
        for i in pids:
            c = ratings.get(i)
            if c and test(c) and castable(c, final_lane) and c["name"] not in seen_names:
                wheel = steps[j + 8][2] if j + 8 < (j // 14 + 1) * 14 else None
                seen_names[c["name"]] = (f"P{pn}P{pk}", i == took[j], took[j], (None if wheel is None else i in wheel), wr_in(c, final_lane))
    taken = sum(v[1] for v in seen_names.values())
    print(f"  distinct cards seen {len(seen_names)}, you took {taken} of them at first sighting (plus later copies)")
    for n, (pk, got, t_id, wh, w) in seen_names.items():
        print(f"  {pk:6} {n[:32]:32} WR {w:5.1f}  {'TAKEN' if got else 'passed for ' + card(t_id)['name'][:28] + ' (%.1f)' % gih(card(t_id)):45} {'' if wh is None else ('wheeled' if wh else 'gone by the wheel')}")
