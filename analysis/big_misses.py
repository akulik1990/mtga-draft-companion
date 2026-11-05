"""Where the drafter disagrees with top drafters by more than GAP points (Akshin, 2026-11-05: find the actual differences and
reasons). Reads a replay log (one entry per pick: draft id, pack, pick, every candidate's score, pool names; written by the replay
wrapper used for analysis/synergy_flips.py) and the drafts file with the human picks.

    python3 analysis/big_misses.py <log.json> <drafts.json.gz> [n_examples]
"""
import gzip, json, random, sys
from collections import Counter
from pathlib import Path

GAP = 4.0
R = Path(__file__).resolve().parents[1]
log = json.loads(Path(sys.argv[1]).read_text())
human = {(d["id"], p[0], p[1]): p[2] for d in json.load(gzip.open(sys.argv[2], "rt")) for p in d["picks"]}
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 20
front = lambda n: n.split(" // ")[0]
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)
info = lambda n: cards.get(front(n), {})
alld = lambda n: info(n).get("deck_colors", {}).get("All Decks") or {}
cols = lambda n: set(info(n).get("colors") or [])
is_land = lambda n: "Land" in (info(n).get("types") or [])

def pair(pool):
    cnt = Counter(c for n in pool if not is_land(n) for c in cols(n))
    top = [c for c, _ in cnt.most_common(2)]
    share = sum(cnt[c] for c in top) / max(sum(cnt.values()), 1)
    return "".join(c for c in "WUBRG" if c in top), share, cnt

def lane_gih(n, lane):
    return (info(n).get("deck_colors", {}).get(lane) or {}).get("gihwr")

rows = []
for e in log:
    key = tuple(e[:3]); h = human.get(key)
    sc = dict(e[3])
    if not e[3] or h is None or h not in sc or e[3][0][0] == h: continue
    A = e[3][0][0]
    gap = sc[A] - sc[h]
    if gap <= GAP: continue
    lane, share, _ = pair(e[4])
    rank = [n for n, _ in e[3]].index(h) + 1
    fits = lambda n: bool(cols(n)) and cols(n) <= set(lane)
    rows.append(dict(key=key, A=A, h=h, gap=gap, rank=rank, lane=lane, share=share, pool=len([n for n in e[4] if not is_land(n)]),
                     A_fit=fits(A), h_fit=fits(h), h_colourless=not cols(h), A_gih=alld(A).get("gihwr") or 0, h_gih=alld(h).get("gihwr") or 0,
                     A_ata=alld(A).get("ata") or 0, h_ata=alld(h).get("ata") or 0, A_rar=info(A).get("rarity"), h_rar=info(h).get("rarity"),
                     h_land=is_land(h), top3=e[3][:3], scores=sc, poolnames=e[4]))

print(f"big misses (engine's pick more than {GAP:g} points over the human's card): {len(rows)}")
def share(label, f):
    k = sum(1 for r in rows if f(r)); print(f"  {label:68s} {k:5d}  {k / len(rows):5.0%}")
share("human's card outside the pool's two main colours", lambda r: not r["h_fit"] and not r["h_colourless"])
share("engine's card outside the pool's two main colours", lambda r: not r["A_fit"] and cols(r["A"]))
share("human off-colour while engine on-colour", lambda r: not r["h_fit"] and not r["h_colourless"] and r["A_fit"])
share("both on-colour", lambda r: r["h_fit"] and r["A_fit"])
share("human's card has the higher 17Lands GIH WR", lambda r: r["h_gih"] > r["A_gih"])
share("human's card is taken earlier by the field (lower ATA)", lambda r: 0 < r["h_ata"] < r["A_ata"])
share("human's card is a land", lambda r: r["h_land"])
share("pack 1 picks 1-5 (colours open)", lambda r: r["key"][1] == 1 and r["key"][2] <= 5)
share("pool committed (80%+ of coloured pips in two colours)", lambda r: r["share"] >= 0.8)
print("  by pack:", dict(sorted(Counter(r["key"][1] for r in rows).items())))
print("  human's card rank in the engine's list:", dict(sorted(Counter(min(r["rank"], 6) for r in rows).items())), "(6 = 6th or lower)")

random.seed(20261105)
ex = random.sample(rows, min(n_ex, len(rows)))
ex.sort(key=lambda r: (r["key"][1], r["key"][2]))
for r in ex:
    _, p, k = r["key"]
    lane, share_, cnt = pair(r["poolnames"])
    def card(n):
        lg = lane_gih(n, lane)
        return (f"{n} [{''.join(sorted(cols(n))) or 'C'}, {info(n).get('rarity', '?')[0]}, mv {info(n).get('cmc', '?')}, "
                f"GIH {alld(n).get('gihwr') or 0:.1f}{f' / {lane} {lg:.1f}' if lg else ''}, ATA {alld(n).get('ata') or 0:.1f}] score {r['scores'][n]:.1f}")
    print(f"\nP{p}p{k}  pool {r['pool']} spells, colours {dict(cnt.most_common())} -> {lane} ({share_:.0%})")
    print(f"  engine : {card(r['A'])}")
    print(f"  human  : {card(r['h'])}  (engine rank {r['rank']}, gap {r['gap']:.1f})")
    print(f"  engine's next: " + "; ".join(f"{n} {s:.1f}" for n, s in r["top3"][1:]))
