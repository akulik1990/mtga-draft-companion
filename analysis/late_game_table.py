"""Per-deck table for reading by eye (Akshin, 2026-11-05): 100 one-deck Premier runs with at least one good late-game payoff
(6+ mana, GIH WR >= set mean + 1 SD), 50 trophies and 50 runs finished at 0-1 wins, random with a fixed seed. Columns: the four
control tags (removal, defence, tempo, draw) as copies in the deck, their total, and how many of the four reach 5+, 6+, 7+.

    python3 analysis/late_game_table.py   ->  analysis/out/late_game_table.csv + summary on stdout
"""
import csv, gzip, json, random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

R = Path(__file__).resolve().parents[1]
CONTROL = ("removal", "defence", "tempo", "draw")
front = lambda n: n.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)
gih = lambda c: (c.get("deck_colors", {}).get("All Decks") or {}).get("gihwr") or 0
wrs = [gih(c) for c in cards.values() if gih(c)]
GOOD = mean(wrs) + pstdev(wrs)

raw = [d for d in json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read()) if d["format"] == "PremierDraft"]
per_draft = Counter(d["draft"] for d in raw)
pool = {"T": [], "L": []}
for d in raw:
    if per_draft[d["draft"]] > 1 or len(d["lane"]) != 2 or sum(d["counts"].values()) != 40: continue
    side = "T" if d["trophy"] else "L" if d["match_wins"] <= 1 and d["deck_games"] - d["deck_wins"] == 3 else None
    if not side: continue
    payoffs = [n for n in d["counts"] if (cards.get(front(n), {}).get("cmc") or 0) >= 6 and gih(cards.get(front(n), {})) >= GOOD]
    if not payoffs: continue
    r = {"result": "trophy" if side == "T" else "loser", "record": f"{int(d['deck_wins'])}-{int(d['deck_games'] - d['deck_wins'])}",
         "lane": d["lane"] + (f"+{d['splash']}" if d["splash"] else ""), "skill": d["skill"],
         "payoffs": "; ".join(f"{d['counts'][n]}x {n}" if d["counts"][n] > 1 else n for n in payoffs)}
    for t in CONTROL:
        r[t] = sum(k for n, k in d["counts"].items() if "Land" not in (cards.get(front(n), {}).get("types") or []) and t in roles.get(front(n), set()))
    r["total"] = sum(r[t] for t in CONTROL)
    for x in (5, 6, 7):
        r[f"tags_{x}+"] = sum(r[t] >= x for t in CONTROL)
    pool[side].append(r)

random.seed(20261105)
rows = random.sample(pool["T"], 50) + random.sample(pool["L"], 50)
out = R / "analysis/out/late_game_table.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"good 6+ payoffs: GIH WR >= {GOOD:.1f}; eligible decks {len(pool['T'])} trophies, {len(pool['L'])} losers; sampled 50 + 50 -> {out}\n")
cols = [*CONTROL, "total", "tags_5+", "tags_6+", "tags_7+"]
print("means      " + "  ".join(f"{c:>8s}" for c in cols))
for side in ("trophy", "loser"):
    s = [r for r in rows if r["result"] == side]
    print(f"{side:10s} " + "  ".join(f"{mean(r[c] for r in s):8.1f}" for c in cols))
print("\nshare of decks with at least k of the four tags at each threshold")
for x in (5, 6, 7):
    for side in ("trophy", "loser"):
        s = [r for r in rows if r["result"] == side]
        print(f"  {x}+ {side:7s} " + "  ".join(f"k>={k}: {sum(r[f'tags_{x}+'] >= k for r in s) / len(s):.0%}" for k in (1, 2, 3, 4)))
