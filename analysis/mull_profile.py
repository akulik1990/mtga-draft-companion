"""Keep or mulligan: win rate of kept 7-card hands with a given profile against games after one mulligan, FRA Premier, 17Lands
public game data (streamed). Profile from argv: on_play (0/1), lands, max cheap spells (cmc <= 2). Mulligan games are selected
on bad hands, so their win rate is the realistic cost of mulliganing, not of a random 6.
Run: python3 analysis/mull_profile.py 0 3 0"""
import csv, gzip, json, math, os, sys
from collections import defaultdict

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
play, lands_want, cheap_max = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cards = {c["name"]: c for c in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
def kind(n):
    c = cards.get(n) or cards.get(n.split(" // ")[0]) or {}
    if "Land" in (c.get("types") or []) or n in ("Plains", "Island", "Swamp", "Mountain", "Forest"):
        return "land"
    return "cheap" if (c.get("cmc") or 0) <= 2 else ("big" if (c.get("cmc") or 0) >= 5 else "mid")

acc = defaultdict(lambda: [0, 0])  # group -> [wins, games]
with gzip.open(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", "rt", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    ix = {k: i for i, k in enumerate(h)}
    oh = [(i, kind(k[len("opening_hand_"):])) for k, i in ix.items() if k.startswith("opening_hand_")]
    for row in r:
        if (row[ix["on_play"]] == "True") != bool(play):
            continue
        won, m = row[ix["won"]] == "True", int(float(row[ix["num_mulligans"]] or 0))
        if m == 0:
            acc["all kept 7s"][0] += won
            acc["all kept 7s"][1] += 1
            n = defaultdict(int)
            for i, k in oh:
                v = row[i]
                if v not in ("", "0"):
                    n[k] += int(float(v))
            if n["land"] != lands_want or n["cheap"] > cheap_max:
                continue
            g = f"keep 7: {lands_want} lands, <= {cheap_max} cheap, {'1+ big' if n['big'] else 'no big'}"
        else:
            g = f"mulligan x{m}"
        acc[g][0] += won
        acc[g][1] += 1
for g, (w, n) in sorted(acc.items()):
    if n:
        p = w / n
        print(f"{g:<42} {n:>7} games  WR {100*p:.1f} +/- {196*math.sqrt(p*(1-p)/n):.1f}")
