"""Fblthp, Knows the Way among top drafters (17Lands FRA Premier top draft log joined to the game data by draft_id): when it is in the
pack, how often it is taken, by the drafter's registered deck (green or not; two colours clean / splash / three+) and by pack and pick;
and, when taken, how often it is in the registered deck. Streams both files (memory rule).
Run: python3 analysis/fblthp_check.py"""
import csv, gzip, math, os
from collections import defaultdict

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "Fblthp, Knows the Way"
deck = {}  # draft_id -> (main colours, splash colours, plays Fblthp)
with gzip.open(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", "rt", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    ix = {k: i for i, k in enumerate(h)}
    fd = ix[f"deck_{NAME}"]
    for row in r:
        deck.setdefault(row[ix["draft_id"]], (row[ix["main_colors"]], row[ix["splash_colors"]], row[fd] not in ("", "0")))

def group(d):
    mc, sp, _ = d
    if "G" not in mc + sp:
        return "no green"
    if len(mc) >= 3:
        return "green, 3+ main colours"
    return "green, two colours + splash" if sp else "green, two colours clean"

seen, took, played = defaultdict(int), defaultdict(int), defaultdict(int)
with gzip.open(f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.top.csv.gz", "rt", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    ix = {k: i for i, k in enumerate(h)}
    pc = ix[f"pack_card_{NAME}"]
    for row in r:
        if row[pc] in ("", "0"):
            continue
        d = deck.get(row[ix["draft_id"]])
        if not d:
            continue
        pick = int(row[ix["pick_number"]]) + 1
        for key in (group(d), f"{group(d)} | pack {int(row[ix['pack_number']]) + 1}", f"{group(d)} | pick {'1-3' if pick <= 3 else '4-7' if pick <= 7 else '8+'}"):
            seen[key] += 1
            if row[ix["pick"]] == NAME:
                took[key] += 1
                played[key] += d[2]
for k in sorted(seen):
    n, t = seen[k], took[k]
    p = t / n
    print(f"{k:<42} in pack {n:>5}  taken {100*p:5.1f}% +/- {196*math.sqrt(p*(1-p)/n):4.1f}   in deck when taken {100*played[k]/t if t else float('nan'):5.1f}%")

# Skipped by green drafters: what they took instead, by GIH WR in their deck's main colours (All Decks fallback).
import json
cards = {c["name"]: c for c in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}
def gih(n, lane):
    dc = (cards.get(n) or {}).get("deck_colors") or {}
    return (dc.get(lane) or {}).get("gihwr") or (dc.get("All Decks") or {}).get("gihwr") or 0.0
inst = defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])  # n skipped, took stronger, took green, sum gap, took land, took rare+
with gzip.open(f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.top.csv.gz", "rt", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    ix = {k: i for i, k in enumerate(h)}
    pc = ix[f"pack_card_{NAME}"]
    for row in r:
        if row[pc] in ("", "0") or row[ix["pick"]] == NAME:
            continue
        d = deck.get(row[ix["draft_id"]])
        if not d or group(d) == "no green":
            continue
        lane, t = d[0][:2], row[ix["pick"]]
        g, gf = gih(t, d[0]), gih(NAME, d[0])
        c = cards.get(t) or {}
        x = inst[group(d)]
        x[0] += 1; x[1] += g > gf; x[2] += "G" in (c.get("colors") or []); x[3] += (g - gf) if g else 0.0
        x[4] += "Land" in (c.get("types") or []); x[5] += c.get("rarity") in ("rare", "mythic")
print()
for k, (n, s, gr, gap, land, rare) in sorted(inst.items()):
    print(f"skipped, {k:<34} n {n:>4}: took a stronger card {100*s/n:.0f}%, mean GIH gap {gap/n:+.1f}, took green {100*gr/n:.0f}%, a land {100*land/n:.0f}%, a rare/mythic {100*rare/n:.0f}%")

# Synergy at the time of the pick: the pool's third-largest colour has 4+ coloured non-land cards (post hoc; the pre-registered
# rule, 3+ colours with 2+ cards each, split almost no pools after pack 1). Green drafters only (registered deck plays green). Take rate by pack x pool.
col = {n: set(c.get("colors") or []) for n, c in cards.items() if "Land" not in (c.get("types") or [])}
tr = defaultdict(lambda: [0, 0])
with gzip.open(f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.top.csv.gz", "rt", newline="") as f:
    r = csv.reader(f)
    h = next(r)
    ix = {k: i for i, k in enumerate(h)}
    pc = ix[f"pack_card_{NAME}"]
    pools = [(i, k[len("pool_"):]) for k, i in ix.items() if k.startswith("pool_")]
    for row in r:
        if row[pc] in ("", "0"):
            continue
        d = deck.get(row[ix["draft_id"]])
        if not d or group(d) == "no green":
            continue
        n = defaultdict(int)
        for i, name in pools:
            v = row[i]
            if v not in ("", "0"):
                for c in col.get(name, ()):
                    n[c] += int(float(v))
        third = sorted(n.values(), reverse=True)[2] if len(n) >= 3 else 0
        multi = third >= 4  # post hoc (2026-11-06): the first rule (3+ colours with 2+ cards) held for almost every pool by pack 2
        pk, pick = int(row[ix["pack_number"]]) + 1, int(row[ix["pick_number"]]) + 1
        for key in (f"pack {pk} | {'third colour 4+' if multi else 'third colour <= 3'}",
                    f"pack {pk} pick {'1-6' if pick <= 6 else '7+'} | {'third colour 4+' if multi else 'third colour <= 3'}"):
            tr[key][0] += 1
            tr[key][1] += row[ix["pick"]] == NAME
print()
for k, (n, t) in sorted(tr.items()):
    p = t / n
    print(f"green drafters, {k:<40} in pack {n:>4}  taken {100*p:5.1f}% +/- {196*math.sqrt(p*(1-p)/n):4.1f}")
