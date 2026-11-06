"""Per-pick table on the saved adv1 replay (no new runs): rank of the player's pick in v2, and off-colour picks.

    python3 analysis/adv2_pick_table.py sp3|two

Off-colour ("splashed") at a pick: the card has a colour outside the top two colours of the player's pool so far
(coloured cards counted; colourless and lands never count). Same pool for both sides, so "engine splashed" means
v2's #1 card is outside those two colours. Picks 1-3 have no colours yet and show "-".
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "Temp" / "adv1"
tag = sys.argv[1] if len(sys.argv) > 1 else "sp3"

ratings = json.load(open(ROOT / "data" / "Sets" / "FRA_PremierDraft_All_Data.json"))["card_ratings"]
colours = {}
for c in ratings.values():
    if "Land" in (c.get("types") or []):
        colours[c["name"]] = set()
    else:
        colours.setdefault(c["name"], set(c.get("colors") or []))

drafts = json.load(gzip.open(D / f"{tag}.drafts.json.gz"))
rank = {}
for d in json.load(open(D / f"{tag}.log.replay.json"))["drafts"]:
    for p in d["picks"]:
        rank[(d["id"], p[0], p[1])] = p[2]
top = {}
for row in json.load(open(D / f"{tag}.log.json")):
    if row[3]:
        top[(row[0], row[1], row[2])] = row[3][0][0]

rows = defaultdict(Counter)
for d in drafts:
    pool = []
    for pack, pick, taken, _pack, _pool in d["picks"]:
        n = (pack - 1) * 14 + pick
        key = (d["id"], pack, pick)
        r = rank.get(key)
        if r is not None:
            rows[n]["n"] += 1
            rows[n][("#1", "#2", "#3")[r] if r < 3 else "other"] += 1
        if len(pool) >= 3 and key in top:
            cnt = Counter(col for name in pool for col in colours.get(name, ()))
            main = {c for c, _ in cnt.most_common(2)}
            off_p = bool(colours.get(taken, set()) - main)
            off_e = bool(colours.get(top[key], set()) - main)
            rows[n]["cn"] += 1
            if off_p and off_e:
                rows[n]["both"] += 1
            elif off_p:
                rows[n]["player"] += 1
            elif off_e:
                rows[n]["engine"] += 1
        pool.append(taken)


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


out = ["| Pick | n | #1 | #2 | #3 | Other | Both splashed | Player splashed | Engine splashed |",
       "|---|---|---|---|---|---|---|---|---|"]
tot = Counter()
for n in sorted(rows):
    c = rows[n]
    tot.update(c)
    label = f"P{(n - 1) // 14 + 1}p{(n - 1) % 14 + 1}"
    out.append(f"| {label} | {c['n']} | {pct(c['#1'], c['n'])} | {pct(c['#2'], c['n'])} | {pct(c['#3'], c['n'])} | "
               f"{pct(c['other'], c['n'])} | {pct(c['both'], c['cn'])} | {pct(c['player'], c['cn'])} | {pct(c['engine'], c['cn'])} |")
c = tot
out.append(f"| All | {c['n']} | {pct(c['#1'], c['n'])} | {pct(c['#2'], c['n'])} | {pct(c['#3'], c['n'])} | "
           f"{pct(c['other'], c['n'])} | {pct(c['both'], c['cn'])} | {pct(c['player'], c['cn'])} | {pct(c['engine'], c['cn'])} |")
text = "\n".join(out)
(ROOT / "analysis" / "out" / f"adv2_pick_table_{tag}.txt").write_text(text + "\n")
print(text)
