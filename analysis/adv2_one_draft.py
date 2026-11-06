"""One saved adv1 draft, pick by pick (no new runs): player's pick, its rank in v2, v2's top 3, colours so far.

    python3 analysis/adv2_one_draft.py [draft_id | seed]    (a number picks a random sp3 draft with that seed)
"""
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "Temp" / "adv1"
ratings = json.load(open(ROOT / "data" / "Sets" / "FRA_PremierDraft_All_Data.json"))["card_ratings"]
colours = {}
for c in ratings.values():
    colours.setdefault(c["name"], "" if "Land" in (c.get("types") or []) else "".join(c.get("colors") or []))

drafts = {d["id"]: d for d in json.load(gzip.open(D / "sp3.drafts.json.gz"))}
meta = json.load(open(D / "meta.json"))
arg = sys.argv[1] if len(sys.argv) > 1 else "1"
did = arg if arg in drafts else random.Random(int(arg)).choice(sorted(drafts))
d, m = drafts[did], meta[did]
log = {(r[1], r[2]): r for r in json.load(open(D / "sp3.log.json")) if r[0] == did}

print(f"Draft {did[:8]}  ({m['kind']}; registered deck {m['lane']}" + (f" + {m['splash']}" if m.get("splash") else "") + f"; skill {m['skill']})\n")
print("| Pick | Player took | Rank | v2 top 3 (score) | Player colours so far | v2 top lane | Off-colour |")
print("|---|---|---|---|---|---|---|")
pool = []
for pack, pick, taken, _p, _pl in d["picks"]:
    r = log.get((pack, pick))
    cnt = Counter(ch for n in pool for ch in colours.get(n, ""))
    main = "".join(c for c, _ in cnt.most_common(2)) if len(pool) >= 3 else ""
    if r and r[3]:
        names = [x[0] for x in r[3]]
        rank = names.index(taken) + 1 if taken in names else "-"
        top3 = ", ".join(f"{n} ({s:.0f})" for n, s in r[3][:3])
        lane = max(r[5], key=r[5].get) if r[5] else ""
        off = []
        if main and set(colours.get(taken, "")) - set(main):
            off.append("player")
        if main and set(colours.get(names[0], "")) - set(main):
            off.append("engine")
        print(f"| P{pack}p{pick} | {taken} ({colours.get(taken, '') or 'C'}) | {rank} | {top3} | {main or '-'} | {lane} | {' + '.join(off)} |")
    pool.append(taken)
print("\nRegistered deck:", ", ".join(f"{v} {k}" if v > 1 else k for k, v in sorted(m["counts"].items())))
