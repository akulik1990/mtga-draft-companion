"""Of the picks where the player did not take v2's #1 (saved adv1 replay of 100 splash/three-colour drafts, no new runs):
how many were a mana-fixing card, and which kind. Also what v2's #1 was in those picks.

    uv run python analysis/adv2_mismatch_fixing.py

Kinds, from rules text (same signals as lanes.pool_fixing / any_colour_fixer):
  dual land      nonbasic land that adds one of two named colours
  multi land     nonbasic land adding any colour / three colours / fetching basics
  other land     nonbasic land with no colour choice (utility, mono)
  landcycler     'Basic landcycling' keyword or landcycling text
  mana source    spell with 'mana of any color'
  other fixing   Heartwood token makers, searchers of basic lands
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import video_draft as V  # noqa: E402

D = ROOT / "data" / "Temp" / "adv1"
rec_ds = "FRA_PremierDraft_All_Data.json"
metrics, cards = V.load_set(rec_ds)
drafts = json.load(gzip.open(D / "sp3.drafts.json.gz"))
meta = json.load(open(D / "meta.json"))
log = {(r[0], r[1], r[2]): r for r in json.load(open(D / "sp3.log.json"))}

names = {name for d in drafts for p in d["picks"] for name in [p[2]] + list(p[3])}
info = {c["name"]: c for c in cards(sorted(names))}


def kind(name):
    c = info.get(name)
    if not c:
        return None
    types, text = c.get("types") or [], (c.get("oracle_text") or "").lower()
    kws = c.get("keywords") or []
    if "Land" in types:
        if "Basic" in types:
            return None
        if "any color" in text or "chosen color" in text or "basic land card" in text or len(re.findall(r"\{[wubrg]\}", text.split("add")[-1])) >= 3:
            return "multi land"
        if re.search(r"add \{[wubrg]\} or \{[wubrg]\}", text):
            return "dual land"
        return "other land"
    if "Basic landcycling" in kws or "landcycling" in text:
        return "landcycler"
    if "mana of any color" in text:
        return "mana source"
    if "heartwood token" in text or "basic land cards with different names" in text:
        return "other fixing"
    return None


FIX = ("dual land", "multi land", "other land", "landcycler", "mana source", "other fixing")
tot = Counter()
by_group = defaultdict(Counter)
by_pack = defaultdict(Counter)
fix_rank = Counter()
fix_all = Counter()
v2_kind = Counter()
v2_also_fix = 0
examples = defaultdict(list)
for d in drafts:
    grp = meta[d["id"]]["kind"]
    for pack, pick, taken, _p, _pl in d["picks"]:
        r = log.get((d["id"], pack, pick))
        if not r or not r[3]:
            continue
        ranked = [x[0] for x in r[3]]
        top = ranked[0]
        rank = ranked.index(taken) if taken in ranked else 99
        k = kind(taken)
        if k:
            fix_all[k] += 1
            fix_rank[(k, "match" if rank == 0 else "miss")] += 1
        if rank == 0:
            continue
        tot["mismatch"] += 1
        by_group[grp]["mismatch"] += 1
        by_pack[pack]["mismatch"] += 1
        if k:
            tot["fix"] += 1
            tot[k] += 1
            by_group[grp]["fix"] += 1
            by_pack[pack]["fix"] += 1
            kt = kind(top)
            v2_kind[kt or "not fixing"] += 1
            if len(examples[k]) < 3:
                examples[k].append((d["id"][:8], pack, pick, taken, top, round(r[3][0][1] - dict(r[3]).get(taken, 0), 1), rank + 1 if rank < 99 else "-"))
    tot["picks"] += sum(1 for p in d["picks"] if log.get((d["id"], p[0], p[1])) and log[(d["id"], p[0], p[1])][3])


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


out = [f"Picks with a v2 ranking: {tot['picks']}; mismatches (player did not take v2's #1): {tot['mismatch']} ({pct(tot['mismatch'], tot['picks'])})", "",
       f"Of the {tot['mismatch']} mismatches, the player took a fixing card in {tot['fix']} ({pct(tot['fix'], tot['mismatch'])}):", "",
       "| Kind the player took | Mismatches | Share of mismatches | Player took it in all picks | v2 matched it |", "|---|---|---|---|---|"]
for k in FIX:
    allp = fix_all[k]
    out.append(f"| {k} | {tot[k]} | {pct(tot[k], tot['mismatch'])} | {allp} | {pct(fix_rank[(k, 'match')], allp)} |")
out += ["", "By draft type:", "", "| Group | Mismatches | Of which fixing |", "|---|---|---|"]
for g in ("splash", "three"):
    out.append(f"| {g} | {by_group[g]['mismatch']} | {by_group[g]['fix']} ({pct(by_group[g]['fix'], by_group[g]['mismatch'])}) |")
out += ["", "By pack:", "", "| Pack | Mismatches | Of which fixing |", "|---|---|---|"]
for p in (1, 2, 3):
    out.append(f"| {p} | {by_pack[p]['mismatch']} | {by_pack[p]['fix']} ({pct(by_pack[p]['fix'], by_pack[p]['mismatch'])}) |")
out += ["", f"In the {tot['fix']} fixing mismatches, v2's #1 was: " + ", ".join(f"{k} {n}" for k, n in v2_kind.most_common()), "", "Examples (draft, pick, player took, v2 #1, score gap, v2's rank of the player's card):"]
for k, ex in examples.items():
    for e in ex:
        out.append(f"- {k}: {e[0]} P{e[1]}p{e[2]}: took {e[3]}; v2 #1 {e[4]} (gap {e[5]}, rank {e[6]})")
text = "\n".join(out)
(ROOT / "analysis" / "out" / "adv2_mismatch_fixing.txt").write_text(text + "\n")
print(text)
