"""Per pick on the saved adv1 replay (no new runs): did the player take a fixing card, did v2's #1 pick one.

    uv run python analysis/adv2_fixing_cells.py

Two views: 'land' = nonbasic land of any kind; 'fixing' = land or landcycler / any-colour mana / token and basic-land
fixers (same text signals as lanes.pool_fixing). Classification as in analysis/adv2_mismatch_fixing.py.
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
metrics, cards = V.load_set("FRA_PremierDraft_All_Data.json")
drafts = json.load(gzip.open(D / "sp3.drafts.json.gz"))
meta = json.load(open(D / "meta.json"))
log = {(r[0], r[1], r[2]): r for r in json.load(open(D / "sp3.log.json"))}
names = {n for d in drafts for p in d["picks"] for n in [p[2]] + list(p[3])}
info = {c["name"]: c for c in cards(sorted(names))}


def kind(name):
    c = info.get(name)
    if not c:
        return None
    types, text = c.get("types") or [], (c.get("oracle_text") or "").lower()
    if "Land" in types:
        if "Basic" in types:
            return None
        if "any color" in text or "chosen color" in text or "basic land card" in text or len(re.findall(r"\{[wubrg]\}", text.split("add")[-1])) >= 3:
            return "multi land"
        return "dual land" if re.search(r"add \{[wubrg]\} or \{[wubrg]\}", text) else "other land"
    if "Basic landcycling" in (c.get("keywords") or []) or "landcycling" in text:
        return "landcycler"
    if "mana of any color" in text:
        return "mana source"
    if "heartwood token" in text or "basic land cards with different names" in text:
        return "other fixing"
    return None


def is_land(k):
    return k in ("dual land", "multi land", "other land")


def cells(view, flt=lambda r, pack, pick, d: True):
    test = (lambda k: k is not None) if view == "fixing" else is_land
    c = Counter()
    for d in drafts:
        for pack, pick, taken, _p, _pl in d["picks"]:
            r = log.get((d["id"], pack, pick))
            if not r or not r[3] or not flt(r, pack, pick, d):
                continue
            top = r[3][0][0]
            pf, ef = test(kind(taken)), test(kind(top))
            c["n"] += 1
            if pf and ef:
                c["both_same" if taken == top else "both_diff"] += 1
            elif pf:
                c["player"] += 1
            elif ef:
                c["engine"] += 1
            else:
                c["neither"] += 1
    return c


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def row(label, c):
    n = c["n"]
    both = c["both_same"] + c["both_diff"]
    return (f"| {label} | {n} | {pct(both, n)} ({pct(c['both_same'], n)} same card, {pct(c['both_diff'], n)} different) | "
            f"{pct(c['player'], n)} | {pct(c['engine'], n)} | {pct(c['neither'], n)} |")


head = ["| | Picks | Both took fixing | Player only | Engine only | Neither |", "|---|---|---|---|---|---|"]
out = []
for view, title in (("land", "LANDS (any nonbasic land)"), ("fixing", "ALL FIXING (lands + landcyclers + any-colour mana + token/basic searchers)")):
    out += ["", title, ""] + head + [row("All picks", cells(view))]
    for pack in (1, 2, 3):
        out.append(row(f"Pack {pack}", cells(view, lambda r, p, k, d, pack=pack: p == pack)))
    for g in ("splash", "three"):
        out.append(row(f"{g} drafts", cells(view, lambda r, p, k, d, g=g: meta[d["id"]]["kind"] == g)))

# engine-only fixing: which kind v2 picked, and what the player took instead (colour-wise: in his final deck or not)
eng_kind, plr_taken, plr_rank = Counter(), Counter(), []
for d in drafts:
    final = set(meta[d["id"]]["counts"])
    for pack, pick, taken, _p, _pl in d["picks"]:
        r = log.get((d["id"], pack, pick))
        if not r or not r[3]:
            continue
        top = r[3][0][0]
        ke, kp = kind(top), kind(taken)
        if ke and not kp:
            eng_kind[ke] += 1
            plr_taken["in his final deck" if taken in final else "not in his final deck"] += 1
            plr_rank.append(r[3][0][1] - dict(r[3]).get(taken, 0))
            eng_kind[("top_in_final", top in final)] += 1
out += ["", "When v2's #1 is a fixing card and the player took something else (engine only):"]
tot = sum(v for k, v in eng_kind.items() if isinstance(k, str))
out.append("  v2's #1 was: " + ", ".join(f"{k} {v}" for k, v in eng_kind.most_common() if isinstance(k, str)))
out.append(f"  the fixer v2 wanted ended in his final deck: {pct(eng_kind[('top_in_final', True)], tot)}; his card was in his final deck: {pct(plr_taken['in his final deck'], tot)}")
plr_rank.sort()
out.append(f"  median score gap between v2's fixer and his card: {plr_rank[len(plr_rank) // 2]:.1f} points")
text = "\n".join(out)
(ROOT / "analysis" / "out" / "adv2_fixing_cells.txt").write_text(text + "\n")
print(text)
