"""Bucket every pick where the player did not take v2's #1 (saved adv1 replay, no new runs).

    uv run python analysis/adv2_mismatch_buckets.py

First rule that fits, in this order (fixed before looking):
  1 near-tie      v2's score gap between its #1 and the player's card is under NEAR points
  2 fixing        the player's card is a fixing card (dual, multi, landcycler, any-colour mana, token/basic searcher)
  3 not in deck   the player's card is not in his final 40
  4 engine off    his card is inside his final colours, v2's #1 has a colour outside them
  5 he went wide  his card has a colour outside his top two (by final-deck cards), v2's #1 is inside the top two
  6 same colours  everything else
"Final colours" = registered lane + splash (meta.json); top two = the two colours with most coloured cards in his final 40.
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

NEAR = 2.0
D = ROOT / "data" / "Temp" / "adv1"
metrics, cards = V.load_set("FRA_PremierDraft_All_Data.json")
drafts = json.load(gzip.open(D / "sp3.drafts.json.gz"))
meta = json.load(open(D / "meta.json"))
log = {(r[0], r[1], r[2]): r for r in json.load(open(D / "sp3.log.json"))}
names = {n for d in drafts for p in d["picks"] for n in [p[2]] + list(p[3])}
info = {c["name"]: c for c in cards(sorted(names))}


def fixing(name):
    c = info.get(name)
    if not c:
        return False
    types, text = c.get("types") or [], (c.get("oracle_text") or "").lower()
    if "Land" in types:
        return "Basic" not in types
    return ("Basic landcycling" in (c.get("keywords") or []) or "landcycling" in text or "mana of any color" in text
            or "heartwood token" in text or "basic land cards with different names" in text)


def cols(name):
    return set((info.get(name) or {}).get("colors") or [])


B = ["1 near-tie", "2 fixing", "3 not in his deck", "4 engine off his colours", "5 he went wide", "6 same colours"]
cnt, by_pack, ex = Counter(), defaultdict(Counter), defaultdict(list)
six = Counter()
gaps = defaultdict(list)
pos, lowscore, took, v2took = defaultdict(Counter), Counter(), Counter(), Counter()
for d in drafts:
    m = meta[d["id"]]
    final_cols = set(m["lane"]) | set(m.get("splash") or "")
    weight = Counter(c for n, k in m["counts"].items() for c in cols(n) for _ in range(k) if "Land" not in (info.get(n, {}).get("types") or []))
    top2 = {c for c, _ in weight.most_common(2)}
    for pack, pick, taken, _p, _pl in d["picks"]:
        r = log.get((d["id"], pack, pick))
        if not r or not r[3]:
            continue
        ranked = r[3]
        top = ranked[0][0]
        if taken == top:
            continue
        sc = dict(ranked)
        gap = ranked[0][1] - sc.get(taken, 0.0)
        if gap < NEAR:
            b = B[0]
        elif fixing(taken):
            b = B[1]
        elif taken not in m["counts"]:
            b = B[2]
        elif cols(taken) <= final_cols and not cols(top) <= final_cols:
            b = B[3]
        elif cols(taken) - top2 and cols(top) <= top2:
            b = B[4]
        else:
            b = B[5]
            a, c = info.get(top, {}), info.get(taken, {})
            kt = lambda x: "creature" if "Creature" in (x.get("types") or []) else "other spell"
            six[f"v2 {kt(a)}, he {kt(c)}"] += 1
            six["his card cheaper" if (c.get("cmc") or 0) < (a.get("cmc") or 0) else "his card same/dearer"] += 1
            six["he took gold" if len(cols(taken)) > 1 else "he took mono/colourless"] += 1
            six["v2 took gold" if len(cols(top)) > 1 else "v2 took mono/colourless"] += 1
        stage = "early (p1-4)" if pick <= 4 else "middle (p5-9)" if pick <= 9 else "late (p10-14)"
        pos[b][stage] += 1
        if b == B[2]:
            lowscore[(pack, "his card v2 score < 35" if sc.get(taken, 0) < 35 else "his card v2 score >= 35")] += 1
        if b == B[5]:
            took[taken] += 1
            v2took[top] += 1
        cnt[b] += 1
        by_pack[pack][b] += 1
        gaps[b].append(gap)
        if len(ex[b]) < 3:
            ex[b].append(f"{d['id'][:8]} P{pack}p{pick}: he took {taken}, v2 #1 {top} (gap {gap:.1f}); his final {m['lane']}" + (f"+{m['splash']}" if m.get("splash") else ""))
total = sum(cnt.values())
pc = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
out = [f"Mismatches: {total} (of 3,900 ranked picks)", "", "| Bucket | Picks | Share | Pack 1 | Pack 2 | Pack 3 | Median gap (pts) |", "|---|---|---|---|---|---|---|"]
for b in B:
    g = sorted(gaps[b])
    out.append(f"| {b} | {cnt[b]} | {pc(cnt[b], total)} | {pc(by_pack[1][b], sum(by_pack[1].values()))} | {pc(by_pack[2][b], sum(by_pack[2].values()))} | "
               f"{pc(by_pack[3][b], sum(by_pack[3].values()))} | {g[len(g) // 2]:.1f} |" if g else f"| {b} | 0 | | | | | |")
out += ["", "Bucket 6 (same colours) detail:"] + [f"  {k}: {v} ({pc(v, cnt[B[5]])})" for k, v in sorted(six.items())]
out += ["", "Where in the pack each bucket sits (share of the bucket):"]
for b in B:
    out.append(f"  {b}: " + ", ".join(f"{s} {pc(pos[b][s], cnt[b])}" for s in ("early (p1-4)", "middle (p5-9)", "late (p10-14)")))
n3 = sum(v for k, v in lowscore.items())
out += ["", "Bucket 3 (not in his deck): how highly v2 scored the card he took (score < 35 = v2 thought it was weak):"]
agg = Counter()
for (_pk, label), v in lowscore.items():
    agg[label] += v
out += [f"  {label}: {v} ({pc(v, n3)})" for label, v in sorted(agg.items())]
out += ["", "Bucket 6, cards he took most often instead:  " + ", ".join(f"{k} {v}" for k, v in took.most_common(8))]
out += ["Bucket 6, v2's #1 most often:  " + ", ".join(f"{k} {v}" for k, v in v2took.most_common(8))]
out += ["", "Examples:"]
for b in B:
    out += [f"- {b}"] + [f"    {e}" for e in ex[b]]
text = "\n".join(out)
(ROOT / "analysis" / "out" / "adv2_mismatch_buckets.txt").write_text(text + "\n")
print(text)
