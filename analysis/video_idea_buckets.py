"""Group the video drafters' ideas (analysis/videos/*/ideas.json) into buckets of the same point, rank by how many videos repeat it.

    python3 analysis/video_idea_buckets.py

Strategy buckets are assigned by hand from reading every non-card idea (ids below); an idea can sit in two buckets when it
makes two points. Card ideas are grouped per card: a card counts for a video when an idea in that video names it in its
`cards` and its claim is about that card (the name appears in the first part of the claim).
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "analysis" / "videos"
index = {x["id"]: x for x in json.load(open(V / "index.json"))}
ideas = {}
for vid in index:
    d = json.load(open(V / vid / "ideas.json"))
    for i in d["ideas"] if isinstance(d, dict) else d:
        i["_v"] = vid
        ideas[i["id"]] = i

BUCKETS = {
    "Fixing / dual land over a marginal playable (when the pack has nothing, playables are covered, or a splash is likely)":
        "mqZ-16 CWs-12 CWs-19 CWs-24 CWs-36 994-11 994-25 re7-11 re7-23 5aM-11 5aM-14 uoy-17 Uy4-06 n6s-17 n6s-31 9nA-13 55a-14 55a-21 "
        "55a-23 55a-29 55a-32 97Z-22 XSY-15 VTU-18 VTU-19 CLK-14 I_C-18 I_C-23 kMo-20 9Db-14 9Db-20 7Yc-08 7Yc-12 7Yc-15 7Yc-18",
    "Signals: read what is missing from a pack and what arrives or wheels late":
        "mqZ-06 mqZ-23 CWs-08 CWs-15 CWs-16 CWs-20 CWs-38 994-13 5aM-03 pdI-18 pdI-25 55a-13 97Z-03 97Z-08 97Z-12 XSY-10 XSY-12 XSY-18 "
        "XSY-19 VTU-05 8Bz-09 CLK-06 CLK-07 I_C-08 kMo-06 kMo-09 kMo-12 kMo-13 9Db-09 7Yc-07 7Yc-13 iGz-07 n6s-11 n6s-29",
    "Removal: take it highly until about 5-6, then other needs come first":
        "mqZ-51 mqZ-62 CWs-11 CWs-31 994-02 994-05 994-06 994-16 994-27 994-32 re7-13 re7-14 re7-21 5aM-04 5aM-16 uoy-05 uoy-19 uoy-23 "
        "uoy-25 uoy-30 n6s-14 pdI-21 pdI-23 pdI-28 9nA-15 9nA-27 9nA-29 55a-30 I_C-13 kMo-15 9Db-02 CLK-14 CLK-19 CLK-20 7Yc-12 8Bz-14 97Z-02",
    "Two-drops are scarce: fill the curve (about 6-7), stop once you have enough":
        "mqZ-07 mqZ-33 mqZ-52 mqZ-59 CWs-45 994-08 5aM-09 5aM-17 5aM-20 uoy-07 n6s-03 n6s-12 n6s-15 n6s-25 n6s-34 pdI-27 9nA-31 9nA-36 "
        "55a-22 55a-37 97Z-10 97Z-31 VTU-10 VTU-28 8Bz-19 8Bz-28 CLK-04 CLK-08 CLK-23 I_C-15 I_C-18",
    "Stay flexible early: take power, keep first picks loose, drop them if their colours do not flow":
        "CWs-04 CWs-06 mqZ-07 n6s-01 n6s-06 pdI-03 pdI-09 pdI-10 97Z-06 Uy4-01 I_C-04 I_C-05 iGz-01 iGz-12 7Yc-06 XSY-13 55a-17 VTU-14 "
        "VTU-20 8Bz-05",
    "Count colour sources for the mana base (splash about 5-6, main colours 7-10; landcyclers and rocks count)":
        "mqZ-60 CWs-42 994-29 994-30 re7-26 re7-29 re7-30 Uy4-13 n6s-35 55a-33 55a-41 VTU-26 VTU-27 kMo-24 9Db-23 7Yc-20 7Yc-21",
    "Splash discipline: only the best cards, cap the splash, fewer splashes the more playables you have":
        "mqZ-25 mqZ-30 mqZ-50 CWs-10 CWs-33 CWs-46 994-14 994-17 994-26 97Z-09 97Z-27 CLK-09 CLK-24 I_C-19 n6s-26 n6s-32 uoy-14 iGz-13",
    "Basic landcyclers: real fixing, about two per deck, being taken higher; count them as part-lands":
        "mqZ-29 mqZ-53 mqZ-66 mqZ-72 CWs-03 994-30 re7-09 re7-20 Uy4-04 Uy4-11 9Db-07 9Db-08 9Db-12 9Db-13 iGz-04 kMo-07 kMo-22 8Bz-25",
    "Commit with reasons: do not jump colours for one card, do not over-read openness":
        "uoy-21 uoy-22 9nA-06 9nA-11 55a-04 55a-07 55a-09 9Db-04 XSY-26 CLK-05 n6s-07 VTU-22",
    "Creature count when building (about 13-16; cut the clunky creature at 16)":
        "mqZ-46 CWs-41 CWs-45 5aM-24 pdI-30 9nA-31 97Z-34 VTU-24 I_C-21 n6s-36",
    "Top end: do not take 6-7 drops highly, lock up one or two finishers, then curve":
        "mqZ-08 mqZ-24 994-04 994-24 re7-21 9nA-19 9nA-36 55a-18 55a-22 CLK-04 I_C-21 I_C-22 9Db-08",
    "Double pips set the base colours; splash single-pip cards (double splash only for the very best)":
        "mqZ-32 mqZ-39 mqZ-61 re7-18 re7-22 7Yc-05 7Yc-09 iGz-11 iGz-13 8Bz-24 CLK-24 kMo-24",
    "One splash colour, carried by two or more good cards; pick it by what it brings (often removal)":
        "mqZ-40 mqZ-44 mqZ-58 mqZ-68 CWs-22 CWs-28 CWs-30 994-31 re7-27 55a-38 kMo-11 8Bz-11",
    "Count playables during the draft to know how selective to be":
        "n6s-22 9nA-23 9nA-27 55a-12 55a-26 97Z-27 7Yc-15 9Db-20 I_C-18",
    "17 lands as the default (adjust for landcyclers, tapped lands, curve)":
        "mqZ-38 5aM-22 5aM-25 5aM-26 uoy-28 uoy-32 Uy4-11 9nA-39 kMo-22 7Yc-19 7Yc-21",
    "Card draw: valuable in ramp, splash and control decks, but enough is enough":
        "mqZ-57 CWs-31 CWs-39 CWs-44 994-15 994-24 re7-10 re7-25 re7-26 5aM-26 55a-24",
    "Late speculative picks are free when nothing is for the deck":
        "5aM-11 uoy-26 n6s-27 pdI-08 9nA-14 CLK-10 CLK-18",
    "The format is balanced: no dominant colour, four- and five-colour decks are viable":
        "mqZ-65 re7-03 re7-18 Uy4-07 XSY-01 8Bz-01 8Bz-12 9Db-22",
    "Enters-tapped lands cost tempo: about two tapped duals at most":
        "mqZ-69 mqZ-38 994-22 994-29 re7-17 re7-28 n6s-26",
    "Hate / sideboard cards: not main-deck in best of one, worth a pick in best of three":
        "994-28 8Bz-22 CLK-21 XSY-34",
    "Build technique: cut everything, add back bombs, removal and two-drops first":
        "n6s-33 pdI-26 55a-36 iGz-17",
    "Aggressive decks are good right now":
        "5aM-21 uoy-02 uoy-11",
}
SIDEBOARD_WORDS = ("sideboard", "best-of-three", "best of three", "hate")


def ids(s):
    return [x for x in s.split() if x in ideas]


bad = [x for s in BUCKETS.values() for x in s.split() if x not in ideas]
assert not bad, f"unknown ids: {bad}"
for i in ideas.values():  # card-evaluation ideas about hate cards join that bucket
    if i["kind"] == "card evaluation" and any(w in i["claim"].lower() for w in SIDEBOARD_WORDS):
        BUCKETS["Hate / sideboard cards: not main-deck in best of one, worth a pick in best of three"] += " " + i["id"]


def nvid(members):
    return len({ideas[m]["_v"] for m in members})


ranked = sorted(((b, ids(s)) for b, s in BUCKETS.items()), key=lambda kv: (-nvid(kv[1]), -len(kv[1])))
out = [f"{len(ideas)} ideas from {len(index)} videos.", "", "## Strategy ideas, by number of videos that make the point", "",
       "| # | Idea | Videos | Ideas |", "|---|---|---|---|"]
for n, (b, m) in enumerate(ranked, 1):
    out.append(f"| {n} | {b} | {nvid(m)} | {len(m)} |")
out += [""]
for b, m in ranked:
    out.append(f"### {b} ({nvid(m)} videos)")
    seen = set()
    for x in m:
        if ideas[x]["_v"] in seen:
            continue
        seen.add(ideas[x]["_v"])
        out.append(f"  - [{x}] {ideas[x]['claim'][:180]}")
        if len(seen) >= 4:
            break
    out.append("")

cards = defaultdict(set)
cardideas = defaultdict(list)
for i in ideas.values():
    head = i["claim"][:70]
    for c in i.get("cards") or []:
        if c.split(",")[0] in head:
            cards[c].add(i["_v"])
            cardideas[c].append(i)
out += ["## Cards discussed in the most videos", "", "| Card | Videos | Typical claim |", "|---|---|---|"]
for c, vs in sorted(cards.items(), key=lambda kv: -len(kv[1]))[:20]:
    out.append(f"| {c} | {len(vs)} | {cardideas[c][0]['claim'][:120]} |")
assigned = {x for s in BUCKETS.values() for x in s.split()}
rest = [i for i in ideas.values() if i["id"] not in assigned and i["kind"] != "card evaluation"]
out += ["", f"Not in a bucket: {len(rest)} strategy ideas (one-off points, synergy pairs, card-specific picks)."]
text = "\n".join(out)
(ROOT / "analysis" / "out" / "video_idea_buckets.txt").write_text(text + "\n")
print(text)
