"""Deck needs: what a card adds to a deck's make-up, against top players' decks in the pair, and which cards top players put in
the same deck (deck-mates).

DIMS are the deck-shape counts top players' decks are measured on (tools/deck_targets.py writes their mean per pair as
"dims" in the set's fitted decks section, and the deck-mate lifts to <data>/Temp/deckmates_<SET>.json).
"""

import json
import math
import re
from functools import lru_cache

from companion.advisor_v2.lanes import any_colour_fixer

DIMS = ("early_creatures", "late_creatures", "noncreature", "one_drops", "two_drops", "three_drops", "four_drops", "five_plus",
        "removal", "card_advantage", "evasion", "combat_trick", "lifegain", "cycling", "finishers", "fixing")
ALPHA = 5.0  # deck-mate lift prior: (together + ALPHA) / (expected + ALPHA)


@lru_cache(maxsize=4)
def set_wr(set_code):
    """(mean, SD) of All Decks GIH WR over the set's cards with data: the finisher bar."""
    from companion import DATA_DIR

    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    wrs = [w for c in data.values() if (w := ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0) > 0]
    mean = sum(wrs) / len(wrs)
    return mean, (sum((w - mean) ** 2 for w in wrs) / len(wrs)) ** 0.5


def dims(card, mean, sd):
    """The DIMS a card counts toward. Finishers: five-plus drops at least one SD above the set's mean GIH WR."""
    types, tags = card.get("types") or [], card.get("tags") or []
    if "Land" in types:
        fixes = len(card.get("colors") or []) >= 2 or "fixing_ramp" in tags or any_colour_fixer(card)
        return {"fixing"} if fixes and "Basic" not in types else set()
    cmc = int(card.get("cmc") or 0)
    out = {t for t in ("removal", "card_advantage", "evasion", "combat_trick", "lifegain") if t in tags}
    if "Creature" in types:
        out.add("early_creatures" if cmc <= 3 else "late_creatures")
    else:
        out.add("noncreature")
    out.add(("one_drops", "one_drops", "two_drops", "three_drops", "four_drops")[cmc] if cmc <= 4 else "five_plus")
    if re.search(r"cycling", (card.get("oracle_text") or "").lower()):
        out.add("cycling")
    gih = ((card.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
    if cmc >= 5 and gih >= mean + sd:
        out.add("finishers")
    if "fixing_ramp" in tags or any_colour_fixer(card):
        out.add("fixing")
    return out


@lru_cache(maxsize=4)
def deckmates(set_code):
    """{pair: {card: {other: log lift}}}; empty without tools/deck_targets.py output."""
    from companion import DATA_DIR

    path = DATA_DIR / "Temp" / f"deckmates_{set_code}.json"
    if not path.exists():
        return {}
    out = {}
    for lane, rows in json.loads(path.read_text()).items():
        m = out.setdefault(lane, {})
        for a, b, v in rows:
            m.setdefault(a, {})[b] = v
            m.setdefault(b, {})[a] = v
    return out


def mate_score(card, pool, lane, set_code):
    """How much top players' decks in the lane pair this card with the pool's cards: the sum of log lifts, per 23 cards."""
    table = deckmates(set_code).get(lane) or {}
    row = table.get(card["name"]) or table.get(card["name"].split(" // ")[0]) or {}
    if not row:
        return 0.0
    return sum(row.get(c["name"], row.get(c["name"].split(" // ")[0], 0.0)) for c in pool) / 23.0


def lift(n_ab, n_a, n_b, n):
    return math.log((n_ab + ALPHA) / (n_a * n_b / n + ALPHA))


def fills(card, pool, lane, target, mean, sd):
    """Per DIM, how much of what the deck still lacks there this card fills: 1 if the card counts toward it, times the
    share of the deck's open slots that must still go to it (top players' count in the pair less the pool's own, in
    cards castable in the pair, over 23 minus those cards; negative once the pool has more than they play)."""
    from companion.advisor_v2.lanes import castable

    mine = dims(card, mean, sd)
    if not mine or not target:
        return [0.0] * len(DIMS)
    spells = [c for c in pool if "Land" not in (c.get("types") or []) and castable(c, lane)]
    lands = [c for c in pool if "Land" in (c.get("types") or [])]
    have = dict.fromkeys(DIMS, 0)
    for c in spells + lands:
        for d in dims(c, mean, sd):
            have[d] += 1
    open_slots = max(1.0, 23.0 - len(spells))
    return [max(-1.0, min(2.0, (target.get(d, 0.0) - have[d]) / open_slots)) if d in mine else 0.0 for d in DIMS]
