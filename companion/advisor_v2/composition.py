"""Bounded multipliers from what the lane's typical deck looks like (tools/synergy/build.py output).

After MIN_POOL cards, project the pool to a 23-spell deck and compare each curve
slot and the interaction count with the lane's typical deck. A card that fills a
slot the projection is short of gets up to +BOUND; one that adds to a slot
already over gets down to -BOUND. Lands and cards with no data get 1.0.
"""

import json
from pathlib import Path

ENABLED = False  # FRA 2026-11-01: -1.5 top-1 on the harness, no mock-deck gain. docs/engine-comparison-2026-11-01.md
MIN_POOL = 12
BOUND = 0.10
DECK_SPELLS = 23.0


def load(set_code, temp_folder):
    path = Path(temp_folder) / f"composition_{set_code}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def bucket(card):
    return str(min(6, max(1, int(card.get("cmc") or 0)))) if int(card.get("cmc") or 0) < 6 else "6+"


def is_interaction(card):
    return "removal" in (card.get("tags") or [])


class Composition:
    def __init__(self, typical, pool):
        self.typical = typical or {}
        self.spells = [c for c in pool if "Land" not in (c.get("types") or [])]
        self.scale = DECK_SPELLS / len(self.spells) if self.spells else 0.0
        self.curve = {}
        self.interaction = 0
        for c in self.spells:
            self.curve[bucket(c)] = self.curve.get(bucket(c), 0) + 1
            self.interaction += is_interaction(c)

    def multiplier(self, card, lane):
        t = self.typical.get(lane)
        if not ENABLED or not t or len(self.spells) < MIN_POOL or "Land" in (card.get("types") or []):
            return 1.0
        need = (t["curve"].get(bucket(card), 0) - self.scale * self.curve.get(bucket(card), 0)) / max(1.0, t["curve"].get(bucket(card), 1))
        m = 1.0 + BOUND * max(-1.0, min(1.0, need))
        if is_interaction(card) and t.get("interaction"):
            gap = (t["interaction"] - self.scale * self.interaction) / t["interaction"]
            m *= 1.0 + BOUND * max(-1.0, min(1.0, gap))
        return max(1.0 - 2 * BOUND, min(1.0 + 2 * BOUND, m))


ROLES = ("creatures", "removal", "card_advantage", "evasion", "two_drops", "five_plus", "lands")


def roles(card):
    """Which ROLES a card counts toward (deck-shape counts for tools/deck_targets.py and the overlay)."""
    types, tags, cmc = card.get("types") or [], card.get("tags") or [], int(card.get("cmc") or 0)
    if "Land" in types:
        return {"lands"}
    out = {t for t in ("removal", "card_advantage", "evasion") if t in tags}
    if "Creature" in types:
        out.add("creatures")
    if cmc == 2:
        out.add("two_drops")
    if cmc >= 5:
        out.add("five_plus")
    return out


def curve_slot(card):
    """A spell's mana-value slot, "1" .. "5" or "6+" (zero-cost spells count as 1); None for lands."""
    if "Land" in (card.get("types") or []):
        return None
    cmc = int(card.get("cmc") or 0)
    return "6+" if cmc >= 6 else str(max(1, cmc))
