"""P(lane) over the ten colour pairs, from pool commitment, signals and the format."""

import math
import re
from collections import Counter

LANES = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
USE_PIPS = True  # round 2: on together with land fixing (it was contested alone in round 1). Castability from mana-cost pips (hybrid aware); False = every colour of the card must be in the lane
ALL = "All Decks"


def all_decks(card):
    return (card.get("deck_colors") or {}).get(ALL) or {}


def pips(card):
    """Coloured pips as lists of options: {U} -> [U], {W/U} -> [W, U]. A land's colours count as pips.

    Generic, X, colourless, twobrid ({2/W}) and Phyrexian ({W/P}) pips are payable
    without the colour, so they are skipped, as upstream's get_strict_colors does.
    Without a mana cost, each colour counts as one pip. An "A // B" card needs every colour
    either face needs (as upstream's mana_base.is_castable judges it), but only as many pips
    of each as the hungrier face: {1}{B} // {B} is a one-pip splash, not two.
    """
    cost = card.get("mana_cost") or ""
    if "Land" in (card.get("types") or []):
        # A land's colours are what it produces; it belongs in a lane only when all of them fit (a WU dual is not a WB card).
        return [[c] for c in card.get("colors") or [] if c in "WUBRG"]
    if not cost:
        return [[c] for c in card.get("colors") or [] if c in "WUBRG"]
    need = Counter()
    for face in cost.split("//"):
        face_need = Counter()
        for pip in re.findall(r"\{(.*?)\}", face):
            parts = pip.split("/")
            if any(p.isdigit() or p in ("X", "C", "P") for p in parts):
                continue
            options = tuple(p for p in parts if p in "WUBRG")
            if options:
                face_need[options] += 1
        need |= face_need  # per pip type, the larger of the faces
    return [list(o) for o, n in need.items() for _ in range(n)]


def off_pips(card, lane):
    """Pips no colour of the lane can pay, as their option lists."""
    return [opts for opts in pips(card) if not any(o in lane for o in opts)]


def castable(card, lane):
    """Every coloured pip payable by the lane's two colours. Hybrid pips need only one option."""
    if not USE_PIPS:
        return all(c in lane for c in card.get("colors") or [])
    return not off_pips(card, lane)


def any_colour_fixer(card):
    """A spell that finds or makes any colour: basic landcycling, or a mana ability for any color."""
    if "Land" in (card.get("types") or []):
        return False
    return "Basic landcycling" in (card.get("keywords") or []) or "mana of any color" in (card.get("oracle_text") or "")


def pool_fixing(pool):
    """Sources per colour a splash can count on: nonbasic lands by the colours they make, plus any-colour lands
    (fetches) and any-colour mana. Basic landcyclers are not sources: they find a basic the deck already plays."""
    fix = dict.fromkeys("WUBRG", 0)
    for c in pool:
        types, text = c.get("types") or [], (c.get("oracle_text") or "").lower()
        if "Land" in types:
            if "Basic" in types:
                continue
            cols = [col for col in c.get("colors") or [] if col in fix]
            if not cols and ("any color" in text or "basic land card" in text):
                cols = list(fix)
        else:
            cols = list(fix) if "mana of any color" in text else []
        for col in cols:
            fix[col] += 1
    return fix


def quality(card, mean, std):
    """0 for an unplayable card, 1 at the format mean, 2 one SD above. Lands count 0."""
    if "Land" in (card.get("types") or []):
        return 0.0
    wr = float(all_decks(card).get("gihwr") or 0.0)
    return max(0.0, (wr - (mean - std)) / std) if wr > 0 else 0.0


GOLD_MIN = 0.0  # a card castable in one pair only pulls toward it by its quality above this (q - GOLD_MIN, at least 0); 0 = full quality
SPLASH_COMMIT = 0.0  # share of a card's quality it adds to pairs where it is one pip off (a splash: {2}{U}{B} in UR, UG, WU, BR...); 0 = off


def commitment(pool, mean, std):
    """Per lane: summed quality of the pool cards castable there."""
    out = {lane: 0.0 for lane in LANES}
    for card in pool:
        q = quality(card, mean, std)
        if q <= 0:
            continue
        lanes = [lane for lane in LANES if castable(card, lane)]
        if GOLD_MIN and len(lanes) == 1 and "Land" not in (card.get("types") or []):
            q = max(0.0, q - GOLD_MIN)  # a medium gold card should not decide the pair on its own
        for lane in lanes:
            out[lane] += q
        if SPLASH_COMMIT:
            for lane in LANES:
                if lane not in lanes and len(off_pips(card, lane)) == 1:
                    out[lane] += q * SPLASH_COMMIT
    return out


def softmax(logits):
    top = max(logits.values())
    weights = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def lane_distribution(commit, signals, prior, alpha, beta, gamma):
    """Softmax over lanes. signals: per-colour z; prior: per-lane strength in SDs."""
    return softmax({
        lane: alpha * commit[lane]
        + beta * (signals.get(lane[0], 0.0) + signals.get(lane[1], 0.0))
        + gamma * prior.get(lane, 0.0)
        for lane in LANES
    })
