"""P(lane) over the ten colour pairs, from pool commitment, signals and the format."""

import math
import re

LANES = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
USE_PIPS = True  # round 2: on together with land fixing (it was contested alone in round 1). Castability from mana-cost pips (hybrid aware); False = every colour of the card must be in the lane
ALL = "All Decks"


def all_decks(card):
    return (card.get("deck_colors") or {}).get(ALL) or {}


def pips(card):
    """Coloured pips as lists of options: {U} -> [U], {W/U} -> [W, U]. A land's colours count as pips.

    Generic, X, colourless, twobrid ({2/W}) and Phyrexian ({W/P}) pips are payable
    without the colour, so they are skipped, as upstream's get_strict_colors does.
    Without a mana cost, each colour counts as one pip. Both faces of an "A // B" card
    are merged, which is how upstream's deck builder (mana_base.is_castable) judges them.
    """
    cost = card.get("mana_cost") or ""
    if "Land" in (card.get("types") or []):
        # A land's colours are what it produces; it belongs in a lane only when all of them fit (a WU dual is not a WB card).
        return [[c] for c in card.get("colors") or [] if c in "WUBRG"]
    if not cost:
        return [[c] for c in card.get("colors") or [] if c in "WUBRG"]
    out = []
    for pip in re.findall(r"\{(.*?)\}", cost):
        parts = pip.split("/")
        if any(p.isdigit() or p in ("X", "C", "P") for p in parts):
            continue
        options = [p for p in parts if p in "WUBRG"]
        if options:
            out.append(options)
    return out


def off_pips(card, lane):
    """Pips no colour of the lane can pay, as their option lists."""
    return [opts for opts in pips(card) if not any(o in lane for o in opts)]


def castable(card, lane):
    """Every coloured pip payable by the lane's two colours. Hybrid pips need only one option."""
    if not USE_PIPS:
        return all(c in lane for c in card.get("colors") or [])
    return not off_pips(card, lane)


def quality(card, mean, std):
    """0 for an unplayable card, 1 at the format mean, 2 one SD above. Lands count 0."""
    if "Land" in (card.get("types") or []):
        return 0.0
    wr = float(all_decks(card).get("gihwr") or 0.0)
    return max(0.0, (wr - (mean - std)) / std) if wr > 0 else 0.0


def commitment(pool, mean, std):
    """Per lane: summed quality of the pool cards castable there."""
    out = {lane: 0.0 for lane in LANES}
    for card in pool:
        q = quality(card, mean, std)
        if q <= 0:
            continue
        for lane in LANES:
            if castable(card, lane):
                out[lane] += q
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
