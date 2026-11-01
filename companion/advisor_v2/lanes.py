"""P(lane) over the ten colour pairs, from pool commitment, signals and the format."""

import math

LANES = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
ALL = "All Decks"


def all_decks(card):
    return (card.get("deck_colors") or {}).get(ALL) or {}


def castable(card, lane):
    """Colourless and lands fit every lane; otherwise every colour must be in the lane. No splashing."""
    return all(c in lane for c in card.get("colors") or [])


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
