"""Per-colour signals from the packs seen so far: good cards arriving later than their ALSA."""

import statistics

from companion.advisor_v2.lanes import all_decks, quality

COLORS = "WUBRG"
PACK_WEIGHT = {1: 1.0, 2: 0.5, 3: 1.0}  # pack 2 comes from the other neighbour
SCALE = "z"  # "z": z-score across colours (divides by their SD); "centred": subtract the mean only, so one dominant colour stays dominant
QUALITY_POWER = 1  # lateness is weighted by quality ** QUALITY_POWER; 2 favours one strong late card over several mediocre ones


def colour_signals(seen_packs, mean, std, alsa_spread):
    """{colour: z}. seen_packs: [(pack_no, pick_no, cards)].

    For each card seen at pick p: lateness = (p - ALSA) / alsa_spread, kept when
    positive, weighted by card quality and pack weight. Totals are divided by the
    number of packs seen so the scale does not grow with the draft, then z-scored
    across the five colours so one strong colour reads as positive and the rest
    as negative.
    """
    raw = {c: 0.0 for c in COLORS}
    n_packs = 0
    for pack_no, pick_no, cards in seen_packs:
        if pick_no <= 1:
            continue  # a fresh pack says nothing about the neighbours
        n_packs += 1
        w = PACK_WEIGHT.get(pack_no, 1.0)
        for card in cards:
            q = quality(card, mean, std)
            alsa = float(all_decks(card).get("alsa") or 0.0)
            if q <= 0 or alsa <= 0:
                continue
            late = (pick_no - alsa) / alsa_spread
            if late <= 0:
                continue
            for c in card.get("colors") or []:
                if c in raw:
                    raw[c] += w * q ** QUALITY_POWER * late
    if n_packs == 0:
        return {c: 0.0 for c in COLORS}
    per_pack = {c: v / n_packs for c, v in raw.items()}
    mu = statistics.mean(per_pack.values())
    if SCALE == "centred":
        return {c: v - mu for c, v in per_pack.items()}
    sd = statistics.pstdev(per_pack.values())
    if sd <= 0:
        return {c: 0.0 for c in COLORS}
    return {c: (v - mu) / sd for c, v in per_pack.items()}
