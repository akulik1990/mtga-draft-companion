"""Our deck builder: per colour pair, the 23 spells that maximise the deck model's predicted win rate while
staying inside the shape top players' decks have in that pair, then lands from the pips.

Why constraints: the deck model (tools/deck_model.py) learnt from real decks. Asked about a deck unlike any of
them (6 creatures) it extrapolates (it rated one 65%). Top players' typical deck per pair (tools/deck_targets.py,
the "decks" section of the set's fitted.json) bounds the search to shapes the model has evidence for.

Search: start from the pair's best cards by win rate, repair the shape, then swap one card in and one out while
the objective improves. Pools are small (about 40 cards), so a few hundred model evaluations per pair suffice.
"""

import re
from collections import Counter

from companion.advisor_v2.composition import curve_slot, roles
from companion.advisor_v2.engine import any_colour_fixer
from companion.advisor_v2.lanes import LANES, castable, off_pips

SPELLS = 23
LANDS = 17
BASIC = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
CURVE = ("1", "2", "3", "4", "5", "6+")

# How far a build may stray from top players' typical deck in the pair; beyond it each card costs PENALTY points.
# All of these are per-set settings (config layers: "deckbuilder.NAME" in a set's manual.json or fitted.json).
CREATURE_BELOW = 2.0  # creatures below the target before the penalty starts
CREATURE_ABOVE = 3.0  # and above it
TWO_DROP_SLACK = 2.0  # below the target
TOP_END_SLACK = 2.0  # five-plus drops above the target
CURVE_SLACK = 2.0  # per mana value, either way
PENALTY = 1.5  # predicted win-rate points per card outside the bounds
MIN_PER_COLOUR = 5  # spells that need each main colour; fewer and the "pair" is a mono-colour deck with a stray card
MAX_SPLASH = 3  # splash cards; more and it is a third main colour
MIN_SPLASH_CMC = 3  # splash late, impactful cards, not early drops
MAX_SPLASH_BASICS = 4  # basics of the splash colour before the main colours suffer


def pips(card):
    """Coloured pips per colour on the front face, hybrid counting toward each option."""
    front = (card.get("mana_cost") or "").split("//")[0]
    out = Counter()
    for pip in re.findall(r"\{(.*?)\}", front):
        for c in pip.split("/"):
            if c in BASIC:
                out[c] += 1
    return out


def lands_for(spells, lane, nonbasics=(), splash="", fixers=0):
    """Basics for a 17-land mana base: each main colour's share follows its pips, weighting double pips and
    cheap cards (needed early) more; at least 6 sources of each main colour. Nonbasic lands count first.
    A splash gets basics only for the sources its other fixing (duals, landcyclers, any-colour mana) leaves short:
    3 sources for one splash card, 4 for two or more."""
    weight = Counter()
    for c in spells:
        p = pips(c)
        for col, n in p.items():
            if col in lane:
                early = 1.5 if (c.get("cmc") or 0) <= 3 else 1.0
                weight[col] += n * (1.5 if n >= 2 else 1.0) * early
    splash_basics = 0
    if splash:
        n_splash = sum(1 for c in spells if not castable(c, lane))
        want = 0 if not n_splash else 3 if n_splash == 1 else 4
        have = fixers + sum(1 for c in nonbasics if splash in (c.get("colors") or []))
        splash_basics = max(0, want - have)
    basics_n = LANDS - len(nonbasics) - splash_basics
    total = sum(weight[c] for c in lane) or 1.0
    counts = {c: max(6 if len(lane) == 2 else 0, round(basics_n * weight[c] / total)) for c in lane}
    while sum(counts.values()) > basics_n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < basics_n:
        counts[max(lane, key=lambda c: weight[c])] += 1
    if splash_basics:
        counts[splash] = splash_basics
    return {BASIC[c]: n for c, n in counts.items() if n}


def splashable(card, lane, splash):
    """One off-colour pip, of the splash colour, on a card worth waiting for (3+ mana): the guides' splash rule."""
    if "Land" in (card.get("types") or []) or castable(card, lane) or (card.get("cmc") or 0) < MIN_SPLASH_CMC:
        return False
    off = off_pips(card, lane)
    return len(off) == 1 and splash in off[0]


def needs(card, colour, lane):
    """The card needs this colour: a pip only it can pay among the lane's colours."""
    front = (card.get("mana_cost") or "").split("//")[0]
    return any(colour in opts and not any(o in lane and o != colour for o in opts)
               for opts in (pip.split("/") for pip in re.findall(r"\{(.*?)\}", front)))


def shape_penalty(spells, target, lane="", min_creatures_slack=None):
    """Win-rate points lost for straying outside top players' shape in the pair."""
    r, cv = Counter(), Counter()
    for c in spells:
        for x in roles(c):
            r[x] += 1
        cv[curve_slot(c)] += 1
    over = 0.0
    below = CREATURE_BELOW if min_creatures_slack is None else min_creatures_slack
    lo, hi = target["creatures"] - below, target["creatures"] + CREATURE_ABOVE
    over += max(0.0, lo - r["creatures"]) + max(0.0, r["creatures"] - hi)
    over += max(0.0, target["two_drops"] - TWO_DROP_SLACK - r["two_drops"])
    over += max(0.0, r["five_plus"] - target["five_plus"] - TOP_END_SLACK)
    for k in CURVE:
        over += max(0.0, abs(cv[k] - (target.get("curve") or {}).get(k, cv[k])) - CURVE_SLACK)
    for colour in lane:
        over += max(0, MIN_PER_COLOUR - sum(needs(c, colour, lane) for c in spells))
    return PENALTY * over


def build_pair(pool, lane, model, target, min_creatures_slack=None, variant="", splash=""):
    """The best 23 spells for one pair (and optional splash colour), with lands; None when the pool cannot make a deck."""
    spells = [c for c in pool if "Land" not in (c.get("types") or []) and (castable(c, lane) or (splash and splashable(c, lane, splash)))]
    lane_colours = lane + splash
    nonbasics = [c for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or [])
                 and (c.get("colors") or []) and all(x in lane_colours for x in c.get("colors"))
                 and (not splash or any(x in lane for x in c.get("colors")))]
    if len(spells) < SPELLS:
        return None

    def wr(card):
        return ((card.get("deck_colors") or {}).get(lane) or {}).get("gihwr") or ((card.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0

    def mana(chosen):
        fixers = sum(1 for c in chosen if any_colour_fixer(c))
        return lands_for(chosen, lane, nonbasics, splash, fixers)

    def penalty(chosen):
        p = shape_penalty(chosen, target, lane, min_creatures_slack)
        if splash:
            n = sum(1 for c in chosen if not castable(c, lane))
            p += PENALTY * max(0, n - MAX_SPLASH)
            p += PENALTY * max(0, mana(chosen).get(BASIC[splash], 0) - MAX_SPLASH_BASICS)
        return p

    def score(chosen):
        counts = Counter(c["name"] for c in chosen)
        counts.update(mana(chosen))
        counts.update(Counter(c["name"] for c in nonbasics))
        used = splash if splash and any(not castable(c, lane) for c in chosen) else ""
        return float(model.win_rate(dict(counts), lane, used)) - penalty(chosen)

    # Start from the pair's own cards (splash cards come in only when a swap pays).
    order = sorted(range(len(spells)), key=lambda i: (not castable(spells[i], lane), -wr(spells[i])))
    deck, side = set(order[:SPELLS]), set(order[SPELLS:])
    best = score([spells[i] for i in deck])
    improved = True
    while improved:
        improved = False
        for i in sorted(deck):
            for j in sorted(side):
                trial = (deck - {i}) | {j}
                s = score([spells[k] for k in trial])
                if s > best + 1e-6:
                    deck, side, best = trial, (side - {j}) | {i}, s
                    improved = True
                    break
            if improved:
                break
    chosen = [spells[i] for i in deck]
    used = splash if splash and any(not castable(c, lane) for c in chosen) else ""
    if splash and not used:
        return None  # the splash did not pay: the plain pair covers it
    lands = mana(chosen)
    full = dict(Counter(c["name"] for c in chosen))
    full.update(lands)
    full.update(Counter(c["name"] for c in nonbasics))
    return {"lane": lane, "splash": used, "variant": variant, "spells": chosen, "nonbasics": nonbasics, "basics": lands,
            "penalty": penalty(chosen), "predicted": float(model.win_rate(full, lane, used)), "objective": best,
            "sideboard": [spells[i] for i in side] + [c for c in pool if c not in spells and c not in nonbasics]}


def build(pool, model, targets, top=3, splash_pairs=2):
    """Best builds across the pairs the pool supports, best first: each pair plain, the best pairs also with each
    splash colour that pays, plus a creature-heavier take on the best pair."""
    plain = []
    for lane in LANES:
        t = (targets or {}).get(lane)
        if t:
            b = build_pair(pool, lane, model, t)
            if b:
                plain.append(b)
    # A build that still breaks the shape after the search (a "pair" with one card of its second colour) is no deck.
    plain = sorted((b for b in plain if b["penalty"] <= 3 * PENALTY), key=lambda b: -b["objective"]) or sorted(plain, key=lambda b: -b["objective"])[:1]
    out = list(plain)
    for b in plain[:splash_pairs]:
        for col in "WUBRG":
            if col not in b["lane"]:
                s = build_pair(pool, b["lane"], model, targets[b["lane"]], splash=col)
                if s and s["penalty"] <= 3 * PENALTY and s["objective"] > b["objective"]:
                    out.append(s)
    out.sort(key=lambda b: -b["objective"])
    if out:
        lane = out[0]["lane"]
        more = build_pair(pool, lane, model, targets[lane], min_creatures_slack=0.0, variant="more creatures", splash=out[0].get("splash", ""))
        if more and Counter(c["name"] for c in more["spells"]) != Counter(c["name"] for c in out[0]["spells"]):
            out.insert(1, more)
    return out[:top]
