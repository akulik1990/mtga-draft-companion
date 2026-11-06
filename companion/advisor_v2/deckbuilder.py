"""Our deck builder: per colour pair, the 23 spells that maximise the deck model's predicted win rate while
staying inside the shape top players' decks have in that pair, then lands from the pips.

Why constraints: the deck model (tools/deck_model.py) learnt from real decks. Asked about a deck unlike any of
them (6 creatures) it extrapolates (it rated one 65%). Top players' typical deck per pair (tools/deck_targets.py,
the "decks" section of the set's fitted.json) bounds the search to shapes the model has evidence for.

Search: start from the pair's best cards by win rate, repair the shape, then swap one card in and one out while
the objective improves. Pools are small (about 40 cards), so a few hundred model evaluations per pair suffice.
"""

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from companion.advisor_v2.composition import curve_slot, helps_mana, roles
from companion.advisor_v2.lanes import LANES, any_colour_fixer, castable, off_pips, pool_fixing
from companion.advisor_v2 import config

SPELLS = 23
LANDS = 17
LAND_STEPS = False  # Akshin's rule: one land fewer with 1+ landcycler and 1+ dual in the deck, two fewer with 2+ of each
BASIC = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
CURVE = ("1", "2", "3", "4", "5", "6+")

# How far a build may stray from top players' typical deck in the pair; beyond it each card costs PENALTY points.
# All of these are per-set settings (config layers: "deckbuilder.NAME" in a set's manual.json or fitted.json).
CREATURE_BELOW = 2.0  # creatures below the target before the penalty starts
CREATURE_ABOVE = 3.0  # and above it
TWO_DROP_SLACK = 2.0  # below the target
TOP_END_SLACK = 2.0  # five-plus drops above the target
CURVE_SLACK = 2.0  # per mana value, either way
CURVE_SKIP = ""  # comma-separated curve slots left out of the curve check ("2": two-drop shortages do not lose, docs/podcast-test-twodrops-FRA.md)
REMOVAL_SLACK = 2.0  # removal spells below the target before the penalty starts (FRA: decks under 6 removal lose ~0.6 WR points beyond their cards, docs/synergy-2026-11-02.md); 99 = off
REMOVAL_SKIP = ""  # ";"-separated removal cards the removal bounds do not count; the pair's target drops by top players' mean count of them (fitted "removal_skip", tools/removal_skip.py). FRA: conditional removal, about 0 in hand within deck (docs/b1-within-removal-FRA.md)
PENALTY = 1.5  # predicted win-rate points per card outside the bounds
MIN_PER_COLOUR = 5  # spells that need each main colour; fewer and the "pair" is a mono-colour deck with a stray card
MIN_PER_COLOUR_THREE = 3  # the same for each colour of a three-colour deck
MIN_BASICS = 6  # basics of each colour of a pair
MIN_BASICS_THREE = 4  # ...and of each colour of a three-colour deck
MAX_SPLASH = 3  # splash cards; more and it is a third main colour
MIN_SPLASH_CMC = 3  # splash late, impactful cards, not early drops
MAX_SPLASH_BASICS = 4  # basics of the splash colour before the main colours suffer
OWN_FIXING = True  # splash and three-colour rule inputs count fixing without basic landcyclers (colour_fixing); refit the rules when switching
THREE_AT = 0.5  # go three colours when the fitted three rule gives at least this chance
SPLASH_FORCE = False  # when the fitted rule says splash, splash even if the splash build scores below the plain pair
OBJECTIVE = "model"  # "model": maximise the fitted deck model; "gihwr": GIH_SLOPE x the mean 17Lands GIH WR (lane, else All Decks) of the spells,
# less SPLASH_TERM for a splash: the cards' own win rates, as Draftsim and top players read them (the deck model's per-card and curve terms
# contradict the card data, docs/engine-comparison-2026-11-03.md)
GIH_SLOPE = 1.72  # deck WR points per +1 mean spell GIH WR (FRA decks of 100+ games players, 2026-11-05)
SPLASH_TERM = 4.07  # WR points for a splash under "gihwr" (the deck model's splash term)
MODEL = "deckmodel"  # fitted.json section of the deck model the builder maximises (the mock yardstick stays "deckmodel")
SPLASH_AT = 0.5  # splash when the fitted splash rule gives at least this chance
# Structure first (STRUCTURE): every category (creatures, removal, card advantage, evasion) and curve slot has a min,
# best and max per pair from 60%+ players' decks (the "structure" section of the set's fitted.json, tools/deck_shapes.py).
# The search first brings the deck inside min..max, with a category low (best - COMP_SLACK or less) made up by its
# replacements (what top players run more of when that category is low: few creatures -> more removal and card
# advantage; manual "structure_replacements" add what people know: each curve slot <- card advantage, mana); then
# swaps for win rate keep that, and only trade a card for one of the same curve slot, the same role,
# or a replacement category (2-drop for 2-drop, removal for removal, creature for removal). Off: one search on win
# rate less the shape penalty.
STRUCTURE = False
WALKER_CREATURE = 0.0  # creatures a planeswalker or Jace-token maker (composition "walker" role) counts as for the creature bounds; FRA 0.62
LEGEND_DUP = 0.0  # GIH WR points off each extra copy of a legendary card under "gihwr" (legend rule: dead while the first is in play); 0 = off
TEMPO_CREATURE = 0.0  # creatures a cheap tempo spell (composition "tempo" role) counts as for the creature bounds; FRA 0.57 from top players' decks (analysis/tempo_substitution.py)
STRUCTURE_DIMS = ("creatures", "removal", "card_advantage", "evasion", "mana", "1", "2", "3", "4", "5", "6+")
COMP_SLACK = 1.0  # cards a category may sit under its best before its replacements must make up the rest
COMP_RATIO = 1.0  # replacement cards above their best needed per card the category is short
BEST_POINTS = 0.0  # WR points per card away from best in each category (a light pull toward best); 0 = off
STRUCTURE_TOL = 0.5  # without a structure section: cards either side of the pair's mean before it counts as off-structure
RISK_RADIUS = 0.8  # whole-shape support (a measure, tools/builder_check.py): real decks within this distance, in the pair's SDs
SYN_MATES = 0.0  # WR points per unit of the deck's deck-mate sum (pairwise log lift of top players' decks in the pair, over 23); 0 = off
SYN_CARDS = 0.0  # WR points per log(1 + enablers in the deck) for each list of a reviewed top card in the deck (manual card_wants); 0 = off
SPLASH_FIX_POINTS = 0.0  # when choosing the splash colour: WR points per land in the pool that makes that colour (up to 3); top players splash the colour their duals support; 0 = off
INCLUDE_WEIGHT = 0.0  # WR points per unit of a card's inclusion log-odds (how often top players put the copies they hold in the deck, in the pair, against the pair's average); 0 = off
BOMB_BONUS = 0.0  # WR points for each of the set's top BOMB_TOP cards in the deck: top players nearly always play them; 0 = off
BOMB_TOP = 30
DIM_PENALTY = 0.0  # WR points per card outside top players' count +- DIM_SLACK on the composition slots in DIM_CHECK; 0 = off
DIM_SLACK = 2.0
DIM_CHECK = "early_creatures,late_creatures,noncreature,pressure,defence,finishers"
SWAP_MARGIN = 0.0  # list sideboard swaps that cost at most this many objective points as close calls ("swaps" in each build); 0 = off
THREE_FALLBACK = True  # a pool where no pair has 23 castable spells gets its best three-colour build instead of none
SPLASH_COST = -1.0  # win-rate points charged for splashing, in place of the deck model's splash term (which mostly
#                     reflects that weaker decks splash: 60%+ FRA players splash 49% of two-colour decks); < 0 = the model's term


def pips(card):
    """Coloured pips per colour on the front face, hybrid counting toward each option."""
    front = (card.get("mana_cost") or "").split("//")[0]
    out = Counter()
    for pip in re.findall(r"\{(.*?)\}", front):
        for c in pip.split("/"):
            if c in BASIC:
                out[c] += 1
    return out


def lands_for(spells, lane, nonbasics=(), splash="", fixers=0, lands=None):
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
        have = fixers + sum(1 for c in nonbasics if splash in (c.get("colors") or []) or not c.get("colors"))  # colourless nonbasics here tap for a chosen colour
        splash_basics = max(0, want - have)
    basics_n = (lands or LANDS) - len(nonbasics) - splash_basics
    total = sum(weight[c] for c in lane) or 1.0
    floor = MIN_BASICS if len(lane) == 2 else MIN_BASICS_THREE
    counts = {c: max(floor, round(basics_n * weight[c] / total)) for c in lane}
    while sum(counts.values()) > basics_n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < basics_n:
        counts[max(lane, key=lambda c: weight[c])] += 1
    if splash_basics:
        counts[splash] = splash_basics
    return {BASIC[c]: n for c, n in counts.items() if n}


def _wr(card, lane):
    d = card.get("deck_colors") or {}
    return (d.get(lane) or {}).get("gihwr") or (d.get("All Decks") or {}).get("gihwr") or 0.0


def splash_features(pool, lane):
    """(edge, fixing, playables, colour) for the pool's best splash colour in this pair: how much the best single-pip
    card of that colour beats the 23rd on-colour spell (All Decks GIH WR), fixing sources for the colour (nonbasic
    lands making it, any-colour fixers), and how many on-colour playables the pool has. The inputs of the fitted
    splash rule (tools/fit_splash.py); colour is None when no single-pip off-colour spell exists."""
    spell = lambda c: "Land" not in (c.get("types") or [])
    on = sorted((_wr(c, lane) for c in pool if spell(c) and castable(c, lane) and _wr(c, lane) > 0), reverse=True)
    cut = on[SPELLS - 1] if len(on) >= SPELLS else 45.0
    best = {}
    for c in pool:
        if spell(c) and not castable(c, lane):
            off = off_pips(c, lane)
            if len(off) == 1:
                for col in off[0]:
                    best[col] = max(best.get(col, 0.0), _wr(c, "All Decks"))
    fix = colour_fixing(pool, lane)
    if not best:
        return -10.0, 0, len(on), None, 0, 0, cut, 0.0, 0
    col = max(best, key=lambda k: best[k] - cut + 0.5 * min(fix[k], 6))
    # Extended inputs (used when the fitted rule lists them): splashable cards of the colour above the 23rd, that
    # colour's own dual lands (any-colour fixers aside), and the pair's depth (its 23rd card's win rate).
    above = sum(1 for c in pool if spell(c) and not castable(c, lane) and len(off_pips(c, lane)) == 1
                and col in off_pips(c, lane)[0] and _wr(c, "All Decks") >= cut)
    duals = sum(1 for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []) and col in (c.get("colors") or []))
    # The best splash card itself: removal (what guides say to splash) and its mana value (late, impactful cards).
    top = max((c for c in pool if spell(c) and not castable(c, lane) and len(off_pips(c, lane)) == 1 and col in off_pips(c, lane)[0]),
              key=lambda c: _wr(c, "All Decks"))
    return (best[col] - cut, min(fix[col], 8), len(on), col, above, min(duals, 6), cut,
            float("removal" in config.tags(top)), min(int(top.get("cmc") or 0), 7))


def colour_fixing(pool, lane):
    """Sources per off-lane colour for the splash and three-colour rules. OWN_FIXING: lanes.pool_fixing (nonbasic lands by
    colour, fetches, any-colour mana); off: the old count, which also took basic landcyclers as sources of every colour."""
    if OWN_FIXING:
        return {col: (0 if col in lane else n) for col, n in pool_fixing(pool).items()}
    fix = dict.fromkeys("WUBRG", 0)
    for c in pool:
        if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []):
            for col in c.get("colors") or []:
                if col not in lane:
                    fix[col] += 1
        if any_colour_fixer(c):
            for col in "WUBRG":
                fix[col] += 1
    return fix


SPLASH_INPUTS = ("edge", "fixing", "playables", "colour", "above", "duals", "depth", "removal", "cmc")


def splash_inputs(rule, f, lane=""):
    """The fitted rule's inputs from splash_features (rules list their input names; old rules use the first three).
    "pair": how often top players splash in this pair (the rule's own pair_rate table)."""
    named = dict(zip(SPLASH_INPUTS, f))
    named["pair"] = (rule.get("pair_rate") or {}).get(lane, rule.get("rate", 0.5))
    return [named[k] for k in rule.get("inputs", SPLASH_INPUTS[:3])]


def three_features(pool, lane):
    """(adds, fixing, playables, colour) for the pool's best third colour on top of this pair: spells the three colours
    cast but the pair cannot that beat the pair's 23rd spell (All Decks GIH WR), fixing sources for the colour, and the
    pair's playables. Inputs of the fitted three-colour rule (tools/fit_splash.py); colour None when nothing adds."""
    spell = lambda c: "Land" not in (c.get("types") or [])
    on = sorted((_wr(c, lane) for c in pool if spell(c) and castable(c, lane) and _wr(c, lane) > 0), reverse=True)
    cut = on[SPELLS - 1] if len(on) >= SPELLS else 45.0
    fix = colour_fixing(pool, lane)
    adds = {col: sum(1 for c in pool if spell(c) and not castable(c, lane) and castable(c, lane + col) and _wr(c, "All Decks") >= cut)
            for col in "WUBRG" if col not in lane}
    col = max(adds, key=lambda k: adds[k] + 0.5 * min(fix[k], 6))
    if not adds[col]:
        return 0, 0, len(on), None, 0, cut
    duals = sum(1 for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []) and col in (c.get("colors") or []))
    return adds[col], min(fix[col], 8), len(on), col, min(duals, 6), cut


THREE_INPUTS = ("adds", "fixing", "playables", "colour", "duals", "depth")


def three_inputs(rule, f):
    named = dict(zip(THREE_INPUTS, f))
    return [named[k] for k in rule.get("inputs", THREE_INPUTS[:3])]


def rule_chance(rule, x):
    """A fitted logistic rule's probability for inputs x."""
    z = rule["w"][0] + sum(w * (v - m) / s for w, v, m, s in zip(rule["w"][1:], x, rule["mu"], rule["sd"]))
    return 1.0 / (1.0 + math.exp(-z))


def splash_chance(rule, pool, lane):
    """P(a top player splashes this pool in this pair), from the set's fitted splash rule; None without one."""
    if not rule:
        return None
    return rule_chance(rule, splash_inputs(rule, splash_features(pool, lane), lane))


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


ROLE_DIMS = ("creatures", "removal", "card_advantage", "evasion", "mana")
SHAPE_DIMS = ROLE_DIMS + CURVE


def card_roles(card):
    """A spell's ROLE_DIMS: composition roles plus "mana" (gets lands or mana: composition.helps_mana)."""
    return (roles(card) & set(ROLE_DIMS)) | ({"mana"} if helps_mana(card) else set())


def shape_vector(spells):
    """A deck's whole shape: role counts and spells per curve slot, in SHAPE_DIMS order (tools/deck_shapes.py)."""
    v = [0.0] * len(SHAPE_DIMS)
    for c in spells:
        r = card_roles(c)
        for i, k in enumerate(ROLE_DIMS):
            v[i] += k in r
        slot = curve_slot(c)
        if slot:
            v[len(ROLE_DIMS) + CURVE.index(slot)] += 1
    return v


@lru_cache(maxsize=64)
def _structure(set_code, lane):
    """(min/best/max per category for the lane, replacements): {category: [option, ...]}, an option being one category
    or a composite (a tuple: all its parts must run high together). Fitted singles plus the manual layer's
    "structure_replacements" (lists in it are composites). A category with no options has no substitute."""
    st = config.fitted(set_code, "structure") if set_code else None
    repl = {k: list(v) for k, v in ((st or {}).get("replacements") or {}).items()}
    for k, extra in ((config.layer(set_code, "manual") if set_code else {}).get("structure_replacements") or {}).items():
        for x in extra:
            x = tuple(x) if isinstance(x, list) else x
            if x not in repl.setdefault(k, []):
                repl[k].append(x)
    return (st or {}).get("bounds", {}).get(lane), repl


def covered(options, x, bounds):
    """Cards a low category's replacements make up: single replacements' excess over their best adds up; a composite
    counts its parts times its weakest part's excess (all must run high); the larger of the two."""
    up = lambda k: max(0.0, x[k] - bounds[k][1])
    singles = sum(up(o) for o in options if isinstance(o, str))
    composite = max((len(o) * min(up(p) for p in o) for o in options if not isinstance(o, str)), default=0.0)
    return max(singles, composite)


def structure_dev(spells, target, lane="", set_code=None):
    """Cards off top players' structure in the pair: counts outside min..max, plus what is left uncovered of a category
    running low (more than COMP_SLACK under best) after its replacements; a category with no substitute is never
    covered. Without a structure section, beyond STRUCTURE_TOL of the pair's mean."""
    x = dict(zip(SHAPE_DIMS, shape_vector(spells)))
    bounds, repl = _structure(set_code, lane)
    if not bounds:
        want = {**{k: target.get(k) for k in ROLE_DIMS}, **(target.get("curve") or {})}
        return sum(max(0.0, abs(x[k] - want[k]) - STRUCTURE_TOL) for k in STRUCTURE_DIMS if want.get(k) is not None)
    out = 0.0
    for k in STRUCTURE_DIMS:
        lo, best, hi = bounds[k]
        out += max(0.0, lo - x[k], x[k] - hi)
        short = best - x[k] - COMP_SLACK
        if short > 0:
            out += max(0.0, COMP_RATIO * short - covered(repl.get(k, ()), x, bounds))
    return out


def off_best(spells, lane, set_code):
    """Cards away from best over every category (BEST_POINTS' pull)."""
    bounds, _ = _structure(set_code, lane)
    if not bounds:
        return 0.0
    x = dict(zip(SHAPE_DIMS, shape_vector(spells)))
    return sum(abs(x[k] - bounds[k][1]) for k in STRUCTURE_DIMS)


def categories(card):
    """A spell's structure categories: its curve slot and its roles."""
    return card_roles(card) | {curve_slot(card)}


def swappable(out, into, repl):
    """Same curve slot or role, or the incoming card fills a replacement for a category the outgoing one leaves."""
    a, b = categories(out), categories(into)
    parts = lambda k: {p for o in repl.get(k, ()) for p in ((o,) if isinstance(o, str) else o)}
    return bool(a & b) or any(parts(x) & b for x in a)


@lru_cache(maxsize=4)
def _shapes(set_code):
    import gzip
    import json

    from companion import DATA_DIR

    path = DATA_DIR / "Temp" / f"deckshapes_{set_code}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    out = {}
    for lane, rows in raw["lanes"].items():
        a = np.array(rows, float)
        k = len(raw["dims"])
        sd = a[:, :k].std(0)
        out[lane] = (a[:, :k], np.where(sd > 0, sd, 1.0), a[:, k], a[:, k + 1], a[:, k + 2])
    return out


def support(set_code, lane, spells):
    """(decks, win-rate points) for real decks of the pair shaped like these spells: how many lie within RISK_RADIUS,
    and how far their wins ran above (+) or below (-) what their cards predicted. (0, None) without the shape table."""
    t = _shapes(set_code).get(lane)
    if t is None:
        return 0, None
    X, sd, games, wins, pred = t
    d = np.sqrt((((X - np.array(shape_vector(spells))) / sd) ** 2).mean(1))
    m = d <= RISK_RADIUS
    if not m.any():
        return 0, None
    return int(m.sum()), float(100 * (wins[m].sum() - pred[m].sum()) / games[m].sum())


def skip_mean(lane):
    """Top players' mean count of the REMOVAL_SKIP cards in the pair (fitted "removal_skip" of the active set); 0 when unknown."""
    return float(((config.fitted(config.ACTIVE["set"], "removal_skip") or {}).get("pairs") or {}).get(lane, 0.0)) if config.ACTIVE["set"] else 0.0


def shape_penalty(spells, target, lane="", min_creatures_slack=None, scale=1.0, colours_only=False):
    """Win-rate points lost for straying outside top players' shape in the pair.
    scale < 1 judges a part-built deck against the same share of the target (the engine's deck-aware picks)."""
    r, cv = Counter(), Counter()
    for c in spells:
        for x in roles(c):
            r[x] += 1
        cv[curve_slot(c)] += 1
    over, f = 0.0, scale
    if colours_only:  # structure-first search: the shape is the search's own business; only the colours stay
        least = MIN_PER_COLOUR if len(lane) == 2 else MIN_PER_COLOUR_THREE
        return PENALTY * sum(max(0.0, least - sum(needs(c, colour, lane) for c in spells)) for colour in lane)
    below = CREATURE_BELOW if min_creatures_slack is None else min_creatures_slack
    lo, hi = f * (target["creatures"] - below), f * (target["creatures"] + CREATURE_ABOVE)
    creatures = r["creatures"] + TEMPO_CREATURE * r["tempo"] + WALKER_CREATURE * r["walker"]
    over += max(0.0, lo - creatures) + max(0.0, creatures - hi)
    over += max(0.0, f * (target["two_drops"] - TWO_DROP_SLACK) - r["two_drops"])
    if "removal" in target:
        removal, want = r["removal"], target["removal"]
        if REMOVAL_SKIP:
            skip = REMOVAL_SKIP.split(";")
            removal -= sum(1 for c in spells if c.get("name") in skip and "removal" in roles(c))
            want -= skip_mean(lane)
        over += max(0.0, f * (want - REMOVAL_SLACK) - removal)
    over += max(0.0, r["five_plus"] - f * (target["five_plus"] + TOP_END_SLACK))
    for k in CURVE:
        if k in CURVE_SKIP.split(","):
            continue
        want = (target.get("curve") or {}).get(k)
        if want is not None:
            over += max(0.0, abs(cv[k] - f * want) - f * CURVE_SLACK)
    least = MIN_PER_COLOUR if len(lane) == 2 else MIN_PER_COLOUR_THREE
    for colour in lane:
        over += max(0.0, f * least - sum(needs(c, colour, lane) for c in spells))
    return PENALTY * over


def make_extras(spells, lane, target, set_code):
    """f(indices of spells in the deck) -> WR points for synergy (SYN_MATES, SYN_CARDS) less composition misses (DIM_PENALTY)."""
    if not set_code or not (SYN_MATES or SYN_CARDS or DIM_PENALTY or BOMB_BONUS or INCLUDE_WEIGHT):
        return lambda idx: 0.0
    import math

    from companion.advisor_v2 import needs as deck_needs

    names = [c["name"] for c in spells]
    n = len(spells)
    lift = [[0.0] * n for _ in range(n)]
    home = [castable(c, lane) for c in spells]
    if SYN_MATES:
        subs = [a + b for i, a in enumerate(lane) for b in lane[i + 1:]]
        tables = [deck_needs.deckmates(set_code).get(x) or {} for x in subs]
        # Only between cards the pair casts: splash cards rarely share a two-colour deck, so their lifts would read as
        # "top players do not play this" and talk the builder out of splashing.
        for i in range(n):
            for j in range(i + 1, n):
                if not (home[i] and home[j]):
                    continue
                vals = [(t.get(names[i]) or {}).get(names[j], 0.0) for t in tables]
                lift[i][j] = max(vals, key=abs) if vals else 0.0
    lists = []
    if SYN_CARDS:
        for payoff, enablers in deck_needs.synergy_lists(set_code):
            if payoff in names:
                lists.append((payoff, [i for i, x in enumerate(names) if x in enablers]))
    dims_of, want = [], {}
    if DIM_PENALTY and target.get("dims"):
        mean, sd = deck_needs.set_wr(set_code)
        dims_of = [deck_needs.dims(c, mean, sd) for c in spells]
        want = {d: target["dims"].get(d, 0.0) for d in DIM_CHECK.split(",")}

    bombs = set()
    if BOMB_BONUS:
        top = deck_needs.top_cards(set_code, BOMB_TOP)
        bombs = {i for i, x in enumerate(names) if x in top}

    incl = [0.0] * n
    if INCLUDE_WEIGHT:
        subs = [a + b for i, a in enumerate(lane) for b in lane[i + 1:]]
        tables = [deck_needs.inclusion(set_code).get(x) or {} for x in subs]
        for i, x in enumerate(names):
            vals = [t[x] for t in tables if x in t]
            incl[i] = sum(vals) / len(vals) if vals and home[i] else 0.0  # splash cards: no evidence in two-colour decks

    def extras(idx):
        idx = sorted(idx)
        out = BOMB_BONUS * sum(1 for i in idx if i in bombs) + INCLUDE_WEIGHT * sum(incl[i] for i in idx)
        if SYN_MATES:
            # Mean over pairs of on-colour cards, scaled to a full deck's pairs: swapping one in for a splash card changes
            # the mean, not how many pairs count, so splashing is not charged for the pairs it leaves out.
            mine = [i for i in idx if home[i]]
            pairs = len(mine) * (len(mine) - 1) / 2
            if pairs:
                mean = sum(lift[i][j] for a, i in enumerate(mine) for j in mine[a + 1:]) / pairs
                out += SYN_MATES * mean * (SPELLS - 1) / 2
        if lists:
            chosen = set(idx)
            for payoff, en in lists:
                if any(names[i] == payoff for i in chosen):
                    out += SYN_CARDS * math.log1p(sum(1 for i in en if i in chosen))
        if want:
            have = Counter(d for i in idx for d in dims_of[i])
            out -= DIM_PENALTY * sum(max(0.0, abs(have[d] - w) - DIM_SLACK) for d, w in want.items())
        return out

    return extras


_LEGENDS = {}


def legendary(set_code):
    """Names of the set's legendary cards, from the Scryfall cache (<data>/Temp/scryfall_cards_SET.json, tools/card_table.py)."""
    if set_code not in _LEGENDS:
        import json
        from companion import DATA_DIR
        path = DATA_DIR / "Temp" / f"scryfall_cards_{set_code}.json"
        cards = json.loads(path.read_text()) if path.exists() else {}
        _LEGENDS[set_code] = {n for n, c in cards.items() if "Legendary" in (c.get("type_line") or "")}
    return _LEGENDS[set_code]


def cycler(card, lane):
    return castable(card, lane) and "landcycling" in (card.get("oracle_text") or "").lower()


def land_count(chosen, nonbasics, lane):
    """LANDS, less LAND_STEPS' steps for the landcyclers and duals (multicolour or any-colour non-basics) the deck plays."""
    if not LAND_STEPS:
        return LANDS
    cyc = sum(1 for c in chosen if cycler(c, lane))
    duals = sum(1 for c in nonbasics if len(c.get("colors") or []) >= 2 or not c.get("colors"))
    return LANDS - (cyc >= 1 and duals >= 1) - (cyc >= 2 and duals >= 2)


def build_pair(pool, lane, model, target, min_creatures_slack=None, variant="", splash="", splash_cost=None, force=False, lands=None):
    """The best 23 spells for one pair (and optional splash colour), with lands; None when the pool cannot make a deck."""
    spells = [c for c in pool if "Land" not in (c.get("types") or []) and (castable(c, lane) or (splash and splashable(c, lane, splash)))]
    by_name = {c["name"]: c for c in pool}
    legends = legendary(next((c.get("set") for c in pool if c.get("set")), "")) if LEGEND_DUP else set()
    lane_colours = lane + splash
    # Colourless lands that tap for a chosen or any colour (Room of Refuge: top players play every copy) count as on-colour.
    nonbasics = [c for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or [])
                 and (((c.get("colors") or []) and all(x in lane_colours for x in c.get("colors"))
                       and (not splash or any(x in lane for x in c.get("colors"))))
                      or (not c.get("colors") and any(s in (c.get("oracle_text") or "") for s in ("chosen color", "any color"))))]
    lands_n = lands or LANDS
    spells_n = 40 - lands_n
    if len(spells) < spells_n:
        return None

    def wr(card):
        return ((card.get("deck_colors") or {}).get(lane) or {}).get("gihwr") or ((card.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0

    def mana(chosen):
        fixers = sum(1 for c in chosen if any_colour_fixer(c))
        return lands_for(chosen, lane, nonbasics, splash, fixers, lands_n)

    def penalty(chosen):
        p = shape_penalty(chosen, target, lane, min_creatures_slack, colours_only=STRUCTURE)
        if splash:
            n = sum(1 for c in chosen if not castable(c, lane))
            p += PENALTY * max(0, n - MAX_SPLASH)
            p += PENALTY * max(0, mana(chosen).get(BASIC[splash], 0) - MAX_SPLASH_BASICS)
        return p

    set_code = next((c.get("set") for c in pool if c.get("set")), None)
    extras = make_extras(spells, lane, target, set_code)

    def score(chosen, idx=None):
        counts = Counter(c["name"] for c in chosen)
        counts.update(mana(chosen))
        counts.update(Counter(c["name"] for c in nonbasics))
        used = splash if splash and any(not castable(c, lane) for c in chosen) else ""
        return predicted(counts, used) - penalty(chosen) + (extras(idx) if idx is not None else 0.0)

    cost = SPLASH_COST if splash_cost is None else splash_cost

    def predicted(counts, used):
        if OBJECTIVE == "gihwr":
            g = [_wr(by_name[n], lane[:2]) - (LEGEND_DUP if i and n in legends else 0.0) for n, k in counts.items()
                 if n in by_name and "Land" not in (by_name[n].get("types") or []) for i in range(k)]
            g = [x for x in g if x]
            return GIH_SLOPE * (sum(g) / len(g) if g else 0.0) - (SPLASH_TERM if used else 0.0)
        if cost < 0:
            return float(model.win_rate(dict(counts), lane, used))
        # A fitted rule decided the colours: score the cards and shape as a plain pair (the model's colour-count and
        # splash terms mostly say that weaker decks add colours).
        return float(model.win_rate(dict(counts), lane[:2], "")) - (cost if used else 0.0)

    # Start from the pair's own cards (splash cards come in only when a swap pays).
    order = sorted(range(len(spells)), key=lambda i: (not castable(spells[i], lane), -wr(spells[i])))
    deck, side = set(order[:spells_n]), set(order[spells_n:])
    best = score([spells[i] for i in deck], deck)
    if STRUCTURE:
        deck, side, best = _structured(spells, deck, side, score, target, set_code, lane)
    improved = not STRUCTURE
    while improved:
        improved = False
        for i in sorted(deck):
            for j in sorted(side):
                trial = (deck - {i}) | {j}
                s = score([spells[k] for k in trial], trial)
                if s > best + 1e-6:
                    deck, side, best = trial, (side - {j}) | {i}, s
                    improved = True
                    break
            if improved:
                break
    if splash and force and not any(not castable(spells[i], lane) for i in deck):
        # The fitted rule decided to splash: bring in the splash card whose best swap costs least.
        trials = [(score([spells[k] for k in (deck - {i}) | {j}], (deck - {i}) | {j}), i, j)
                  for j in side if not castable(spells[j], lane) for i in deck if castable(spells[i], lane)]
        if trials:
            _, i, j = max(trials)
            deck, side = (deck - {i}) | {j}, (side - {j}) | {i}
    chosen = [spells[i] for i in deck]
    used = splash if splash and any(not castable(c, lane) for c in chosen) else ""
    if splash and not used:
        return None  # the splash did not pay: the plain pair covers it
    swaps = close_calls(spells, deck, side, score, best) if SWAP_MARGIN else []
    want = land_count(chosen, nonbasics, lane)
    if lands is None and want != lands_n:  # rebuild once at the rule's count; keep it only if the deck still qualifies
        again = build_pair(pool, lane, model, target, min_creatures_slack, variant, splash, splash_cost, force, lands=want)
        if again and land_count(again["spells"], nonbasics, lane) == want:
            return again
    lands = mana(chosen)
    full = dict(Counter(c["name"] for c in chosen))
    full.update(lands)
    full.update(Counter(c["name"] for c in nonbasics))
    return {"lane": lane, "splash": used, "variant": variant, "spells": chosen, "nonbasics": nonbasics, "basics": lands,
            "penalty": penalty(chosen), "predicted": predicted(full, used), "objective": best, "extras": round(extras(deck), 2),
            "structure_dev": round(structure_dev(chosen, target, lane, set_code), 1),
            "sideboard": [spells[i] for i in side] + [c for c in pool if c not in spells and c not in nonbasics], "swaps": swaps}


def close_calls(spells, deck, side, score, best, top=3):
    """Sideboard cards within SWAP_MARGIN of a deck card: [{out, in, cost, when}]. "when" reads the trade: a cheaper interactive card in for
    a top-end or engine card is for fast opponents; the reverse is for slow ones."""
    seen, out = set(), []
    for i in deck:
        for j in side:
            key = (spells[i]["name"], spells[j]["name"])
            if key in seen or key[0] == key[1]:
                continue
            seen.add(key)
            trial = (deck - {i}) | {j}
            cost = best - score([spells[k] for k in trial], trial)
            if cost <= SWAP_MARGIN:
                a, b = spells[i], spells[j]
                cheap = lambda c: (c.get("cmc") or 0) <= 2 or "tempo" in roles(c) or "removal" in roles(c)
                when = ("against fast decks" if cheap(b) and (a.get("cmc") or 0) >= 4 else
                        "against slow decks" if cheap(a) and (b.get("cmc") or 0) >= 4 else "either way")
                out.append({"out": key[0], "in": key[1], "cost": round(cost, 2), "when": when})
    return sorted(out, key=lambda x: x["cost"])[:top]


def _structured(spells, deck, side, score, target, set_code, lane):
    """Structure first, then win rate within it through like-for-like or replacement swaps. Returns (deck, side, objective)."""
    cards = lambda idx: [spells[k] for k in idx]
    _, repl = _structure(set_code, lane)
    pull = lambda idx: BEST_POINTS * off_best(cards(idx), lane, set_code) if BEST_POINTS else 0.0
    dev = structure_dev(cards(deck), target, lane, set_code)
    plain = score(cards(deck), deck)
    best = plain - pull(deck)
    while True:
        # each round: the best-scoring swap that gets nearer the structure, else the best like-for-like or
        # replacement swap that stays as near and scores higher
        pick = None
        for i in sorted(deck):
            for j in sorted(side):
                trial = (deck - {i}) | {j}
                d = structure_dev(cards(trial), target, lane, set_code)
                if d > dev + 1e-9 or (d >= dev - 1e-9 and not swappable(spells[i], spells[j], repl)):
                    continue
                s = score(cards(trial), trial)
                key = (d < dev - 1e-9, s - pull(trial))
                if (d < dev - 1e-9 or key[1] > best + 1e-6) and (pick is None or key > pick[0]):
                    pick = (key, i, j, s, d)
        if pick is None:
            break
        (_, adj), i, j, plain, dev = pick
        deck, side, best = (deck - {i}) | {j}, (side - {j}) | {i}, adj
    return deck, side, plain


def build(pool, model, targets, top=3, splash_pairs=2, rule=None, three=None):
    """Best builds across the pairs the pool supports, best first: each pair plain, the best pairs also with a splash
    or a third colour, plus a creature-heavier take on the first build.

    With the set's fitted rules (rule: "splash", three: "three"), the colours are the rules' call for the best pair:
    three colours if top players would go three colours with this pool at least half the time, else a splash if they
    would splash, else the plain pair. The deck model then only picks the cards: ruled builds are scored without its
    colour-count and splash terms, which mostly say that weaker decks add colours. Without rules, every variant
    competes on the model's score as before."""
    plain = []
    for lane in LANES:
        t = (targets or {}).get(lane)
        if t:
            b = build_pair(pool, lane, model, t)
            if b:
                plain.append(b)
    # A build that still breaks the shape after the search (a "pair" with one card of its second colour) is no deck.
    plain = sorted((b for b in plain if b["penalty"] <= 3 * PENALTY), key=lambda b: -b["objective"]) or sorted(plain, key=lambda b: -b["objective"])[:1]
    if not plain and THREE_FALLBACK:
        # No pair has 23 castable spells (top players' three-colour pools: 24% of them): build the best three-colour deck outright.
        triples = [b for lane, t in (targets or {}).items() if len(lane) == 3 and t
                   for b in [build_pair(pool, lane, model, t, splash_cost=0.0)] if b and b["penalty"] <= 3 * PENALTY]
        if triples:
            best = max(triples, key=lambda b: b["objective"])
            best["three_chance"] = None
            return [best]
    if not plain:
        return []

    def splashes(b, chance):
        out = []
        for col in "WUBRG":
            if col not in b["lane"]:
                force = SPLASH_FORCE and chance is not None
                s = build_pair(pool, b["lane"], model, targets[b["lane"]], splash=col, splash_cost=None if chance is None else 0.0, force=force)
                if s and s["penalty"] <= 3 * PENALTY and (force or s["objective"] > b["objective"]):
                    s["splash_chance"] = None if chance is None else round(chance, 2)
                    out.append(s)

        def fixing(col):
            return min(3, sum(1 for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or [])
                              and col in (c.get("colors") or [])))
        return sorted(out, key=lambda x: -(x["objective"] + SPLASH_FIX_POINTS * fixing(x["splash"])))

    def third(b):
        f3 = three_features(pool, b["lane"])
        col = f3[3]
        lane = "".join(sorted(b["lane"] + (col or ""), key="WUBRG".index))
        if not col or lane not in (targets or {}):
            return None, None
        chance = rule_chance(three, three_inputs(three, f3))
        t = build_pair(pool, lane, model, targets[lane], splash_cost=0.0)
        if t and t["penalty"] <= 3 * PENALTY:
            t["three_chance"] = round(chance, 2)
            return t, chance
        return None, chance

    if rule or three:
        first, extra = plain[0], []
        t, p3 = third(first) if three else (None, None)
        ps = splash_chance(rule, pool, first["lane"])
        variants = splashes(first, ps) if ps is not None and ps >= SPLASH_AT else []
        if t and p3 >= THREE_AT:
            first, extra = t, [plain[0]] + variants
        elif variants:
            first, extra = variants[0], [plain[0]] + variants[1:]
        out = [first] + sorted(extra + plain[1:], key=lambda b: -b["objective"])
    else:
        out = list(plain)
        for b in plain[:splash_pairs]:
            out += splashes(b, None)
        out.sort(key=lambda b: -b["objective"])
    lane = out[0]["lane"]
    ruled = out[0].get("splash_chance") is not None or out[0].get("three_chance") is not None
    more = build_pair(pool, lane, model, targets[lane], min_creatures_slack=0.0, variant="more creatures", splash=out[0].get("splash", ""),
                      splash_cost=0.0 if ruled else None)
    if more and Counter(c["name"] for c in more["spells"]) != Counter(c["name"] for c in out[0]["spells"]):
        out.insert(1, more)
    return out[:top]
