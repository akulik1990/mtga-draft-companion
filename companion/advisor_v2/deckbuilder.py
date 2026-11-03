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

from companion.advisor_v2.composition import curve_slot, roles
from companion.advisor_v2.lanes import LANES, any_colour_fixer, castable, off_pips

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
REMOVAL_SLACK = 2.0  # removal spells below the target before the penalty starts (FRA: decks under 6 removal lose ~0.6 WR points beyond their cards, docs/synergy-2026-11-02.md); 99 = off
PENALTY = 1.5  # predicted win-rate points per card outside the bounds
MIN_PER_COLOUR = 5  # spells that need each main colour; fewer and the "pair" is a mono-colour deck with a stray card
MIN_PER_COLOUR_THREE = 3  # the same for each colour of a three-colour deck
MIN_BASICS = 6  # basics of each colour of a pair
MIN_BASICS_THREE = 4  # ...and of each colour of a three-colour deck
MAX_SPLASH = 3  # splash cards; more and it is a third main colour
MIN_SPLASH_CMC = 3  # splash late, impactful cards, not early drops
MAX_SPLASH_BASICS = 4  # basics of the splash colour before the main colours suffer
THREE_AT = 0.5  # go three colours when the fitted three rule gives at least this chance
SPLASH_AT = 0.5  # splash when the fitted splash rule gives at least this chance
SYN_MATES = 0.0  # WR points per unit of the deck's deck-mate sum (pairwise log lift of top players' decks in the pair, over 23); 0 = off
SYN_CARDS = 0.0  # WR points per log(1 + enablers in the deck) for each list of a reviewed top card in the deck (manual card_wants); 0 = off
SPLASH_FIX_POINTS = 0.0  # when choosing the splash colour: WR points per land in the pool that makes that colour (up to 3); top players splash the colour their duals support; 0 = off
BOMB_BONUS = 0.0  # WR points for each of the set's top BOMB_TOP cards in the deck: top players nearly always play them; 0 = off
BOMB_TOP = 30
DIM_PENALTY = 0.0  # WR points per card outside top players' count +- DIM_SLACK on the composition slots in DIM_CHECK; 0 = off
DIM_SLACK = 2.0
DIM_CHECK = "early_creatures,late_creatures,noncreature,pressure,defence,finishers"
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
    fix = dict.fromkeys("WUBRG", 0)
    for c in pool:
        if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []):
            for col in c.get("colors") or []:
                if col not in lane:
                    fix[col] += 1
        if any_colour_fixer(c):
            for col in "WUBRG":
                fix[col] += 1
    if not best:
        return -10.0, 0, len(on), None
    col = max(best, key=lambda k: best[k] - cut + 0.5 * min(fix[k], 6))
    return best[col] - cut, min(fix[col], 8), len(on), col


def three_features(pool, lane):
    """(adds, fixing, playables, colour) for the pool's best third colour on top of this pair: spells the three colours
    cast but the pair cannot that beat the pair's 23rd spell (All Decks GIH WR), fixing sources for the colour, and the
    pair's playables. Inputs of the fitted three-colour rule (tools/fit_splash.py); colour None when nothing adds."""
    spell = lambda c: "Land" not in (c.get("types") or [])
    on = sorted((_wr(c, lane) for c in pool if spell(c) and castable(c, lane) and _wr(c, lane) > 0), reverse=True)
    cut = on[SPELLS - 1] if len(on) >= SPELLS else 45.0
    fix = dict.fromkeys("WUBRG", 0)
    for c in pool:
        if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []):
            for col in c.get("colors") or []:
                if col not in lane:
                    fix[col] += 1
        if any_colour_fixer(c):
            for col in "WUBRG":
                fix[col] += 1
    adds = {col: sum(1 for c in pool if spell(c) and not castable(c, lane) and castable(c, lane + col) and _wr(c, "All Decks") >= cut)
            for col in "WUBRG" if col not in lane}
    col = max(adds, key=lambda k: adds[k] + 0.5 * min(fix[k], 6))
    if not adds[col]:
        return 0, 0, len(on), None
    return adds[col], min(fix[col], 8), len(on), col


def rule_chance(rule, x):
    """A fitted logistic rule's probability for inputs x."""
    z = rule["w"][0] + sum(w * (v - m) / s for w, v, m, s in zip(rule["w"][1:], x, rule["mu"], rule["sd"]))
    return 1.0 / (1.0 + math.exp(-z))


def splash_chance(rule, pool, lane):
    """P(a top player splashes this pool in this pair), from the set's fitted splash rule; None without one."""
    if not rule:
        return None
    return rule_chance(rule, splash_features(pool, lane)[:3])


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


def shape_penalty(spells, target, lane="", min_creatures_slack=None, scale=1.0):
    """Win-rate points lost for straying outside top players' shape in the pair.
    scale < 1 judges a part-built deck against the same share of the target (the engine's deck-aware picks)."""
    r, cv = Counter(), Counter()
    for c in spells:
        for x in roles(c):
            r[x] += 1
        cv[curve_slot(c)] += 1
    over, f = 0.0, scale
    below = CREATURE_BELOW if min_creatures_slack is None else min_creatures_slack
    lo, hi = f * (target["creatures"] - below), f * (target["creatures"] + CREATURE_ABOVE)
    over += max(0.0, lo - r["creatures"]) + max(0.0, r["creatures"] - hi)
    over += max(0.0, f * (target["two_drops"] - TWO_DROP_SLACK) - r["two_drops"])
    if "removal" in target:
        over += max(0.0, f * (target["removal"] - REMOVAL_SLACK) - r["removal"])
    over += max(0.0, r["five_plus"] - f * (target["five_plus"] + TOP_END_SLACK))
    for k in CURVE:
        want = (target.get("curve") or {}).get(k)
        if want is not None:
            over += max(0.0, abs(cv[k] - f * want) - f * CURVE_SLACK)
    least = MIN_PER_COLOUR if len(lane) == 2 else MIN_PER_COLOUR_THREE
    for colour in lane:
        over += max(0.0, f * least - sum(needs(c, colour, lane) for c in spells))
    return PENALTY * over


def make_extras(spells, lane, target, set_code):
    """f(indices of spells in the deck) -> WR points for synergy (SYN_MATES, SYN_CARDS) less composition misses (DIM_PENALTY)."""
    if not set_code or not (SYN_MATES or SYN_CARDS or DIM_PENALTY or BOMB_BONUS):
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

    def extras(idx):
        idx = sorted(idx)
        out = BOMB_BONUS * sum(1 for i in idx if i in bombs)
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


def build_pair(pool, lane, model, target, min_creatures_slack=None, variant="", splash="", splash_cost=None):
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
        if cost < 0:
            return float(model.win_rate(dict(counts), lane, used))
        # A fitted rule decided the colours: score the cards and shape as a plain pair (the model's colour-count and
        # splash terms mostly say that weaker decks add colours).
        return float(model.win_rate(dict(counts), lane[:2], "")) - (cost if used else 0.0)

    # Start from the pair's own cards (splash cards come in only when a swap pays).
    order = sorted(range(len(spells)), key=lambda i: (not castable(spells[i], lane), -wr(spells[i])))
    deck, side = set(order[:SPELLS]), set(order[SPELLS:])
    best = score([spells[i] for i in deck], deck)
    improved = True
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
    chosen = [spells[i] for i in deck]
    used = splash if splash and any(not castable(c, lane) for c in chosen) else ""
    if splash and not used:
        return None  # the splash did not pay: the plain pair covers it
    lands = mana(chosen)
    full = dict(Counter(c["name"] for c in chosen))
    full.update(lands)
    full.update(Counter(c["name"] for c in nonbasics))
    return {"lane": lane, "splash": used, "variant": variant, "spells": chosen, "nonbasics": nonbasics, "basics": lands,
            "penalty": penalty(chosen), "predicted": predicted(full, used), "objective": best, "extras": round(extras(deck), 2),
            "sideboard": [spells[i] for i in side] + [c for c in pool if c not in spells and c not in nonbasics]}


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
    if not plain:
        return []

    def splashes(b, chance):
        out = []
        for col in "WUBRG":
            if col not in b["lane"]:
                s = build_pair(pool, b["lane"], model, targets[b["lane"]], splash=col, splash_cost=None if chance is None else 0.0)
                if s and s["penalty"] <= 3 * PENALTY and s["objective"] > b["objective"]:
                    s["splash_chance"] = None if chance is None else round(chance, 2)
                    out.append(s)

        def fixing(col):
            return min(3, sum(1 for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or [])
                              and col in (c.get("colors") or [])))
        return sorted(out, key=lambda x: -(x["objective"] + SPLASH_FIX_POINTS * fixing(x["splash"])))

    def third(b):
        adds, fixing, playables, col = three_features(pool, b["lane"])
        lane = "".join(sorted(b["lane"] + (col or ""), key="WUBRG".index))
        if not col or lane not in (targets or {}):
            return None, None
        chance = rule_chance(three, (adds, fixing, playables))
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
