"""Top drafters' picks beyond v2's score: the features of tools/choice_model.py and the engine's correction from them.

The set's fitted "choices" section (tools/choice_model.py fit) holds, per draft stage, conditional-logit coefficients
for v2's score and each feature. With DraftAdvisorV2.CHOICE_ADJUST on, a card's final score gains
sum(beta_f / beta_v2 * f) for its stage: v2's ranking moved toward how top drafters choose, in v2's own units.
"""

import math
from collections import Counter
from functools import lru_cache

from companion.advisor_v2 import config
from companion.advisor_v2.lanes import LANES, any_colour_fixer, castable, off_pips
from companion.advisor_v2.synergy import roles

FEATURES = ["v2", "syn_payoff", "syn_enabler", "lonely_payoff", "two_need", "creature_need", "removal_need",
            "on_colour", "splash_fixed", "fixer", "cmc"]
# Draft phases (the user's structure, 2026-11-02): pack 1 opens on raw card quality and signals, closes on synergy;
# pack 2 opens on synergy and settling final colours (two or three, splash or not); from mid pack 2 the deck's fit
# (curve, needs, composition) takes over; picks 12-14 of any pack fill the deck.
STAGES = {"P1 picks 1-7": lambda p, k: p == 1 and k <= 7,
          "P1 picks 8-11": lambda p, k: p == 1 and 8 <= k <= 11,
          "P2 picks 1-6": lambda p, k: p == 2 and k <= 6,
          "fit phase": lambda p, k: ((p == 2 and 7 <= k) or p == 3) and k <= 11,
          "picks 12-14": lambda p, k: k >= 12}

SHORT = {"P1 picks 1-7": "p1open", "P1 picks 8-11": "p1close", "P2 picks 1-6": "p2open", "fit phase": "fit", "picks 12-14": "late"}


def stage_of(pack_no, pick):
    return next(s for s, test in STAGES.items() if test(pack_no, pick))


@lru_cache(maxsize=4)
def theme_tags(set_code):
    """{theme: {card name: (enabler, payoff)}} for the set's manual.json themes, over the set's cards."""
    import json

    from companion import DATA_DIR

    themes = config.layer(set_code, "manual").get("themes") or {}
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    tags = {t: {} for t in themes}
    for c in data.values():
        for t, th in themes.items():
            r = roles(c, th)
            if r[0] or r[1]:
                for n in (c["name"], c["name"].split(" // ")[0]):
                    tags[t][n] = r
    return tags


def top_lane(pool):
    def wr(c):
        return ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 50.0
    spells = [c for c in pool if "Land" not in (c.get("types") or [])]
    return max(LANES, key=lambda lane: sum(max(0.0, wr(c) - 50.0) for c in spells if castable(c, lane)))


def features(card, pool, lane, themes, tags, targets, score, best):
    """One row of FEATURES for a candidate card given the pool before the pick."""
    spells = [c for c in pool if "Land" not in (c.get("types") or [])]
    n = len(spells)
    scale = 23.0 / n if n else 0.0
    types = card.get("types") or []
    is_spell = "Land" not in types
    syn_p = syn_e = 0.0
    payoff_themes, supported = 0, 0
    for t in themes:
        en, pa = tags[t].get(card["name"], (False, False))
        e_pool = sum(tags[t].get(c["name"], (False, False))[0] for c in pool)
        p_pool = sum(tags[t].get(c["name"], (False, False))[1] for c in pool)
        if pa:
            syn_p += math.log1p(e_pool)
            payoff_themes += 1
            supported += e_pool > 0
        if en:
            syn_e += math.log1p(p_pool)
    lonely = float(payoff_themes > 0 and supported == 0 and n >= 6)
    tgt = targets.get(lane) or {}

    def need(role, want):
        have = sum(1 for c in spells if role(c)) * scale
        return max(0.0, want - have) / want if want and n >= 6 else 0.0

    two = is_spell and int(card.get("cmc") or 0) == 2
    creature = "Creature" in types
    removal = is_spell and "removal" in (card.get("tags") or [])
    on = castable(card, lane)
    off = off_pips(card, lane)
    fixing = Counter()
    for c in pool:
        if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or []):
            for col in c.get("colors") or []:
                if col not in lane:
                    fixing[col] += 1
        if any_colour_fixer(c):
            for col in "WUBRG":
                fixing[col] += 1
    splash_fixed = float(is_spell and not on and len(off) == 1 and any(fixing[col] for col in off[0]))
    fixer = float(("Land" in types and any(col not in lane for col in card.get("colors") or []) and any(col in lane for col in card.get("colors") or []))
                  or any_colour_fixer(card))
    return [score - best, syn_p, syn_e, lonely,
            float(two) * need(lambda c: int(c.get("cmc") or 0) == 2, tgt.get("two_drops", 8.5)),
            float(creature) * need(lambda c: "Creature" in (c.get("types") or []), tgt.get("creatures", 13.5)),
            float(removal) * need(lambda c: "removal" in (c.get("tags") or []), max(1.0, tgt.get("removal", 8.0) - 2.0)),
            float(on), splash_fixed, fixer, float(card.get("cmc") or 0) if is_spell else 0.0]





SYNERGY = (1, 2, 3)  # FEATURES indices: syn_payoff, syn_enabler, lonely_payoff
GROUPS = {"synergy": SYNERGY, "needs": (4, 5, 6), "colour": (7,), "splash": (8,), "fixer": (9,), "cmc": (10,),
          "all": tuple(range(1, len(FEATURES)))}


def allowed(spec, stage):
    """FEATURES indices a CHOICE_FEATURES spec ("early:synergy;middle:synergy,fixer;late:all") applies at this stage;
    None when the spec is empty (fall back to CHOICE_STAGES)."""
    if not spec:
        return None
    for part in filter(None, spec.split(";")):
        name, groups = part.split(":")
        if name.strip() == SHORT[stage]:
            return {i for g in groups.split(",") if g.strip() for i in GROUPS[g.strip()]}
    return set()


def synergy_points(card, pool, set_code, pack_no, pick):
    """The synergy part of the choice model for this card and pool, in v2's WR points at this stage: payoffs whose
    enablers are in the pool, enablers for payoffs in the pool, payoffs with none of their enablers. 0 without a model."""
    model = ((config.fitted(set_code, "choices") or {}).get("stages") or {}).get(stage_of(pack_no, pick))
    if not model:
        return 0.0
    tags = theme_tags(set_code)
    x = features(card, pool, top_lane(pool), list(tags), tags, {}, 0.0, 0.0)
    b = model["beta"]
    return sum(b[i] / b[0] * x[i] for i in SYNERGY)


def correction(card, pool, set_code, pack_no, pick, lane=None, stages="all", skip=(), spec=""):
    """WR points to add to v2's final score for this card, from the fitted choice model; 0 without one or outside stages.
    skip: FEATURES indices another term already applies (SYNERGY when SYNERGY_FIT is on)."""
    stage = stage_of(pack_no, pick)
    use = allowed(spec, stage)
    if use is None:
        if stages != "all" and SHORT[stage] not in stages.split(","):
            return 0.0
        use = set(GROUPS["all"])
    if not use:
        return 0.0
    model = ((config.fitted(set_code, "choices") or {}).get("stages") or {}).get(stage)
    if not model:
        return 0.0
    tags = theme_tags(set_code)
    targets = (config.fitted(set_code, "decks") or {}).get("pairs") or {}
    x = features(card, pool, lane or top_lane(pool), list(tags), tags, targets, 0.0, 0.0)
    b = model["beta"]
    return sum(b[i] / b[0] * x[i] for i in range(1, len(b)) if i in use and i not in skip)
