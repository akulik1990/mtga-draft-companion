"""Everything readable about each card of a set, one row per card: the table deck measures are built from.

    uv run python tools/card_table.py FRA      # writes <data>/Temp/cards_FRA.csv

From the dataset (types, subtypes, mana cost, keywords, rules text, tags) and Scryfall's power/toughness
(needs.scryfall): card types, stats, keywords, effects matched in the rules text, our roles (removal, card advantage,
evasion, mana, creature) and the dataset's tags. A new measure is one more entry in FEATURES; decks pick it up from
their card counts (tools/deck_table.py), without reading the game data again.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402

KEYWORDS = ("flying", "deathtouch", "lifelink", "trample", "haste", "vigilance", "reach", "menace", "first strike",
            "double strike", "flash", "defender", "hexproof", "ward", "indestructible", "prowess")

# name -> rules-text pattern (case-insensitive); counted 1 when the card's text matches
EFFECTS = {
    "draw": r"\bdraws? (?:a|an|one|two|three|x|\d+) cards?",
    "scry": r"\bscry\b",
    "surveil": r"\bsurveil\b",
    "mill": r"\bmills?\b",
    "discard": r"\bdiscards?\b",
    "loot": r"draw[^.]*then discard|discard[^.]*then draw",
    "plus_counters": r"\+1/\+1 counters?",
    "pump": r"gets? \+\d+/\+\d+|get \+\d+/\+\d+|gets? \+x/\+x",
    "anthem": r"creatures you control get \+",
    "shrink": r"gets? -\d+/-\d+|get -\d+/-\d+|-x/-x|-1/-1 counters?",
    "destroy": r"\bdestroy target",
    "exile_target": r"\bexile target",
    "damage": r"deals? (?:\d+|x|that much) damage to (?:any target|target creature|target player|each)",
    "fight": r"\bfights?\b|deals damage equal to its power to",
    "counterspell": r"counter target (?:spell|creature spell|noncreature spell|activated)",
    "bounce": r"return target [^.]*to (?:its|their) owner's hand",
    "tap_down": r"tap target|doesn't untap|don't untap|can't (?:attack|block)",
    "sacrifice": r"\bsacrifices?\b",
    "edict": r"(?:each opponent|target player|target opponent) sacrifices",
    "gain_life": r"gains? (?:\d+|x|that much) life|\blifelink\b",
    "treasure": r"\btreasure\b",
    "food_clue": r"\b(?:food|clue|blood|map) tokens?\b",
    "creature_tokens": r"create[^.]*creature tokens?|create[^.]*\d+/\d+[^.]*tokens?",
    "tokens": r"\bcreates? [^.]*\btokens?\b",
    "drain": r"damage to (?:each opponent|target opponent|that player|each player|target player|defending player)"
             r"|(?:each opponent|target opponent|target player|that player|defending player|each player) loses (?:\d+|x|that much) life",
    "any_target": r"damage to any target",
    "noncombat_payoff": r"dealt noncombat damage|would deal noncombat damage",
    "combat_damage_trigger": r"deals combat damage to a player|are dealt combat damage",
    "landcycling": r"landcycling",
    "land_search": r"search your library for (?:a|up to \w+) (?:basic )?land",
    "etb": r"when (?:this creature|this permanent|[^.,]*) enters",
    "dies": r"when (?:this creature|[^.,]*) dies|whenever (?:a|another) creature (?:you control )?dies",
    "attack_trigger": r"whenever [^.]*attacks",
    "recursion": r"return [^.]*from (?:your|a) graveyard",
    "copy": r"\bcopy\b|\bcopies\b",
    "empower_jace": r"empower jace",
    "prepared": r"\bprepared\b",
    "threshold": r"\bthreshold\b",
    "flashback": r"\bflashback\b",
    "cycling": r"(?<!land)cycling",
    "mana_sink": r"\{\d*[wubrgcx]*\}(?:\{[wubrgcx\d]\})*: |activate only as a sorcery",
}

FEATURES = None  # filled by card_rows: the column names after "name"
RANK_BUCKETS = 10  # rank_1_10 .. rank_91_100


def _scryfall(set_code):
    from companion.advisor_v2 import needs

    return needs.scryfall(set_code)


def features(card, stats=None):
    """{feature: value} for one card (non-land and land alike)."""
    from companion.advisor_v2 import composition
    from companion.advisor_v2.deckbuilder import card_roles, curve_slot, pips

    types = card.get("types") or []
    text = (card.get("oracle_text") or "").lower()
    kw = {k.lower() for k in card.get("keywords") or []}
    tags = set(config.tags(card))
    st = (stats or {}).get(card["name"]) or (stats or {}).get(card["name"].split(" // ")[0]) or {}
    num = lambda v: float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()) else None
    out = {
        "cmc": int(card.get("cmc") or 0), "slot": curve_slot(card) or "land", "colours": len(card.get("colors") or []),
        "pips": sum(pips(card).values()), "double_pip": int(any(n >= 2 for n in pips(card).values())),
        "rarity": card.get("rarity") or "", "land": int("Land" in types), "basic": int("Basic" in types),
    }
    for r in ("common", "uncommon", "rare", "mythic"):
        out[r] = int(card.get("rarity") == r)
    for t in ("Creature", "Artifact", "Enchantment", "Instant", "Sorcery", "Planeswalker", "Battle"):  # no supertypes in the dataset
        out[t.lower()] = int(t in types)
    subs = {s.lower() for s in card.get("subtypes") or []}
    out.update(equipment=int("equipment" in subs), aura=int("aura" in subs), vehicle=int("vehicle" in subs),
               double_faced=int("//" in (card.get("mana_cost") or "")), creature_like=int(composition.is_creature(card)))
    out["power"], out["toughness"] = num(st.get("power")), num(st.get("toughness"))
    for k in KEYWORDS:
        out[k.replace(" ", "_")] = int(k in kw or bool(re.search(rf"\b{k}\b", text)))
    for name, pat in EFFECTS.items():
        out[name] = int(bool(re.search(pat, text)) or name in kw)
    roles = card_roles(card) if "Land" not in types else set()
    for r in ("creatures", "removal", "card_advantage", "evasion", "mana"):
        out[f"role_{r}"] = int(r in roles)
    for t in ("combat_trick", "lifegain", "mana_sink", "protection", "fixing_ramp", "hate"):
        out[f"tag_{t}"] = int(t in tags)
    return out


def card_rows(set_code):
    """[(name, features)] for every distinct card name in the set's dataset."""
    global FEATURES
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    stats, seen, rows = _scryfall(set_code), set(), []
    wr = lambda c: ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
    ranked = sorted({c["name"]: wr(c) for c in data.values() if wr(c) > 0}.items(), key=lambda kv: -kv[1])
    rank = {n: i + 1 for i, (n, _) in enumerate(ranked)}
    # the set's own mechanics: every keyword the dataset lists on its cards, one column each
    mechs = sorted({k for c in data.values() for k in c.get("keywords") or []})
    for c in data.values():
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        c = dict(c, set=set_code)
        ft = features(c, stats)
        r = rank.get(c["name"])
        ft["gihwr"], ft["rank"] = round(wr(c), 1), r or ""
        for lo in range(1, RANK_BUCKETS * 10, 10):  # rank_1_10, rank_11_20, ...: set rank by All Decks GIH WR
            ft[f"rank_{lo}_{lo + 9}"] = int(bool(r) and lo <= r <= lo + 9)
        for m in mechs:
            ft["mech_" + m.lower().replace(" ", "_")] = int(m in (c.get("keywords") or []))
        rows.append((c["name"], ft))
    FEATURES = list(rows[0][1])
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401
    config.activate(a.set)
    rows = card_rows(a.set)
    path = DATA_DIR / "Temp" / f"cards_{a.set}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + FEATURES)
        for name, ft in sorted(rows):
            w.writerow([name] + [ft[k] for k in FEATURES])
    print(f"{len(rows)} cards x {len(FEATURES)} features -> {path}")


if __name__ == "__main__":
    main()
