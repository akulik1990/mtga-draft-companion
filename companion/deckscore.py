"""Score a deck you edit yourself: the deck model's win rate, shape against top players, mana, and a per-card breakdown.

The same pieces the builder decides with (companion/advisor_v2/deckbuilder.py), shown instead of searched over:
predicted win rate (tools/deck_model.py) less the shape penalty is the builder's objective without its synergy extras.
"""

import json
from collections import Counter
from functools import lru_cache

from companion.advisor_v2 import config, deckbuilder, needs
from companion.advisor_v2.lanes import LANES, castable, off_pips, pips

ROLES = ("creatures", "removal", "card_advantage", "evasion", "two_drops", "five_plus")


@lru_cache(maxsize=4)
def bomb_ranks(set_code, n=30):
    """{name: rank} of the set's n best cards by All Decks GIH WR (needs.top_cards' list, ranked)."""
    from companion import DATA_DIR

    path = DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json"
    if not set_code or not path.exists():
        return {}
    data = json.loads(path.read_text())["card_ratings"].values()
    wr = {c["name"]: _gihwr(c, "All Decks") for c in data}
    return {name: i for i, name in enumerate(sorted((k for k in wr if wr[k] > 0), key=lambda k: -wr[k])[:n], 1)}


def _gihwr(card, lane):
    return float(((card.get("deck_colors") or {}).get(lane) or {}).get("gihwr") or 0.0)


def score(set_code, pool, counts, model):
    """counts: {name: copies} of the 40 (basics by name). Non-basic copies are capped at what the pool holds."""
    by_name, held = {}, Counter()
    for c in pool:
        by_name.setdefault(c["name"], c)
        held[c["name"]] += 1
    basic_names = set(deckbuilder.BASIC.values())
    deck = Counter()
    for name, n in counts.items():
        n = int(n)
        if n > 0 and (name in basic_names or name in by_name):
            deck[name] = n if name in basic_names else min(n, held[name])
    cards = lambda d: [by_name[n] for n, k in d.items() if n in by_name for _ in range(k)]
    spells_of = lambda d: [c for c in cards(d) if "Land" not in (c.get("types") or [])]

    from companion.session import mana_identity  # session imports this module's callers, not this module

    rows = [{**by_name[n], "count": k} for n, k in deck.items() if n in by_name]
    ident = mana_identity(rows)
    main, splash = ident["main"], ident["splash"]
    ruled = bool(config.fitted(set_code, "splash"))  # with fitted colour rules the builder scores without the splash term
    targets = (config.fitted(set_code, "decks") or {}).get("pairs") or {}
    target = targets.get(main)

    def wr(d):
        return float(model.win_rate(dict(d), main, "" if ruled else splash)) if model and main else None

    # The builder's synergy extras (inclusion rates, deck-mates, reviewed-card synergy) over the pool's spells by index.
    pool_spells = [c for c in pool if "Land" not in (c.get("types") or [])]
    extras = deckbuilder.make_extras(pool_spells, main, target, set_code) if target else (lambda idx: 0.0)
    slots = {}
    for i, c in enumerate(pool_spells):
        slots.setdefault(c["name"], []).append(i)

    def objective(d):
        w = wr(d)
        if w is None:
            return None
        idx = [i for n, k in d.items() for i in slots.get(n, [])[:k]]
        return w - (deckbuilder.shape_penalty(spells_of(d), target, main) if target else 0.0) + extras(idx)

    spells = spells_of(deck)
    have = Counter(r for c in spells for r in deckbuilder.roles(c))
    curve = Counter(deckbuilder.curve_slot(c) for c in spells)
    shape = [{"name": k, "deck": have[k], "top": target.get(k) if target else None} for k in ROLES]
    shape += [{"name": f"{k} mana", "deck": curve[k], "top": (target.get("curve") or {}).get(k) if target else None} for k in deckbuilder.CURVE]

    sources = Counter()
    for n, k in deck.items():
        c = by_name.get(n)
        if n in basic_names:
            sources[next(col for col, b in deckbuilder.BASIC.items() if b == n)] += k
        elif c and "Land" in (c.get("types") or []):
            for col in c.get("colors") or []:
                sources[col] += k
    mana = []
    for col in "WUBRG":
        need = [c for c in spells if any(col in opts and len(opts) == 1 for opts in pips(c))]
        if need or sources[col]:
            mana.append({"colour": col, "sources": sources[col], "spells": len(need),
                         "max_pips": max((sum(1 for o in pips(c) if o == [col]) for c in need), default=0)})

    base = objective(deck)
    mates = (needs.deckmates(set_code) or {}).get(main) or {}
    incl = (needs.inclusion(set_code) or {}).get(main) or {}
    deck_spell_names = sorted({c["name"] for c in spells})
    side_spell_names = sorted(n for n in held if held[n] > deck.get(n, 0) and "Land" not in (by_name[n].get("types") or []))
    bombs = bomb_ranks(set_code)
    out = []
    for name in sorted(held):
        c = by_name[name]
        if name in basic_names:
            continue
        land = "Land" in (c.get("types") or [])
        homes = [lane for lane in LANES if castable(c, lane) and _gihwr(c, lane) > 0]
        power_lane = max(homes, key=lambda lane: _gihwr(c, lane)) if homes else "All Decks"
        status = "on" if not main or castable(c, main) else ("splash" if splash and castable(c, main + splash) or len(off_pips(c, main)) == 1 and splash and off_pips(c, main)[0][0] in splash else "off")
        row = mates.get(name) or {}
        others = [n for n in deck_spell_names if n != name]
        r = {
            "name": name, "in_deck": deck.get(name, 0), "held": held[name], "land": land, "status": status,
            "power": round(_gihwr(c, power_lane) or _gihwr(c, "All Decks"), 1), "power_lane": power_lane,
            "fit": round(_gihwr(c, main) or _gihwr(c, "All Decks"), 1) if main else None,
            "mates": round(sum(row.get(n, 0.0) for n in others) / max(1, len(others)), 3) if not land else None,
            "inclusion": round(incl[name], 2) if name in incl else None,
            "bomb": bombs.get(name),
        }
        if base is not None and not land:
            if deck.get(name):
                # Worth over the best sideboard spell that could take its slot (an empty slot when there is none).
                best = None
                for in_name in side_spell_names:
                    trial = deck.copy()
                    trial[name] -= 1
                    trial[in_name] += 1
                    o = objective(+trial)
                    if best is None or o > best[0]:
                        best = (o, in_name)
                if best is None:
                    without = deck.copy()
                    without[name] -= 1
                    best = (objective(+without), None)
                r["value"], r["swap_for"] = round(base - best[0], 2), best[1]
            else:
                best = None
                for out_name in deck_spell_names:
                    trial = deck.copy()
                    trial[out_name] -= 1
                    trial[name] += 1
                    gain = objective(+trial) - base
                    if best is None or gain > best[0]:
                        best = (gain, out_name)
                if best:
                    r["value"], r["swap_for"] = round(best[0], 2), best[1]
        out.append(r)

    total = sum(deck.values())
    return {
        "main": main, "splash": splash, "total": total, "spells": len(spells), "lands": total - len(spells),
        "deck_wr": None if base is None else round(wr(deck), 2),
        "shape_penalty": round(deckbuilder.shape_penalty(spells, target, main), 2) if target else None,
        "objective": None if base is None else round(base, 2),
        "extras": round(extras([i for n, k in deck.items() for i in slots.get(n, [])[:k]]), 2) if target else None,
        "shape": shape, "mana": mana, "cards": out,
        "bombs": {"held": sum(held[n] for n in bombs if n in held), "in_deck": sum(deck.get(n, 0) for n in bombs)},
    }
