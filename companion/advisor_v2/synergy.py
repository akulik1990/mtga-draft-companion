"""Pairwise synergy with the pool, per lane, from tools/synergy/build.py. Off unless ENABLED."""

import json
from pathlib import Path

ENABLED = False
K = 300.0  # confidence: games_with_both / (games + K)


def load(set_code, temp_folder):
    path = Path(temp_folder) / f"synergy_{set_code}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def bonus(table, card, pool, lane):
    """Win-rate points to add to V(card | lane): confidence-weighted sum of lifts with pool cards."""
    rows = table.get(lane)
    if not rows:
        return 0.0
    name, total = card.get("name"), 0.0
    for other in pool:
        key = f"{name}|{other.get('name')}" if name <= other.get("name", "") else f"{other.get('name')}|{name}"
        row = rows.get(key)
        if row:
            games, _, lift = row
            total += lift * games / (games + K)
    return total


def roles(card, theme):
    """(enabler, payoff) for a card under one theme definition (a set's manual.json "themes" entry): patterns over the
    lower-cased oracle text, or types / minimum mana value / names where the theme says so."""
    import re

    text = re.sub(r"\([^)]*\)", "", (card.get("oracle_text") or "").lower())  # reminder text is not the card's own
    types = card.get("types") or []
    enabler = bool(theme.get("enabler") and re.search(theme["enabler"], text)) or any(t in types for t in theme.get("enabler_types", ()))
    payoff = bool(theme.get("payoff") and re.search(theme["payoff"], text)) or card.get("name") in theme.get("payoff_names", ())
    if theme.get("payoff_min_cmc"):
        payoff = payoff or ((card.get("cmc") or 0) >= theme["payoff_min_cmc"] and all(t in types for t in theme.get("payoff_types", ())))
    return enabler, payoff
