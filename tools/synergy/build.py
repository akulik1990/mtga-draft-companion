"""From 17Lands public game data: what a colour pair's typical deck looks like, and which cards win together.

    uv run python tools/synergy/build.py FRA            # downloads the game CSV once into <data>/Temp/synergy/
    → <data>/Temp/composition_FRA.json  {pair: {games, curve: {1..6+: mean count}, creatures, interaction}}
    → <data>/Temp/synergy_FRA.json      {pair: {"A|B": [games_with_both, wr_both, lift]}}  (n ≥ 100)

Composition is per main colour pair over every deck in the data (what people
actually register, not what they should). Synergy: for cards a, b in the same
deck, wr_both is the win rate of games with both; lift is wr_both minus the mean
of the two cards' own win rates in that pair, in win-rate points. The engine
treats composition as targets (companion/advisor_v2/composition.py) and synergy
as a small additive term that is off by default (synergy.py).
"""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402

GAME_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{set}.PremierDraft.csv.gz"
PAIRS = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
MIN_GAMES = 100


def fetch(set_code):
    import requests

    dest = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {GAME_URL.format(set=set_code)}")
        with requests.get(GAME_URL.format(set=set_code), stream=True, timeout=60) as r, open(dest, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def card_facts(set_code):
    """name -> (cmc, is_creature, is_interaction, is_land) from the app's dataset."""
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())
    facts = {}
    for c in data["card_ratings"].values():
        types, tags = c.get("types") or [], c.get("tags") or []
        f = (int(c.get("cmc") or 0), "Creature" in types, "removal" in tags, "Land" in types)
        facts.setdefault(c["name"], f)
        facts.setdefault(c["name"].split(" // ")[0], f)
    return facts


def load(path):
    """(card names, deck count matrix uint8, main_colors list, won array)."""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        idx = [i for i, _ in deck_cols]
        names = [n for _, n in deck_cols]
        rows, colors, won = [], [], []
        for row in reader:
            rows.append([int(row[i] or 0) for i in idx])
            colors.append(row[col["main_colors"]])
            won.append(row[col["won"]] == "True")
            if len(rows) % 50000 == 0:
                print(f"  {len(rows)} games", flush=True)
    return names, np.array(rows, dtype=np.uint8), colors, np.array(won, dtype=bool)


def normalize_pair(main_colors):
    s = "".join(c for c in "WUBRG" if c in main_colors)
    return s if s in PAIRS else None


def build(set_code, names, decks, colors, won, facts):
    composition, synergy = {}, {}
    cmc = np.array([min(6, max(1, facts.get(n, (0,))[0])) for n in names])
    is_creature = np.array([facts.get(n, (0, False))[1] for n in names])
    is_interaction = np.array([facts.get(n, (0, False, False))[2] for n in names])
    is_land = np.array([facts.get(n, (0, False, False, False))[3] for n in names])
    spells = ~is_land
    pair_of = np.array([normalize_pair(c) or "" for c in colors])
    for pair in PAIRS:
        sel = pair_of == pair
        n = int(sel.sum())
        if n < MIN_GAMES:
            continue
        d = decks[sel].astype(np.float32)
        counts = d[:, spells]
        curve = {str(b) if b < 6 else "6+": round(float(counts[:, cmc[spells] == b].sum(axis=1).mean()), 2) for b in range(1, 7)}
        composition[pair] = {
            "games": n,
            "spells": round(float(counts.sum(axis=1).mean()), 2),
            "curve": curve,
            "creatures": round(float(d[:, is_creature].sum(axis=1).mean()), 2),
            "interaction": round(float(d[:, is_interaction].sum(axis=1).mean()), 2),
        }
        # Synergy: binary presence, co-occurrence and co-wins by matrix products.
        present = (d > 0).astype(np.float32)
        w = won[sel].astype(np.float32)
        both = present.T @ present
        both_won = (present * w[:, None]).T @ present
        solo = np.diag(both)
        solo_wr = np.divide(np.diag(both_won), solo, out=np.zeros_like(solo), where=solo > 0)
        out = {}
        ii, jj = np.where(np.triu(both, 1) >= MIN_GAMES)
        for i, j in zip(ii, jj):
            g = both[i, j]
            wr = both_won[i, j] / g
            out[f"{names[i]}|{names[j]}"] = [int(g), round(float(100 * wr), 2), round(float(100 * (wr - (solo_wr[i] + solo_wr[j]) / 2)), 2)]
        synergy[pair] = out
        print(f"{pair}: {n} games, {len(out)} card pairs with {MIN_GAMES}+ games", flush=True)
    return composition, synergy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    a = p.parse_args(argv)
    path = fetch(a.set)
    print(f"reading {path}")
    names, decks, colors, won = load(path)
    composition, synergy = build(a.set, names, decks, colors, won, card_facts(a.set))
    out_c = DATA_DIR / "Temp" / f"composition_{a.set}.json"
    out_s = DATA_DIR / "Temp" / f"synergy_{a.set}.json"
    out_c.write_text(json.dumps(composition, indent=1))
    out_s.write_text(json.dumps(synergy))
    print(f"wrote {out_c} and {out_s}")


if __name__ == "__main__":
    main()
