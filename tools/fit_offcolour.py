"""How often top players play a card outside their core pair, by its strength: the picker's "chance of doubt".

    uv run python tools/fit_offcolour.py FRA        # writes the offcolour section of the set's fitted.json

From 17Lands' public game data (deck + sideboard = pool) for 60%+ players' two- and three-colour builds: every spell
in the pool that their core pair (the pair casting most of their deck) cannot cast, and whether it made the deck.
Logistic on the card's All Decks GIH WR (in SDs of the set) and whether it needs two or more off-colour pips.
FRA: 25% overall; 89% of single-pip cards at 62%+ GIH WR, 5% below 54%.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.lanes import LANES, castable, off_pips  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--builds", type=int, default=8000)
    p.add_argument("--min-skill", type=float, default=0.60)
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by.setdefault(n, c)
    wrs = [w for c in data.values() if (w := ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0) > 0]
    mean, std = float(np.mean(wrs)), float(np.std(wrs))
    X, y, seen = [], [], set()
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[5:]) for i, h in enumerate(header) if h.startswith("deck_")]
        side = {h[10:]: i for i, h in enumerate(header) if h.startswith("sideboard_")}
        for row in reader:
            skill, main = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
            key = (row[col["draft_id"]], row[col["build_index"]])
            if len(main) not in (2, 3) or skill in ("", "None") or float(skill) < a.min_skill or key in seen:
                continue
            seen.add(key)
            deck, pool = {}, {}
            for i, n in deck_cols:
                d = int(float(row[i] or 0))
                s = int(float(row[side[n]] or 0)) if n in side else 0
                if d:
                    deck[n] = d
                if d + s:
                    pool[n] = d + s
            spells = [by[n] for n in deck if n in by and "Land" not in (by[n].get("types") or [])]
            core = max(LANES, key=lambda lane: sum(castable(c, lane) for c in spells))
            for n, k in pool.items():
                c = by.get(n)
                w = ((c or {}).get("deck_colors") or {}).get("All Decks", {}).get("gihwr") or 0
                if not c or "Land" in (c.get("types") or []) or castable(c, core) or w <= 0:
                    continue
                for i in range(k):
                    X.append([(w - mean) / std, float(len(off_pips(c, core)) >= 2)])
                    y.append(1.0 if i < deck.get(n, 0) else 0.0)
            if len(seen) >= a.builds:
                break
    X, y = np.array(X), np.array(y)
    Z = np.c_[np.ones(len(X)), X]

    def nll(w):
        q = np.clip(1 / (1 + np.exp(-Z @ w)), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    w = minimize(nll, np.zeros(3)).x
    print(f"{len(seen)} builds, {len(y)} off-colour cards, played {100 * y.mean():.1f}%")
    print(f"logit = {w[0]:+.2f} {w[1]:+.2f} x WR z {w[2]:+.2f} x [2+ off pips]")
    for z in (-1, 0, 1, 2):
        print(f"  WR {mean + z * std:.1f}: 1 pip {100 / (1 + np.exp(-(w[0] + w[1] * z))):.0f}%, 2+ pips {100 / (1 + np.exp(-(w[0] + w[1] * z + w[2]))):.0f}%")
    rule = {"w": [round(float(v), 4) for v in w], "mean": round(mean, 3), "std": round(std, 3), "n": int(len(y)), "played": round(float(y.mean()), 3)}
    path = config.write_fitted(a.set, "offcolour", rule, "tools/fit_offcolour.py: share of cards outside 60%+ players' core pair that made their deck, logistic on GIH WR z and 2+ off pips")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
