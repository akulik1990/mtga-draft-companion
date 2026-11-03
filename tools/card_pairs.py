"""Where the set's setup bombs shine: the colour pairs each reviewed top card that wants something wins most in.

    uv run python tools/card_pairs.py FRA [--within 1.5] [--min 150]

For every card in the set's manual.json card_wants that is not standalone, the pairs (castable or splashing it) with at
least --min games in hand whose GIH WR is within --within points of its best pair. Written to the card_pairs section of
the set's fitted.json; the engine's CARD_PAIRS_PULL adds an extra pull toward those pairs while the card is in the pool
(Ajani Unrelenting: UR and RG, not WR; Vraska, the Cutting Glare: UB with green splashed).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.lanes import LANES  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--within", type=float, default=1.5)
    p.add_argument("--min", type=int, default=150)
    a = p.parse_args(argv)
    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {c["name"].split(" // ")[0]: c for c in data.values()}
    wants = config.layer(a.set, "manual").get("card_wants") or {}
    standalone = set(wants.get("_standalone") or [])
    out = {}
    for name, spec in wants.items():
        if name.startswith("_") or name in standalone or not isinstance(spec, dict) or name not in by:
            continue
        rows = {lane: r["gihwr"] for lane in LANES if (r := (by[name].get("deck_colors") or {}).get(lane) or {}).get("gihwr")
                and (r.get("samples") or 0) >= a.min}
        if not rows:
            continue
        best = max(rows.values())
        out[name] = sorted((lane for lane, w in rows.items() if w >= best - a.within), key=lambda lane: -rows[lane])
        print(f"{name:32} {', '.join(f'{lane} {rows[lane]:.1f}' for lane in out[name])}")
    path = config.write_fitted(a.set, "card_pairs", out,
                               f"tools/card_pairs.py: setup cards' pairs within {a.within} GIH WR points of their best (>= {a.min} games in hand)")
    print(f"wrote card_pairs for {len(out)} cards to {path}")


if __name__ == "__main__":
    main()
