"""Rebuild the 2026-11-02 Quick Draft pool from its Player.log and show what the current builder makes of it.

    uv run python tools/ub_deck_check.py [LOG]

Card ids are looked up the way tools/live_diff.py does (replay.App ratings keyed by grpId).
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import composition, config, deckbuilder  # noqa: E402
import companion.advisor_v2.engine  # noqa: E402,F401
from tools.deck_model import DeckModel  # noqa: E402
from tools.live_diff import picks  # noqa: E402
from tools.replay.run import replay  # noqa: E402

SET = "FRA"
LOG = ROOT / "data" / "Temp" / "live" / "FRA-quickdraft-2026-11-02.Player.log"


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG
    config.activate(SET)
    app = replay.App(str(DATA_DIR / "Sets" / f"{SET}_PremierDraft_All_Data.json"))
    ratings = {k: dict(v, set=SET) for k, v in app.dataset.get_card_ratings().items()}
    steps = picks(log)
    last = steps[-1]
    ids = list(last[3]) + [i for i in last[2]]  # picks before the last one + the last pack's remaining card
    pool = [ratings[i] for i in ids if i in ratings]
    print(f"{len(steps)} picks, pool {len(pool)} of {len(ids)} ids resolved")
    spells = [c for c in pool if "Land" not in (c.get("types") or [])]
    print("pool:", ", ".join(sorted(c["name"] for c in pool)))

    model = DeckModel(SET, deckbuilder.MODEL)
    targets = config.fitted(SET, "decks")["pairs"]
    structure = config.fitted(SET, "structure")
    ub_cards = [c for c in spells if set(c.get("colors") or []) <= {"U", "B"}]
    rl = lambda cs: sum("removal" in config.tags(c) for c in cs)
    print(f"\nUB-castable spells in pool: {len(ub_cards)}; creatures {sum(composition.is_creature(c) for c in ub_cards)}; removal {rl(ub_cards)}")
    print("UB creatures:", ", ".join(sorted(c['name'] for c in ub_cards if composition.is_creature(c))))
    print("UB removal:", ", ".join(sorted(c['name'] for c in ub_cards if 'removal' in config.tags(c))))

    builds = deckbuilder.build(pool, model, targets, top=4, rule=config.fitted(SET, "splash"), three=config.fitted(SET, "three"))
    for b in builds:
        sp = b["spells"]
        print(f"\n== {b['lane']} {b.get('variant', '')} splash={b.get('splash', '')} predicted {b['predicted']:.2f} penalty {b['penalty']:.2f}")
        print(" creatures", sum(composition.is_creature(c) for c in sp), "removal", rl(sp), "shape", deckbuilder.shape_vector(sp))
        print(" structure dev", deckbuilder.structure_dev(sp, targets[b["lane"]], b["lane"], SET))
        print(" ", ", ".join(sorted(c["name"] for c in sp)))
    if structure and "UB" in structure["bounds"]:
        print("\nUB bounds (min/best/max):", json.dumps(structure["bounds"]["UB"]))


if __name__ == "__main__":
    main()
