"""Per pair, the mean count of the given removal cards in 60%+ players' decks (same filter as tools/deck_targets.py), so the
builder and picker can count removal without them against a target without them (REMOVAL_SKIP, docs/b1-within-removal-FRA.md).

    uv run python tools/removal_skip.py FRA "Card A;Card B;..."   # writes the removal_skip section of the set's fitted.json
"""

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402


def main(set_code, names, min_skill=0.6):
    skip = set(names.split(";"))
    sums, decks, seen = defaultdict(float), defaultdict(int), set()
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        cols = [i for i, h in enumerate(header) if h.startswith("deck_") and (h[5:] in skip or h[5:].split(" // ")[0] in skip)]
        for row in reader:
            skill, pair = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
            if len(pair) not in (2, 3) or skill in ("", "None") or float(skill) < min_skill:
                continue
            key = (row[col["draft_id"]], row[col["build_index"]])
            if key in seen:
                continue
            seen.add(key)
            lane = "".join(sorted(pair, key="WUBRG".index))
            decks[lane] += 1
            sums[lane] += sum(int(float(row[i])) for i in cols if row[i] and row[i] != "0")
    out = {lane: round(sums[lane] / n, 2) for lane, n in sorted(decks.items())}
    path = config.write_fitted(set_code, "removal_skip", {"cards": sorted(skip), "pairs": out},
                               f"tools/removal_skip.py: mean count of the listed removal cards in {min_skill:.0%}+ players' decks per pair")
    print(path, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
