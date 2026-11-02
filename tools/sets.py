"""Per-set engine settings (companion/advisor_v2/config.py): write the baseline list, show where each value comes from.

    uv run python tools/sets.py baseline          # regenerate sets/baseline.json from the code
    uv run python tools/sets.py show FRA          # every value for FRA and its layer (baseline / fitted / manual), and the fitted sections
    uv run python tools/sets.py new SET           # empty fitted.json and manual.json for a new set
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion  # noqa: E402,F401
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.engine import DraftAdvisorV2  # noqa: E402,F401  (fills config.BASELINE)

EMPTY = {"values": {}, "pairs": {}, "types": {}, "cards": {}, "combos": []}


def baseline_json():
    return json.dumps({"values": dict(sorted(config.BASELINE.items())), **{k: v for k, v in EMPTY.items() if k != "values"}}, indent=1) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline")
    sub.add_parser("show").add_argument("set")
    sub.add_parser("new").add_argument("set")
    a = p.parse_args(argv)
    if a.cmd == "baseline":
        (config.SETS / "baseline.json").write_text(baseline_json())
        print(f"wrote {len(config.BASELINE)} values to {config.SETS / 'baseline.json'}")
    elif a.cmd == "new":
        for which in ("fitted", "manual"):
            path = config.set_file(a.set, which)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(json.dumps(EMPTY, indent=1) + "\n")
                print(f"wrote {path}")
    else:
        values, tables, source = config.resolve(a.set)
        for name in sorted(values):
            print(f"{name:28} {values[name]!s:22} {source[name]}")
        for t in config.TABLES:
            if tables[t]:
                print(f"{t}: {json.dumps(tables[t])}")
        for section, src in (config.layer(a.set, "fitted").get("sources") or {}).items():
            print(f"fitted {section:14} {src.get('date', '')} {src.get('commit', '')}  {src.get('how', '')}")


if __name__ == "__main__":
    main()
