"""Replay a finished Player.log into another file line by line, so the app follows it live.

    uv run python tools/stream_log.py data/demo/Player.log data/demo/Live.log --delay 3
    MTGA_COMPANION_DATA=$PWD/data uv run python -m companion --no-sync --log data/demo/Live.log --browser

Same idea as upstream's simulator.py. Stdlib only.
"""

import argparse
import time
from pathlib import Path

EVENTS = ("Event_Join", "Draft.Notify", "Event_PlayerDraftMakePick", "CardsInPack", "BotDraft")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--delay", type=float, default=3.0, help="seconds between draft events")
    args = p.parse_args()

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("MTGA log stream started\n", encoding="utf-8")
    with open(args.src, encoding="utf-8") as src, open(dst, "a", encoding="utf-8") as out:
        for line in src:
            out.write(line)
            out.flush()
            if any(k in line for k in EVENTS):
                print(line[:90].rstrip())
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
