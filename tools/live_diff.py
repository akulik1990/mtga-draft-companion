"""Replay a real Arena draft from Player.log through v2 under two settings and list the picks where they differ.

    uv run python tools/live_diff.py data/Temp/live/FRA-quickdraft-2026-11-02.Player.log --b DECK_WEIGHT=1
    uv run python tools/live_diff.py LOG --a DECK_WEIGHT=0 --b DECK_WEIGHT=1 [--all]

Reads Quick Draft (BotDraftDraftStatus) and Premier (Draft.Notify is not parsed yet) status lines: each pick's pack
and what had been taken before it. "A" is the engine as configured (plus any --a), "B" adds --b on top.
Shows, per pick, what you took, each side's top 3 and the reason line of B's pick when it changed.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR, LAUNCH_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.engine import DraftAdvisorV2  # noqa: E402
from tools.mock_draft import apply_overrides  # noqa: E402
from tools.replay.run import replay  # noqa: E402

STATUS = re.compile(r'"Payload":"(\{.*?\})"')


def picks(log):
    """[(pack 1-based, pick 1-based, pack grpIds, picked grpIds before this pick)] in order, from Quick Draft status lines."""
    out, seen = [], set()
    for line in Path(log).read_text(errors="replace").splitlines():
        if '\\"DraftPack\\"' not in line:
            continue
        m = STATUS.search(line)
        if not m:
            continue
        p = json.loads(m.group(1).encode().decode("unicode_escape"))
        key = (p["PackNumber"], p["PickNumber"])
        if p.get("DraftStatus") != "PickNext" or key in seen:
            continue
        seen.add(key)
        out.append((p["PackNumber"] + 1, p["PickNumber"] + 1, p["DraftPack"], p["PickedCards"]))
    return out


def run(app, ratings, steps, overrides):
    config.RUNTIME.clear()
    config.ACTIVE["set"] = None
    apply_overrides(overrides)
    by_id = lambda ids: [ratings[i] for i in ids if i in ratings]
    rows, seen = [], []
    for pack_no, pick_no, pack_ids, picked_ids in steps:
        pack = by_id(pack_ids)
        seen.append((pack_no, pick_no, pack))
        recs = DraftAdvisorV2(app.metrics, by_id(picked_ids), seen_packs=list(seen)).evaluate_pack(pack, pick_no, current_pack=pack_no)
        rows.append(recs)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("log")
    p.add_argument("--set", default="FRA")
    p.add_argument("--a", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--b", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--all", action="store_true", help="show every pick, not only the changed ones")
    a = p.parse_args(argv)

    app = replay.App(str(DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json"))
    ratings = {k: dict(v, set=a.set) for k, v in app.dataset.get_card_ratings().items()}
    steps = picks(LAUNCH_DIR / a.log)
    if not steps:
        raise SystemExit("no draft picks found in that log")
    side_a = run(app, ratings, steps, a.a)
    side_b = run(app, ratings, steps, a.a + a.b)
    took = [next((ratings[i]["name"] for i in (Counter(nxt[3]) - Counter(cur[3])) if i in ratings), "?")
            for cur, nxt in zip(steps, steps[1:])] + ["(last pick)"]
    top = lambda recs: [f"{r.card_name} {r.contextual_score:.1f}" for r in recs[:3]]
    changed = 0
    for (pack_no, pick_no, *_), ra, rb, t in zip(steps, side_a, side_b, took):
        if not ra or not rb:
            continue
        diff = ra[0].card_name != rb[0].card_name
        changed += diff
        if diff or a.all:
            mark = "*" if diff else " "
            print(f"{mark} P{pack_no}P{pick_no:<2} took {t}")
            print(f"     A: {' | '.join(top(ra))}")
            print(f"     B: {' | '.join(top(rb))}")
            if diff:
                print(f"     B's reasons: {'; '.join(rb[0].reasoning[:3])}")
    agree = lambda side: sum(bool(r) and r[0].card_name == t for r, t in zip(side, took))
    print(f"\n{len(steps)} picks; top pick changed on {changed}; matched your pick A {agree(side_a)}, B {agree(side_b)}  (B = A + {' '.join(a.b) or 'nothing'})")


if __name__ == "__main__":
    main()
