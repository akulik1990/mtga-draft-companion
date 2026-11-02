"""How often top drafters take v2's #1, #2 or #3, by how far #1 is ahead of #2 (the overlay's "close call" line).

    uv run python tools/close_calls.py FRA        # train sample; writes companion/advisor_v2/sets/FRA/closecalls.json

The gap is between the numbers the overlay shows: a switch pick's Power score, otherwise the ranking score.
"""

import argparse
import json
from pathlib import Path

import companion  # noqa: F401
from companion.advisor_v2 import config
from companion.advisor_v2.engine import DraftAdvisorV2
from tools.replay import run

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [0.5, 1.0, 2.0, 4.0, 8.0]  # upper edges, in points; the last bucket is everything above


def shown_scores(recs, split):
    """The top three as the overlay shows them."""
    scores = [r.contextual_score for r in recs[:3]]
    b = (split or {}).get("branch")
    if b and b["take"] and recs and recs[0].card_name == split["power"][0]["card"]:
        scores[0] = split["power"][0]["score"]
    return scores


def bucket(gap):
    return next((i for i, edge in enumerate(BUCKETS) if gap < edge), len(BUCKETS))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--split", default="train")
    a = p.parse_args(argv)

    last = {}
    evaluate = DraftAdvisorV2.evaluate_pack

    def capture(self, pack, pick, current_pack=1):
        recs = evaluate(self, pack, pick, current_pack)
        last.update(recs=recs, split=self.split)
        return recs

    DraftAdvisorV2.evaluate_pack = capture
    counts = [[0, 0, 0, 0] for _ in range(len(BUCKETS) + 1)]  # took #1, #2, #3, other
    install = run.use_advisor

    def use_advisor(name):
        install(name)
        advise = run.replay.App.advise

        def wrapped(self, pack, pool, pack_no, pick_no):
            out = advise(self, pack, pool, pack_no, pick_no)
            taken = next((t for pk, k, t, *_ in self.current_draft["picks"] if (pk, k) == (pack_no, pick_no)), None)
            recs = last.get("recs") or []
            if len(recs) >= 3 and taken:
                s = shown_scores(recs, last.get("split"))
                names = [r.card_name for r in recs[:3]]
                counts[bucket(s[0] - s[1])][names.index(taken) if taken in names else 3] += 1
            return out

        run.replay.App.advise = wrapped

    run.use_advisor = use_advisor
    drafts = run.top_sample_paths(a.set, 600)[a.split]
    out = run.CACHE / f"closecalls.{a.set}.head.json"
    run.replay_one(a.set, drafts, "v2", out)
    out.unlink(missing_ok=True)

    edges = [0.0] + BUCKETS
    table = []
    for i, c in enumerate(counts):
        n = sum(c)
        if n:
            table.append({"from": edges[i], "to": BUCKETS[i] if i < len(BUCKETS) else None, "n": n,
                          "took": [round(100 * x / n, 1) for x in c]})
    for row in table:
        print(f"gap {row['from']:>4}-{row['to'] or '':<4} n={row['n']:5}  #1 {row['took'][0]:5.1f}%  #2 {row['took'][1]:5.1f}%  #3 {row['took'][2]:5.1f}%  other {row['took'][3]:5.1f}%")
    path = config.set_file(a.set, "closecalls")
    path.write_text(json.dumps({"sample": a.split, "buckets": table}, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
