"""Replay a top-drafter sample through v2 and log every pick: [draft id, pack, pick, candidates' scores high to low, pool names].
The input of analysis/big_misses.py and synergy_flips.py. KEY=VALUE engine overrides as in tools/mock_draft.py.

    env -C <repo> uv run python analysis/replay_log.py <drafts.json.gz> <out.json> [KEY=VALUE ...]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
drafts, out = (Path(p).resolve() for p in sys.argv[1:3])  # absolute before the import moves the working directory
from tools.replay import run  # noqa: E402
from tools.mock_draft import apply_overrides  # noqa: E402

apply_overrides(sys.argv[3:])
run.use_advisor("v2")
log, advise = [], run.replay.App.advise


def logged(self, pack, pool, pack_no, pick_no):
    scores = advise(self, pack, pool, pack_no, pick_no)
    names = [c if isinstance(c, str) else c.get("name") for c in pool or []]
    log.append([(self.current_draft or {}).get("id"), pack_no, pick_no, sorted(scores.items(), key=lambda kv: -kv[1]), names])
    return scores


run.replay.App.advise = logged  # after use_advisor, which replaces advise; so not run.replay_one, which calls it again
run.replay.main(["--src", str(run.VENDOR), "--drafts", str(drafts), "--dataset", str(run.dataset_path("FRA")),
                 "--set", "FRA", "--label", "head", "--out", str(out.with_suffix(".replay.json"))])
out.write_text(json.dumps(log))
print(f"{len(log)} picks -> {out}")
