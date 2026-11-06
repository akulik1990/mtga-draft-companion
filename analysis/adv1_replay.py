"""replay_log.py plus v2's lane distribution (advisor.lanes, incl. three-colour keys) at each pick.
    env -C <repo> uv run python analysis/adv1_replay.py <drafts.json.gz> <out.json>"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
drafts, out = (Path(p).resolve() for p in sys.argv[1:3])
from tools.replay import run  # noqa: E402
run.use_advisor("v2")
cls = run.advisor_class("v2")
last = {}
ev = cls.evaluate_pack
def evaluate_pack(self, *a, **k):
    r = ev(self, *a, **k)
    last["lanes"] = {l: round(p, 4) for l, p in self.lanes.items() if p >= 0.005}
    return r
cls.evaluate_pack = evaluate_pack
log, advise = [], run.replay.App.advise
def logged(self, pack, pool, pack_no, pick_no):
    scores = advise(self, pack, pool, pack_no, pick_no)
    names = [c if isinstance(c, str) else c.get("name") for c in pool or []]
    log.append([(self.current_draft or {}).get("id"), pack_no, pick_no, sorted(scores.items(), key=lambda kv: -kv[1]), names, last.get("lanes")])
    return scores
run.replay.App.advise = logged
run.replay.main(["--src", str(run.VENDOR), "--drafts", str(drafts), "--dataset", str(run.dataset_path("FRA")),
                 "--set", "FRA", "--label", "head", "--out", str(out.with_suffix(".replay.json"))])
out.write_text(json.dumps(log))
print(f"{len(log)} picks -> {out}")
