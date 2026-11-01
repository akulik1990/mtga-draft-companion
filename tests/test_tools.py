"""Converter and replay wrapper: a sampled 17Lands draft round-trips through the session and the harness."""

import gzip
import json

from companion import VENDOR
from tests.conftest import OTJ, append
from tools import draft_to_log
from tools.replay import run as replay_run

MKM = VENDOR / "tests" / "data" / "MKM_PremierDraft_Data_2024_5_3.json"


def sampled_draft(dataset, n_picks, pack_size=6):
    """A sampler-shaped draft that takes the first card of each pack."""
    names = sorted(c["name"] for c in json.loads(dataset.read_text())["card_ratings"].values() if "Land" not in c["types"])
    picks = []
    for i in range(n_picks):
        pack = names[i : i + pack_size]
        picks.append([1 + i // 14, 1 + i % 14, pack[0], pack, [p[2] for p in picks]])
    return {"id": "d1", "skill": 0.6, "games": 100, "wins": 7, "losses": 2, "picks": picks}


def test_draft_to_log_replays_through_session(session):
    draft = sampled_draft(OTJ, 5)
    lines = draft_to_log.to_log_lines(draft, draft_to_log.name_to_id(OTJ), "OTJ", until=(1, 5))
    for line in lines:
        append(session.log, line)
    session.poll()
    snap = session.snapshot()
    assert (snap["pack_number"], snap["pick_number"]) == (1, 5)
    assert [c["name"] for c in snap["pool"]] == [p[2] for p in draft["picks"][:4]]
    assert len(snap["pack"]) == 6
    assert set(snap["advisors"]) == {"upstream", "v2"}


def test_replay_wrapper_uses_v2(tmp_path, monkeypatch):
    seen = []

    class Spy(replay_run.advisor_class("v2")):
        def __init__(self, *a, **k):
            seen.append(type(self).__name__)
            super().__init__(*a, **k)

    monkeypatch.setattr("companion.advisor_v2.engine.DraftAdvisorV2", Spy)
    drafts = tmp_path / "d.json.gz"
    with gzip.open(drafts, "wt") as f:
        json.dump([sampled_draft(MKM, 3)], f)
    out = tmp_path / "head.json"
    replay_run.main(["replay", "--advisor", "v2", "--drafts", str(drafts), "--dataset", str(MKM),
                     "--set", "MKM", "--label", "head", "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["label"] == "head" and len(result["drafts"]) == 1
    assert len(result["drafts"][0]["picks"]) == 3
    assert seen and set(seen) == {"Spy"}
