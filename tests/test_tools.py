"""Replay wrapper: the harness constructs our advisor when asked for v2."""

import gzip
import json

from companion import VENDOR
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
