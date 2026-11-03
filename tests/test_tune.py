"""The tuner's coordinate search, with a fake evaluator instead of replays."""

from tools import results, tune


def test_search_moves_only_on_real_gains(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    monkeypatch.setattr(tune, "GRID", {"STEER": [2.0, 4.0, 6.0], "K": [150.0, 300.0, 600.0]})

    def fake(set_code, split, params):
        # STEER 6 is clearly better; K 150 is better only by noise-sized 0.1 (the set's own K is not 150)
        v = 50.0 + (2.0 if params["STEER"] == 6.0 else 0.0) + (0.1 if params["K"] == 150.0 else 0.0) + (0.5 if split == "test" else 0.0)
        return {"weighted_top1": v, "top1_se": 0.4, "top1": v, "outcome_lift": 1.0}

    monkeypatch.setattr(tune, "evaluate", fake)
    tune.main(["--sets", "FRA", "TDM", "--passes", "2", "--jobs", "2"])
    rec = [l for l in (tmp_path / "runs.jsonl").read_text().splitlines()][-1]
    import json

    m = json.loads(rec)["metrics"]
    assert m["constants"]["STEER"] == 6.0 and m["constants"]["K"] != 150.0  # K stays where it started
    assert m["train"] == [50.0, 52.0] and m["test"] == [50.5, 52.5]
