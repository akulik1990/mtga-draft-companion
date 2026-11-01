"""The run ledger: records land in runs.jsonl with a saved report; metrics parse from both tools' output."""

import json

from tools import results

HARNESS_MD = """## Picks

| advisor | all drafters | strong drafters | strong, pack 1 | strong, pack 3 | strong, picks 1-4 |
|---|---|---|---|---|---|
| base | 44.8 / 79.7 | 47.2 / 81.0 | 47.6 / 81.8 | 48.7 / 82.3 | 41.7 / 72.5 |
| head | 43.4 / 78.0 | 45.7 / 79.7 | 48.7 / 83.1 | 48.5 / 81.6 | 40.3 / 74.3 |

Top-1, head minus base: **-1.5 points** (paired bootstrap SE 0.6).

| measure | base | head | head minus base (SE) | better is |
|---|---|---|---|---|
| regret: how far the drafter's picks fell below the advisor's top choice | -0.329 | -0.342 | **-0.013** (0.145) | more negative |
"""


def test_harness_metrics_parse():
    m = results.harness_metrics(HARNESS_MD)
    assert m["base"] == {"top1": 44.8, "top3": 79.7, "strong_top1": 47.2, "strong_top3": 81.0}
    assert m["head"]["top1"] == 43.4
    assert m["top1_diff"] == {"points": -1.5, "se": 0.6}
    assert m["regret"] == {"base": -0.329, "head": -0.342, "diff": -0.013, "se": 0.145}


def row(seed, up, v2):
    def eng(rating):
        return {"rating": rating, "lane_matches_deck": {"P2P1": True, "P3P1": False},
                "signals": [{"pack": p, "engine": "x", "spearman": 0.5, "top_signal_open": p == 1} for p in (1, 2, 3)]}
    return {"seed": seed, "upstream": eng(up), "v2": eng(v2), "divergences": [{}] * 3}


def test_mock_metrics_summarise_rows():
    m = results.mock_metrics([row(1, 80.0, 82.0), row(2, 81.0, 80.0)])
    assert m["tables"] == 2 and m["diverging_picks"] == 3.0
    assert m["v2_minus_upstream"]["deck_rating"] == 0.5 and m["v2_minus_upstream"]["better"] == 1
    assert m["v2"]["lane_matches_deck"] == {"P2P1": 1.0, "P3P1": 0.0}
    assert m["v2"]["open_colour"]["P1"] == {"spearman": 0.5, "top_open": 1.0}


def test_record_appends_and_saves_report(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    rec = results.record("mock", "OTJ", {"seeds": 1}, {"tables": 1}, "## report", "trial", config={"K": 1})
    rec2 = results.record("harness", "OTJ", {"drafts": 5}, {}, "## other", config={"K": 1})
    lines = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [l["tool"] for l in lines] == ["mock", "harness"]
    assert (tmp_path / rec["report"]).read_text() == "## report\n"
    assert lines[0]["note"] == "trial" and lines[0]["engine"] == {"K": 1} and lines[1]["report"] == rec2["report"]
