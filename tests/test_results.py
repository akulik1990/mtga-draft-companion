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


def test_same_second_reports_do_not_collide(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(results, "RESULTS", tmp_path)
    t = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = results.record("mock", "FRA", {}, {}, "a", stamp=t, config={})
    b = results.record("mock", "FRA", {}, {}, "b", stamp=t, config={})
    assert a["report"] != b["report"] and (tmp_path / b["report"]).read_text() == "b\n"


def test_card_kinds():
    assert results.card_kind({"types": ["Land"], "colors": ["W", "U"]}) == "dual land"
    assert results.card_kind({"types": ["Land"], "colors": []}) == "other land"
    assert results.card_kind({"types": ["Creature"], "colors": ["W", "B"], "mana_cost": "{1}{W/B}"}) == "hybrid"
    assert results.card_kind({"types": ["Creature"], "colors": ["W", "B", "R"], "mana_cost": "{W}{B}{R}"}) == "gold3"
    assert results.card_kind({"types": ["Instant"], "colors": ["U", "R"], "mana_cost": "{U}{R}"}) == "gold2"
    assert results.card_kind({"types": ["Artifact"], "colors": [], "mana_cost": "{3}"}) == "colourless"


def test_harness_by_kind_counts_strong_drafters_only(tmp_path):
    import gzip

    cards = {"1": {"name": "A", "types": ["Creature"], "colors": ["W"], "mana_cost": "{W}"},
             "2": {"name": "L", "types": ["Land"], "colors": ["W", "U"]}}
    (tmp_path / "set.json").write_text(json.dumps({"card_ratings": cards}))
    drafts = [{"id": "s", "skill": 0.62, "games": 80, "picks": [[1, 1, "A", ["A", "L"], []], [1, 2, "L", ["L"], ["A"]]]},
              {"id": "w", "skill": 0.50, "games": 80, "picks": [[1, 1, "A", ["A", "L"], []]]}]
    with gzip.open(tmp_path / "d.json.gz", "wt") as f:
        json.dump(drafts, f)
    base = {"drafts": [{"id": "s", "picks": [[1, 1, 0, 0.0, 0], [1, 2, 0, 0.0, 0]]}, {"id": "w", "picks": [[1, 1, 0, 0.0, 0]]}]}
    head = {"drafts": [{"id": "s", "picks": [[1, 1, 1, 1.5, 0], [1, 2, 0, 0.0, 0]]}, {"id": "w", "picks": [[1, 1, 1, 2.0, 0]]}]}
    (tmp_path / "b.json").write_text(json.dumps(base))
    (tmp_path / "h.json").write_text(json.dumps(head))
    out = results.harness_by_kind(tmp_path / "b.json", tmp_path / "h.json", tmp_path / "d.json.gz", tmp_path / "set.json")
    assert out["kind:mono"] == {"n": 1, "base_top1": 100.0, "head_top1": 0.0, "base_regret": 0.0, "head_regret": 1.5}
    assert out["kind:dual land"]["n"] == 1 and out["pack:1"]["n"] == 2
    assert "| mono | 1 |" in results.by_kind_markdown(out)
