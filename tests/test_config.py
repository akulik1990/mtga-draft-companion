import json

from companion.advisor_v2 import config
from companion.advisor_v2.engine import DraftAdvisorV2
from tests.test_advisor_v2 import world  # noqa: F401  (fixture)
from tools.sets import baseline_json


def test_baseline_file_matches_the_code():
    # Regenerate with: uv run python tools/sets.py baseline
    assert (config.SETS / "baseline.json").read_text() == baseline_json()


def test_layers_resolve_runtime_manual_stats_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETS", tmp_path)
    (tmp_path / "XYZ").mkdir()
    (tmp_path / "XYZ" / "stats.json").write_text(json.dumps({"values": {"STEER": 3.0, "K": 200.0}, "pairs": {"UB": 0.5}}))
    (tmp_path / "XYZ" / "manual.json").write_text(json.dumps({"values": {"STEER": 2.0}, "pairs": {"UB": 1.0}, "cards": {"Foo": 2.0}}))
    monkeypatch.setattr(config, "RUNTIME", {"SPLASH": 0.5})
    values, tables, source = config.resolve("XYZ")
    assert (values["STEER"], source["STEER"]) == (2.0, "manual")
    assert (values["K"], source["K"]) == (200.0, "stats")
    assert (values["SPLASH"], source["SPLASH"]) == (0.5, "runtime")
    assert source["ALPHA"] == "baseline" and values["ALPHA"] == config.BASELINE["ALPHA"]
    assert tables["pairs"] == {"UB": 1.0} and tables["cards"] == {"Foo": 2.0}


def test_activate_sets_and_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETS", tmp_path)
    monkeypatch.setattr(config, "ACTIVE", {"set": None, "tables": {}, "source": {}, "applied": set()})
    (tmp_path / "AAA").mkdir()
    (tmp_path / "AAA" / "manual.json").write_text(json.dumps({"values": {"STEER": 1.5}}))
    base = DraftAdvisorV2.STEER
    try:
        config.activate("AAA")
        assert DraftAdvisorV2.STEER == 1.5
        config.activate("BBB")  # no layers: back to the code's value
        assert DraftAdvisorV2.STEER == base
    finally:
        DraftAdvisorV2.STEER = base


def test_manual_card_and_pair_points_move_values(world, monkeypatch):
    metrics, cards = world
    card = next(c for c in cards if "Land" not in (c.get("types") or []) and len(c.get("colors") or []) == 1)
    lane = next(l for l in ("WU", "UB", "BR", "RG", "WG") if card["colors"][0] in l)
    plain = DraftAdvisorV2(metrics, []).value(card, lane)
    monkeypatch.setitem(config.ACTIVE, "tables", {"pairs": {lane: 1.0}, "types": {}, "cards": {card["name"]: 2.0}, "combos": []})
    assert abs(DraftAdvisorV2(metrics, []).value(card, lane) - plain - 3.0) < 1e-9
