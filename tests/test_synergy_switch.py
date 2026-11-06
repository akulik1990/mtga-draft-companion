import json
from pathlib import Path

import pytest

from companion import DATA_DIR
from companion.advisor_v2 import config, needs

GOLDEN = Path(__file__).with_name("synergy_lists_FRA.golden.json")


def c(**k):
    return {"name": "X", "cmc": 3, "types": ["Creature"], "subtypes": ["Elf"], "colors": ["G"], "keywords": ["Flying"],
            "oracle_text": "When X enters, mill two. (reminder)", **k}


def test_match_keys():
    assert needs.matches(c(), {"text": r"\bmill", "types": ["Creature"], "cmc_min": 3, "cmc_max": 3})
    assert not needs.matches(c(), {"text": "reminder"})  # reminder text stripped
    assert needs.matches(c(), {"subtypes": ["Elf"], "colours": ["G"], "keywords": ["Flying"], "names": ["X"]})
    assert not needs.matches(c(), {"types": ["Creature"], "not": {"keywords": ["Flying"]}})
    assert needs.matches(c(), {"types": ["Land"], "keywords": ["Flying"], "mode": "any"})
    assert not needs.matches(c(), {"types": ["Land"], "keywords": ["Flying"]})
    assert not needs.matches(c(), {"power_min": 1})  # no power known


@pytest.mark.skipif(not (DATA_DIR / "Sets" / "FRA_PremierDraft_All_Data.json").exists(), reason="no FRA data")
def test_switch_defaults_off(tmp_path, monkeypatch):
    m = tmp_path / "manual.json"
    m.write_text(json.dumps({"card_wants": {"A": {"enabler": "x"}, "B": {"enabler": "x", "synergy": True}, "C": {"enabler": "x", "synergy": False}}}))
    monkeypatch.setenv("MTGA_V2_MANUAL", str(m))
    needs.synergy_lists.cache_clear()
    try:
        assert [n for n, _ in needs.synergy_lists("FRA", every=True)] == ["A", "B", "C"]
        assert [n for n, _ in needs.synergy_lists("FRA")] == ["B"]
    finally:
        needs.synergy_lists.cache_clear()
        config._tag_fixes.cache_clear()  # filled from the temp manual


@pytest.mark.skipif(not (DATA_DIR / "Sets" / "FRA_PremierDraft_All_Data.json").exists(), reason="no FRA data")
def test_legacy_rules_match_same_enablers():
    needs.synergy_lists.cache_clear()
    gold = [tuple(x) for x in json.loads(GOLDEN.read_text())]  # the 26 entries reviewed before the match format; later entries are not in it
    new = {k for k, v in config.layer("FRA", "manual")["card_wants"].items() if isinstance(v, dict) and "match" in v}  # rewritten in the match format
    got = [(n, sorted(e)) for n, e in needs.synergy_lists("FRA", every=True) if n not in new]
    assert got[:len([g for g in gold if g[0] not in new])] == [g for g in gold if g[0] not in new]


def test_deck_condition(tmp_path, monkeypatch):
    def card(name, col, types=("Creature",)):
        return {"name": name, "colors": [col], "types": list(types)}
    two = [card("a", "G"), card("b", "G"), card("c", "U"), card("d", "U"), card("l", "R", ("Land",))]
    three = two + [card("e", "R"), card("f", "R")]
    cond = {"colours_min": 3}
    assert not needs.deck_ok(cond, two) and needs.deck_ok(cond, three)  # lands do not count
    assert not needs.deck_ok(cond, two + [card("e", "R")]) and needs.deck_ok({"colours_min": 3, "colour_spells": 1}, two + [card("e", "R")])
    assert not needs.deck_ok({"basic_types_min": 3}, three, {"Forest": 8, "Island": 8, "Mountain": 0})
    assert needs.deck_ok({"basic_types_min": 3}, three, {"Forest": 8, "Island": 4, "Mountain": 2})
    m = tmp_path / "manual.json"
    m.write_text(json.dumps({"card_wants": {"P": {"enabler": "boost", "synergy": True, "deck": cond}}}))
    monkeypatch.setenv("MTGA_V2_MANUAL", str(m))
    needs.synergy_deck.cache_clear()
    monkeypatch.setattr(needs, "synergy_lists", lambda s: [("P", frozenset({"e"}))])
    try:
        assert needs.card_synergy(card("P", "G"), two, "FRA") == 0.0  # no third colour with the pool
        assert needs.card_synergy(card("P", "G"), three, "FRA") > 0.0
    finally:
        needs.synergy_deck.cache_clear()
        config._tag_fixes.cache_clear()
