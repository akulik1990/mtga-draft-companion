"""v2 behaviours the harness cannot check cheaply: empty pool, lane pull, signals, shrinkage."""

import json

import pytest

from companion.advisor_v2.engine import DraftAdvisorV2
from companion.advisor_v2.lanes import LANES
from src.dataset import Dataset
from src.set_metrics import SetMetrics
from tests.conftest import OTJ


@pytest.fixture(scope="module")
def world():
    ds = Dataset()
    ds.open_file(str(OTJ))
    cards = list(ds.get_card_ratings().values())
    return SetMetrics(ds), cards


def by_colors(cards, colors, n, min_wr=0):
    out = [c for c in cards if "Land" not in c["types"] and set(c["colors"]) == set(colors)
           and (c["deck_colors"]["All Decks"].get("gihwr") or 0) > min_wr]
    return sorted(out, key=lambda c: -c["deck_colors"]["All Decks"]["gihwr"])[:n]


def test_empty_pool_orders_like_all_decks_gihwr(world):
    metrics, cards = world
    pack = by_colors(cards, "W", 3) + by_colors(cards, "U", 3) + by_colors(cards, "B", 3)
    recs = DraftAdvisorV2(metrics, []).evaluate_pack(pack, 1, 1)
    wrs = [r.base_win_rate for r in recs]
    # wheel tiebreak may swap near-equal neighbours, so compare loosely: top is within 0.5 of best WR
    assert max(wrs) - recs[0].base_win_rate <= 0.5
    assert all(r.cast_probability > 0.9 for r in recs), "nothing is committed yet, so every card is castable somewhere"


def test_committed_pool_pulls_off_colour_to_replacement(world):
    metrics, cards = world
    pool = by_colors(cards, "W", 10, 54) + by_colors(cards, "U", 10, 54)
    adv = DraftAdvisorV2(metrics, pool)
    assert max(adv.lanes, key=adv.lanes.get) == "WU"
    on = by_colors(cards, "U", 1)[0]
    off = by_colors(cards, "R", 1)[0]
    recs = {r.card_name: r for r in adv.evaluate_pack([on, off], 30, 3)}
    assert recs[off["name"]].cast_probability < 0.2
    assert recs[off["name"]].contextual_score < recs[on["name"]].contextual_score


def test_late_good_cards_signal_their_colour(world):
    metrics, cards = world
    good_red = by_colors(cards, "R", 6, 56)
    seen = [(1, 6, good_red), (1, 7, good_red[1:]), (1, 8, good_red[2:])]
    adv = DraftAdvisorV2(metrics, [], seen_packs=seen)
    assert adv.signals["R"] == max(adv.signals.values()) and adv.signals["R"] > 1.0
    red_lanes = sum(p for lane, p in adv.lanes.items() if "R" in lane)
    assert red_lanes > sum(p for lane, p in DraftAdvisorV2(metrics, []).lanes.items() if "R" in lane)


def test_shrinkage_moves_with_sample_size(world):
    metrics, _ = world
    card = {"name": "x", "colors": ["W"], "types": ["Creature"], "cmc": 2,
            "deck_colors": {"All Decks": {"gihwr": 55.0, "samples": 5000}, "WU": {"gihwr": 61.0, "samples": 300}}}
    adv = DraftAdvisorV2(metrics, [])
    assert adv.shrunk_wr(card, "WU") == (58.0, 300)  # w = 300/(300+300) = 0.5
    assert adv.shrunk_wr(card, "WB") == (55.0 - adv.SPARSE_SD * adv.std, 0)
    assert adv.value(card, "BR") == adv.replacement


def test_composition_fills_gaps_within_bounds(monkeypatch):
    from companion.advisor_v2 import composition
    from companion.advisor_v2.composition import BOUND, Composition

    monkeypatch.setattr(composition, "ENABLED", True)

    typical = {"WU": {"curve": {"1": 1, "2": 5, "3": 5, "4": 4, "5": 2, "6+": 1}, "interaction": 4, "creatures": 14, "spells": 23}}
    two = {"name": "a", "cmc": 2, "types": ["Creature"], "tags": []}
    five = {"name": "b", "cmc": 5, "types": ["Creature"], "tags": []}
    removal = {"name": "c", "cmc": 3, "types": ["Instant"], "tags": ["removal"]}
    pool = [dict(five, name=f"x{i}") for i in range(12)]  # twelve five-drops, nothing else
    comp = Composition(typical, pool)
    assert comp.multiplier(two, "WU") == 1.0 + BOUND
    assert comp.multiplier(five, "WU") == 1.0 - BOUND
    assert comp.multiplier(removal, "WU") > comp.multiplier(two, "WU") * 0.99  # short on interaction too
    assert comp.multiplier(two, "BR") == 1.0  # no data for that lane
    assert Composition(typical, pool[:5]).multiplier(two, "WU") == 1.0  # too early


def test_pips_hybrid_and_splash(world, monkeypatch):
    from companion.advisor_v2 import lanes
    from companion.advisor_v2.lanes import castable, off_pips, pips

    monkeypatch.setattr(lanes, "USE_PIPS", True)
    monkeypatch.setattr(DraftAdvisorV2, "SPLASH", 0.85)

    hybrid = {"name": "h", "mana_cost": "{1}{W/U}", "colors": ["W", "U"], "types": ["Creature"]}
    assert pips(hybrid) == [["W", "U"]]
    assert castable(hybrid, "WB") and castable(hybrid, "UB") and not castable(hybrid, "BR")
    assert pips({"mana_cost": "{2/W}{U/P}{X}{R}", "types": ["Sorcery"]}) == [["R"]]
    assert castable({"name": "dual", "colors": ["W", "U"], "types": ["Land"]}, "WU")
    assert not castable({"name": "dual", "colors": ["W", "U"], "types": ["Land"]}, "WB")

    metrics, cards = world
    stats = {"All Decks": {"gihwr": 60.0, "samples": 5000}}
    one_red = {"name": "one red", "mana_cost": "{3}{R}", "colors": ["R"], "types": ["Creature"], "deck_colors": stats}
    two_red = dict(one_red, name="two red", mana_cost="{2}{R}{R}")
    fixer = {"name": "fixer", "mana_cost": "", "colors": ["R", "W"], "types": ["Land"], "deck_colors": stats}
    assert len(off_pips(two_red, "WU")) == 2

    plain = DraftAdvisorV2(metrics, [])
    fixed = DraftAdvisorV2(metrics, [fixer])
    assert plain.value(one_red, "WU") == plain.replacement
    assert fixed.splash_colour(one_red, "WU") == "R"
    assert plain.replacement < fixed.value(one_red, "WU") < fixed.value(one_red, "WR")
    assert fixed.value(two_red, "WU") == fixed.replacement


def test_sparse_lane_is_valued_below_all_decks(world):
    metrics, _ = world
    card = {"name": "s", "colors": ["W"], "types": ["Creature"], "cmc": 2,
            "deck_colors": {"All Decks": {"gihwr": 56.0, "samples": 9000}, "WU": {"gihwr": 57.0, "samples": 6000},
                            "WB": {"gihwr": 58.0, "samples": 10}, "UB": {"gihwr": 0.0, "samples": 91}}}
    adv = DraftAdvisorV2(metrics, [])
    wu, _ = adv.shrunk_wr(card, "WU")
    wb, n = adv.shrunk_wr(card, "WB")
    wr, _ = adv.shrunk_wr(card, "WR")  # no row at all
    assert n == 10 and wb < 56.0 < wu
    assert wr == 56.0 - adv.SPARSE_SD * adv.std
    assert adv.shrunk_wr(card, "UB") == (wr, 0)  # samples but no published win rate


def test_overrides_reach_engine_and_switches():
    from companion.advisor_v2 import lanes
    from companion.advisor_v2.engine import DraftAdvisorV2
    from tools import mock_draft

    before = (DraftAdvisorV2.SPLASH, lanes.USE_PIPS)
    try:
        mock_draft.apply_overrides(["SPLASH=0.5", "lanes.USE_PIPS=0"])
        assert DraftAdvisorV2.SPLASH == 0.5 and lanes.USE_PIPS is False
    finally:
        DraftAdvisorV2.SPLASH, lanes.USE_PIPS = before


def test_steer_decays_with_pool(world, monkeypatch):
    metrics, cards = world
    monkeypatch.setattr(DraftAdvisorV2, "COMMIT_BY", 24)
    monkeypatch.setattr(DraftAdvisorV2, "ALPHA_LATE", 1.5)
    red = by_colors(cards, "R", 1, 57)[0]
    q_pool = lambda n: by_colors(cards, "W", n // 2, 50) + by_colors(cards, "U", n - n // 2, 50)

    def pull(pool):
        adv = DraftAdvisorV2(metrics, pool)
        before = sum(p for lane, p in adv.lanes.items() if "R" in lane)
        commit = dict(adv.commit)
        from companion.advisor_v2.lanes import LANES, castable, quality
        for lane in LANES:
            if castable(red, lane):
                commit[lane] += adv.steer * quality(red, adv.mean, adv.std)
        after = sum(p for lane, p in adv._lanes(commit).items() if "R" in lane)
        return after - before

    early, late = pull(q_pool(2)), pull(q_pool(24))
    assert early > 0.3 and late < 0.05


def test_signal_variants_keep_a_dominant_colour(world, monkeypatch):
    from companion.advisor_v2 import signals

    metrics, cards = world
    good_red = by_colors(cards, "R", 6, 56)
    seen = [(1, 6, good_red), (1, 7, good_red[1:]), (1, 8, good_red[2:])]
    z = DraftAdvisorV2(metrics, [], seen_packs=seen).signals
    monkeypatch.setattr(signals, "SCALE", "centred")
    monkeypatch.setattr(signals, "QUALITY_POWER", 2)
    centred = DraftAdvisorV2(metrics, [], seen_packs=seen).signals
    assert max(z, key=z.get) == max(centred, key=centred.get) == "R"
    assert abs(sum(centred.values())) < 1e-9 and centred["R"] > z["R"]
