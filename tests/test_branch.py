"""Power vs Fit orders and the learnt branch rule."""

import json

import numpy as np
import pytest

from companion.advisor_v2.engine import DraftAdvisorV2, stage_of
from src.dataset import Dataset
from src.set_metrics import SetMetrics
from tests.conftest import OTJ
from tools import fit_branch


@pytest.fixture(scope="module")
def metrics():
    ds = Dataset()
    ds.open_file(str(OTJ))
    return SetMetrics(ds)


def card(name, cost, colours, wr):
    return {"name": name, "mana_cost": cost, "colors": colours, "types": ["Creature"],
            "deck_colors": {"All Decks": {"gihwr": wr, "samples": 9000}}}


def blue_pool(n):
    return [card(f"u{i}", "{1}{U}", ["U"], 56.0) for i in range(n)]


def test_power_ignores_pool_and_fit_follows_it(metrics, monkeypatch):
    monkeypatch.setattr(DraftAdvisorV2, "ORDER_WEIGHT", 0.0)
    red = card("red", "{2}{R}", ["R"], 60.0)
    blue = card("blue", "{2}{U}", ["U"], 56.0)
    empty, committed = DraftAdvisorV2(metrics, []), DraftAdvisorV2(metrics, blue_pool(8))
    assert empty.power(red, 3) == committed.power(red, 3)  # power ignores the pool
    assert committed.fit(blue, 3) > committed.fit(red, 3)  # fit follows the blue pool
    assert committed.power(red, 3) > committed.power(blue, 3)


def test_branch_only_above_the_learnt_gap(metrics, monkeypatch, tmp_path):
    monkeypatch.setattr(DraftAdvisorV2, "ORDER_WEIGHT", 0.0)
    monkeypatch.setattr(DraftAdvisorV2, "SPLIT", True)
    rule = tmp_path / "rule.json"
    # gap needed = -(a + c*strength)/b = 3.0 when a = -3, b = 1, c = 0
    rule.write_text(json.dumps({"stages": {stage_of(8): {"a": -3.0, "b": 1.0, "c": 0.0}}}))
    monkeypatch.setattr(DraftAdvisorV2, "BRANCH_FILE", str(rule))
    blue = card("blue", "{2}{U}", ["U"], 56.0)
    for red_wr, branches in ((58.0, False), (62.0, True)):
        adv = DraftAdvisorV2(metrics, blue_pool(8))
        recs = adv.evaluate_pack([blue, card("red", "{2}{R}", ["R"], red_wr)], 10, 1)
        b = adv.split["branch"]
        assert b["take"] is branches and recs[0].card_name == ("red" if branches else "blue")
        assert recs[0].contextual_score > recs[1].contextual_score  # the score agrees with the order
        assert b["threshold"] == 3.0


def test_prior_fades_through_the_pack(metrics, monkeypatch):
    monkeypatch.setattr(DraftAdvisorV2, "ORDER_FADE_BY", 10)
    adv = DraftAdvisorV2(metrics, [])
    adv.order = {"x": 1.0}
    assert adv.order_prior("x", 1) == DraftAdvisorV2.ORDER_WEIGHT
    assert adv.order_prior("x", 6) == pytest.approx(DraftAdvisorV2.ORDER_WEIGHT * 0.5)
    assert adv.order_prior("x", 11) == 0.0


def test_logistic_recovers_a_known_crossing():
    rng = np.random.default_rng(1)
    gap = rng.uniform(-5, 10, 4000)
    strength = rng.uniform(-3, 3, 4000)
    p = 1 / (1 + np.exp(-(-2.0 + 0.5 * gap - 0.3 * strength)))
    y = (rng.uniform(size=4000) < p).astype(float)
    w, se = fit_branch.logistic(np.column_stack([np.ones(4000), gap, strength]), y)
    assert abs(w[0] + 2.0) < 3 * se[0] and abs(w[1] - 0.5) < 3 * se[1] and abs(w[2] + 0.3) < 3 * se[2]
    assert fit_branch.gap_needed({"a": -2.0, "b": 0.5, "c": -0.3}, 0.0) == pytest.approx(4.0)
