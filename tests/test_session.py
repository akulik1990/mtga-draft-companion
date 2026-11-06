"""Drive a DraftSession from log lines, using upstream's OTJ dataset fixture."""

import json
from fastapi.testclient import TestClient

from companion.api.app import create_app
from tests.conftest import JOIN, KNOWN_IN_PACK, P1P1, append


def test_idle_snapshot(session):
    snap = session.snapshot()
    assert snap["pack"] == []
    assert snap["advisors"] == {}


def test_pack_flows_to_both_advisors(session):
    append(session.log, JOIN)
    assert session.poll()
    append(session.log, P1P1)
    session.poll()

    snap = session.snapshot()
    assert snap["event"]["set"] == "OTJ"
    assert (snap["pack_number"], snap["pick_number"]) == (1, 1)
    assert len(snap["pack"]) == KNOWN_IN_PACK
    assert set(snap["advisors"]) == {"upstream", "v2"}
    for key in ("upstream", "v2"):
        recs = snap["advisors"][key]
        assert len(recs) == KNOWN_IN_PACK
        assert [r["rank"] for r in recs] == list(range(1, KNOWN_IN_PACK + 1))
    # Both rank the same cards; v2 scores in win-rate units and carries its lane distribution.
    assert {r["card_name"] for r in snap["advisors"]["v2"]} == {r["card_name"] for r in snap["advisors"]["upstream"]}
    assert abs(sum(snap["lanes"].values()) - 1.0) < 1e-6


def test_api_state_and_ws(session):
    append(session.log, JOIN)
    session.poll()
    append(session.log, P1P1)
    session.poll()

    with TestClient(create_app(session, run_scanner=False)) as client:
        state = client.get("/api/state").json()
        assert len(state["pack"]) == KNOWN_IN_PACK
        with client.websocket_connect("/api/ws") as ws:
            pushed = ws.receive_json()
            assert pushed["pick_number"] == 1
            assert len(pushed["advisors"]["upstream"]) == KNOWN_IN_PACK


def test_mana_identity_ignores_hybrid_halves():
    from companion.session import mana_identity

    def c(cost, n=1):
        return {"name": cost, "mana_cost": cost, "types": ["Creature"], "count": n, "colors": []}

    deck = [c("{1}{U}", 8), c("{2}{G}", 8), c("{1}{W/U}", 3), c("{R/G}", 2), c("{3}{B}")]
    out = mana_identity(deck + [{"name": "Island", "types": ["Basic", "Land"], "count": 8, "colors": ["U"]}])
    assert out == {"main": "UG", "splash": "B", "splash_cards": 1}


def test_deck_win_rate_needs_a_model():
    from companion.session import deck_win_rate

    assert deck_win_rate("XYZ", {"deck": [{"name": "Island", "count": 17}]}) is None
    assert deck_win_rate("", {"deck": []}) is None


def test_completed_picks_and_overlay_builds():
    import json

    from companion.session import completed_picks, overlay_builds

    compact = {"separators": (",", ":")}  # Arena writes compact JSON
    payload = json.dumps({"DraftStatus": "Completed", "PickedCards": ["1", "2", "3"]}, **compact)
    tail = "noise\n" + json.dumps({"CurrentModule": "DeckSelect", "Payload": payload}, **compact) + "\nmore\n"
    assert completed_picks(tail) == ["1", "2", "3"]
    assert completed_picks("nothing here") == []

    pool = [{"name": "A", "types": ["Creature"]}, {"name": "A", "types": ["Creature"]}, {"name": "B", "types": ["Instant"]}]
    builds = [{"deck_wr": 55.0, "main": "UB", "deck": [{"name": "A", "count": 1, "types": ["Creature"], "cmc": 2},
                                                       {"name": "Island", "count": 17, "types": ["Basic", "Land"]}]},
              {"deck_wr": 60.0, "main": "UR", "deck": [{"name": "B", "count": 1, "types": ["Instant"], "cmc": 1}]}]
    out = overlay_builds(builds, pool)
    assert [b["main"] for b in out] == ["UR", "UB"]
    builds[1]["realistic"] = False  # far too few creatures: ranked after realistic builds whatever its predicted win rate
    assert [b["main"] for b in overlay_builds(builds, pool)] == ["UB", "UR"]
    assert out[1]["remove"] == [{"name": "A", "count": 1}, {"name": "B", "count": 1}]
    assert out[1]["lands"] == [{"name": "Island", "count": 17}]


def test_course_pool_takes_latest_draft():
    from companion.session import course_pool

    def course(event, pool, updated):
        return {"InternalEventName": event, "CardPool": pool,
                "CourseDeckSummary": {"Attributes": [{"name": "LastUpdated", "value": updated}]}}

    line = json.dumps({"Courses": [course("QuickDraft_A", [1], "2026-01-01"), course("Play", [9], "2027-01-01"),
                                   course("PremierDraft_B", [2, 3], "2026-02-01")]})
    assert course_pool(f"junk\n{line}\nmore") == ([2, 3], "PremierDraft_B")
    assert course_pool("no courses") == ([], None)


def test_yardstick_is_deckmodel_v2_for_fra(monkeypatch):
    from companion.advisor_v2 import config
    from tools import deck_model
    from tools.deck_model import DeckModel

    monkeypatch.setattr(deck_model, "card_facts", lambda s: {})
    assert DeckModel("FRA").m == config.fitted("FRA", "deckmodel_v2")
    assert DeckModel("FRA").m.get("version") == 2
