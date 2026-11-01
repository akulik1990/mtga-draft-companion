"""Drive a DraftSession from log lines, using upstream's OTJ dataset fixture."""

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
