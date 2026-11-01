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
