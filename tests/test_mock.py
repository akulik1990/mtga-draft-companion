"""Mock drafts run through the app: the session's own log, both engines scoring, the API, the tool's driver."""

from fastapi.testclient import TestClient

from companion.api.app import create_app
from companion.mock import PACK_SIZE, Table
from tools import mock_draft


def test_table_is_seeded_and_bots_draft(session):
    from companion.mock import Cards
    from tests.conftest import OTJ

    cards = Cards(OTJ)
    a, b = Table(cards, 5), Table(cards, 5)
    assert a.pack() == b.pack() and len(a.pack()) == PACK_SIZE
    first = a.pack()[0]
    a.take(first)
    assert a.pick == 2 and len(a.pack()) == PACK_SIZE - 1 and all(len(p) == 1 for p in a.pools)
    assert set(a.openness(1)) == set("WUBRG")


def test_session_runs_a_mock_and_scores_it(session):
    info = session.start_mock("table", "OTJ", seed=3)
    assert info["kind"] == "table" and (info["pack_number"], info["pick_number"]) == (1, 1)
    session.poll()
    snap = session.snapshot()
    assert (snap["pack_number"], snap["pick_number"]) == (1, 1)
    assert len(snap["pack"]) == PACK_SIZE and snap["mock"]["id"] == 3
    assert set(snap["advisors"]) == {"upstream", "v2"}
    top = snap["advisors"]["v2"][0]["card_name"]
    session.mock_pick(top)
    session.poll()
    snap = session.snapshot()
    assert (snap["pack_number"], snap["pick_number"]) == (1, 2)
    assert [c["name"] for c in snap["pool"]] == [top]
    session.stop_mock()
    assert session.snapshot()["mock"] is None


def test_mock_api_roundtrip(session):
    with TestClient(create_app(session, run_scanner=False)) as client:
        info = client.post("/api/mock/start", json={"kind": "table", "set": "OTJ", "seed": 9}).json()
        assert info["pick_number"] == 1
        state = client.get("/api/state").json()
        assert len(state["pack"]) == PACK_SIZE
        assert client.post("/api/mock/pick", json={"card_name": "not in pack"}).status_code == 400
        after = client.post("/api/mock/pick", json={"card_name": state["pack"][0]["name"]}).json()
        assert after["picks"] == 1 and after["pick_number"] == 2
        assert client.post("/api/mock/stop").json() == {"ok": True}


def test_tool_drives_a_full_draft_in_process(session):
    app = mock_draft.InProcess(session)
    log, checks = mock_draft.run_draft(app, "OTJ", "upstream", "table", seed=1)
    assert len(log) == 3 * PACK_SIZE
    assert {c["pack"] for c in checks} == {1, 2, 3} and {c["engine"] for c in checks} == {"upstream", "v2"}
    assert len(app.state()["pool"]) == 3 * PACK_SIZE and app.state()["mock"]["done"]
    builds = app.build()
    assert builds and sum(c["count"] for c in builds[0]["deck"]) == 40
