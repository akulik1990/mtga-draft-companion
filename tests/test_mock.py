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


def test_lane_score_uses_pair_rates_and_fills():
    def card(name, colors, all_wr, pair=None, pair_wr=0.0):
        dc = {"All Decks": {"gihwr": all_wr}}
        if pair:
            dc[pair] = {"gihwr": pair_wr}
        return {"name": name, "colors": colors, "types": ["Creature"], "deck_colors": dc}

    pool = [card(f"w{i}", ["W"], 55.0, "WU", 60.0) for i in range(23)] + [card("r", ["R"], 70.0)]
    score, pair = mock_draft.lane_score(pool)
    assert pair == "WU" and score == 60.0  # pair rates used; red bomb is off-pair
    short = [card("u", ["U"], 50.0)]
    assert mock_draft.lane_score(short)[0] == round((50.0 + 45.0 * 22) / 23, 3)


def test_lane_score_counts_hybrid_in_either_colour():
    hybrid = {"name": "h", "colors": ["W", "B"], "mana_cost": "{1}{W/B}", "types": ["Creature"],
              "deck_colors": {"All Decks": {"gihwr": 60.0}, "WU": {"gihwr": 61.0}}}
    score, pair = mock_draft.lane_score([hybrid])
    assert pair == "WU" and score == round((61.0 + 45.0 * 22) / 23, 3)


def test_upstream_side_is_played_once_per_seed(session, tmp_path, monkeypatch):
    monkeypatch.setattr(mock_draft, "upstream_cache_dir", lambda set_code: tmp_path / set_code)
    made = []

    def make_app(engine):
        made.append(engine)
        return mock_draft.InProcess(session)

    first = mock_draft.compare_table("OTJ", 4, make_app)
    second = mock_draft.compare_table("OTJ", 4, make_app)
    assert made == ["upstream", "v2", "v2"]  # upstream replayed only the first time
    assert second["upstream"] == first["upstream"] and (tmp_path / "OTJ" / "4.json").exists()
    assert [d["pack"] for d in second["divergences"]] == [d["pack"] for d in first["divergences"]]
