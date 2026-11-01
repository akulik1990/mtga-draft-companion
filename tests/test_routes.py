"""Datasets, settings and deck routes over the OTJ fixture."""

import pytest
import json

from fastapi.testclient import TestClient

from companion.api.app import create_app
from tests.conftest import JOIN, OTJ, append

PICK = (
    '[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick {"id":"%d","request":"{\\"DraftId\\":\\"1\\",'
    '\\"GrpIds\\":[%s],\\"Pack\\":%d,\\"Pick\\":%d}"}\n'
)
NOTIFY = '[UnityCrossThreadLogger]Draft.Notify {"draftId":"1","SelfPick":%d,"SelfPack":%d,"PackCards":"%s"}\n'


def wu_spells(n):
    ratings = json.loads(OTJ.read_text())["card_ratings"]
    ok = [
        (v["deck_colors"]["All Decks"]["gihwr"], k)
        for k, v in ratings.items()
        if "Land" not in v["types"] and v["colors"] and set(v["colors"]) <= {"W", "U"}
    ]
    return [k for _, k in sorted(ok, reverse=True)[:n]]


def draft_full_pool(session, ids):
    append(session.log, JOIN)
    for i, grp in enumerate(ids):
        pack_no, pick = divmod(i, 15)
        pack_no, pick = pack_no + 1, pick + 1
        remaining = ids[i : i + 16 - pick]
        append(session.log, NOTIFY % (pick, pack_no, ",".join(remaining)))
        append(session.log, PICK % (i, grp, pack_no, pick))
    session.poll()


def test_datasets_list_and_select(session):
    with TestClient(create_app(session, run_scanner=False)) as client:
        listing = client.get("/api/datasets").json()
        assert listing["active"] == "OTJ_PremierDraft_All_Data.json"
        assert [(d["set"], d["format"], d["user_group"]) for d in listing["datasets"]] == [("OTJ", "PremierDraft", "All")]

        r = client.post("/api/datasets/select", json={"set": "OTJ", "format": "PremierDraft"})
        assert r.status_code == 200 and r.json()["active"] == "OTJ_PremierDraft_All_Data.json"
        assert client.post("/api/datasets/select", json={"set": "XXX", "format": "PremierDraft"}).status_code == 404


def test_settings_roundtrip(session):
    with TestClient(create_app(session, run_scanner=False)) as client:
        st = client.get("/api/settings").json()
        assert st["active_log"] == str(session.log)
        assert st["arena_log_pinned"] is False

        st = client.put("/api/settings", json={"arena_log_location": str(session.log), "auto_sync_datasets": False}).json()
        assert st["arena_log_pinned"] is True and st["auto_sync_datasets"] is False
        st = client.put("/api/settings", json={"arena_log_location": ""}).json()
        assert st["arena_log_pinned"] is False and st["arena_log_location"] == ""


def test_deck_build_over_full_pool(session):
    ids = wu_spells(45)
    assert len(ids) == 45
    draft_full_pool(session, ids)
    assert len(session.snapshot()["pool"]) == 45

    with TestClient(create_app(session, run_scanner=False)) as client:
        builds = client.post("/api/deck/build").json()
    assert builds, "builder returned nothing"
    best = builds[0]
    assert set(best) >= {"label", "deck", "sideboard", "arena_export", "stats", "colors"}
    deck_size = sum(c["count"] for c in best["deck"])
    assert deck_size == 40, deck_size
    assert "Deck" in best["arena_export"]
    assert all({"name", "count", "gihwr"} <= set(c) for c in best["deck"])


def test_snapshot_carries_table_controls(session):
    from tests.conftest import P1P1

    append(session.log, JOIN)
    append(session.log, P1P1)
    session.poll()
    snap = session.snapshot()
    assert "Auto" in snap["deck_filter_options"].values() and snap["deck_filter"] in snap["deck_filter_options"].values()
    assert "UB" in snap["metrics"]
    assert set(snap["metrics"]["All Decks"]) == {"gihwr", "ohwr", "gpwr", "gnswr", "gdwr"}
    assert snap["grade_thresholds"][0][0] == "A+"
    card = snap["pack"][0]
    assert set(card["grades"]["All Decks"]) == {"gihwr", "ohwr", "gpwr", "gnswr", "gdwr"} and card["tiers"] == {}


def test_deck_filter_and_result_format_settings(session):
    from tests.conftest import P1P1

    append(session.log, JOIN)
    append(session.log, P1P1)
    session.poll()
    with TestClient(create_app(session, run_scanner=False)) as client:
        st = client.put("/api/settings", json={"deck_filter": "UB", "result_format": "Grade"}).json()
        assert st["deck_filter"] == "UB" and st["result_format"] == "Grade"
        assert client.put("/api/settings", json={"deck_filter": "nonsense"}).status_code == 400
        assert client.put("/api/settings", json={"result_format": "nonsense"}).status_code == 400


def test_card_search_and_tiers(session):
    append(session.log, JOIN)
    session.poll()
    with TestClient(create_app(session, run_scanner=False)) as client:
        found = client.get("/api/cards", params={"q": "thought"}).json()
        assert found and all("thought" in c["name"].lower() for c in found) and "grades" in found[0]
        assert client.get("/api/tiers").json() == []
        assert client.post("/api/tiers", json={"url": "https://example.com/x"}).status_code == 400
        assert client.delete("/api/tiers/..%2Fconfig.json").status_code != 200
    with pytest.raises(ValueError):
        session.delete_tier_list("../config.json")


def test_engine_switches_toggle_and_persist(session, monkeypatch, tmp_path):
    from companion.advisor_v2.engine import DraftAdvisorV2 as V

    monkeypatch.setattr(type(session), "_engine_file", lambda self: tmp_path / "engine.json")
    for attr in ("PREF_WEIGHT", "ORDER_WEIGHT", "SPLIT"):
        monkeypatch.setattr(V, attr, getattr(V, attr))
    with TestClient(create_app(session, run_scanner=False)) as client:
        sw = client.get("/api/engine").json()
        assert set(sw) == {"learnt_prefs", "split"} and sw["learnt_prefs"]["on"] is True
        sw = client.put("/api/engine", json={"learnt_prefs": False}).json()
        assert sw["learnt_prefs"]["on"] is False and V.PREF_WEIGHT == 0.0 and V.ORDER_WEIGHT == V.ORDER_OFF_WEIGHT
        assert json.loads((tmp_path / "engine.json").read_text())["learnt_prefs"] is False
        client.put("/api/engine", json={"learnt_prefs": True})
        assert V.PREF_WEIGHT == V.PREF_ON_WEIGHT and V.ORDER_WEIGHT == 0.0
        assert client.put("/api/engine", json={"nonsense": True}).status_code == 400
    session.update_engine_settings({"split": False}, save=False)
    assert V.SPLIT is False
    session.load_engine_settings()  # back to the saved file: split on, prefs on
    assert V.SPLIT is True and V.PREF_WEIGHT == V.PREF_ON_WEIGHT


def test_pages_are_revalidated_but_bundles_cached(session, tmp_path):
    (tmp_path / "_app" / "immutable").mkdir(parents=True)
    (tmp_path / "index.html").write_text("<p>app</p>")
    (tmp_path / "_app" / "immutable" / "x.js").write_text("1")
    with TestClient(create_app(session, run_scanner=False, frontend=tmp_path)) as client:
        assert client.get("/").headers.get("cache-control") == "no-cache"
        assert "cache-control" not in client.get("/_app/immutable/x.js").headers
