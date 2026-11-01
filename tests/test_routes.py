"""Datasets, settings and deck routes over the OTJ fixture."""

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
