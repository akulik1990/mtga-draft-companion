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
    for attr in ("PREF_WEIGHT", "ORDER_WEIGHT", "SPLIT", "BRANCH_COLOUR"):
        monkeypatch.setattr(V, attr, getattr(V, attr))
    with TestClient(create_app(session, run_scanner=False)) as client:
        sw = client.get("/api/engine").json()
        assert set(sw) == {"learnt_prefs", "split", "branch_colour"} and sw["learnt_prefs"]["on"] is True
        assert sw["branch_colour"]["on"] is True
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


def test_overlay_uses_arena_order_and_ranks(session):
    from companion.api.app import arena_order
    from tests.conftest import P1P1

    append(session.log, JOIN)
    append(session.log, P1P1)
    session.poll()
    with TestClient(create_app(session, run_scanner=False)) as client:
        o = client.get("/api/overlay").json()
    assert [c["name"] for c in o["cards"]] == [c["name"] for c in sorted(session.snapshot()["pack"], key=arena_order)]
    assert sorted(c["rank"] for c in o["cards"] if c["rank"] is not None)[0] == 1


def test_arena_order_matches_a_live_fra_pack():
    from companion.api.app import arena_order

    def card(name, rarity, cost, types=("Creature",), cmc=3):
        return {"name": name, "rarity": rarity, "mana_cost": cost, "types": list(types), "cmc": cmc}

    # Pack 1 pick 1 of a live Reality Fracture draft, 2026-11-02, in the order Arena showed it.
    screen = [card("Jace", "rare", "{3}{U}{U}", cmc=5), card("Master of Barbs", "rare", "{1}{R}", cmc=2),
              card("Precise Redaction", "uncommon", "{1}{U}", cmc=2), card("Mabel, Bitter Recluse", "uncommon", "{B}", cmc=1),
              card("Twisted Fates", "uncommon", "{2}{W}{W}{B}", cmc=5), card("Mabel, Valley Hero", "uncommon", "{1}{R}{W}"),
              card("Graft Surgeon", "common", "{2}{W}"), card("Silence the Echo", "common", "{1}{B}", cmc=2),
              card("Hallway Heckler", "common", "{2}{R} // {B/R}"), card("Greenhouse Propagator", "common", "{2}{G}"),
              card("Theorix Metamage", "common", "{2}{U/B} // {U/B}"), card("Charge the Sanctum", "common", "{2}{R/W}"),
              card("Afterthought Sentry", "common", "{2}", cmc=2)]
    assert sorted(reversed(screen), key=arena_order) == screen


def test_heat_marks_top_twenty_and_early_picks():
    from companion.api.app import heat

    def card(name, wr, ata, n=1000, types=("Creature",)):
        return {"name": name, "types": list(types), "deck_colors": {"All Decks": {"gihwr": wr, "ata": ata, "samples": n}}}

    ratings = {str(i): card(f"c{i}", 70 - i * 0.1, 5.0) for i in range(30)}  # c0..c19 are the top 20
    ratings["x"] = card("early", 50.0, 1.8)
    ratings["y"] = card("strong", 61.0, 6.0)
    ratings["z"] = card("rare sample", 99.0, 1.0, n=100)
    h = heat(ratings, mean=56.0, std=4.0)
    assert h["c0"] == 2 and h["c19"] == 2 and h["early"] == 1 and h["strong"] == 1
    assert "c25" in h and h["c25"] == 1  # 67.5 >= 60: fire by win rate
    assert "rare sample" not in h


def test_plain_reasoning():
    from companion.api.app import plain

    assert plain("UB 28% → 61.9 (n=937)") == "If taken, UB 28% likely; wins 61.9% there, 937 games"
    assert plain("UB 100% → 61.3 (n=7419)") == "Plays in UB: wins 61.3%, 7,419 games"
    assert plain("UG 34% → 56.6 (rarely played there)") == "If taken, UG 34% likely; wins 56.6% there, rarely played there"
    assert plain("Branch: +9.4 over Hapatra, the Desert Frost, above the -1.6 top drafters switch at") == \
        "9.4 points stronger than Hapatra, the Desert Frost; top drafters switch at -1.6 or more this early"
