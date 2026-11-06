"""Video routes over a temporary analysis/videos tree."""

import json

import pytest
from fastapi.testclient import TestClient

from companion.api.app import create_app


@pytest.fixture
def client(session, tmp_path):
    d = tmp_path / "abc"
    (d / "engine").mkdir(parents=True)
    (tmp_path / "index.json").write_text(json.dumps([{"id": "abc", "title": "T", "player": "P", "set": "FRA", "url": "u", "added": "x"}]))
    (d / "picks.json").write_text(json.dumps({"picks": []}))
    (d / "ideas.json").write_text(json.dumps([{"id": "i1", "claim": "c"}]))
    (d / "engine" / "latest.json").write_text(json.dumps({"commit": "c0ffee", "top1": 3, "top3": 5, "n": 6}))
    (d / "engine" / "c0ffee.json").write_text(json.dumps({"commit": "c0ffee"}))
    with TestClient(create_app(session, run_scanner=False, videos=tmp_path)) as c:
        c.dir = d
        yield c


def test_list_and_get(client):
    (row,) = client.get("/api/videos").json()
    assert (row["id"], row["top1"], row["top3"]) == ("abc", 3, 5)
    v = client.get("/api/videos/abc").json()
    assert v["commits"] == ["c0ffee"] and v["engine"]["commit"] == "c0ffee" and v["deck"] is None and v["status"] == {}
    assert client.get("/api/videos/abc/engine/c0ffee").json() == {"commit": "c0ffee"}
    assert client.get("/api/videos/abc/engine/nope").status_code == 404
    assert client.get("/api/videos/zzz").status_code == 404


def test_put_status(client):
    r = client.put("/api/videos/abc/ideas/i1", json={"status": "testing", "result": "r", "doc": "d"})
    assert r.json() == {"status": "testing", "result": "r", "doc": "d"}
    assert client.get("/api/videos/abc").json()["status"]["i1"]["status"] == "testing"
    assert client.put("/api/videos/abc/ideas/i1", json={"status": "bogus"}).status_code == 400
    assert client.put("/api/videos/abc/ideas/nope", json={"status": "shipped"}).status_code == 404
    assert client.put("/api/videos/zzz/ideas/i1", json={"status": "shipped"}).status_code == 404
    # static files are never written
    assert json.loads((client.dir / "ideas.json").read_text()) == [{"id": "i1", "claim": "c"}]
    assert json.loads((client.dir / "picks.json").read_text()) == {"picks": []}
    assert client.put("/api/videos/abc/ideas/picks", json={"status": "shipped"}).status_code == 404
