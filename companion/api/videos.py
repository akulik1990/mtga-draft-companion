"""Video drafts: analysis/videos records plus v2's replay of each (tools/video_draft.py). Only ideas-status.json is written."""

import json
import re
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from companion import ROOT

VIDEOS = ROOT / "analysis" / "videos"
STATUSES = ("untested", "testing", "confirmed", "rejected", "shipped")


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def ideas(path: Path):
    """ideas.json is a list, or {"video": ..., "ideas": [...]} with the extraction's metadata."""
    data = read(path, [])
    return data.get("ideas", []) if isinstance(data, dict) else data


def videos_router(root: Path = VIDEOS) -> APIRouter:
    r = APIRouter(prefix="/api/videos")

    def video_dir(vid):
        if vid not in {v["id"] for v in read(root / "index.json", [])}:  # also keeps path parts out of the filesystem
            raise HTTPException(404, f"No video {vid}")
        return root / vid

    @r.get("")
    def listing():
        out = []
        for v in read(root / "index.json", []):
            latest = read(root / v["id"] / "engine" / "latest.json") or {}
            out.append({**v, **{k: latest.get(k) for k in ("top1", "top3", "n", "commit")}})
        return out

    @r.get("/{vid}")
    def one(vid: str):
        d = video_dir(vid)
        engine = d / "engine"
        files = sorted((p for p in engine.glob("*.json") if p.stem != "latest"), key=lambda p: -p.stat().st_mtime) if engine.is_dir() else []
        return {"video": next(v for v in read(root / "index.json") if v["id"] == vid), "picks": read(d / "picks.json"), "deck": read(d / "deck.json"),
                "ideas": ideas(d / "ideas.json"), "status": read(d / "ideas-status.json", {}),
                "engine": read(engine / "latest.json"), "commits": [p.stem for p in files]}

    @r.get("/{vid}/engine/{commit}")
    def engine_at(vid: str, commit: str):
        data = read(video_dir(vid) / "engine" / f"{commit}.json") if re.fullmatch(r"[0-9a-zA-Z]+", commit) else None
        if data is None:
            raise HTTPException(404, f"No engine record {commit}")
        return data

    @r.put("/{vid}/ideas/{idea_id}")
    def put_idea(vid: str, idea_id: str, status: str = Body(...), result: str | None = Body(None), doc: str | None = Body(None)):
        d = video_dir(vid)
        if status not in STATUSES:
            raise HTTPException(400, f"status must be one of {', '.join(STATUSES)}")
        if idea_id not in {str(i["id"]) for i in ideas(d / "ideas.json")}:
            raise HTTPException(404, f"No idea {idea_id}")
        all_ = read(d / "ideas-status.json", {})
        all_[idea_id] = {"status": status, "result": result or "", "doc": doc or ""}
        tmp = d / "ideas-status.json.tmp"
        tmp.write_text(json.dumps(all_, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "ideas-status.json")
        return all_[idea_id]

    return r
