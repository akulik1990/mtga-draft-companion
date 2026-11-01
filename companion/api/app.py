"""FastAPI app: snapshot endpoint, WebSocket push, static frontend."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from companion import ROOT
from companion.session import DraftSession

log = logging.getLogger(__name__)

FRONTEND_BUILD = ROOT / "frontend" / "build"
POLL_SECONDS = 0.5


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def broadcast(self, payload):
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.clients.discard(ws)


def create_app(session: DraftSession, *, run_scanner: bool = True, frontend: Path = FRONTEND_BUILD) -> FastAPI:
    hub = Hub()

    def snapshot():
        return jsonable_encoder(session.snapshot())

    async def pump():
        while True:
            try:
                if await asyncio.to_thread(session.poll) and hub.clients:
                    await hub.broadcast(await asyncio.to_thread(snapshot))
            except Exception:
                log.exception("pump failed")
            await asyncio.sleep(POLL_SECONDS)

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if run_scanner:
            session.start()
            task = asyncio.create_task(pump())
        yield
        if task:
            task.cancel()
        session.stop()

    app = FastAPI(title="MTGA Draft Companion", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/state")
    def state():
        return snapshot()

    @app.post("/api/draft/rescan")
    def rescan():
        session.orchestrator.trigger_full_scan()
        return {"ok": True}

    @app.websocket("/api/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            await websocket.send_json(await asyncio.to_thread(snapshot))
            while True:
                await websocket.receive_text()  # keepalive; client messages ignored
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(websocket)

    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return "<p>Frontend not built. API: <a href='/api/state'>/api/state</a></p>"

    return app
