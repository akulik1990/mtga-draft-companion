"""FastAPI app: snapshot endpoint, WebSocket push, static frontend."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
        session.poll()  # no-op drain when the scanner thread runs; advances the scanner when it does not (tests, tools)
        return snapshot()

    @app.post("/api/draft/rescan")
    def rescan():
        session.rescan()
        return {"ok": True}

    @app.post("/api/draft/reset")
    def reset():
        session.reset()
        return {"ok": True}

    @app.get("/api/datasets")
    def datasets():
        return session.datasets()

    @app.post("/api/datasets/select")
    def select_dataset(set: str = Body(...), format: str = Body(...), user_group: str = Body("All")):
        if not session.select_dataset(set, format, user_group):
            raise HTTPException(404, f"No dataset for {set} {format} {user_group}")
        return session.datasets()

    @app.post("/api/datasets/sync")
    async def sync_datasets():
        messages = await asyncio.to_thread(session.sync_datasets)
        return {"messages": messages, **session.datasets()}

    @app.get("/api/settings")
    def get_settings():
        return session.settings()

    @app.put("/api/settings")
    def put_settings(changes: dict = Body(...)):
        try:
            return session.update_settings(changes)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/engine")
    def engine():
        return session.engine_settings()

    @app.put("/api/engine")
    def put_engine(changes: dict = Body(...)):
        try:
            return session.update_engine_settings(changes)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/tiers")
    def tiers():
        return session.tier_lists()

    @app.post("/api/tiers")
    def import_tier(url: str = Body(...), label: str = Body("")):
        try:
            return session.import_tier_list(url, label)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/tiers/{filename}")
    def delete_tier(filename: str):
        try:
            return session.delete_tier_list(filename)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/cards")
    def cards(q: str = ""):
        return jsonable_encoder(session.search_cards(q))

    @app.post("/api/deck/build")
    async def build_decks():
        # Monte Carlo + numba JIT on first call: seconds, so keep it off the event loop.
        return jsonable_encoder(await asyncio.to_thread(session.build_decks))

    @app.post("/api/mock/start")
    def mock_start(kind: str = Body("random"), set: str | None = Body(None), seed: int | None = Body(None)):
        try:
            return session.start_mock(kind, set, seed)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/mock/pick")
    def mock_pick(card_name: str = Body(..., embed=True)):
        try:
            return session.mock_pick(card_name)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/mock/stop")
    def mock_stop():
        session.stop_mock()
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

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        # Hashed bundles under _app/immutable never change; the page itself must be re-checked after a rebuild.
        response = await call_next(request)
        path = request.url.path
        if not path.startswith("/api") and "/_app/immutable/" not in path:
            response.headers["Cache-Control"] = "no-cache"
        return response

    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return "<p>Frontend not built. API: <a href='/api/state'>/api/state</a></p>"

    return app
