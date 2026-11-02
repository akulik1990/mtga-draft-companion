"""FastAPI app: snapshot endpoint, WebSocket push, static frontend."""

import asyncio
import json
import re
from functools import lru_cache
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from companion import ROOT
from companion.advisor_v2.lanes import pips
from companion.session import DraftSession

log = logging.getLogger(__name__)

FRONTEND_BUILD = ROOT / "frontend" / "build"
POLL_SECONDS = 0.5
RARITY = {"mythic": 0, "rare": 1, "uncommon": 2, "common": 3}
PAIRS = ["WU", "UB", "BR", "RG", "GW", "WB", "UR", "BG", "RW", "GU"]


def heat(ratings, mean, std):
    """{name: 2 | 1} for the set's standout cards. 2 (blue fire): top 20 by All Decks GIH WR. 1 (fire): taken by pick 2
    on average (ATA <= 2.0) or GIH WR >= mean + 1 SD. Cards in hand for under 500 games are left out."""
    rows = {}
    for c in (ratings or {}).values():
        a = (c.get("deck_colors") or {}).get("All Decks") or {}
        if "Land" not in (c.get("types") or []) and (a.get("samples") or 0) >= 500:
            rows[c["name"]] = (a.get("gihwr") or 0, a.get("ata") or 99)
    top = set(sorted(rows, key=lambda n: -rows[n][0])[:20])
    return {n: 2 if n in top else 1 for n, (wr, ata) in rows.items() if n in top or ata <= 2.0 or (std and wr >= mean + std)}


def plain(reason):
    """v2's reasoning lines, reworded for the overlay panel."""
    m = re.match(r"Branch: \+([\d.]+) over (.+), above the (-?[\d.]+) top drafters switch at", reason)
    if m:
        return f"{m[1]} points stronger than {m[2]}; top drafters switch at {m[3]} or more this early"
    m = re.match(r"(\w+) (\d+)% → ([\d.]+) \((?:n=(\d+)|rarely played there)\)", reason)
    if m:
        games = f"{int(m[4]):,} games" if m[4] else "rarely played there"
        return f"Plays in {m[1]}: wins {m[3]}%, {games}" if m[2] == "100" else f"{m[1]} ({m[2]}% likely): wins {m[3]}%, {games}"
    return reason


@lru_cache
def close_calls(set_code):
    path = ROOT / "companion" / "advisor_v2" / f"closecalls_{set_code}.json"
    return json.loads(path.read_text())["buckets"] if path.exists() else []


def side_panel(s, ranked=()):
    """What the overlay's left panel shows: v2's top three and how top drafters split between them in this spot,
    v2's reasons, likeliest lanes, open colours, pool summary."""
    top = (s["advisors"].get("v2") or [None])[0]
    why = {r["card_name"]: next((plain(w) for w in r.get("reasoning") or [] if re.match(r"\w+ \d+% →", w)), "")
           for r in s["advisors"].get("v2") or []}
    close = None
    if len(ranked) >= 2 and ranked[0]["score"] and ranked[1]["score"]:
        gap = ranked[0]["score"] - ranked[1]["score"]
        row = next((r for r in close_calls((s.get("event") or {}).get("set") or "")
                    if r["from"] <= gap and (r["to"] is None or gap < r["to"])), None)
        close = {"gap": round(gap, 1), "took": row and row["took"][:3]}
    lanes = sorted((s.get("lanes") or {}).items(), key=lambda kv: -kv[1])[:4]
    pair_wr = s.get("color_ratings") or {}
    sides = s.get("signals_v2_sides") or {}
    spells = [c for c in s.get("pool") or [] if "Land" not in (c.get("types") or [])]
    curve = [0] * 6
    for c in spells:
        curve[min(max(int(c.get("cmc") or 0), 1), 6) - 1] += 1
    return {
        "pick": top and top["card_name"],
        "top3": [{"name": c["name"], "score": c["score"], "why": why.get(c["name"], "")} for c in ranked],
        "close": close,
        "why": [plain(r) for r in (top or {}).get("reasoning") or [] if not r.startswith("Fit #")][:3],
        "lanes": [{"lane": l, "p": round(p, 3), "wr": pair_wr.get(l)} for l, p in lanes],
        "signals": sides.get(sides.get("active")) or s.get("signals_v2") or {},
        "pool": {"n": len(s.get("pool") or []), "colours": {k: sum(k in (c.get("colors") or []) for c in spells) for k in "WUBRG"}, "curve": curve},
    }


def arena_order(card):
    """Where Arena's draft screen puts a card: rarity, then mono colours WUBRG, multicolour, colourless, lands.

    Colours come from the front face's mana cost, hybrid pips counting both colours.
    ponytail: from one FRA pack (2026-11-02); pair order, three colours and same-group ties (cmc, name) are guesses.
    """
    front = (card.get("mana_cost") or "").split("//")[0]
    colours = "".join(c for c in "WUBRG" if any(c in o for o in pips({"mana_cost": front, "types": []})))
    if "Land" in (card.get("types") or []):
        group = (3, 0)
    elif len(colours) == 1:
        group = (0, "WUBRG".index(colours))
    elif colours:
        pair = next((i for i, p in enumerate(PAIRS) if set(p) == set(colours)), None)
        group = (1, pair if pair is not None else len(PAIRS) + len(colours))
    else:
        group = (2, 0)
    return RARITY.get(card.get("rarity"), 4), group, card.get("cmc") or 0, card.get("name") or ""


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

    heat_cache = {}

    @app.get("/api/overlay")
    def overlay():
        """The pack in log order with v2's verdict per card; what overlay/draft.qml draws over Arena's cards."""
        s = session.snapshot()
        mean, std = ((s.get("metrics") or {}).get("All Decks") or {}).get("gihwr") or (0.0, 0.0)
        if heat_cache.get("dataset") != s["dataset"]:
            heat_cache.update(dataset=s["dataset"], heat=heat(session.scanner.set_data.get_card_ratings(), mean, std))
        v2 = {r["card_name"]: r for r in s["advisors"].get("v2") or []}
        up = {r["card_name"]: r["rank"] for r in s["advisors"].get("upstream") or []}
        cards = []
        for c in sorted(s["pack"], key=arena_order):
            r = v2.get(c["name"])
            gih = ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0
            why = (r or {}).get("reasoning") or [""]
            ranks = next(filter(None, (re.match(r"Fit #(\d+) · Power #(\d+)", w) for w in why)), None)  # after "Branch: ..." on a switch
            cards.append({"name": c["name"], "rank": r and r["rank"], "score": r and r["contextual_score"],
                          "orders": ranks[0] if ranks else "", "fit": ranks and int(ranks[1]), "power": ranks and int(ranks[2]),
                          "heat": heat_cache["heat"].get(c["name"], 0),
                          "upstream": up.get(c["name"]), "gihwr": gih})
        b = (s.get("v2_split") or {}).get("branch")
        if b and b["take"]:  # the switch pick was chosen on Power; show that, not its blended Fit-side score
            power_top = s["v2_split"]["power"][0]
            for c in cards:
                if c["name"] == power_top["card"]:
                    c["score"] = power_top["score"]
        if not b:
            line = ""
        elif b["take"]:
            line = f"Switch: {b['card']} over {b['over']} (+{b['gap']:.1f})"
        else:
            line = (f"Not switching to {b['card']}: {b['gap']:+.1f} stronger, needs {b['threshold']:+.1f} at this stage"
                    + ("" if b.get("shares_colour", True) else " for a card outside both your colours"))
        ranked = sorted((c for c in cards if c["rank"]), key=lambda c: c["rank"])[:3]
        return {"status": s["status"], "pick": [s["pack_number"], s["pick_number"]], "cards": cards, "branch": line,
                "mean": mean, "std": std, "panel": side_panel(s, ranked)}

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
