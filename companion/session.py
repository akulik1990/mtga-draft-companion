"""One draft session over upstream's scanner and its Tk-free background orchestrator.

Everything here calls vendor functions; nothing reimplements their logic.
"""

import logging
import os
import queue
import random
import threading
from functools import lru_cache
from pathlib import Path

from src import constants
from src.card_logic import copy_deck, filter_options, format_win_rate, suggest_deck
from src.configuration import write_configuration
from src.dataset_updater import DatasetUpdater
from src.dataset_selection import select_event_dataset
from src.signals import SignalCalculator
from src.ui.orchestrator import DraftOrchestrator  # threading only, no Tk
from src.tier_list import TIER_FILE_PREFIX, TIER_FOLDER, TierList
from src.utils import normalize_color_string, retrieve_local_set_list

from companion.advisors import rank_pack
from companion.mock import Cards, MockDraft, sampled_drafts

log = logging.getLogger(__name__)

GRADED_FIELDS = tuple(constants.WIN_RATE_OPTIONS)  # gihwr, ohwr, gpwr, gnswr, gdwr
DECK_CARD_FIELDS = ("name", "count", "cmc", "mana_cost", "types", "colors", "image", "rarity")


def set_ranks(ratings):
    """{name: rank} of the set's cards by All Decks GIH WR, 1 = best; lands and cards in hand for under 500 games
    are left out (the same pool the overlay's fire markers use)."""
    rows = {}
    for c in (ratings or {}).values():
        a = (c.get("deck_colors") or {}).get("All Decks") or {}
        if "Land" not in (c.get("types") or []) and (a.get("samples") or 0) >= 500 and a.get("gihwr"):
            rows[c["name"]] = a["gihwr"]
    return {n: i for i, n in enumerate(sorted(rows, key=lambda n: -rows[n]), 1)}


class DraftSession:
    def __init__(self, scanner, config):
        self._ranks = (None, {})  # (dataset, {name: set rank by GIH WR})
        self.scanner = scanner
        self.config = config
        self.orchestrator = DraftOrchestrator(scanner, config, refresh_callback=None)
        self.status = "Waiting for draft..."
        self.mock = None  # MockDraft while one is running
        self._deckbuild = {}  # overlay builds for the finished draft's pool: key, busy, builds

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self.orchestrator.start()

    def stop(self):
        self.orchestrator.stop()

    def poll(self) -> bool:
        """Advance the scanner if its thread is not running and drain its queue.

        Returns True when the snapshot may have changed.
        """
        if not self.orchestrator.is_alive():
            self._swap_pending_file()
            self.orchestrator.step_process()
        changed = False
        while True:
            try:
                item = self.orchestrator.update_queue.get_nowait()
            except queue.Empty:
                return changed
            if isinstance(item, dict) and "status" in item:
                self.status = item["status"]
            elif item == "REFRESH":
                self.status = None  # transient message done; snapshot() derives the steady state
            changed = True

    # ── draft state ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Mirror of upstream AppController.refresh_ui_data, returned as data."""
        s = self.scanner
        with s.lock:
            event_set, event_type = s.retrieve_current_limited_event()
            pack_no, pick = s.retrieve_current_pack_and_pick()
            metrics = s.retrieve_set_metrics()
            taken = s.retrieve_taken_cards()
            pack = s.retrieve_current_pack_cards()
            missing = s.retrieve_current_missing_cards()
            picked_now = s.retrieve_current_picked_cards()
            history = s.retrieve_draft_history()
            color_ratings = dict(s.set_data.get_color_ratings() or {})
            draft_id = s.current_draft_id
            event_string = s.event_string
            cards_per_pick = s.cards_per_pick

        seen_packs = [(e["Pack"], e["Pick"], self.scanner.set_data.get_data_by_id(e["Cards"])) for e in history]
        signals = self._signals(metrics, seen_packs)
        deck_filter = filter_options(taken, self.config.settings.deck_filter, metrics, self.config)
        archetype = deck_filter[0] if deck_filter else constants.FILTER_OPTION_ALL_DECKS
        advisors, lanes, signals_v2, signal_sides, v2_split = rank_pack(metrics, taken, pack, pick, pack_no, seen_packs) if pack else ({}, {}, {}, {}, None)

        filter_options_map = s.retrieve_color_win_rate(self.config.settings.filter_format) or {}
        tier_data = s.retrieve_tier_data() or {}
        tier_names = dict((v, k) for k, v in (TierList.retrieve_data(event_set)[1] or {}).items()) if event_set else {}
        if self._ranks[0] != self.config.card_data.latest_dataset:
            self._ranks = (self.config.card_data.latest_dataset, set_ranks(self.scanner.set_data.get_card_ratings()))
        for card in (*pack, *missing, *taken):
            card["set_rank"] = self._ranks[1].get(card.get("name"))
            card["grades"] = self._grades(card, metrics, archetype)
            card["tiers"] = {key: tl.ratings[card["name"]].rating.strip() for key, tl in tier_data.items() if card.get("name") in tl.ratings}

        if self.status is None:
            self.status = f"Pack {pack_no} Pick {pick}" if pack_no > 0 else "Waiting for draft..."
        return {
            "status": self.status,
            "mock": self.mock.describe() if self.mock else None,
            "event": {"set": event_set, "type": event_type, "label": event_string, "draft_id": draft_id},
            "pack_number": pack_no,
            "pick_number": pick,
            "cards_per_pick": cards_per_pick,
            "dataset": self.config.card_data.latest_dataset,
            "archetype": archetype,
            "color_ratings": color_ratings,
            "signals": signals,
            "lanes": lanes,
            "signals_v2": signals_v2,
            "signals_v2_sides": signal_sides,
            "v2_split": v2_split,
            "pack": pack,
            "missing": missing,
            "picked_from_pack": picked_now,
            "pool": taken,
            "advisors": advisors,
            # What the UI needs to show any lane and field the way upstream's table does.
            "deck_filter": self.config.settings.deck_filter,
            "deck_filter_options": filter_options_map,
            "result_format": self.config.settings.result_format,
            "metrics": self._lane_metrics(metrics, filter_options_map),
            "grade_thresholds": [[g.strip(), z] for g, z in constants.GRADE_DEVIATION_DICT.items()],
            "tiers": [{"key": k, "label": tier_names.get(k, k)} for k in tier_data],
        }

    def _signals(self, metrics, seen_packs):
        """Upstream's lateness signal, for display; v2 computes its own z-scores."""
        calc = SignalCalculator(metrics)
        scores = {c: 0.0 for c in constants.CARD_COLORS}
        for pack_no, pick, cards in seen_packs:
            if pack_no == 2:  # upstream excludes pack 2 (other direction)
                continue
            for color, value in calc.calculate_pack_signals(cards, pick).items():
                scores[color] += value
        return scores

    @staticmethod
    def _lane_metrics(metrics, options):
        """{lane: {field: [mean, std]}} for All Decks and every lane the deck filter offers ({label: key})."""
        if not metrics:
            return {}
        lanes = {constants.FILTER_OPTION_ALL_DECKS}
        lanes |= {normalize_color_string(k) for k in options.values() if k not in (constants.FILTER_OPTION_AUTO, constants.FILTER_OPTION_ALL_DECKS)}
        return {lane: {f: list(metrics.get_metrics(lane, f)) for f in GRADED_FIELDS} for lane in lanes}

    @staticmethod
    def _grades(card, metrics, archetype):
        """Letter grades for All Decks and the active archetype, via upstream's grading."""
        out = {}
        stats = card.get("deck_colors", {})
        for key in {constants.FILTER_OPTION_ALL_DECKS, archetype}:
            row = stats.get(key) or {}
            out[key] = {
                f: format_win_rate(row.get(f, 0.0), key, f, metrics, constants.RESULT_FORMAT_GRADE)
                for f in GRADED_FIELDS
            }
        return out

    def rescan(self):
        self.orchestrator.trigger_full_scan()

    def _swap_pending_file(self):
        """Without the orchestrator thread (tests, tools), apply a queued file swap the way its loop does."""
        new_file = None
        while not self.orchestrator._file_swap_queue.empty():
            new_file = self.orchestrator._file_swap_queue.get_nowait()
        if new_file:
            self.scanner.set_arena_file(new_file)
            self.scanner.draft_start_search()
            self.orchestrator.sync_dataset_to_event()
            self.scanner.draft_data_search()
            self.status = None

    # ── mock drafts ──────────────────────────────────────────────────────────

    def start_mock(self, kind="random", set_code=None, seed=None):
        """Begin a mock draft on its own log and point the scanner at it. kind: random | table | 17lands."""
        set_code = set_code or (self.config.card_data.latest_dataset or "").split("_")[0]
        if not set_code:
            raise ValueError("no active dataset; select one first")
        dataset = Path(constants.SETS_FOLDER) / f"{set_code}_PremierDraft_All_Data.json"
        if not dataset.exists():
            raise FileNotFoundError(f"no Premier dataset for {set_code}")
        cards = Cards(dataset)
        # One file per mock: the scanner ignores a swap to the path it already watches. Only this
        # session's previous mock is removed; other processes (a compare run) may be using the folder.
        folder = Path(constants.TEMP_FOLDER) / "mock"
        folder.mkdir(parents=True, exist_ok=True)
        if self.mock and self.mock.log.exists():
            self.mock.log.unlink()
        log = folder / f"mock-{random.randrange(16**6):06x}.Player.log"
        real = sampled_drafts(Path(constants.TEMP_FOLDER).parent, set_code) if kind in ("random", "17lands") else []
        if kind == "random":
            kind = "17lands" if real and random.random() < 0.5 else "table"
        if kind == "17lands" and real:
            self.mock = MockDraft.real(cards, log, random.choice(real))
        else:
            self.mock = MockDraft.table(cards, log, seed)
        self.mock.start()
        self.orchestrator.set_file_and_scan(str(log))
        return self.mock.describe()

    def mock_pick(self, card_name):
        if not self.mock:
            raise ValueError("no mock draft running")
        self.mock.pick(card_name)
        return self.mock.describe()

    def stop_mock(self):
        self.mock = None
        live = getattr(self.orchestrator, "live_log_path", None) or self.scanner.arena_file
        self.orchestrator.set_file_and_scan(live)

    def reset(self):
        with self.scanner.lock:
            self.scanner.clear_draft(True)
        self.orchestrator.request_math_update()

    # ── datasets ─────────────────────────────────────────────────────────────

    def datasets(self) -> dict:
        files, _ = retrieve_local_set_list()
        keys = ("set", "format", "user_group", "start_date", "end_date", "games", "path", "collected")
        rows = [dict(zip(keys, f)) for f in files if f]
        for r in rows:
            r["file"] = os.path.basename(r.pop("path"))
        return {"active": self.config.card_data.latest_dataset, "datasets": rows}

    def select_dataset(self, set_code: str, fmt: str, user_group: str) -> bool:
        ok = self.orchestrator.sync_dataset_to_event(set_code, fmt, user_group)
        self.orchestrator.request_math_update()
        return bool(ok)

    def sync_datasets(self) -> list[str]:
        messages: list[str] = []
        DatasetUpdater(self.config).sync_datasets(messages.append)
        self.orchestrator.sync_dataset_to_event()
        self.orchestrator.request_math_update()
        return messages

    # ── settings ─────────────────────────────────────────────────────────────

    def settings(self) -> dict:
        st = self.config.settings
        return {
            "arena_log_location": st.arena_log_location,
            "arena_log_pinned": st.arena_log_pinned,
            "auto_sync_datasets": st.auto_sync_datasets,
            "active_log": self.scanner.arena_file,
            "deck_filter": st.deck_filter,
            "result_format": st.result_format,
            "result_formats": list(constants.RESULT_FORMAT_LIST),
        }

    def update_settings(self, changes: dict) -> dict:
        st = self.config.settings
        if "auto_sync_datasets" in changes:
            st.auto_sync_datasets = bool(changes["auto_sync_datasets"])
        if "deck_filter" in changes:
            options = self.scanner.retrieve_color_win_rate(st.filter_format) or {}  # {label: key}
            if changes["deck_filter"] not in options.values():
                raise ValueError(f"unknown deck filter {changes['deck_filter']!r}")
            st.deck_filter = changes["deck_filter"]
            self.orchestrator.request_math_update()
        if "result_format" in changes:
            if changes["result_format"] not in constants.RESULT_FORMAT_LIST:
                raise ValueError(f"unknown result format {changes['result_format']!r}")
            st.result_format = changes["result_format"]
        if "arena_log_location" in changes:
            path = (changes["arena_log_location"] or "").strip()
            # Same rule as upstream Settings: a manual path is pinned; clearing re-enables discovery.
            st.arena_log_location = path
            st.arena_log_pinned = bool(path)
            if path:
                self.orchestrator.live_log_path = path
                self.orchestrator.set_file_and_scan(path)
        write_configuration(self.config)
        return self.settings()

    # ── engine switches (v2), saved in the data dir ───────────────────────────

    ENGINE_SWITCHES = {
        "learnt_prefs": ("PREF_WEIGHT", "Learnt top-drafter preferences", "Bonus for the cards top drafters choose head to head (tools/learn_prefs.py)."),
        "split": ("SPLIT", "Power / Fit split", "Two orders and a learnt switch between them; off = one blended order."),
        "own_builder": ("OWN_BUILDER", "Our deck builder", "Builds from the deck win model within top players' deck shape (creatures, curve, two colours); off = upstream's builder."),
        "branch_colour": ("BRANCH_COLOUR", "Colour-aware switch", "The switch to the stronger card also weighs whether it keeps one of your colours and how settled your pool is. Matches top drafters better, mock decks slightly worse."),
    }

    def _engine_file(self):
        from companion import DATA_DIR

        return DATA_DIR / "companion_engine.json"

    def engine_settings(self) -> dict:
        from companion.advisor_v2.engine import DraftAdvisorV2 as V

        return {key: {"on": bool(getattr(V, attr)), "label": label, "help": hint}
                for key, (attr, label, hint) in self.ENGINE_SWITCHES.items()}

    def update_engine_settings(self, changes: dict, save: bool = True) -> dict:
        """Switch v2 features on or off for this app; the change applies from the next snapshot."""
        import json

        from companion.advisor_v2 import config
        from companion.advisor_v2.engine import DraftAdvisorV2 as V

        for key, on in changes.items():
            if key not in self.ENGINE_SWITCHES:
                raise ValueError(f"unknown engine switch {key!r}")
            attr = self.ENGINE_SWITCHES[key][0]
            if attr == "PREF_WEIGHT":
                # Learnt preferences replace the pick-order prior; switching them off brings the prior back.
                wanted = {"PREF_WEIGHT": V.PREF_ON_WEIGHT if on else 0.0, "ORDER_WEIGHT": 0.0 if on else V.ORDER_OFF_WEIGHT}
            else:
                wanted = {attr: bool(on)}
            for name, value in wanted.items():
                # A switch left at the code's default does not pin the value: a set's manual layer can still change it.
                if value == config.BASELINE.get(name):
                    config.RUNTIME.pop(name, None)
                    setattr(V, name, value)
                else:
                    config.override(name, value)
        if save:
            self._engine_file().write_text(json.dumps({k: v["on"] for k, v in self.engine_settings().items()}))
        self.orchestrator.request_math_update()
        return self.engine_settings()

    def load_engine_settings(self):
        import json

        f = self._engine_file()
        if f.exists():
            try:
                self.update_engine_settings(json.loads(f.read_text()), save=False)
            except (ValueError, json.JSONDecodeError) as e:
                log.warning("ignoring %s: %s", f, e)

    # ── tier lists and card search ───────────────────────────────────────────

    def tier_lists(self) -> list[dict]:
        event_set, _ = self.scanner.retrieve_current_limited_event()
        return [{"set": s_, "label": l, "date": d, "file": f} for s_, l, d, f in TierList.retrieve_files(event_set or "")]

    def import_tier_list(self, url: str, label: str = "") -> list[dict]:
        """Same as upstream's tier list panel: fetch a 17Lands tier list and save it under Tier/."""
        tl = TierList.from_api(url)
        if not tl:
            raise ValueError("could not fetch that tier list; it must be a https://www.17lands.com/tier_list/ URL")
        if label:
            tl.meta.label = label
        import time
        tl.to_file(os.path.join(TIER_FOLDER, f"{TIER_FILE_PREFIX}_{tl.meta.set}_{int(time.time())}.txt"))
        self.orchestrator.request_math_update()
        return self.tier_lists()

    def delete_tier_list(self, filename: str) -> list[dict]:
        if os.path.basename(filename) != filename:
            raise ValueError("bad file name")
        TierList.delete_file(filename)
        self.orchestrator.request_math_update()
        return self.tier_lists()

    def search_cards(self, query: str, limit: int = 25) -> list[dict]:
        """Cards of the active dataset whose name contains `query`, for Compare."""
        q = (query or "").strip().lower()
        with self.scanner.lock:
            ratings = self.scanner.set_data.get_card_ratings() or {}
            metrics = self.scanner.retrieve_set_metrics()
            taken = self.scanner.retrieve_taken_cards()
        deck_filter = filter_options(taken, self.config.settings.deck_filter, metrics, self.config)
        archetype = deck_filter[0] if deck_filter else constants.FILTER_OPTION_ALL_DECKS
        seen, out = set(), []
        for card in ratings.values():
            name = card.get("name", "")
            if q and q not in name.lower() or name in seen or not (card.get("deck_colors") or {}).get(constants.FILTER_OPTION_ALL_DECKS, {}).get("gihwr"):
                continue
            seen.add(name)
            out.append({**card, "grades": self._grades(card, metrics, archetype)})
            if len(out) >= limit:
                break
        return out

    # ── deck building ────────────────────────────────────────────────────────

    def deckbuild_state(self) -> dict:
        """After a draft: Arena's deck builder is open, so the overlay suggests builds. {"phase": "draft"} otherwise.

        Upstream's scanner stops before the last pick ("Completed" carries it), so the pool comes from that message.
        Builds take seconds, so they are made once per pool in a thread; until then "building" is true.
        """
        tail = log_tail(self.scanner.arena_file)
        scene = tail.rfind('"toSceneName":"DeckBuilder"')
        if scene < 0 or tail.rfind('"toSceneName":"Draft"') > scene:
            return {"phase": "draft"}
        ids = completed_picks(tail)
        if not ids:
            return {"phase": "draft"}
        key = tuple(sorted(ids))
        cache = self._deckbuild
        if cache.get("key") != key and not cache.get("busy"):
            cache.update(key=key, busy=True, builds=None)

            def run():
                try:
                    with self.scanner.lock:
                        pool = self.scanner.set_data.get_data_by_id(list(ids))
                    cache["builds"] = overlay_builds(self.build_decks(pool), pool)
                finally:
                    cache["busy"] = False

            threading.Thread(target=run, daemon=True).start()
        return {"phase": "deckbuild", "building": cache.get("builds") is None, "builds": cache.get("builds") or []}

    def deck_pool(self) -> list[dict]:
        """The pool to build from: the scanner's picks, else (Arena restarted, so the new log holds no draft) the last
        Completed draft in the log, else the card pool of the latest draft event in Arena's course list."""
        with self.scanner.lock:
            taken = self.scanner.retrieve_taken_cards()
        if taken:
            return taken
        tail = log_tail(self.scanner.arena_file)
        ids, event = completed_picks(tail), None
        if not ids:
            ids, event = course_pool(tail)
        if not ids:
            return []
        with self.scanner.lock:
            pool = self.scanner.set_data.get_data_by_id(list(ids))
        parts = (event or "").split("_")  # "QuickDraft_FRA_20261027": no draft seen yet, so load that set's dataset
        if len(pool) < len(ids) // 2 and len(parts) >= 2:
            with self.scanner.lock:  # look first: upstream's sync clears the active dataset when nothing matches
                found = select_event_dataset(self.scanner, parts[1], parts[0], "All")
            if not (found and self.select_dataset(parts[1], parts[0], "All")):
                return pool
            with self.scanner.lock:
                pool = self.scanner.set_data.get_data_by_id(list(ids))
        return pool

    def score_deck(self, counts: dict, taken=None) -> dict:
        """The deck editor's live stats for a 40 of the user's choosing (companion/deckscore.py)."""
        from companion import deckscore

        taken = taken if taken is not None else self.deck_pool()
        set_code = (self.config.card_data.latest_dataset or "").split("_")[0]
        from companion.advisor_v2 import config, deckbuilder
        from companion.advisor_v2.engine import DraftAdvisorV2  # noqa: F401  (registers the settings activate sets)

        config.activate(set_code)
        MODEL = deckbuilder.MODEL  # the editor judges as the builder does

        out = deckscore.score(set_code, taken, counts, _deck_model(set_code, MODEL) if set_code else None)
        deck, side = [], []
        held = {}
        for c in taken:
            held.setdefault(c["name"], [c, 0])[1] += 1
        for name, (c, k) in held.items():
            n = min(int(counts.get(name, 0)), k)
            if n:
                deck.append({**c, "count": n})
            if k - n:
                side.append({**c, "count": k - n})
        for col, name in deckbuilder_basics().items():
            if counts.get(name):
                deck.append({"name": name, "count": int(counts[name]), "types": ["Basic", "Land"], "colors": [col]})
        out["arena_export"] = copy_deck(deck, side)
        out["pool"] = [_trim({**c, "count": k}) | {"count": k} for c, k in held.values()]
        return out

    def build_decks(self, taken=None) -> list[dict]:
        """Upstream's builds for the pool (taken, else the scanner's picks), each with our label and predicted win rate."""
        taken = taken if taken is not None else self.deck_pool()
        with self.scanner.lock:
            metrics = self.scanner.retrieve_set_metrics()
            _, event_type = self.scanner.retrieve_current_limited_event()
        set_code = (self.config.card_data.latest_dataset or "").split("_")[0]
        ours = own_builds(set_code, taken)
        if ours is not None:
            return ours
        results = suggest_deck(
            taken, metrics, self.config, event_type or "PremierDraft", None,
            self.config.card_data.latest_dataset,
        )
        builds = []
        for label, data in (results or {}).items():
            deck, side = data.get("deck_cards", []), data.get("sideboard_cards", [])
            builds.append({
                "label": label,
                "variant": data.get("label_prefix", ""),
                "rating": data.get("rating"),
                "record": data.get("record"),
                "colors": data.get("colors", []),
                "breakdown": data.get("breakdown"),
                "stats": data.get("stats"),
                "note": data.get("optimization_note"),
                "deck": [_trim(c) for c in deck],
                "sideboard": [_trim(c) for c in side],
                "arena_export": copy_deck(deck, side),
                **mana_identity(deck),
            })
        for b in builds:
            b["deck_wr"] = deck_win_rate(set_code, b)
            b["creatures"] = sum(c.get("count", 1) for c in b["deck"] if "Creature" in (c.get("types") or []))
            # The deck model learnt from real decks; far fewer creatures than any of them run is outside its evidence.
            b["realistic"] = b["creatures"] >= MIN_CREATURES
        return builds


def deckbuilder_basics():
    from companion.advisor_v2.deckbuilder import BASIC

    return BASIC


@lru_cache(maxsize=8)
def _deck_model(set_code, section=None):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/ sits beside companion/
    from tools.deck_model import DeckModel

    try:
        return DeckModel(set_code, section)
    except FileNotFoundError:
        return None


def deck_win_rate(set_code: str, build: dict) -> float | None:
    """Predicted game win rate of the 40 for an average player (tools/deck_model.py), or None without a model."""
    model = _deck_model(set_code) if set_code else None
    if not model or not build.get("deck"):
        return None
    counts = {c["name"]: c.get("count", 1) for c in build["deck"]}
    return round(float(model.win_rate(counts, build.get("main", ""), build.get("splash", ""))), 2)


def mana_identity(deck: list[dict]) -> dict:
    """The pair that casts the most spells, and the colours the rest need, hybrid aware.

    Upstream's label counts every pip outside the top two colours as a splash, so a W/U hybrid in
    a UG deck reads "Splash W". Here a colour is a splash only if some spell cannot be cast
    without it.
    """
    from companion.advisor_v2.lanes import LANES, castable, off_pips, pips

    spells = [c for c in deck if "Land" not in (c.get("types") or [])]
    weight = lambda lane: (sum(c.get("count", 1) for c in spells if castable(c, lane)),
                           sum(c.get("count", 1) * len(pips(c)) for c in spells if castable(c, lane)))
    main = max(LANES, key=weight) if spells else ""
    splash, cards = set(), 0
    for c in spells:
        off = off_pips(c, main)
        if off:
            cards += c.get("count", 1)
            for options in off:
                splash.add(options[0] if len(options) == 1 else next((o for o in options if o in splash), options[0]))
    order = "WUBRG"
    return {"main": main, "splash": "".join(sorted(splash, key=order.index)), "splash_cards": cards}


def _trim(card: dict) -> dict:
    out = {k: card.get(k) for k in DECK_CARD_FIELDS if k in card}
    out["gihwr"] = (card.get("deck_colors", {}).get("All Decks") or {}).get("gihwr")
    return out


def own_builds(set_code, taken):
    """Our builder's decks in the app's build format; None when it is off or the set lacks a deck model or targets."""
    from collections import Counter

    from companion.advisor_v2 import deckbuilder
    from companion.advisor_v2 import config
    from companion.advisor_v2.engine import DraftAdvisorV2

    config.activate(set_code)  # else the set's builder settings wait until a pack has been scored
    targets = config.fitted(set_code, "decks").get("pairs")
    model = _deck_model(set_code, deckbuilder.MODEL) if set_code else None
    if not DraftAdvisorV2.OWN_BUILDER or not targets or not model or not taken:
        return None

    def grouped(cards):
        out = {}
        for c in cards:
            e = out.setdefault(c["name"], {**_trim(c), "count": 0})
            e["count"] += 1
        return sorted(out.values(), key=lambda e: (e.get("cmc") or 0, e["name"]))

    builds = []
    for b in deckbuilder.build(taken, model, targets, rule=config.fitted(set_code, "splash"), three=config.fitted(set_code, "three")):
        deck = grouped(b["spells"] + b["nonbasics"])
        for col, name in deckbuilder.BASIC.items():
            if b["basics"].get(name):
                deck.append({"name": name, "count": b["basics"][name], "cmc": 0, "mana_cost": "", "types": ["Basic", "Land"], "colors": [col]})
        side = grouped(b["sideboard"])
        creatures = sum(1 for c in b["spells"] if "Creature" in (c.get("types") or []))
        builds.append({
            "label": b["lane"] + (f" + {b['splash']}" if b.get("splash") else "") + (f" ({b['variant']})" if b["variant"] else ""), "variant": b["variant"],
            "rating": None, "record": None, "colors": list(b["lane"]), "breakdown": None, "stats": None,
            "note": "companion/advisor_v2/deckbuilder.py", "deck": deck, "sideboard": side, "arena_export": copy_deck(deck, side),
            **mana_identity(deck), "deck_wr": round(b["predicted"], 2), "creatures": creatures, "realistic": True,
        })
    return builds


MIN_CREATURES = 11  # top players' decks run 12-15 creatures in every FRA pair (tools/deck_targets.py)


def log_tail(path, size=2_000_000):
    """The last part of Arena's Player.log, as text ('' when unreadable)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode("utf-8", "replace")
    except (OSError, TypeError):
        return ""


def completed_picks(tail):
    """Card ids of the draft Arena last reported as Completed (all picks, the last one included)."""
    import json

    i = tail.rfind('\\"DraftStatus\\":\\"Completed')
    if i < 0:
        return []
    start = tail.rfind("\n", 0, i) + 1
    end = tail.find("\n", i)
    try:
        return json.loads(json.loads(tail[start:end if end > 0 else None])["Payload"]).get("PickedCards") or []
    except (ValueError, KeyError, TypeError):
        return []


def course_pool(tail):
    """(card ids, event name) of the latest draft event's pool in Arena's course list, written at every start."""
    import json

    i = tail.rfind('{"Courses":')
    if i < 0:
        return [], None
    end = tail.find("\n", i)
    try:
        courses = json.loads(tail[i:end if end > 0 else None])["Courses"]
    except (ValueError, KeyError, TypeError):
        return [], None
    drafts = [c for c in courses if "Draft" in (c.get("InternalEventName") or "") and c.get("CardPool")]
    # ponytail: latest by the deck's LastUpdated attribute; a missing one sorts first
    def updated(c):
        attrs = (c.get("CourseDeckSummary") or {}).get("Attributes") or []
        return next((a.get("value", "") for a in attrs if a.get("name") == "LastUpdated"), "")

    if not drafts:
        return [], None
    c = max(drafts, key=updated)
    return c["CardPool"], c["InternalEventName"]


def overlay_builds(builds, pool, top=3):
    """The best builds by predicted win rate, shaped for the overlay: cards, lands, and what to take out of a deck
    that holds the whole pool (Arena starts the draft deck with every pick)."""
    from collections import Counter

    out = []
    pool_counts = Counter(c.get("name") for c in pool if "Basic" not in (c.get("types") or []))
    for b in sorted(builds, key=lambda b: (not b.get("realistic", True), -(b.get("deck_wr") or 0)))[:top]:
        deck = b.get("deck") or []
        spells = sorted((c for c in deck if "Land" not in (c.get("types") or [])), key=lambda c: (c.get("cmc") or 0, c.get("name") or ""))
        lands = [c for c in deck if "Land" in (c.get("types") or [])]
        in_deck = Counter()
        for c in deck:
            in_deck[c.get("name")] += c.get("count", 1)
        remove = sorted(((n, k - in_deck.get(n, 0)) for n, k in pool_counts.items() if k > in_deck.get(n, 0)), key=lambda x: x[0])
        out.append({
            "label": b.get("label"), "main": b.get("main"), "splash": b.get("splash"), "deck_wr": b.get("deck_wr"),
            "creatures": b.get("creatures"), "realistic": b.get("realistic", True),
            "spells": [{"name": c.get("name"), "count": c.get("count", 1), "cmc": c.get("cmc"), "mana_cost": c.get("mana_cost") or ""} for c in spells],
            "lands": [{"name": c.get("name"), "count": c.get("count", 1)} for c in lands],
            "remove": [{"name": n, "count": k} for n, k in remove],
        })
    return out
