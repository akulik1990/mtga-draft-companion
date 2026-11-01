"""One draft session over upstream's scanner and its Tk-free background orchestrator.

Everything here calls vendor functions; nothing reimplements their logic.
"""

import logging
import os
import queue
import random
from pathlib import Path

from src import constants
from src.card_logic import copy_deck, filter_options, format_win_rate, suggest_deck
from src.configuration import write_configuration
from src.dataset_updater import DatasetUpdater
from src.signals import SignalCalculator
from src.ui.orchestrator import DraftOrchestrator  # threading only, no Tk
from src.tier_list import TIER_FILE_PREFIX, TIER_FOLDER, TierList
from src.utils import normalize_color_string, retrieve_local_set_list

from companion.advisors import rank_pack
from companion.mock import Cards, MockDraft, sampled_drafts

log = logging.getLogger(__name__)

GRADED_FIELDS = tuple(constants.WIN_RATE_OPTIONS)  # gihwr, ohwr, gpwr, gnswr, gdwr
DECK_CARD_FIELDS = ("name", "count", "cmc", "mana_cost", "types", "colors", "image", "rarity")


class DraftSession:
    def __init__(self, scanner, config):
        self.scanner = scanner
        self.config = config
        self.orchestrator = DraftOrchestrator(scanner, config, refresh_callback=None)
        self.status = "Waiting for draft..."
        self.mock = None  # MockDraft while one is running

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
        for card in (*pack, *missing, *taken):
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

        from companion.advisor_v2.engine import DraftAdvisorV2 as V

        for key, on in changes.items():
            if key not in self.ENGINE_SWITCHES:
                raise ValueError(f"unknown engine switch {key!r}")
            attr = self.ENGINE_SWITCHES[key][0]
            if attr == "PREF_WEIGHT":
                # Learnt preferences replace the pick-order prior; switching them off brings the prior back.
                V.PREF_WEIGHT = V.PREF_ON_WEIGHT if on else 0.0
                V.ORDER_WEIGHT = 0.0 if on else V.ORDER_OFF_WEIGHT
            else:
                setattr(V, attr, bool(on))
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

    def build_decks(self) -> list[dict]:
        with self.scanner.lock:
            taken = self.scanner.retrieve_taken_cards()
            metrics = self.scanner.retrieve_set_metrics()
            _, event_type = self.scanner.retrieve_current_limited_event()
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
            })
        return builds


def _trim(card: dict) -> dict:
    out = {k: card.get(k) for k in DECK_CARD_FIELDS if k in card}
    out["gihwr"] = (card.get("deck_colors", {}).get("All Decks") or {}).get("gihwr")
    return out
