"""One draft session over upstream's scanner and its Tk-free background orchestrator.

Everything here calls vendor functions; nothing reimplements their logic.
"""

import logging
import os
import queue

from src import constants
from src.card_logic import copy_deck, filter_options, format_win_rate, suggest_deck
from src.configuration import write_configuration
from src.dataset_updater import DatasetUpdater
from src.signals import SignalCalculator
from src.ui.orchestrator import DraftOrchestrator  # threading only, no Tk
from src.utils import retrieve_local_set_list

from companion.advisors import rank_pack

log = logging.getLogger(__name__)

GRADED_FIELDS = ("gihwr", "ohwr", "gpwr")
DECK_CARD_FIELDS = ("name", "count", "cmc", "mana_cost", "types", "colors", "image", "rarity")


class DraftSession:
    def __init__(self, scanner, config):
        self.scanner = scanner
        self.config = config
        self.orchestrator = DraftOrchestrator(scanner, config, refresh_callback=None)
        self.status = "Waiting for draft..."

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

        signals = self._signals(metrics, history)
        deck_filter = filter_options(taken, self.config.settings.deck_filter, metrics, self.config)
        archetype = deck_filter[0] if deck_filter else constants.FILTER_OPTION_ALL_DECKS
        advisors = rank_pack(metrics, taken, pack, pick, pack_no) if pack else {}

        for card in (*pack, *missing, *taken):
            card["grades"] = self._grades(card, metrics, archetype)

        if self.status is None:
            self.status = f"Pack {pack_no} Pick {pick}" if pack_no > 0 else "Waiting for draft..."
        return {
            "status": self.status,
            "event": {"set": event_set, "type": event_type, "label": event_string, "draft_id": draft_id},
            "pack_number": pack_no,
            "pick_number": pick,
            "cards_per_pick": cards_per_pick,
            "dataset": self.config.card_data.latest_dataset,
            "archetype": archetype,
            "color_ratings": color_ratings,
            "signals": signals,
            "pack": pack,
            "missing": missing,
            "picked_from_pack": picked_now,
            "pool": taken,
            "advisors": advisors,
        }

    def _signals(self, metrics, history):
        calc = SignalCalculator(metrics)
        scores = {c: 0.0 for c in constants.CARD_COLORS}
        for entry in history:
            if entry["Pack"] == 2:  # upstream excludes pack 2 (other direction)
                continue
            cards = self.scanner.set_data.get_data_by_id(entry["Cards"])
            for color, value in calc.calculate_pack_signals(cards, entry["Pick"]).items():
                scores[color] += value
        return scores

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
        }

    def update_settings(self, changes: dict) -> dict:
        st = self.config.settings
        if "auto_sync_datasets" in changes:
            st.auto_sync_datasets = bool(changes["auto_sync_datasets"])
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
