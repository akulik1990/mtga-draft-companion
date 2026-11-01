"""One draft session over upstream's scanner and its Tk-free background orchestrator."""

import logging
import queue

from src import constants
from src.card_logic import filter_options
from src.signals import SignalCalculator
from src.ui.orchestrator import DraftOrchestrator  # threading only, no Tk

from companion.advisors import rank_pack

log = logging.getLogger(__name__)


class DraftSession:
    def __init__(self, scanner, config):
        self.scanner = scanner
        self.config = config
        self.orchestrator = DraftOrchestrator(scanner, config, refresh_callback=None)
        self.status = "Waiting for draft..."

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
            changed = True

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
            draft_id = s.current_draft_id
            event_string = s.event_string

        signals = self._signals(metrics, history)
        deck_filter = filter_options(taken, self.config.settings.deck_filter, metrics, self.config)
        advisors = rank_pack(metrics, taken, pack, pick, pack_no, signals) if pack else {}

        return {
            "status": self.status,
            "event": {"set": event_set, "type": event_type, "label": event_string, "draft_id": draft_id},
            "pack_number": pack_no,
            "pick_number": pick,
            "dataset": self.config.card_data.latest_dataset,
            "deck_filter": deck_filter,
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
